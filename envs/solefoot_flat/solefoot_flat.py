import torch
from torch import nn, Tensor
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float, CubicSpline
)
import math
from time import time
from warnings import WarningMessage
import numpy as np
import os
from typing import Tuple, Dict
import random

from .solefoot_flat_config import BipedCfgSF
from legged_gym.algorithm.mlp_encoder import MLP_Encoder
from legged_gym.algorithm.actor_critic import ActorCritic

class BipedSF(BaseTask):
    def __init__(
        self, cfg: BipedCfgSF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        ###### !!! ADD_BEGIN [初始化Encoder/ActorCritic模块]
        # 1. 初始化网络
        obs_history_dim = self.num_obs * self.obs_history_length
        encoder_output_dim = 64   # 可调
        self.encoder = MLP_Encoder(
            num_input_dim=obs_history_dim,
            num_output_dim=encoder_output_dim,
            hidden_dims=[256, 256],
            activation="elu",
            orthogonal_init=True
        ).to(self.device)

        actor_input_dim = self.num_obs 
        critic_input_dim = self.num_obs 

        self.actor_critic = ActorCritic(
            num_actor_obs=actor_input_dim,
            num_critic_obs=critic_input_dim,
            num_actions=self.num_actions,  # 你的动作维度
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
            activation="elu",
            orthogonal_init=True
        ).to(self.device)
        # 2. 编码器权重加载逻辑
        self._load_encoder_weights()
        
        # 3. 参数冻结/微调配置
        # 编码器参数配置（从配置文件读取）
        if hasattr(self.cfg, 'encoder_loading'):
            freeze_encoder = getattr(self.cfg.encoder_loading, 'freeze_encoder', True)
        else:
            freeze_encoder = True  # 如果没有配置文件，默认冻结
        
        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder
        
        if freeze_encoder:
            print("[INFO] 编码器参数已冻结，不会更新")
        else:
            print("[INFO] 编码器参数已解冻，将使用小学习率进行微调")
        
        # 策略网络参数冻结（通常不冻结，让其适应新任务）
        for param in self.actor_critic.parameters():
            param.requires_grad = True
        ###### !!! ADD_END [初始化Encoder/ActorCritic模块]

    def _load_encoder_weights(self):
        """加载预训练的编码器权重"""
        # 检查配置是否存在
        if not hasattr(self.cfg, 'encoder_loading'):
            return
            
        # 检查是否启用权重加载
        if not self.cfg.encoder_loading.load_stage1:
            if self.cfg.encoder_loading.verbose:
                print("[INFO] 编码器权重加载未启用")
            return
        
        # 获取配置参数
        stage1_path = self.cfg.encoder_loading.stage1_path
        checkpoint_name = self.cfg.encoder_loading.stage1_checkpoint
        load_actor_critic = self.cfg.encoder_loading.load_actor_critic
        verbose = self.cfg.encoder_loading.verbose
        
        # 封装打印函数，减少重复的verbose检查
        def log_info(message):
            if verbose:
                print(message)
        
        # 构建检查点文件路径
        # 展开用户目录路径（处理~符号）
        expanded_path = os.path.expanduser(stage1_path)
        checkpoint_path = os.path.join(expanded_path, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            log_info(f"[WARNING] 检查点文件不存在: {checkpoint_path}")
            return
        
        try:
            # 加载检查点
            log_info(f"[INFO] 正在加载检查点: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 尝试加载编码器权重
            encoder_loaded = False
            
            # 方法1: 尝试直接加载独立的编码器权重
            if 'encoder_state_dict' in checkpoint:
                try:
                    encoder_state_dict = checkpoint['encoder_state_dict']
                    
                    # 智能键名映射：检查当前编码器的键名格式
                    current_encoder_keys = list(self.encoder.state_dict().keys())
                    checkpoint_keys = list(encoder_state_dict.keys())
                    
                    log_info(f"[DEBUG] 当前编码器键名示例: {current_encoder_keys[:3]}")
                    log_info(f"[DEBUG] 检查点键名示例: {checkpoint_keys[:3]}")
                    
                    cleaned_encoder_weights = {}
                    
                    # 判断是否需要添加或移除 'encoder.' 前缀
                    current_has_prefix = any(key.startswith('encoder.') for key in current_encoder_keys)
                    checkpoint_has_prefix = any(key.startswith('encoder.') for key in checkpoint_keys)
                    
                    for key, value in encoder_state_dict.items():
                        if current_has_prefix and not checkpoint_has_prefix:
                            # 当前模型期望有前缀，但检查点没有 -> 添加前缀
                            new_key = f'encoder.{key}'
                            cleaned_encoder_weights[new_key] = value
                        elif not current_has_prefix and checkpoint_has_prefix:
                            # 当前模型期望没有前缀，但检查点有 -> 移除前缀
                            new_key = key.replace('encoder.', '')
                            cleaned_encoder_weights[new_key] = value
                        else:
                            # 格式一致，直接使用
                            cleaned_encoder_weights[key] = value
                    
                    missing_keys, unexpected_keys = self.encoder.load_state_dict(
                        cleaned_encoder_weights, strict=False
                    )
                    encoder_loaded = True
                    log_info(f"[SUCCESS] 成功加载独立编码器权重")
                    log_info(f"[INFO] 加载了 {len(cleaned_encoder_weights)} 个编码器参数")
                    if missing_keys:
                        log_info(f"[INFO] 缺失的键: {missing_keys}")
                    if unexpected_keys:
                        log_info(f"[INFO] 意外的键: {unexpected_keys}")
                except Exception as e:
                    log_info(f"[WARNING] 加载独立编码器权重失败: {e}")
            
            # 方法2: 如果没有独立的编码器权重，尝试从完整模型中提取
            if not encoder_loaded and 'model_state_dict' in checkpoint:
                try:
                    # 提取编码器相关的权重
                    encoder_weights = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith('encoder.'):
                            # 移除 'encoder.' 前缀
                            new_key = key.replace('encoder.', '')
                            encoder_weights[new_key] = value
                    
                    if encoder_weights:
                        missing_keys, unexpected_keys = self.encoder.load_state_dict(
                            encoder_weights, strict=False
                        )
                        encoder_loaded = True
                        log_info(f"[SUCCESS] 成功从完整模型中提取并加载编码器权重")
                        log_info(f"[INFO] 加载了 {len(encoder_weights)} 个编码器参数")
                        if missing_keys:
                            log_info(f"[INFO] 缺失的键: {missing_keys}")
                        if unexpected_keys:
                            log_info(f"[INFO] 意外的键: {unexpected_keys}")
                    else:
                        log_info(f"[WARNING] 在完整模型中未找到编码器权重")
                except Exception as e:
                    log_info(f"[WARNING] 从完整模型提取编码器权重失败: {e}")
            
            # 可选：加载策略网络权重
            if load_actor_critic and encoder_loaded:
                actor_critic_loaded = False
                
                # 尝试加载策略网络权重
                if 'actor_critic_state_dict' in checkpoint:
                    try:
                        missing_keys, unexpected_keys = self.actor_critic.load_state_dict(
                            checkpoint['actor_critic_state_dict'], strict=False
                        )
                        actor_critic_loaded = True
                        log_info(f"[SUCCESS] 成功加载策略网络权重")
                    except Exception as e:
                        log_info(f"[WARNING] 加载策略网络权重失败: {e}")
                
                # 从完整模型中提取策略网络权重
                if not actor_critic_loaded and 'model_state_dict' in checkpoint:
                    try:
                        actor_critic_weights = {}
                        for key, value in checkpoint['model_state_dict'].items():
                            if key.startswith('actor_critic.') or key.startswith('policy.'):
                                new_key = key.replace('actor_critic.', '').replace('policy.', '')
                                actor_critic_weights[new_key] = value
                        
                        if actor_critic_weights:
                            missing_keys, unexpected_keys = self.actor_critic.load_state_dict(
                                actor_critic_weights, strict=False
                            )
                            log_info(f"[SUCCESS] 成功从完整模型中提取并加载策略网络权重")
                            log_info(f"[INFO] 加载了 {len(actor_critic_weights)} 个策略网络参数")
                    except Exception as e:
                        log_info(f"[WARNING] 从完整模型提取策略网络权重失败: {e}")
            
            if not encoder_loaded:
                log_info(f"[ERROR] 未能加载任何编码器权重")
                log_info(f"[INFO] 检查点中可用的键: {list(checkpoint.keys())}")
            
        except Exception as e:
            log_info(f"[ERROR] 加载检查点时发生错误: {e}")

    def save_obs_buf_to_json(self, filepath=None, env_indices=None, include_metadata=True):
        """
        将obs_buf整体数据保存为JSON文件
        
        Args:
            filepath (str, optional): 保存文件路径，如果为None则自动生成
            env_indices (list, optional): 要保存的环境索引，如果为None则保存所有环境
            include_metadata (bool): 是否包含元数据信息
        
        Returns:
            str: 保存的文件路径
        """
        import json
        from datetime import datetime
        
        # 确定要保存的环境索引
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        elif isinstance(env_indices, int):
            env_indices = [env_indices]
        
        # 生成文件路径
        if filepath is None:
            save_dir = "/home/xinyun/limx_rl/observation_logs"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"obs_buf_complete_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)
        
        # 准备保存的数据
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "num_envs_saved": len(env_indices),
            "obs_buf_shape": list(self.obs_buf.shape),
            "device": str(self.device),
        }
        
        # 添加元数据
        if include_metadata:
            save_data["metadata"] = {
                "num_envs_total": self.num_envs,
                "num_obs": self.num_obs,
                "obs_history_length": self.obs_history_length,
                "episode_length_buf": self.episode_length_buf[env_indices].cpu().numpy().tolist() if hasattr(self, 'episode_length_buf') else None,
                "dt": self.dt,
                "cfg_info": {
                    "env_spacing": self.cfg.env.env_spacing,
                    "episode_length_s": self.cfg.env.episode_length_s,
                    "control_type": self.cfg.control.control_type,
                    "action_scale": self.cfg.control.action_scale
                }
            }
        
        # 保存obs_buf数据
        obs_data = []
        for i, env_idx in enumerate(env_indices):
            env_data = {
                "env_index": env_idx,
                "obs_buf": self.obs_buf[env_idx].cpu().numpy().tolist(),
            }
            
            # 添加其他相关观测数据
            if hasattr(self, 'critic_obs_buf'):
                env_data["critic_obs_buf"] = self.critic_obs_buf[env_idx].cpu().numpy().tolist()
            
            if hasattr(self, 'obs_history'):
                env_data["obs_history"] = self.obs_history[env_idx].cpu().numpy().tolist()
            
            if hasattr(self, 'encoder_feature'):
                env_data["encoder_feature"] = self.encoder_feature[env_idx].cpu().numpy().tolist()
            
            # 添加当前状态信息
            if hasattr(self, 'base_lin_vel'):
                env_data["base_lin_vel"] = self.base_lin_vel[env_idx].cpu().numpy().tolist()
            
            if hasattr(self, 'base_ang_vel'):
                env_data["base_ang_vel"] = self.base_ang_vel[env_idx].cpu().numpy().tolist()
            
            if hasattr(self, 'dof_pos'):
                env_data["dof_pos"] = self.dof_pos[env_idx].cpu().numpy().tolist()
            
            if hasattr(self, 'dof_vel'):
                env_data["dof_vel"] = self.dof_vel[env_idx].cpu().numpy().tolist()
            
            if hasattr(self, 'actions'):
                env_data["actions"] = self.actions[env_idx].cpu().numpy().tolist()
            
            if hasattr(self, 'commands'):
                env_data["commands"] = self.commands[env_idx].cpu().numpy().tolist()
            
            obs_data.append(env_data)
        
        save_data["observations"] = obs_data
        
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ obs_buf数据已保存到: {filepath}")
            print(f"📊 保存了 {len(env_indices)} 个环境的观测数据")
            print(f"📏 obs_buf形状: {self.obs_buf.shape}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ 保存obs_buf数据时发生错误: {e}")
            return None
    
    def save_obs_buf_batch(self, save_interval=1000, max_envs_per_file=100):
        """
        批量保存obs_buf数据，适用于大规模环境
        
        Args:
            save_interval (int): 每隔多少步保存一次
            max_envs_per_file (int): 每个文件最多保存多少个环境的数据
        """
        if not hasattr(self, '_obs_save_counter'):
            self._obs_save_counter = 0
        
        self._obs_save_counter += 1
        
        if self._obs_save_counter % save_interval == 0:
            # 分批保存环境数据
            num_batches = (self.num_envs + max_envs_per_file - 1) // max_envs_per_file
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_envs_per_file
                end_idx = min((batch_idx + 1) * max_envs_per_file, self.num_envs)
                env_indices = list(range(start_idx, end_idx))
                
                # 生成批次文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = "/home/xinyun/limx_rl/observation_logs"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"obs_buf_batch_{batch_idx+1}of{num_batches}_{timestamp}.json"
                filepath = os.path.join(save_dir, filename)
                
                self.save_obs_buf_to_json(filepath, env_indices, include_metadata=(batch_idx == 0))
            
            print(f"🎯 完成第 {self._obs_save_counter} 步的批量保存，共 {num_batches} 个文件")

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)

        self.base_lin_acc = (self.base_lin_vel - self.last_base_lin_vel) / self.dt
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, self.base_lin_acc)

        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.dof_pos_int += (self.dof_pos - self.raw_default_dof_pos) * self.dt
        self.power = torch.abs(self.torques * self.dof_vel)

        # self.dof_jerk = (self.last_dof_acc - self.dof_acc) / self.dt

        self.compute_foot_state()
        
        # 计算机械臂状态
        self.compute_arm_state()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        self._post_physics_step_callback()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # self.last_dof_acc[:] = self.dof_acc[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_foot_positions[:] = self.foot_positions[:]

    def compute_foot_state(self):
        super().compute_foot_state()
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
    
    def compute_arm_state(self):
        """计算机械臂相关状态"""
        # 计算机械臂关节速度 (从dof_vel中提取机械臂关节)
        if len(self.arm_joint_indices) > 0:
            self.arm_vel[:] = self.dof_vel[:, self.arm_joint_indices]
            
            # 计算机械臂功率 (扭矩 * 角速度)
            self.arm_power[:] = torch.abs(self.torques[:, self.arm_joint_indices] * self.dof_vel[:, self.arm_joint_indices])
            
            # 计算机械臂姿态正则化 (当前位置与初始位置的差异)
            self.arm_pose_reg[:] = self.dof_pos[:, self.arm_joint_indices] - self.arm_init_pos.unsqueeze(0)
        
        # 计算机械臂末端执行器位置
        # 根据URDF分析，机械臂的末端执行器是link6
        try:
            if 'link6' in self.rigid_body_indices:
                ee_body_idx = self.rigid_body_indices['link6']
                self.arm_ee_pos[:] = self.rigid_body_states[:, ee_body_idx, :3]
            else:
                raise KeyError("link6 not found in rigid_body_indices")
                
        except (KeyError, AttributeError):
            # 如果无法获取link6位置，使用基座位置加偏移作为近似
            self.arm_ee_pos[:] = self.base_position + torch.tensor([0.3, 0.0, 0.5], device=self.device)
        
        # 生成机械臂末端执行器轨迹
        self._generate_des_arm_ee_ref()

    def compute_observations(self):
        """计算观测"""
        self.obs_buf, self.critic_obs_buf = self.compute_self_observations()
        
        # NaN检测函数
        def debug_nan_check(tensor, name):
            if torch.any(torch.isnan(tensor)):
                print(f"NaN detected in {name}!")
                print(f"Shape: {tensor.shape}")
                nan_positions = torch.isnan(tensor).nonzero()
                print(f"NaN positions (first 10): {nan_positions[:10]}")
                print(f"NaN count: {torch.sum(torch.isnan(tensor))}")
                return True
            return False
        
        # 检查初始观测数据
        debug_nan_check(self.obs_buf, "obs_buf after compute_self_observations")
        debug_nan_check(self.critic_obs_buf, "critic_obs_buf")

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs :], self.obs_buf), dim=-1
        )
        
        # 检查历史观测数据
        debug_nan_check(self.obs_history, "obs_history")
        
        ###### !!! ADD_BEGIN [观测经编码器处理]
        obs_history_for_encoder = self.obs_history  # [num_envs, num_obs * obs_history_length]
        
        # 检查编码器输入
        debug_nan_check(obs_history_for_encoder, "obs_history_for_encoder")
        
        with torch.no_grad():
            self.encoder_feature = self.encoder(obs_history_for_encoder)
            
        # 检查编码器输出
        debug_nan_check(self.encoder_feature, "encoder_feature")
        ###### !!! ADD_END [观测经编码器处理]
            
        # print("[维度诊断]")
        # print(f"原始观测维度: {self.obs_buf.shape}")
        # print(f"历史观测维度: {obs_history_for_encoder.shape}")
        # print(f"编码特征维度: {self.encoder_feature.shape}")
        # print(f"Actor输入维度: {self.obs_buf.shape}")


    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:14] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[14:22] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[22:] = 0.0  # previous actions
        return noise_vec

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.rigid_body_names = body_names  # 保存刚体名称到类属性
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        contact_names = []
        if hasattr(self.cfg.asset, "contact_name"):
            contact_names = [s for s in body_names if self.cfg.asset.contact_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.friction_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.restitution_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.whole_body_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_com = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)

            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        self.contact_indices = torch.zeros(
            len(contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )
        for i in range(len(contact_names)):
            self.contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], contact_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )
    
    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._check_walk_stability(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # # 如果配置了机械臂随机化，则进行随机化
        # if hasattr(self.cfg.domain_rand, 'randomize_arm_position') and self.cfg.domain_rand.randomize_arm_position:
        #     self._randomize_arm_pose(env_ids) 


        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_self_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        
        # 记录机械臂末端执行器的初始位置（重置后）
        # 需要先刷新机械臂状态以获取正确的末端位置
        self.compute_arm_state()
        # 保存重置环境的机械臂末端初始位置
        self.arm_ee_init_pos[env_ids] = self.arm_ee_pos[env_ids].clone()
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        n = env_ids.size(0)
        indices = torch.randperm(n)

        half_size = n // 2
        half_indices = indices[:half_size]
        remaining_indices = indices[half_size:]

        half_list = env_ids[half_indices]
        remaining_list = env_ids[remaining_indices]
        self.dof_pos[half_list] = self.default_dof_pos[half_list, :] + torch_rand_float(
            -0.5, 0.5, (len(half_list), self.num_dof), device=self.device
        )
        self.dof_pos[remaining_list] = self.init_stand_dof_pos[remaining_list, :] + torch_rand_float(
            -0.5, 0.5, (len(remaining_list), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """   
        
        full_actions = torch.zeros((actions.shape[0], self.num_dof), device=actions.device)
        
        # 下半身关节名在dof_names里的顺序，假如为[0,1,2,3,4,5,6,7]，可确认
        # 或直接用下半身关节名字定位
        leg_dof_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint"
        ]
        
        # 对应的索引
        leg_indices = [self.dof_names.index(name) for name in leg_dof_names]
        
        # 给下半身的关节分配 actions
        full_actions[:, leg_indices] = actions[:, :8]  # 只给下半身动作分配前8个

        # 给机械臂的6个关节分配剩余的动作
        arm_indices = list(range(8, 14))  # 机械臂对应的6个关节的索引
        full_actions[:, arm_indices] = actions[:, 8:]  # 给机械臂分配剩余的6个动作
        
        self._action_clip(full_actions)
        
        # step physics and render each frame
        self.render()
        self.pre_physics_step()

        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (full_actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :5] * self.commands_scale,  # 5 commands
            self.critic_obs_buf
        )


    def compute_self_observations(self):
        # note that observation noise need to modified accordingly !!!
        # print('base_ang_vel:', self.base_ang_vel.shape)
        # print('projected_gravity:', self.projected_gravity.shape)
        # print('dof_pos:', self.dof_pos.shape)
        # print('default_dof_pos:', self.default_dof_pos.shape)
        # print('dof_vel:', self.dof_vel.shape)
        # print('actions:', self.actions.shape)
        # print('clock_inputs_sin:', self.clock_inputs_sin.view(self.num_envs, 1).shape)
        # print('clock_inputs_cos:', self.clock_inputs_cos.view(self.num_envs, 1).shape)
        # print('gaits:', self.gaits.shape)

        # 计算各个观测分量
        base_ang_vel_scaled = self.base_ang_vel * self.obs_scales.ang_vel
        projected_gravity = self.projected_gravity
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_scaled = self.dof_vel * self.obs_scales.dof_vel
        actions = self.actions
        clock_sin = self.clock_inputs_sin.view(self.num_envs, 1)
        clock_cos = self.clock_inputs_cos.view(self.num_envs, 1)
        gaits = self.gaits
        
        # 在play.py运行时打印和保存观测数据
        if hasattr(self, '_is_play_mode') and self._is_play_mode:
            # 每100步打印一次，避免输出过多
            if not hasattr(self, '_obs_print_counter'):
                self._obs_print_counter = 0
                self._obs_data_log = []
            
            self._obs_print_counter += 1
            
            if self._obs_print_counter % 100 == 0:
                env_idx = 0  # 只打印第一个环境的数据
                # print(f"\n=== Observation Components (Step {self._obs_print_counter}) ===")
                # print(f"base_ang_vel_scaled: {base_ang_vel_scaled[env_idx].cpu().numpy()}")
                # print(f"projected_gravity: {projected_gravity[env_idx].cpu().numpy()}")
                # print(f"dof_pos_scaled: {dof_pos_scaled[env_idx].cpu().numpy()}")
                # print(f"dof_vel_scaled: {dof_vel_scaled[env_idx].cpu().numpy()}")
                # print(f"actions: {actions[env_idx].cpu().numpy()}")
                # print(f"clock_sin: {clock_sin[env_idx].cpu().numpy()}")
                # print(f"clock_cos: {clock_cos[env_idx].cpu().numpy()}")
                # print(f"gaits: {gaits[env_idx].cpu().numpy()}")
                
                # 保存数据到日志
                obs_data = {
                    'step': self._obs_print_counter,
                    'base_ang_vel_scaled': base_ang_vel_scaled[env_idx].cpu().numpy().tolist(),
                    'projected_gravity': projected_gravity[env_idx].cpu().numpy().tolist(),
                    'dof_pos_scaled': dof_pos_scaled[env_idx].cpu().numpy().tolist(),
                    'dof_vel_scaled': dof_vel_scaled[env_idx].cpu().numpy().tolist(),
                    'actions': actions[env_idx].cpu().numpy().tolist(),
                    'clock_sin': clock_sin[env_idx].cpu().numpy().tolist(),
                    'clock_cos': clock_cos[env_idx].cpu().numpy().tolist(),
                    'gaits': gaits[env_idx].cpu().numpy().tolist()
                }
                self._obs_data_log.append(obs_data)
                
                # 每1000步保存一次数据到文件
                if self._obs_print_counter % 1000 == 0:
                    import json
                    import os
                    from datetime import datetime
                    
                    # 创建保存目录
                    save_dir = "/home/xinyun/limx_rl/observation_logs"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 生成文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"obs_data_{timestamp}.json"
                    filepath = os.path.join(save_dir, filename)
                    
                    # 保存数据
                    with open(filepath, 'w') as f:
                        json.dump(self._obs_data_log, f, indent=2)
                    
                    print(f"\n💾 Observation data saved to: {filepath}")
                    print(f"📊 Total logged steps: {len(self._obs_data_log)}")

        obs_buf = torch.cat(
            (
                base_ang_vel_scaled,
                projected_gravity,
                dof_pos_scaled,
                dof_vel_scaled,
                actions,
                clock_sin,
                clock_cos,
                gaits,
            ),
            dim=-1,
        )
        # print("obs_buf.shape:", obs_buf.shape)
        # compute critic_obs_buf
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf

    def get_observations(self):
        return (
            self.obs_buf,
            self.obs_history,
            self.commands[:, :5] * self.commands_scale, # 5 commands
            self.critic_obs_buf
        )


    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                    self.episode_length_buf
                    % int(self.cfg.commands.resampling_time / self.dt)
                    == 0
            )
                .nonzero(as_tuple=False)
                .flatten()
        )
        self._resample_commands(env_ids, False)
        self._resample_gaits(env_ids)
        self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 1.0 * wrap_to_pi(self.commands[:, 5] - heading)

        self._resample_zero_commands(env_ids)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _step_contact_targets(self):
        super()._step_contact_targets()
        self._generate_des_ee_ref()

    def _generate_des_ee_ref(self):
        frequencies = self.gaits[:, 0]
        mask_0 = (self.gait_indices < 0.25) & (self.gait_indices >= 0.0)  # lift up
        mask_1 = (self.gait_indices < 0.5) & (self.gait_indices >= 0.25)  # touch down
        mask_2 = (self.gait_indices < 0.75) & (self.gait_indices >= 0.5)  # lift up
        mask_3 = (self.gait_indices <= 1.0) & (self.gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.cfg.gait.touch_down_vel
        swing_end_vel[mask_3] = self.cfg.gait.touch_down_vel

        # generate desire foot z trajectory
        swing_height = self.gaits[:, 3]
        # self.des_foot_height = 0.5 * swing_height * (1 - torch.cos(4 * np.pi * self.gait_indices))
        # self.des_foot_velocity_z = 2 * np.pi * swing_height * frequencies * torch.sin(
        #     4 * np.pi * self.gait_indices)

        start = {'time': swing_start_time, 'position': swing_start_pos * swing_height,
                 'velocity': torch.zeros(self.num_envs, device=self.device)}
        end = {'time': swing_end_time, 'position': swing_end_pos * swing_height,
               'velocity': swing_end_vel}
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(self.gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(self.gait_indices / frequencies)

    def _generate_des_arm_ee_ref(self):
        """生成机械臂末端执行器的期望轨迹（简化版）"""
        # 相对于机械臂末端执行器初始位置的固定偏移
        target_offset = torch.tensor([-0.15, 0.0, 0.1], device=self.device)  # X方向 -0.15m，Z方向 0.1m
        
        # 设置机械臂末端执行器目标位置（世界坐标系）
        # 目标 = 初始位置 + 偏移
        self.arm_ee_target_pos[:] = self.arm_ee_init_pos + target_offset.unsqueeze(0)

    def _resample_gaits(self, env_ids):
        super()._resample_gaits(env_ids)
        self._resample_stand_still_gait_commands(env_ids)

    def _check_walk_stability(self, env_ids):
        if len(env_ids) != 0:
            self.mean_episode_len = torch.mean(self.episode_length_buf[env_ids].float(), dim=0).cpu().item()
        if self.mean_episode_len > 950:
            self.stable_episode_length_count += 1
            # print("Stable Episode Length:{}, count:{}.".format(self.mean_episode_len, self.stable_episode_length_count))
        else:
            self.stable_episode_length_count = 0

    def _resample_commands(self, env_ids, is_start=True):
        """Randommly select commands of some environments

                Args:
                    env_ids (List[int]): Environments ids for which new commands are needed
                """
        self.commands[env_ids, 0] = (self.command_ranges["lin_vel_x"][env_ids, 1]
                                     - self.command_ranges["lin_vel_x"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["lin_vel_x"][env_ids, 0]
        self.commands[env_ids, 1] = (self.command_ranges["lin_vel_y"][env_ids, 1]
                                     - self.command_ranges["lin_vel_y"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["lin_vel_y"][env_ids, 0]
        self.commands[env_ids, 2] = (self.command_ranges["ang_vel_yaw"][env_ids, 1]
                                     - self.command_ranges["ang_vel_yaw"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["ang_vel_yaw"][env_ids, 0]
        self.commands[env_ids, 3] = (self.command_ranges["base_height"][env_ids, 1]
                                     - self.command_ranges["base_height"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["base_height"][env_ids, 0]

        self._resample_stand_still_commands(env_ids, is_start)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 5] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _resample_zero_commands(self, env_ids):
        thresh = 0.25
        indices_to_update = env_ids[(self.commands[env_ids, 0] < thresh) & (self.commands[env_ids, 0] > -thresh)]
        self.commands[indices_to_update, :3] = 0.0

    def _resample_stand_still_commands(self, env_ids, is_start=True):
        if (not self.walk_stability) and self.stable_episode_length_count >= 10:
            self.walk_stability = True
            self.stable_episode_length_count = 0
        if self.walk_stability and (not self.stand_still_stability) and self.stable_episode_length_count >= 10:
            self.stand_still_stability = True
        if self.walk_stability and (not self.stand_still_stability):
            if not is_start:
                indices_to_update = env_ids[self.commands[env_ids, 4] == 0]
                self.commands[indices_to_update, 4] = (self.command_ranges["stand_still"][indices_to_update, 1]
                                                       - self.command_ranges["stand_still"][indices_to_update, 0]) \
                                                      * torch.randint(0, 2, (len(indices_to_update),),
                                                                      device=self.device) \
                                                      + self.command_ranges["stand_still"][indices_to_update, 0]
                indices_to_update1 = indices_to_update[self.commands[indices_to_update, 4] == 1]
                self.commands[indices_to_update1, :3] = 0.0
            else:
                self.commands[env_ids, 4] = 0
        elif self.walk_stability and self.stand_still_stability:
            self.commands[env_ids, 4] = (self.command_ranges["stand_still"][env_ids, 1]
                                         - self.command_ranges["stand_still"][env_ids, 0]) \
                                        * torch.randint(0, 2, (len(env_ids),), device=self.device) \
                                        + self.command_ranges["stand_still"][env_ids, 0]
            indices_to_update = env_ids[self.commands[env_ids, 4] == 1]
            self.commands[indices_to_update, :3] = 0.0
        else:
            self.commands[env_ids, 4] = 0

    def _resample_stand_still_gait_commands(self, env_ids):
        # indices_to_update = env_ids[self.commands[env_ids, 4] == 1]
        # self.gaits[indices_to_update, :] = 0.0
        pass

    def _resample_stand_still_gait_clock(self):
        indices_to_update = torch.nonzero(self.commands[:, 4] == 1).squeeze()
        gait_indices = self.gait_indices[indices_to_update]

        mask_0_5_to_0_55 = (gait_indices >= 0.5) & (gait_indices < 0.55)
        mask_0_0_to_0_05 = (gait_indices >= 0.0) & (gait_indices < 0.05)
        mask_0_95_to_1_0 = (gait_indices >= 0.95) & (gait_indices < 1.0)

        self.gait_indices[indices_to_update[mask_0_5_to_0_55]] = 0.5
        self.gait_indices[indices_to_update[mask_0_0_to_0_05]] = 0.0
        self.gait_indices[indices_to_update[mask_0_95_to_1_0]] = 0.0

        mask_else = ~(mask_0_5_to_0_55 | mask_0_0_to_0_05 | mask_0_95_to_1_0)
        self.commands[indices_to_update[mask_else], 4] = 0

    def _init_buffers(self):
        super()._init_buffers()
        self.foot_heights = torch.zeros_like(self.foot_positions[:, :, 2])
        self.last_base_lin_vel = self.base_lin_vel.clone()

        self.base_lin_acc = torch.zeros_like(self.base_lin_vel)
        self.variances_per_env = 0
        self.init_stand_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            if hasattr(self.cfg.init_state, "init_stand_joint_angles"):
                stand_angle = self.cfg.init_state.init_stand_joint_angles[name]
                self.init_stand_dof_pos[:, i] = stand_angle

        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, 1, 1],
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["base_height"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["base_height"][:] = torch.tensor(
            self.cfg.commands.ranges.base_height
        )
        self.command_ranges["stand_still"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["stand_still"][:] = torch.tensor(
            self.cfg.commands.ranges.stand_still
        )

        self.des_foot_height = torch.zeros(self.num_envs,
                                           dtype=torch.float,
                                           device=self.device, requires_grad=False, ) # TODO
        self.des_foot_velocity_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                               requires_grad=False, ) # TODO

        # 机械臂相关变量初始化
        # 机械臂末端执行器位置 (x, y, z)
        self.arm_ee_pos = torch.zeros(
            self.num_envs, 
            3,  # xyz坐标
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # 机械臂末端执行器目标位置 (x, y, z)
        self.arm_ee_target_pos = torch.zeros(
            self.num_envs, 
            3,  # xyz坐标
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # 机械臂末端执行器初始位置 (x, y, z) - 用于相对偏移计算
        self.arm_ee_init_pos = torch.zeros(
            self.num_envs, 
            3,  # xyz坐标
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # 机械臂功率 (6个关节)
        self.arm_power = torch.zeros(
            self.num_envs, 
            6,  # 机械臂6个关节
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # 机械臂关节速度 (6个关节)
        self.arm_vel = torch.zeros(
            self.num_envs, 
            6,  # 机械臂6个关节
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # 机械臂姿态正则化 (6个关节与初始位置的差异)
        self.arm_pose_reg = torch.zeros(
            self.num_envs, 
            6,  # 机械臂6个关节
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # 机械臂关节索引 (J1-J6对应的DOF索引)
        self.arm_joint_indices = []
        for i, name in enumerate(self.dof_names):
            if name.startswith('J'):  # J1, J2, J3, J4, J5, J6
                self.arm_joint_indices.append(i)
        self.arm_joint_indices = torch.tensor(self.arm_joint_indices, device=self.device, dtype=torch.long)
        
        # 机械臂初始关节位置 (用于姿态正则化)
        self.arm_init_pos = torch.zeros(6, dtype=torch.float, device=self.device)
        arm_joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        for i, joint_name in enumerate(arm_joint_names):
            if joint_name in self.cfg.init_state.default_joint_angles:
                self.arm_init_pos[i] = self.cfg.init_state.default_joint_angles[joint_name]

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        self.rwd_orientationPrev = self._reward_orientation()
        # self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_baseHeightPrev = self._reward_base_height()
        if "tracking_contacts_shaped_height" in self.reward_scales.keys():
            self.rwd_swingHeightPrev = self._reward_tracking_contacts_shaped_height()

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x) / self.cfg.rewards.tracking_sigma)
    
    # ----------------------rewards----------------------
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (
                    1
                    - torch.exp(
                        -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                    )
                )

        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (
                    1
                    - torch.exp(
                        -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                    )
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (1 - torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma)
                )
        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_tracking_contacts_shaped_height(self):
        foot_heights = self.foot_heights
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_height"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * torch.exp(
                    -(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma)
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * (
                        1 - torch.exp(-(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma)
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma))
        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        return torch.where(
            torch.logical_and(self.commands[:, 4] == 1, self.commands[:, 3] <= 0.3),
            torch.clip(torch.abs(self.cfg.rewards.min_feet_distance - feet_distance), 0, 1),
            torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1),
        )

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.025
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        return reward

    def _reward_power(self):
        # Penalize torques
        joint_array = [i for i in range(self.num_dof)]
        joint_array.remove(3)
        joint_array.remove(7)
        return torch.sum(torch.abs(self.torques[:, joint_array] * self.dof_vel[:, joint_array]), dim=1)

    def _reward_collision(self):
        reward = torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        # reward = torch.square(base_height - self.commands[:, 3])
        reward = torch.abs(base_height - self.cfg.rewards.base_height_target)
        # return torch.where(self.commands[:, 4] == 0, reward, reward * 1.5)
        return reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_ankle_torque_limits(self):
        torque_limit = torch.cat((self.torque_limits[3].view(1) * self.cfg.rewards.soft_torque_limit,
                                  self.torque_limits[7].view(1) * self.cfg.rewards.soft_torque_limit),
                                 dim=-1, )
        torque = torch.cat((self.torques[:, 3].view(self.num_envs, 1),
                            self.torques[:, 7].view(self.num_envs, 1)), dim=-1)
        return torch.sum(
            torch.pow(torque / torque_limit, 8),
            dim=1,
        )

    def _reward_relative_feet_height_tracking(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        feet_height_in_body_frame = base_height.view(self.num_envs, 1) - self.foot_heights
        reward = torch.exp(
            -torch.sum(
                torch.square(
                    feet_height_in_body_frame - self.commands[:, 3].view(self.num_envs, 1)
                ),
                dim=-1) / self.cfg.rewards.height_tracking_sigma
        )
        return torch.where(self.commands[:, 4] == 1, reward, 0)

    def _reward_zero_command_nominal_state(self):
        # Penalize the hip joint pos in zero command
        dof_pos = self.dof_pos - self.raw_default_dof_pos
        reward = torch.sum(
            torch.square(dof_pos[:, [1, 5]]), dim=1
        )
        return reward * torch.logical_and(torch.norm(self.commands[:, :3], dim=1) < 0.05, self.commands[:, 4] == 0)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_keep_ankle_pitch_zero_in_air(self):
        ankle_pitch = torch.abs(self.dof_pos[:, 3]) * ~self.contact_filt[:, 0] + torch.abs(
            self.dof_pos[:, 7]) * ~self.contact_filt[:, 1]
        return torch.exp(-torch.abs(ankle_pitch) / 0.2)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~(self.time_out_buf | self.edge_reset_buf)

    def _reward_fail(self):
        return self.fail_buf > 0

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        torque_limit = self.torque_limits * self.cfg.rewards.soft_torque_limit
        return torch.sum(
            torch.pow(self.torques / torque_limit, 8),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_stand_still(self):

        return torch.sum(self.foot_heights, dim=1) * (
            torch.norm(self.commands[:, :3], dim=1) < self.cfg.commands.min_norm
        )

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                self.contact_forces[:, self.feet_indices, 2]
                - self.base_mass.mean() * 9.8 / 2
            ).clip(min=0.0),
            dim=1,
        )
        
    # 机械臂奖励
    def _reward_arm_ee_distance(self):
        # 末端到目标点距离奖励
        ee_distance = torch.norm(self.arm_ee_pos - self.arm_ee_target_pos, dim=1)
        return torch.exp(-ee_distance / self.cfg.rewards.scales.arm_ee_distance)
    
    def _reward_arm_power(self):
        # 机械臂能耗惩罚
        avg_power = torch.abs(self.arm_power).mean(dim=1)  # 关键修改
        return torch.exp(-avg_power / self.cfg.rewards.scales.arm_power)
    
    def _reward_arm_vel(self):
        # 完整修复方案 (必须与arm_vel=0.5一起实施)
        # 1. 修改速度计算方式（关键！）
        vel = self.arm_vel.abs().amax(dim=1)  # 取最大关节速度而非范数
        
        # 2. 添加数值安全保护
        vel = vel.clamp(max=8.0)  # 超速保护
        
        # 3. 应用新参数
        return torch.exp(-vel / self.cfg.rewards.scales.arm_vel) * 1000  # 乘数放大
    
    def _reward_arm_pose_reg(self):
        # 机械臂保持初始姿态惩罚
        pose_reg = torch.norm(self.arm_pose_reg, dim=1)
        return torch.exp(-pose_reg / self.cfg.rewards.scales.arm_pose_reg)

