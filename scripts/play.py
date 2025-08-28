# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 30
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 9)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 4
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 启用观测数据打印和保存功能
    env._is_play_mode = True
    print("\n🔍 Observation data logging enabled for play mode")
    print("📊 Data will be printed every 100 steps and saved every 1000 steps")
    print("💾 Save location: /home/xinyun/limx_rl/observation_logs/")
    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE")
    commands_val = to_torch([0.5, 0.0, 0, 0], device=env.device) if robot_type.startswith("PF")\
        else to_torch([1.0, 0.0, 0.0], device=env.device) if robot_type == "WF_TRON1A" else to_torch([1.5, 0.0, 0.0, 0.0, 0.0])#第一个数据1.5
    action_scale = env.cfg.control.action_scale_pos if robot_type == "WF_TRON1A"\
        else env.cfg.control.action_scale
    obs, obs_history, commands, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    # train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)
    
    # 打印policy和encoder的具体内容
    print("=== Policy and Encoder Details ===")
    print(f"Policy type: {type(policy)}")
    print(f"Policy: {policy}")
    print(f"Encoder type: {type(encoder)}")
    print(f"Encoder: {encoder}")

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    logger = Logger(env.dt)
    robot_index = 5  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    
    # obs_buf保存相关配置
    save_obs_interval = getattr(args, 'save_obs_interval', 200)  # 每 N 步保存一次obs_buf数据
    save_obs_envs = getattr(args, 'save_obs_envs', min(100, env.num_envs))  # 保存前 N 个环境的数据
    enable_obs_save = getattr(args, 'enable_obs_save', True)  # 是否启用obs_buf保存
    obs_save_counter = 0
    
    if enable_obs_save:
        print(f"\n💾 obs_buf保存已启用:")
        print(f"  - 保存间隔: 每 {save_obs_interval} 步")
        print(f"  - 保存环境数: {save_obs_envs} 个")
        print(f"  - 保存目录: /home/xinyun/limx_rl/observation_logs/")
    else:
        print("\n⚠️ obs_buf保存已禁用")
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    for i in range(10 * int(env.max_episode_length)):
        est = encoder(obs_history)
        actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        
        # 打印est、actions和env.commands的内容
        if i % 50 == 0:  # 每50步打印一次，避免输出过多
            print(f"\n=== Step {i} Debug Info ===")
            print(f"est (encoder output) shape: {est.shape}")
            print(f"est content: {est}")
            print(f"actions shape: {actions.shape}")
            print(f"actions content: {actions}")
            print(f"commands shape: {commands.shape}")
            print(f"commands content: {commands}")
            print(f"obs_history shape: {obs_history.shape}")
            print(f"obs shape: {obs.shape}")
            print(f"concatenated input shape: {torch.cat((est, obs, commands), dim=-1).shape}")
            
            # 打印机械臂相对位置信息
            if hasattr(env, 'arm_ee_pos') and hasattr(env, 'arm_ee_target_pos') and hasattr(env, 'arm_ee_init_pos'):
                print(f"\n=== 机械臂位置信息 ===")
                print(f"当前末端位置: {env.arm_ee_pos[0].cpu().numpy()}")
                print(f"目标位置: {env.arm_ee_target_pos[0].cpu().numpy()}")
                print(f"初始位置: {env.arm_ee_init_pos[0].cpu().numpy()}")
                
                # 计算相对位置
                relative_to_target = env.arm_ee_pos[0] - env.arm_ee_target_pos[0]
                relative_to_init = env.arm_ee_pos[0] - env.arm_ee_init_pos[0]
                target_relative_to_init = env.arm_ee_target_pos[0] - env.arm_ee_init_pos[0]
                
                print(f"相对于目标的位置: {relative_to_target.cpu().numpy()}")
                print(f"相对于初始的位置: {relative_to_init.cpu().numpy()}")
                print(f"目标相对于初始的位置: {target_relative_to_init.cpu().numpy()}")
                
                # 计算距离
                distance_to_target = torch.norm(relative_to_target).item()
                distance_from_init = torch.norm(relative_to_init).item()
                
                print(f"到目标的距离: {distance_to_target:.4f}m")
                print(f"离初始的距离: {distance_from_init:.4f}m")

        env.commands[:, :] = commands_val
        
        # 打印env.commands内容
        if i % 50 == 0:
            print(f"env.commands shape: {env.commands.shape}")
            print(f"env.commands content: {env.commands}")
            print(f"commands_val: {commands_val}")

        obs, rews, dones, infos, obs_history, commands, _ = env.step(
            actions.detach()
        )
        
        # 保存obs_buf数据
        if enable_obs_save:
            obs_save_counter += 1
            if obs_save_counter % save_obs_interval == 0:
                print(f"\n💾 正在保存obs_buf数据 (Step {i+1})...")
                # 保存前 N 个环境的数据
                env_indices = list(range(save_obs_envs))
                # # 保存后 N 个环境的数据
                # start_env = max(0, env.num_envs - save_obs_envs)
                # env_indices = list(range(start_env, env.num_envs))
                
                # 使用环境的保存方法
                if hasattr(env, 'save_obs_buf_to_json'):
                    from datetime import datetime
                    
                    # 生成文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = "/home/xinyun/limx_rl/observation_logs"
                    os.makedirs(save_dir, exist_ok=True)
                    filename = f"play_obs_buf_step{i+1}_{timestamp}.json"
                    filepath = os.path.join(save_dir, filename)
                    
                    # 保存数据
                    saved_path = env.save_obs_buf_to_json(
                        filepath=filepath,
                        env_indices=env_indices,
                        include_metadata=True
                    )
                    
                    if saved_path:
                        print(f"✅ 成功保存obs_buf数据到: {saved_path}")
                        print(f"📊 保存了 {save_obs_envs} 个环境的数据")
                    else:
                        print("❌ 保存obs_buf数据失败")
                else:
                    print("⚠️ 环境不支持obs_buf保存功能")
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            # env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                    "dof_pos": (
                        env.dof_pos[robot_index, joint_index]
                        - env.raw_default_dof_pos[joint_index]
                    ).item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "power": torch.sum(env.power[robot_index, :]).item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            # print(torch.sum(env.power[robot_index, :]).item())
            if est != None:
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
