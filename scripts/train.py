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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import matplotlib.pyplot as plt


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )
    
    # 仅复用编码器权重的功能
    def load_encoder_only(ppo_runner, stage1_path, checkpoint_name="model_8000.pt"):
        """
        仅加载第一阶段编码器权重，策略网络保持随机初始化
        Args:
            ppo_runner: PPO运行器实例
            stage1_path: 第一阶段权重文件路径
            checkpoint_name: 权重文件名
        """
        import torch
        import os
        
        checkpoint_path = os.path.join(stage1_path, checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            print(f"\n=== Loading Encoder Weights Only ===")
            print(f"Loading from: {checkpoint_path}")
            
            try:
                # 加载权重文件
                checkpoint = torch.load(checkpoint_path, map_location=ppo_runner.device)
                
                # 仅加载编码器权重
                if 'encoder_state_dict' in checkpoint:
                    ppo_runner.alg.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                    print("✅ Successfully loaded encoder weights")
                else:
                    # 如果没有单独的encoder_state_dict，尝试从完整模型中提取
                    if 'model_state_dict' in checkpoint:
                        model_state = checkpoint['model_state_dict']
                        # 提取编码器相关的权重
                        encoder_state = {}
                        for key, value in model_state.items():
                            if 'encoder' in key:
                                # 移除'encoder.'前缀（如果存在）
                                new_key = key.replace('encoder.', '') if key.startswith('encoder.') else key
                                encoder_state[new_key] = value
                        
                        if encoder_state:
                            ppo_runner.alg.encoder.load_state_dict(encoder_state, strict=False)
                            print("✅ Successfully extracted and loaded encoder weights")
                        else:
                            print("⚠️  No encoder weights found in checkpoint")
                    else:
                        print("⚠️  No model state dict found in checkpoint")
                
                print(f"✅ Encoder weights loaded successfully!")
                print(f"📝 Policy network remains randomly initialized for new learning")
                print(f"Previous training iterations: {checkpoint.get('iter', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Error loading encoder weights: {e}")
                print("Continuing with random initialization...")
        else:
            print(f"❌ Stage-1 checkpoint not found: {checkpoint_path}")
            print("Continuing with random initialization...")
    
    # 检查是否需要加载编码器权重
    if args.load_stage1:
        print(f"\n=== Transfer Learning: Encoder Only ===")
        load_encoder_only(ppo_runner, args.stage1_path, args.stage1_checkpoint)
    else:
        print(f"\n=== Training from scratch ===")
    
    task_registry.save_cfgs(name=args.task)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )
    return ppo_runner.alg.storage


if __name__ == "__main__":
    args = get_args()
    storage = train(args)
