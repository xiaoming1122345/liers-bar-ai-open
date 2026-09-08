"""
PPO-Clip 独立 Actor 与 Critic 神经网络定义 (v27 纯 RL 路线)。

设计规范：
1. Actor 与 Critic 彻底解耦，各自拥有独立骨干网络与参数，杜绝梯度交叉干扰；
2. Actor: 104 维 control 特征输入 -> 512x4 MLP (LayerNorm + ReLU + Dropout(0.0)) -> 58 维动作 logits；
   - 严格关闭 Dropout (dropout=0.0)，确保采样时与训练更新时概率分布绝对一致；
   - 骨干结构与 CardNet 完全一致，支持直接 1:1 复制 Init0 的 strategy_net 权重进行蒸馏热启动；
   - 策略分布使用真实合法动作上的 masked softmax 分布；
3. Critic: 104 维 control 特征输入 -> 256x2 MLP (LayerNorm + ReLU) -> 1 维标量期望回报 (无激活层)；
4. 提供批量推理、动作采样、log_prob 评估及策略熵计算工具函数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class PPOActor(nn.Module):
    """PPO 策略网络：输出 58 维抽象动作的未归一化 logits。"""

    def __init__(
        self,
        input_dim: int = 104,
        action_space_size: int = 58,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_space_size = action_space_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        current_in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            # 设为 Dropout(0.0)，彻底关闭随机丢弃，保持与 CardNet 索引 1:1 对齐
            layers.append(nn.Dropout(0.0))
            current_in_dim = hidden_dim

        layers.append(nn.Linear(current_in_dim, action_space_size))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回未加掩码的 raw logits: shape (batch_size, 58) 或 (58,)."""
        return self.net(features)

    def get_action_probs(
        self,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """在合法动作掩码上应用 softmax 计算动作概率。
        
        legal_mask: 合法动作位置为 0.0，非法动作位置为 -1e9 或 -inf。
        返回 probs: 与 legal_mask 形状相同。
        """
        logits = self.forward(features)
        temp = max(float(temperature), 1e-6)
        masked_logits = (logits + legal_mask) / temp
        probs = torch.softmax(masked_logits, dim=-1)
        return probs

    def evaluate_actions(
        self,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量评估给定动作的 log_prob 与当前策略熵 (用于 PPO 更新)。
        
        features: (B, 104)
        legal_mask: (B, 58)
        actions: (B,) 动作索引 int64
        
        返回:
            log_probs: (B,) 实际所选动作的 log 概率
            entropy: (B,) 掩码策略分布的香农熵
        """
        logits = self.forward(features)
        masked_logits = logits + legal_mask
        log_probs_all = torch.log_softmax(masked_logits, dim=-1)
        probs_all = torch.softmax(masked_logits, dim=-1)

        # 选出指定 actions 对应的 log_prob
        log_probs = log_probs_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        # 掩码下的熵: -sum(p * log_p) (仅对合法动作求和，非法动作 p=0 且 p*log_p=0)
        safe_log_probs = torch.where(probs_all > 1e-12, log_probs_all, torch.zeros_like(log_probs_all))
        entropy = -(probs_all * safe_log_probs).sum(dim=-1)

        return log_probs, entropy


class PPOCritic(nn.Module):
    """PPO 价值网络：输入局面特征，输出标量期望终局回报 (256x2 紧凑架构)。"""

    def __init__(
        self,
        input_dim: int = 104,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        current_in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.0))
            current_in_dim = hidden_dim

        # Output layer -> 1 维标量 (无激活层)
        layers.append(nn.Linear(current_in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """输出局面价值预测: shape (batch_size, 1) 或 (1,)."""
        return self.net(features).squeeze(-1)
