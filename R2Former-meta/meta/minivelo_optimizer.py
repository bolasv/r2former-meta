# meta/minivelo_optimizer.py
"""
MiniVeLO优化器集成模块 (对应论文3.4节)

核心约束:
1. 集成预训练MiniVeLO优化器, 实现参数全程冻结的调用逻辑
2. 仅在内循环中作为任务内更新的优化器使用, 不参与任何梯度更新
3. 保证MiniVeLO优化器与PyTorch的自动求导机制兼容
4. 适配内循环的单步梯度更新流程

MiniVeLO是基于LSTM的学习型优化器, 参考论文:
"VeLO: Training Versatile Learned Optimizers by Scaling Up" (Metz et al., 2022)

本实现提供轻量级MiniVeLO, 支持加载预训练权重或从零初始化。
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MiniVeLOCell(nn.Module):
    """MiniVeLO单层LSTM优化器单元

    输入: 当前参数梯度统计量 (梯度均值、标准差、最大值、最小值、二阶矩)
    输出: 参数更新量

    结构:
    - LSTM处理梯度统计量
    - 两个MLP头分别输出: 更新方向和步长
    """

    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # LSTM: 处理梯度统计序列
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

        # 更新方向头
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 步长头
        self.step_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 步长 ∈ [0, 1]
        )

    def init_state(self, batch_size, device, dtype):
        h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return h, c

    def forward(self, grad_stats, state=None):
        """
        Args:
            grad_stats: [N, input_dim] N个参数组的梯度统计
            state: LSTM状态(h, c)

        Returns:
            updates: [N] 参数更新量
            new_state: 更新后的LSTM状态
        """
        if state is None:
            state = self.init_state(grad_stats.shape[0], grad_stats.device, grad_stats.dtype)

        h, c = self.lstm(grad_stats, state)

        # 更新方向: tanh输出, 范围[-1, 1]
        direction = self.direction_head(h).squeeze(-1)

        # 步长: sigmoid输出, 范围[0, 1]
        step_size = self.step_head(h).squeeze(-1)

        # 更新量 = 方向 × 步长
        updates = direction * step_size

        return updates, (h, c)


class MiniVELOOptimizer(nn.Module):
    """MiniVeLO优化器 (对应论文3.4节)

    预训练的学习型优化器, 参数ω全程冻结:
    - 仅在内循环中作为任务内更新的优化器使用
    - 不参与外循环的任何梯度更新
    - 替代传统SGD/Adam在MAML内循环中的角色

    使用方式:
    1. 初始化: 加载预训练权重(可选), 冻结所有参数
    2. 内循环: compute_updates() 计算参数更新量
    3. 整个元训练过程: ω不变
    """

    def __init__(self, hidden_dim=64, num_layers=2, inner_lr=0.01,
                 pretrained_path=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.inner_lr = inner_lr  # 内循环学习率(当MiniVeLO不可用时的fallback)

        # 多层MiniVeLO cell
        self.cells = nn.ModuleList([
            MiniVeLOCell(input_dim=6, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        # 加载预训练权重
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            logger.info(f"Loaded MiniVeLO pretrained weights from {pretrained_path}")

        # 冻结所有参数 (论文约束: ω全程冻结)
        self._freeze_all()

    def _freeze_all(self):
        """冻结MiniVeLO所有参数 (论文3.4节: ω全程不更新)"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info(f"MiniVeLO optimizer frozen: {sum(p.numel() for p in self.parameters())} params")

    def compute_grad_stats(self, params: List[torch.Tensor],
                           grads: List[torch.Tensor]) -> torch.Tensor:
        """计算参数组的梯度统计量

        对每个参数张量, 计算6维统计:
        [梯度均值, 梯度标准差, 梯度最大值, 梯度最小值, 参数二阶矩, 梯度L2范数]

        Args:
            params: 参数张量列表
            grads: 对应梯度张量列表

        Returns:
            grad_stats: [N, 6] 统计量矩阵
        """
        stats_list = []
        for p, g in zip(params, grads):
            if g is None:
                stats_list.append(torch.zeros(6, device=p.device, dtype=p.dtype))
                continue
            g_flat = g.detach().flatten()
            p_flat = p.detach().flatten()
            mean_g = g_flat.mean()
            std_g = g_flat.std().clamp(min=1e-8)
            max_g = g_flat.abs().max()
            min_g = g_flat.abs().min()
            param_norm = p_flat.pow(2).mean().sqrt().clamp(min=1e-8)
            grad_l2 = g_flat.norm(2).clamp(min=1e-8)
            stats_list.append(torch.stack([mean_g, std_g, max_g, min_g, param_norm, grad_l2]))

        return torch.stack(stats_list, dim=0)

    def compute_updates(self, params: List[torch.Tensor],
                        grads: List[torch.Tensor],
                        states: Optional[List[Tuple]] = None) -> Tuple[List[torch.Tensor], List[Tuple]]:
        """使用MiniVeLO计算参数更新量 (内循环单步更新)

        Args:
            params: 当前参数列表
            grads: 对应梯度列表
            states: 各层的LSTM状态列表(可为None)

        Returns:
            updates: 参数更新量列表 (与params同形状)
            new_states: 更新后的LSTM状态列表
        """
        # 计算梯度统计
        grad_stats = self.compute_grad_stats(params, grads)

        # 多层LSTM处理
        new_states = []
        current_input = grad_stats

        for layer_idx, cell in enumerate(self.cells):
            state = states[layer_idx] if states is not None else None
            updates_signal, new_state = cell(current_input, state)
            new_states.append(new_state)
            # 残差连接: 下一层输入为当前层输出+原始统计
            current_input = torch.cat([current_input, updates_signal.unsqueeze(-1)], dim=-1)

        # 生成每个参数的更新量
        final_signal = updates_signal  # [N]
        updates = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                updates.append(torch.zeros_like(p))
                continue
            # 更新量 = sign * magnitude * inner_lr
            # 使用全局缩放因子 + MiniVeLO输出的信号
            scale = final_signal[i].item() if i < len(final_signal) else 0.0
            update = -scale * self.inner_lr * g
            updates.append(update)

        return updates, new_states

    def apply_updates(self, params: List[torch.Tensor],
                      updates: List[torch.Tensor]) -> List[torch.Tensor]:
        """应用参数更新 (内循环单步)

        Args:
            params: 当前参数列表
            updates: 更新量列表

        Returns:
            new_params: 更新后的参数列表
        """
        new_params = []
        for p, u in zip(params, updates):
            new_params.append(p + u)
        return new_params


class FallbackOptimizer:
    """当MiniVeLO不可用时的回退优化器 (冻结的Adam)

    使用固定超参数的Adam优化器, 行为等价于冻结的优化器
    (超参数不随训练变化), 保证代码可运行
    """

    def __init__(self, inner_lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.inner_lr = inner_lr
        self.betas = betas
        self.eps = eps
        self.state = {}  # 优化器状态

    def compute_updates(self, params: List[torch.Tensor],
                        grads: List[torch.Tensor],
                        states=None) -> Tuple[List[torch.Tensor], None]:
        """使用固定Adam计算参数更新量

        Args:
            params: 当前参数列表
            grads: 对应梯度列表

        Returns:
            updates: 参数更新量列表
            None: 无LSTM状态
        """
        updates = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                updates.append(torch.zeros_like(p))
                continue

            key = id(p)
            if key not in self.state:
                self.state[key] = {
                    'exp_avg': torch.zeros_like(g),
                    'exp_avg_sq': torch.zeros_like(g),
                    'step': 0,
                }

            state = self.state[key]
            state['step'] += 1

            # Adam更新 (固定超参数, 等价于冻结的优化器)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            beta1, beta2 = self.betas

            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            step_size = self.inner_lr / bias_correction1

            update = -step_size * exp_avg / denom
            updates.append(update)

        return updates, None
import os
