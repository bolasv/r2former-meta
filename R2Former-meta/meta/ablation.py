# meta/ablation.py
"""
消融实验代码分支 (对应论文4.3节)

3组消融实验, 通过配置项快速切换:
1. 基线: 固定单参数GeM (p=3.0)
2. 对照: 固定多参数GeM (8组p值均固定为3.0)
3. 本文方法: 动态多参数GeM

核心约束:
- 消融实验仅修改池化模块, 其余训练/评估逻辑完全一致
- 确保实验公平性
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FixedSingleGeM(nn.Module):
    """固定单参数GeM (基线, 论文4.3节消融实验1)

    经典GeM池化: 所有通道共享同一个固定的幂次p
    f(X) = (1/|X| * Σ x_i^p)^(1/p), p固定为3.0

    这等同于R2Former原始的GeM池化, 作为消融实验的基线
    """

    def __init__(self, p=3.0, eps=1e-6, features_dim=384):
        super().__init__()
        self.p = p
        self.eps = eps
        self.features_dim = features_dim

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图

        Returns:
            descriptors: [B, C] 全局描述子
        """
        # GeM池化: f(X) = (mean(x^p) + eps)^(1/p)
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=[2, 3])  # [B, C]
        x = x.pow(1.0 / self.p)
        return x

    def extra_repr(self):
        return f"p={self.p}, eps={self.eps}"


class FixedMultiGeM(nn.Module):
    """固定多参数GeM (消融对照, 论文4.3节消融实验2)

    8组通道各有独立的幂次p_k, 但p_k固定不更新
    用于验证动态调整p_k带来的增益

    对应论文公式(3)-(6), 但p_k不可学习
    """

    def __init__(self, p=3.0, num_groups=8, eps=1e-6, features_dim=384):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.features_dim = features_dim
        self.channels_per_group = features_dim // num_groups

        # 固定的幂次参数 (所有组均设为p=3.0)
        self.register_buffer('group_p', torch.full((num_groups,), p))

        # 固定的融合权重 (均匀权重)
        self.register_buffer('group_weights', torch.ones(num_groups) / num_groups)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图

        Returns:
            descriptors: [B, C] 全局描述子
        """
        B, C, H, W = x.shape
        x = x.clamp(min=self.eps)

        # 按通道分组 [B, K, C/K, H, W]
        x_groups = x.reshape(B, self.num_groups, self.channels_per_group, H, W)

        # 每组独立GeM池化 (论文公式3)
        group_descriptors = []
        for k in range(self.num_groups):
            p_k = self.group_p[k].item()
            group_feat = x_groups[:, k]  # [B, C/K, H, W]
            group_feat = group_feat.pow(p_k)
            group_feat = group_feat.mean(dim=[2, 3])  # [B, C/K]
            group_feat = group_feat.pow(1.0 / p_k)
            group_descriptors.append(group_feat)

        # 加权融合 (论文公式5-6)
        weighted_groups = []
        for k in range(self.num_groups):
            weighted_groups.append(
                group_descriptors[k] * self.group_weights[k].item()
            )

        # 拼接输出 [B, C]
        descriptors = torch.cat(weighted_groups, dim=1)

        return descriptors

    def extra_repr(self):
        return (f"num_groups={self.num_groups}, "
                f"channels_per_group={self.channels_per_group}, "
                f"p=3.0(fixed), eps={self.eps}")


class DynamicMultiGeM(nn.Module):
    """动态多参数GeM (本文方法, 论文3.2节, 消融实验3)

    8组通道各有独立的动态幂次p_k(t), 由LSTM控制器实时预测
    对应论文公式(3)-(13)

    注意: 此类是model/aggregation.py中DynamicMultiGeM的封装版本,
    用于消融实验中的统一接口
    """

    def __init__(self, num_groups=8, eps=1e-6, features_dim=384,
                 p_min=1.0, p_max=10.0, lstm_hidden_dim=128):
        super().__init__()
        from model.aggregation import DynamicMultiGeM as _DynamicMultiGeM
        from model.aggregation import LSTMController

        self.num_groups = num_groups
        self.eps = eps
        self.features_dim = features_dim
        self.channels_per_group = features_dim // num_groups

        # LSTM控制器
        self.controller = LSTMController(
            stats_dim=5,
            hidden_dim=lstm_hidden_dim,
            num_groups=num_groups,
            p_min=p_min,
            p_max=p_max,
        )

        # 可学习的基准幂次
        self.base_p = nn.Parameter(torch.full((num_groups,), 3.0))

    def forward(self, x, p_values=None):
        """
        Args:
            x: [B, C, H, W] 特征图
            p_values: 可选, [K] 幂次值 (由LSTM控制器提供)

        Returns:
            descriptors: [B, C] 全局描述子
        """
        B, C, H, W = x.shape
        x = x.clamp(min=self.eps)

        # 使用提供的p值或base_p
        if p_values is not None:
            p = p_values
        else:
            p = self.base_p

        # 按通道分组
        x_groups = x.reshape(B, self.num_groups, self.channels_per_group, H, W)

        # 计算每组的自适应融合权重α_k (论文公式5)
        alpha = torch.softmax(p - p.mean(), dim=0)

        # 每组独立GeM池化 (论文公式3)
        group_descriptors = []
        for k in range(self.num_groups):
            p_k = p[k].item() if p.dim() == 1 else p[k].item()
            group_feat = x_groups[:, k]
            group_feat = group_feat.pow(p_k)
            group_feat = group_feat.mean(dim=[2, 3])
            group_feat = group_feat.pow(1.0 / p_k)
            group_descriptors.append(group_feat)

        # 加权拼接 (论文公式6)
        weighted_groups = []
        for k in range(self.num_groups):
            weighted_groups.append(group_descriptors[k] * alpha[k])

        descriptors = torch.cat(weighted_groups, dim=1)

        return descriptors

    def extra_repr(self):
        return (f"num_groups={self.num_groups}, "
                f"channels_per_group={self.channels_per_group}, "
                f"dynamic_p=True, eps={self.eps}")


def create_gem_module(gem_mode: str = "dynamic", **kwargs):
    """创建GeM池化模块的工厂函数 (对应论文4.3节消融实验)

    统一接口, 通过gem_mode参数切换消融实验配置:
    - "fixed_single": 固定单参数GeM (基线)
    - "fixed_multi": 固定多参数GeM (对照)
    - "dynamic": 动态多参数GeM (本文方法)

    Args:
        gem_mode: GeM模式
        **kwargs: 传递给具体模块的参数

    Returns:
        GeM池化模块实例
    """
    defaults = {
        'p': 3.0,
        'num_groups': 8,
        'eps': 1e-6,
        'features_dim': 384,
        'p_min': 1.0,
        'p_max': 10.0,
        'lstm_hidden_dim': 128,
    }
    defaults.update(kwargs)

    if gem_mode == "fixed_single":
        logger.info("Ablation: Using Fixed Single-Parameter GeM (baseline)")
        return FixedSingleGeM(
            p=defaults['p'],
            eps=defaults['eps'],
            features_dim=defaults['features_dim'],
        )
    elif gem_mode == "fixed_multi":
        logger.info("Ablation: Using Fixed Multi-Parameter GeM (control)")
        return FixedMultiGeM(
            p=defaults['p'],
            num_groups=defaults['num_groups'],
            eps=defaults['eps'],
            features_dim=defaults['features_dim'],
        )
    elif gem_mode == "dynamic":
        logger.info("Using Dynamic Multi-Parameter GeM (proposed method)")
        return DynamicMultiGeM(
            num_groups=defaults['num_groups'],
            eps=defaults['eps'],
            features_dim=defaults['features_dim'],
            p_min=defaults['p_min'],
            p_max=defaults['p_max'],
            lstm_hidden_dim=defaults['lstm_hidden_dim'],
        )
    else:
        raise ValueError(f"Unknown gem_mode: {gem_mode}. "
                         f"Must be 'fixed_single', 'fixed_multi', or 'dynamic'")


def create_model_with_ablation(base_model, gem_mode: str = "dynamic", **kwargs):
    """用消融实验的GeM模块替换R2Former模型中的池化层

    零侵入原则: 仅替换池化模块, 不修改其他部分

    Args:
        base_model: R2Former模型
        gem_mode: GeM模式
        **kwargs: 传递给create_gem_module的参数

    Returns:
        修改后的模型 (原地修改, 同时返回引用)
    """
    gem_module = create_gem_module(gem_mode, **kwargs)

    # 替换模型中的聚合模块
    if hasattr(base_model, 'aggregation'):
        base_model.aggregation = gem_module
        logger.info(f"Replaced aggregation module with {gem_mode} GeM")
    else:
        logger.warning("Model has no 'aggregation' attribute, cannot replace")

    return base_model
