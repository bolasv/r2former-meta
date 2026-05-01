# meta/config.py
"""
元学习配置模块 (对应论文4.1节实验设置)

严格对齐论文超参数:
- GEM分组数K=8
- 内循环步数τ=20
- 元训练总步数500
- 元批次大小2
- 训练批次大小4
- 外循环学习率1e-4
- 随机种子42
- GeM幂次输出范围[1.0, 10.0]
"""
import os
import torch
import random
import numpy as np
from typing import List, Optional


class MetaConfig:
    """元学习框架完整配置, 严格对齐论文4.1节"""

    def __init__(self):
        # ==================== 论文核心超参数 (不可修改) ====================
        self.gem_groups = 8                 # GEM分组数K (论文4.1节)
        self.inner_steps = 20              # 内循环步数τ (论文4.1节)
        self.meta_train_steps = 500        # 元训练总步数 (论文4.1节)
        self.meta_batch_size = 2           # 元批次大小B (论文4.1节)
        self.train_batch_size = 4          # 训练批次大小 (论文4.1节)
        self.outer_lr = 1e-4              # 外循环学习率β (论文4.1节)
        self.seed = 42                     # 随机种子 (论文4.1节)
        self.gem_p_min = 1.0              # GeM幂次下界 (论文3.3节, 公式13)
        self.gem_p_max = 10.0             # GeM幂次上界 (论文3.3节, 公式13)
        self.gem_p_init = 3.0             # GeM幂次初始值 (论文基线)
        self.gem_eps = 1e-6               # GeM数值稳定项

        # ==================== LSTM控制器参数 ====================
        self.lstm_hidden_dim = 128         # LSTM隐藏维度 (论文3.3节)
        self.paper_mode = True             # 使用5维统计输入 (论文3.3节, s_t∈R^5)

        # ==================== MiniVeLO优化器参数 ====================
        self.minivelo_hidden_dim = 64      # MiniVeLO LSTM隐藏维度
        self.minivelo_num_layers = 2       # MiniVeLO LSTM层数
        self.minivelo_pretrained_path = None  # 预训练MiniVeLO权重路径
        self.use_minivelo = True           # 是否使用MiniVeLO (False则用Adam)

        # ==================== MSLS数据集参数 ====================
        # 训练城市 (5个城市, 用于元任务采样) - 论文4.1节
        self.meta_train_cities = [
            "trondheim", "london", "melbourne", "amsterdam", "helsinki"
        ]
        # 测试城市 (5-6个未见城市) - 论文表1
        self.meta_test_cities = [
            "amman", "boston", "goa", "nairobi", "sf"
        ]
        # 每个元任务内支持集/查询集划分比例
        self.support_ratio = 0.5           # 支持集占比
        self.negs_num_per_query = 5        # 每个查询的负样本数 (论文: 5)
        self.pos_num_per_query = 1         # 每个查询的正样本数

        # ==================== 模型参数 ====================
        self.backbone = "deit"             # 主干网络: DeiT-S
        self.fc_output_dim = 384           # 输出维度
        self.features_dim = 384            # 特征维度
        self.aggregation = "dynamic_gem"   # 池化方式
        self.resize = [224, 224]           # 输入图像尺寸
        self.datasets_folder = "datasets"  # 数据集根目录
        self.dataset_name = "msls"         # 数据集名称
        self.infer_batch_size = 16         # 推理批次大小

        # ==================== 消融实验参数 (论文4.3节) ====================
        # gem_mode: "dynamic" | "fixed_multi" | "fixed_single"
        self.gem_mode = "dynamic"          # 默认为本文方法

        # ==================== 训练日志与保存 ====================
        self.save_dir = "logs/meta_learning"
        self.log_interval = 10             # 日志输出间隔(步)
        self.eval_interval = 50            # 评估间隔(步)
        self.save_best = True              # 保存最优模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ==================== 收敛速度统计 (论文4.2节) ====================
        self.baseline_best_map1 = 79.0     # 固定GeM基线平均mAP@1 (论文表2)
        self.track_convergence = True      # 是否记录收敛速度

    def make_deterministic(self):
        """设置随机种子保证可复现性"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def count_controller_params(self, model=None):
        """统计LSTM-GeM控制器可训练参数量, 验证约70K (论文4.1节)"""
        if model is None:
            # 理论计算
            if self.paper_mode:
                # LSTMCell(5, 128): 4*(5*128 + 128*128 + 128) = 68608
                # Linear(128, 8): 128*8 + 8 = 1032
                # base_p: 8
                total = 68608 + 1032 + 8
            else:
                # LSTMCell(40, 128): 4*(40*128 + 128*128 + 128) = 86528
                # Linear(128, 8): 1032
                # base_p: 8
                total = 86528 + 1032 + 8
            return total
        # 实际模型统计
        total = 0
        for name, param in model.named_parameters():
            if "controller" in name or "base_p" in name:
                total += param.numel()
        return total

    def __repr__(self):
        lines = ["=" * 60]
        lines.append("Meta-Learning Configuration (Paper Section 4.1)")
        lines.append("=" * 60)
        for key, value in sorted(self.__dict__.items()):
            if not key.startswith("_"):
                lines.append(f"  {key}: {value}")
        lines.append(f"  Controller params (theoretical): {self.count_controller_params()}")
        lines.append("=" * 60)
        return "\n".join(lines)
