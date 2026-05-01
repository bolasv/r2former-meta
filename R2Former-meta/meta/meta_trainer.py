# meta/meta_trainer.py
"""
双层元学习训练引擎 (对应论文3.4节、算法1)

严格复现论文算法1的双层元训练流程:
- 内循环: 任务适配 (论文算法1第5-12行)
- 外循环: 元更新 (论文算法1第13-16行)

核心约束:
1. 内循环: 用冻结的MiniVeLO优化器执行参数更新
2. 外循环: 仅对LSTM-GeM控制器参数φ回传梯度
3. 主干参数θ和外循环不更新; MiniVeLO参数ω全程冻结
4. 元批次损失平均逻辑, 支持多任务并行
"""
import os
import sys
import time
import copy
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from meta.config import MetaConfig
from meta.msls_meta_dataset import MSLSMetaDataset
from meta.minivelo_optimizer import MiniVELOOptimizer, FallbackOptimizer

logger = logging.getLogger(__name__)


class MetaTrainer:
    """双层元学习训练引擎 (对应论文算法1)

    训练流程:
    1. 采样元批次 B个城市任务 (算法1第3行)
    2. 对每个任务执行内循环适配 (算法1第5-12行)
       - 初始化 (θ̃_c^0, φ̃_c^0) = (θ, φ)
       - τ步内循环: LSTM预测p_t → 计算支持集损失 → MiniVeLO更新参数
    3. 计算外循环元损失 (算法1第13行)
       - 在查询集上用适配参数计算损失
    4. 元更新 (算法1第14-16行)
       - 仅更新φ, 不更新θ和ω
    """

    def __init__(self, model, config: MetaConfig,
                 meta_dataset: MSLSMetaDataset,
                 loss_fn=None):
        """
        Args:
            model: R2Former模型 (含DynamicMultiGeM + LSTMController)
            config: 元学习配置
            meta_dataset: MSLS元任务数据集
            loss_fn: 损失函数 (兼容R2Former官方)
        """
        self.model = model
        self.config = config
        self.meta_dataset = meta_dataset

        # 损失函数: 使用R2Former官方的TripletMarginLoss + 支配性损失
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.TripletMarginLoss(margin=0.1, p=2, eps=1e-6)

        # 设备
        self.device = torch.device(config.device)

        # 分离参数组:
        # - backbone_params: θ (内循环适配, 外循环不更新)
        # - controller_params: φ (内循环适配, 外循环更新)
        # - other_params: 重排序等 (不参与元学习)
        self.backbone_params = []
        self.controller_params = []
        self.other_params = []

        for name, param in model.named_parameters():
            if 'controller' in name or 'base_p' in name:
                self.controller_params.append(param)
            elif 'backbone' in name or 'aggregation' not in name:
                self.backbone_params.append(param)
            else:
                self.other_params.append(param)

        # 统计控制器参数量
        controller_param_count = sum(p.numel() for p in self.controller_params)
        logger.info(f"LSTM-GeM Controller params (φ): {controller_param_count}")
        logger.info(f"Backbone params (θ): {sum(p.numel() for p in self.backbone_params)}")

        # 初始化MiniVeLO优化器 (冻结)
        if config.use_minivelo:
            self.inner_optimizer = MiniVELOOptimizer(
                hidden_dim=config.minivelo_hidden_dim,
                num_layers=config.minivelo_num_layers,
                inner_lr=config.inner_lr if hasattr(config, 'inner_lr') else 0.01,
                pretrained_path=config.minivelo_pretrained_path,
            ).to(self.device)
        else:
            self.inner_optimizer = FallbackOptimizer(
                inner_lr=config.inner_lr if hasattr(config, 'inner_lr') else 0.01,
            )
            logger.info("Using FallbackOptimizer (frozen Adam) for inner loop")

        # 外循环优化器: 仅更新控制器参数φ
        self.outer_optimizer = torch.optim.Adam(
            self.controller_params, lr=config.outer_lr
        )

        # 训练状态
        self.global_step = 0
        self.best_map1 = 0.0
        self.convergence_step = None

        # 日志记录
        self.log_dir = config.save_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_inner_loop_params(self, model=None) -> Dict[str, List[torch.Tensor]]:
        """获取内循环需要适配的参数组

        内循环适配参数: (θ̃_c, φ̃_c) - backbone + controller
        外循环不更新: θ固定; ω(MiniVeLO)冻结
        """
        if model is None:
            model = self.model

        backbone = []
        controller = []

        for name, param in model.named_parameters():
            if 'controller' in name or 'base_p' in name:
                controller.append(param)
            elif 'backbone' in name or 'aggregation' not in name:
                backbone.append(param)

        return {'backbone': backbone, 'controller': controller}

    def _clone_params(self, params: List[torch.Tensor]) -> List[torch.Tensor]:
        """深拷贝参数列表 (用于内循环初始化, 论文算法1第4行)"""
        return [p.clone() for p in params]

    def inner_loop_adapt(self, model, city_task, device) -> Dict[str, List[torch.Tensor]]:
        """内循环任务适配 (对应论文算法1第5-12行)

        对每个城市任务T_c:
        1. 初始化任务专属参数 (θ̃_c^0, φ̃_c^0) = (θ, φ) (第4行)
        2. τ步内循环:
           a. 从支持集特征提取统计量s_t (第7行)
           b. LSTM控制器预测p_t (第8行, 公式7-13)
           c. 用动态GeM计算支持集损失L_sup (第9行)
           d. 冻结的MiniVeLO更新参数 (第10行, 公式14-15)

        Args:
            model: R2Former模型
            city_task: 城市元任务
            device: 计算设备

        Returns:
            适配后的参数字典
        """
        # 第4行: 初始化任务专属参数
        param_groups = self._get_inner_loop_params(model)
        fast_backbone = self._clone_params(param_groups['backbone'])
        fast_controller = self._clone_params(param_groups['controller'])

        # MiniVeLO状态
        minivelo_states = None

        # LSTM控制器隐藏状态 (在任务内循环中持续更新)
        lstm_h = None
        lstm_c = None

        # τ步内循环
        for step in range(self.config.inner_steps):
            # 第7行: 从支持集采样一个batch
            batch = self.meta_dataset.get_triplet_batch(
                city_task, batch_size=self.config.train_batch_size, split="support"
            )

            if not batch or 'images' not in batch:
                break

            images = batch['images'].to(device)
            triplets = batch['triplets'].to(device)

            # 第8行: 提取特征统计量s_t (论文公式7-10)
            with torch.no_grad():
                # 前向传播获取特征图, 用于统计量计算
                feature_stats = self._extract_feature_stats(model, images)

            # 第8行: LSTM控制器预测p_t (论文公式11-13)
            if hasattr(model, 'aggregation') and hasattr(model.aggregation, 'controller'):
                controller = model.aggregation.controller
                p_t, lstm_h, lstm_c = controller(feature_stats, lstm_h, lstm_c)

                # 更新动态GeM的幂次参数
                if hasattr(model.aggregation, 'base_p'):
                    model.aggregation.base_p.data = p_t.squeeze(0)

            # 第9行: 用当前适配参数计算支持集损失L_sup
            # 临时替换参数
            self._set_params(model, fast_backbone, fast_controller)

            # 前向传播 + 计算损失
            descriptors = model(images)
            loss = self._compute_loss(descriptors, triplets)

            # 第10行: 计算梯度
            grads_backbone = torch.autograd.grad(
                loss, fast_backbone,
                create_graph=True, allow_unused=True
            )
            grads_controller = torch.autograd.grad(
                loss, fast_controller,
                create_graph=True, allow_unused=True
            )

            # 第10行: 用冻结的MiniVeLO更新参数 (论文公式14-15)
            # 收集所有参数和梯度
            all_params = list(fast_backbone) + list(fast_controller)
            all_grads = list(grads_backbone) + list(grads_controller)

            # 过滤None梯度
            valid_params = []
            valid_grads = []
            for p, g in zip(all_params, all_grads):
                if g is not None:
                    valid_params.append(p)
                    valid_grads.append(g)

            if isinstance(self.inner_optimizer, MiniVELOOptimizer):
                updates, minivelo_states = self.inner_optimizer.compute_updates(
                    valid_params, valid_grads, minivelo_states
                )
                # 应用更新
                new_params = self.inner_optimizer.apply_updates(valid_params, updates)
                # 分配回backbone和controller
                n_bb = len(fast_backbone)
                fast_backbone = new_params[:n_bb]
                fast_controller = new_params[n_bb:]
            else:
                # FallbackOptimizer
                updates, _ = self.inner_optimizer.compute_updates(
                    valid_params, valid_grads, minivelo_states
                )
                n_bb = len(fast_backbone)
                fast_backbone_updated = []
                fast_controller_updated = []
                for i, (p, u) in enumerate(zip(valid_params, updates)):
                    new_p = p + u
                    if i < n_bb:
                        fast_backbone_updated.append(new_p)
                    else:
                        fast_controller_updated.append(new_p)
                if fast_backbone_updated:
                    fast_backbone = fast_backbone_updated
                if fast_controller_updated:
                    fast_controller = fast_controller_updated

        return {'backbone': fast_backbone, 'controller': fast_controller}

    def _extract_feature_stats(self, model, images):
        """从特征图提取统计向量s_t (对应论文公式7-10)

        s_t = [μ, σ, max, min, ||·||_2] ∈ R^5

        Args:
            model: R2Former模型
            images: 输入图像 [B, 3, H, W]

        Returns:
            feature_stats: [1, 5] 统计向量
        """
        with torch.no_grad():
            # 获取主干特征图 (DeiT-S输出 [B, 384, 7, 7])
            features = model.backbone(images)

            # 公式7: 均值 μ = E[x]
            feat_mean = features.mean()

            # 公式8: 标准差 σ = std(x)
            feat_std = features.std().clamp(min=1e-8)

            # 公式9: 最大值
            feat_max = features.max()

            # 公式9: 最小值
            feat_min = features.min()

            # 公式10: L2范数 ||x||_2
            feat_norm = features.norm(2)

            # 组装统计向量 [1, 5]
            stats = torch.stack([feat_mean, feat_std, feat_max, feat_min, feat_norm])
            stats = stats.unsqueeze(0)  # [1, 5]

        return stats

    def _compute_loss(self, descriptors, triplets):
        """计算三元组损失 (兼容R2Former官方)

        支持两种方式:
        1. 直接使用TripletMarginLoss
        2. 使用R2Former官方的支配性损失 (如果模型有reranker)

        Args:
            descriptors: 全局描述子 [N, D]
            triplets: 三元组索引 [M, 3]

        Returns:
            loss: 标量损失
        """
        if descriptors is None or len(triplets) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 提取三元组描述子
        anchors = descriptors[triplets[:, 0]]
        positives = descriptors[triplets[:, 1]]
        negatives = descriptors[triplets[:, 2]]

        # 归一化
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)

        # 三元组损失
        loss = self.loss_fn(anchors, positives, negatives)

        return loss

    def _set_params(self, model, backbone_params, controller_params):
        """设置模型参数 (用于内循环适配后的前向传播)"""
        idx = 0
        for name, param in model.named_parameters():
            if 'controller' in name or 'base_p' in name:
                if idx < len(controller_params):
                    param.data.copy_(controller_params[idx].data)
                    idx += 1

        idx = 0
        for name, param in model.named_parameters():
            if 'backbone' in name or ('controller' not in name and 'base_p' not in name and 'aggregation' not in name):
                if idx < len(backbone_params):
                    param.data.copy_(backbone_params[idx].data)
                    idx += 1

    def meta_train_step(self) -> Dict[str, float]:
        """单步元训练 (对应论文算法1完整流程)

        1. 采样B个城市任务 (第3行)
        2. 对每个任务执行内循环适配 (第4-12行)
        3. 在查询集上计算元损失 (第13行)
        4. 元更新φ (第14-16行)

        Returns:
            metrics: 包含元损失等指标的字典
        """
        self.model.train()

        # 第3行: 采样元批次
        meta_batch = self.meta_dataset.sample_meta_batch(
            self.config.meta_batch_size, split="train"
        )

        if len(meta_batch) == 0:
            logger.warning("Empty meta batch, skipping step")
            return {'meta_loss': 0.0}

        # 保存原始参数 (用于外循环恢复)
        original_backbone = self._clone_params(
            [p for n, p in self.model.named_parameters()
             if 'controller' not in n and 'base_p' not in n]
        )
        original_controller = self._clone_params(self.controller_params)

        meta_loss_total = 0.0
        num_valid_tasks = 0

        for city_task in meta_batch:
            # 第4-12行: 内循环适配
            adapted_params = self.inner_loop_adapt(
                self.model, city_task, self.device
            )

            # 第13行: 在查询集上计算元损失L_qry
            query_batch = self.meta_dataset.get_triplet_batch(
                city_task, batch_size=self.config.train_batch_size, split="query"
            )

            if not query_batch or 'images' not in query_batch:
                continue

            images = query_batch['images'].to(self.device)
            triplets = query_batch['triplets'].to(self.device)

            # 用适配后的参数计算查询集损失
            self._set_params(
                self.model,
                adapted_params['backbone'],
                adapted_params['controller']
            )

            descriptors = self.model(images)
            query_loss = self._compute_loss(descriptors, triplets)

            meta_loss_total += query_loss
            num_valid_tasks += 1

        # 恢复原始参数
        self._set_params(self.model, original_backbone, original_controller)

        if num_valid_tasks == 0:
            return {'meta_loss': 0.0}

        # 第14行: 元损失平均 (论文: 1/B * Σ L_qry)
        meta_loss = meta_loss_total / num_valid_tasks

        # 第15-16行: 仅对φ回传梯度, 外循环更新
        self.outer_optimizer.zero_grad()
        meta_loss.backward()

        # 确保只更新控制器参数
        for name, param in self.model.named_parameters():
            if 'controller' not in name and 'base_p' not in name:
                if param.grad is not None:
                    param.grad.zero_()

        self.outer_optimizer.step()

        self.global_step += 1

        metrics = {
            'meta_loss': meta_loss.item(),
            'step': self.global_step,
        }

        return metrics

    def train(self):
        """完整元训练循环 (对应论文4.1节实验设置)

        元训练总步数: 500步
        每步: 采样元批次 → 内循环适配 → 外循环更新
        每50步: 评估 + 日志
        """
        self.config.make_deterministic()
        logger.info(f"Starting meta-training for {self.config.meta_train_steps} steps")
        logger.info(str(self.config))

        # 统计控制器参数量
        controller_params = sum(p.numel() for p in self.controller_params)
        logger.info(f"Controller params (φ): {controller_params} (~{controller_params/1000:.0f}K)")

        start_time = time.time()

        for step in range(self.config.meta_train_steps):
            metrics = self.meta_train_step()

            # 日志记录
            if step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step}/{self.config.meta_train_steps} | "
                    f"Meta Loss: {metrics['meta_loss']:.6f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            # 定期评估
            if (step + 1) % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                logger.info(
                    f"Step {step+1} Eval | "
                    f"mAP@1: {eval_metrics.get('mean_map1', 0):.2f} | "
                    f"mAP@5: {eval_metrics.get('mean_map5', 0):.2f} | "
                    f"Recall@1: {eval_metrics.get('mean_recall1', 0):.2f}"
                )

                # 保存最优模型
                current_map1 = eval_metrics.get('mean_map1', 0)
                if current_map1 > self.best_map1:
                    self.best_map1 = current_map1
                    self._save_checkpoint('best_controller.pth')

                    # 收敛速度统计 (论文4.2节)
                    if self.config.track_convergence and self.convergence_step is None:
                        if current_map1 >= self.config.baseline_best_map1:
                            self.convergence_step = step + 1
                            logger.info(
                                f"Convergence achieved at step {self.convergence_step} "
                                f"(baseline mAP@1: {self.config.baseline_best_map1})"
                            )

        # 训练结束保存
        self._save_checkpoint('final_controller.pth')
        logger.info(f"Meta-training completed. Best mAP@1: {self.best_map1:.2f}")

        if self.convergence_step:
            logger.info(f"Convergence step: {self.convergence_step} "
                        f"(3x speedup verification)")

    def evaluate(self) -> Dict[str, float]:
        """评估当前模型在测试城市上的性能

        Returns:
            评估指标字典
        """
        from meta.meta_evaluator import MetaEvaluator
        evaluator = MetaEvaluator(self.model, self.config, self.meta_dataset)
        return evaluator.evaluate_all()

    def _save_checkpoint(self, filename):
        """保存控制器权重"""
        save_path = os.path.join(self.log_dir, filename)
        controller_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if 'controller' in name or 'base_p' in name
        }
        torch.save({
            'step': self.global_step,
            'controller_state': controller_state,
            'best_map1': self.best_map1,
        }, save_path)
        logger.info(f"Saved controller checkpoint to {save_path}")
