#!/usr/bin/env python3
"""
元学习训练入口脚本

运行命令:
    python meta_train.py --datasets_folder /path/to/datasets --dataset_name msls

参数说明:
    --datasets_folder: MSLS数据集根目录
    --dataset_name: 数据集名称 (默认: msls)
    --gem_mode: GeM池化模式 (dynamic/fixed_multi/fixed_single, 默认: dynamic)
    --use_minivelo: 是否使用MiniVeLO优化器 (默认: True)
    --meta_train_steps: 元训练步数 (默认: 500)
    --meta_batch_size: 元批次大小 (默认: 2)
    --seed: 随机种子 (默认: 42)
"""
import os
import sys
import logging
import argparse
import torch
from argparse import Namespace

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta.config import MetaConfig
from meta.msls_meta_dataset import MSLSMetaDataset
from meta.meta_trainer import MetaTrainer
from meta.ablation import create_gem_module, create_model_with_ablation


def setup_logging(log_dir):
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'meta_train.log')),
        ]
    )


def build_model(config):
    """构建R2Former模型 + 动态GeM池化

    零侵入原则: 使用R2Former官方模型架构, 仅替换GeM池化模块。
    GeoLocalizationNetRerank是network.py中的DeiT+R2Former完整模型,
    包含DeiT-S主干、GeM池化/动态GeM池化、重排序分支。
    """
    from model.network import GeoLocalizationNetRerank

    # 构造与GeoLocalizationNetRerank.__init__兼容的args namespace
    # 对齐network.py中该类的所有必要参数
    args = Namespace(
        # 主干网络参数
        backbone=config.backbone,                      # 'deitsmall'
        fc_output_dim=config.fc_output_dim,            # 384
        features_dim=config.features_dim,              # 384
        resize=config.resize,                          # [224, 224]
        # 池化聚合参数
        aggregation='dynamic_gem',
        l2='before_pool',
        work_with_tokens=True,
        # DynamicMultiGeM参数 (论文3.2-3.3节)
        dynamic_gem_init_p=config.gem_p_init,          # 3.0
        dynamic_gem_eps=config.gem_eps,                # 1e-6
        dynamic_gem_groups=config.gem_groups,          # K=8 (论文4.1节)
        dynamic_gem_hidden_dim=config.lstm_hidden_dim,  # 128
        gem_mode=config.gem_mode,                      # 'dynamic'
        paper_mode=True,
        # 重排序分支参数
        hypercolumn=0,                                 # 不使用hypercolumn
        local_dim=128,
        num_local=196,
        rerank_model='r2former',
        finetune=True,
        # 其他必要参数
        non_local=False,
        channel_bottleneck=128,
        num_non_local=1,
    )

    try:
        model = GeoLocalizationNetRerank(args)
        logging.info("Built GeoLocalizationNetRerank model with dynamic_gem aggregation")
    except Exception as e:
        logging.warning(f"Could not build GeoLocalizationNetRerank: {e}")
        logging.info("Falling back to manual model construction with ablation")

        # 回退方案: 手动构建基础模型 (不含重排序分支)
        try:
            from model.network import GeoLocalizationNet
            model = GeoLocalizationNet(args)
            logging.info("Built GeoLocalizationNet (no reranker) as fallback")
        except Exception as e2:
            logging.error(f"Could not build model: {e2}")
            raise

    return model


def main():
    parser = argparse.ArgumentParser(description='Meta-Learning Training (Paper Algorithm 1)')
    parser.add_argument('--datasets_folder', type=str, default='datasets',
                        help='MSLS datasets root folder')
    parser.add_argument('--dataset_name', type=str, default='msls',
                        help='Dataset name')
    parser.add_argument('--gem_mode', type=str, default='dynamic',
                        choices=['dynamic', 'fixed_multi', 'fixed_single'],
                        help='GeM pooling mode (ablation)')
    parser.add_argument('--use_minivelo', action='store_true', default=True,
                        help='Use MiniVeLO optimizer (frozen)')
    parser.add_argument('--no_minivelo', action='store_true',
                        help='Disable MiniVeLO, use fallback Adam')
    parser.add_argument('--meta_train_steps', type=int, default=500,
                        help='Total meta-training steps')
    parser.add_argument('--meta_batch_size', type=int, default=2,
                        help='Meta batch size')
    parser.add_argument('--inner_steps', type=int, default=20,
                        help='Inner loop adaptation steps')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Training batch size per task')
    parser.add_argument('--outer_lr', type=float, default=1e-4,
                        help='Outer loop learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='logs/meta_learning',
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # 创建配置
    config = MetaConfig()
    config.datasets_folder = args.datasets_folder
    config.dataset_name = args.dataset_name
    config.gem_mode = args.gem_mode
    config.use_minivelo = args.use_minivelo and not args.no_minivelo
    config.meta_train_steps = args.meta_train_steps
    config.meta_batch_size = args.meta_batch_size
    config.inner_steps = args.inner_steps
    config.train_batch_size = args.train_batch_size
    config.outer_lr = args.outer_lr
    config.seed = args.seed
    config.save_dir = args.save_dir

    if args.device:
        config.device = args.device

    # 设置日志
    setup_logging(config.save_dir)
    logger = logging.getLogger(__name__)

    # 设置随机种子
    config.make_deterministic()
    logger.info(f"Configuration:\n{config}")

    # 构建模型
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(config.device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    controller_params = config.count_controller_params(model)
    logger.info(f"Total model params: {total_params:,}")
    logger.info(f"Controller params (φ): {controller_params:,} (~{controller_params/1000:.1f}K)")

    # 创建元任务数据集
    logger.info("Initializing MSLS meta-dataset...")
    meta_dataset = MSLSMetaDataset(config)

    # 创建训练器
    logger.info("Initializing meta-trainer...")
    trainer = MetaTrainer(model, config, meta_dataset)

    # 开始训练
    logger.info("Starting meta-training...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
