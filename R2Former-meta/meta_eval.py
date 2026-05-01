#!/usr/bin/env python3
"""
元学习评估入口脚本

运行命令:
    python meta_eval.py --datasets_folder /path/to/datasets --checkpoint logs/meta_learning/best_controller.pth

参数说明:
    --datasets_folder: MSLS数据集根目录
    --checkpoint: 控制器权重文件路径
    --gem_mode: GeM池化模式 (dynamic/fixed_multi/fixed_single)
    --device: 计算设备
"""
import os
import sys
import logging
import argparse
import torch
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta.config import MetaConfig
from meta.msls_meta_dataset import MSLSMetaDataset
from meta.meta_evaluator import MetaEvaluator
from meta.ablation import create_model_with_ablation


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )


def build_model(config):
    """构建R2Former模型 + 动态GeM池化

    使用GeoLocalizationNetRerank (network.py中的DeiT+R2Former完整模型)
    """
    from model.network import GeoLocalizationNetRerank

    args = Namespace(
        # 主干网络参数
        backbone=config.backbone,
        fc_output_dim=config.fc_output_dim,
        features_dim=config.features_dim,
        resize=config.resize,
        # 池化聚合参数
        aggregation='dynamic_gem',
        l2='before_pool',
        work_with_tokens=True,
        # DynamicMultiGeM参数
        dynamic_gem_init_p=config.gem_p_init,
        dynamic_gem_eps=config.gem_eps,
        dynamic_gem_groups=config.gem_groups,
        dynamic_gem_hidden_dim=config.lstm_hidden_dim,
        gem_mode=config.gem_mode,
        paper_mode=True,
        # 重排序分支参数
        hypercolumn=0,
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
        logging.info("Built GeoLocalizationNetRerank model for evaluation")
    except Exception as e:
        logging.warning(f"Could not build GeoLocalizationNetRerank: {e}")
        from model.network import GeoLocalizationNet
        model = GeoLocalizationNet(args)
        logging.info("Built GeoLocalizationNet (no reranker) as fallback")

    return model


def main():
    parser = argparse.ArgumentParser(description='Meta-Learning Evaluation (Paper Section 4)')
    parser.add_argument('--datasets_folder', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='msls')
    parser.add_argument('--checkpoint', type=str, default='logs/meta_learning/best_controller.pth',
                        help='Path to controller checkpoint')
    parser.add_argument('--gem_mode', type=str, default='dynamic',
                        choices=['dynamic', 'fixed_multi', 'fixed_single'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='logs/meta_eval')

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # 创建配置
    config = MetaConfig()
    config.datasets_folder = args.datasets_folder
    config.dataset_name = args.dataset_name
    config.gem_mode = args.gem_mode
    config.seed = args.seed
    if args.device:
        config.device = args.device

    config.make_deterministic()

    # 构建模型
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(config.device)

    # 加载控制器权重
    if os.path.exists(args.checkpoint):
        logger.info(f"Loading controller checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        controller_state = checkpoint.get('controller_state_dict', checkpoint)
        # 只加载LSTM控制器的参数
        controller_params = {}
        model_state = model.state_dict()
        for k, v in controller_state.items():
            if 'dynamic_aggregation' in k:
                controller_params[k] = v
        model.load_state_dict(controller_params, strict=False)
        logger.info(f"Loaded controller params: {len(controller_params)} keys")
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}, using random init")

    # 创建元任务数据集
    logger.info("Initializing MSLS meta-dataset...")
    meta_dataset = MSLSMetaDataset(config)

    # 创建评估器
    logger.info("Initializing meta-evaluator...")
    evaluator = MetaEvaluator(model, config, meta_dataset)

    # 执行评估
    logger.info("Starting meta-evaluation...")
    results = evaluator.evaluate()

    # 输出结果
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    for city, metrics in results.items():
        if isinstance(metrics, dict):
            logger.info(f"  {city}: mAP@1={metrics.get('mAP@1', 0):.2f}, "
                        f"mAP@5={metrics.get('mAP@5', 0):.2f}, "
                        f"Recall@1={metrics.get('Recall@1', 0):.2f}")
        else:
            logger.info(f"  {city}: {metrics}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
