#!/usr/bin/env python3
"""
消融实验运行脚本 (对应论文4.3节)

3组消融实验:
1. fixed_single: 固定单参数GeM (p=3.0) - 基线
2. fixed_multi: 固定多参数GeM (8组p值均固定为3.0) - 对照
3. dynamic: 动态多参数GeM (本文方法)

运行命令:
    python run_ablation.py --datasets_folder /path/to/datasets
"""
import os
import sys
import logging
import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta.config import MetaConfig


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'ablation.log')),
        ]
    )


def run_single_ablation(gem_mode, config, logger):
    """运行单个消融实验

    Args:
        gem_mode: GeM模式 (fixed_single/fixed_multi/dynamic)
        config: 基础配置
        logger: 日志器

    Returns:
        Dict: 实验结果
    """
    from meta.msls_meta_dataset import MSLSMetaDataset
    from meta.meta_trainer import MetaTrainer

    # 创建该模式的配置副本
    mode_config = MetaConfig()
    for key, value in config.__dict__.items():
        setattr(mode_config, key, value)
    mode_config.gem_mode = gem_mode
    mode_config.save_dir = os.path.join(config.save_dir, f"ablation_{gem_mode}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Running ablation: gem_mode={gem_mode}")
    logger.info(f"{'='*60}")

    # 构建模型
    from meta_train import build_model
    model = build_model(mode_config)
    model = model.to(mode_config.device)

    # 创建数据集和训练器
    meta_dataset = MSLSMetaDataset(mode_config)
    trainer = MetaTrainer(model, mode_config, meta_dataset)

    # 训练
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # 评估
    metrics = trainer.evaluate()

    result = {
        'gem_mode': gem_mode,
        'mean_map1': metrics.get('mean_map1', 0),
        'mean_map5': metrics.get('mean_map5', 0),
        'mean_recall1': metrics.get('mean_recall1', 0),
        'training_time': elapsed,
        'convergence_step': trainer.convergence_step,
    }

    logger.info(f"Result for {gem_mode}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Ablation Experiments (Paper Section 4.3)')
    parser.add_argument('--datasets_folder', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='msls')
    parser.add_argument('--save_dir', type=str, default='logs/ablation')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--meta_train_steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--modes', type=str, default='fixed_single,fixed_multi,dynamic',
                        help='Comma-separated ablation modes to run')

    args = parser.parse_args()

    setup_logging(args.save_dir)
    logger = logging.getLogger(__name__)

    # 基础配置
    config = MetaConfig()
    config.datasets_folder = args.datasets_folder
    config.dataset_name = args.dataset_name
    config.save_dir = args.save_dir
    config.meta_train_steps = args.meta_train_steps
    config.seed = args.seed
    if args.device:
        config.device = args.device

    config.make_deterministic()

    # 运行消融实验
    modes = args.modes.split(',')
    all_results = {}

    for mode in modes:
        mode = mode.strip()
        try:
            result = run_single_ablation(mode, config, logger)
            all_results[mode] = result
        except Exception as e:
            logger.error(f"Ablation {mode} failed: {e}")
            all_results[mode] = {'error': str(e)}

    # 打印对比表
    print("\n" + "=" * 80)
    print("Ablation Study Results (Paper Table 3)")
    print("=" * 80)
    print(f"{'Method':<25} {'mAP@1':>10} {'mAP@5':>10} {'Recall@1':>10} {'Convergence':>12}")
    print("-" * 80)

    mode_labels = {
        'fixed_single': 'Fixed Single-Param GeM',
        'fixed_multi': 'Fixed Multi-Param GeM',
        'dynamic': 'Dynamic Multi-Param GeM (Ours)',
    }

    for mode in modes:
        mode = mode.strip()
        if mode in all_results and 'error' not in all_results[mode]:
            r = all_results[mode]
            conv = str(r.get('convergence_step', 'N/A'))
            print(f"{mode_labels.get(mode, mode):<25} "
                  f"{r.get('mean_map1', 0):>10.2f} "
                  f"{r.get('mean_map5', 0):>10.2f} "
                  f"{r.get('mean_recall1', 0):>10.2f} "
                  f"{conv:>12}")

    print("=" * 80)

    # 保存结果
    results_path = os.path.join(args.save_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
