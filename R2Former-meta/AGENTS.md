## 项目概述
- **名称**: R2Former 动态GeM元学习框架
- **功能**: 基于论文《跨城市视觉定位动态GeM元学习》，在R2Former基础上实现LSTM控制的动态多参数GeM池化+双层元学习训练，实现零侵入、轻量级的跨城市视觉地点识别

### 节点清单
| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| DynamicGeMController | `model/aggregation.py` | core | LSTM-GeM控制器(论文3.3节,公式7-13),5维统计→8维幂次 | - | - |
| DynamicMultiGeM | `model/aggregation.py` | core | 动态多参数GeM池化(论文3.2节,公式3-6),8组独立幂次+加权融合 | - | - |
| MiniVELOOptimizer | `meta/minivelo_optimizer.py` | core | 冻结的MiniVeLO优化器(论文3.4节),内循环任务适配 | - | - |
| MetaTrainer | `meta/meta_trainer.py` | core | 双层元学习训练引擎(论文算法1) | 内循环→外循环 | - |
| MSLSMetaDataset | `meta/msls_meta_dataset.py` | core | MSLS城市级元任务构建(论文3.1节) | 城市→支持集/查询集 | - |
| MetaEvaluator | `meta/meta_evaluator.py` | core | 性能评估(mAP@1/5, Recall@1)(论文4.1-4.2节) | - | - |
| AblationSwitch | `meta/ablation.py` | core | 消融实验切换(论文4.3节) | dynamic/fixed_multi/fixed_single | - |
| MetaConfig | `meta/config.py` | config | 元学习配置与超参数(论文4.1节) | - | - |

**类型说明**: core(核心模块) / config(配置) / ablation(消融实验)

## 模块清单
| 模块 | 文件位置 | 功能描述 | 论文对应 |
|------|---------|---------|---------|
| DynamicMultiGeM | `model/aggregation.py` | 8组通道独立动态GeM池化 | 论文3.2节,公式(3)-(6) |
| DynamicGeMController | `model/aggregation.py` | LSTM控制器预测池化幂次 | 论文3.3节,公式(7)-(13) |
| MSLSMetaDataset | `meta/msls_meta_dataset.py` | MSLS城市级元任务采样与划分 | 论文3.1节,4.1节 |
| MetaTrainer | `meta/meta_trainer.py` | 双层元学习训练(算法1) | 论文3.4节,算法1 |
| MiniVELOOptimizer | `meta/minivelo_optimizer.py` | 冻结MiniVeLO优化器 | 论文3.4节 |
| MetaEvaluator | `meta/meta_evaluator.py` | mAP@1/5, Recall@1评估 | 论文4.1-4.2节 |
| Ablation | `meta/ablation.py` | 消融实验切换 | 论文4.3节 |
| MetaConfig | `meta/config.py` | 超参数配置 | 论文4.1节 |

## 论文公式与代码对应
| 论文公式 | 代码位置 | 描述 |
|---------|---------|------|
| 公式(3) | `DynamicMultiGeM.pool_tokens()` | 分组GeM池化 z_j = (Σx^p_k)^{1/p_k} |
| 公式(4) | `DynamicMultiGeM.pool_tokens()` | 通道分组G_k,每组48通道 |
| 公式(5) | `DynamicMultiGeM.pool_tokens()` | 自适应融合权重α_k = softmax(p) |
| 公式(6) | `DynamicMultiGeM.pool_tokens()` | 加权拼接输出 z_j = α_k·z_j |
| 公式(7)-(12) | `DynamicGeMController.forward()` | LSTM单元更新 |
| 公式(13) | `DynamicGeMController.forward()` | p_t = 1 + 9·sigmoid(W_p·h_t + b_p) |
| 算法1第4行 | `MetaTrainer.inner_loop_adapt()` | 初始化(θ̃,φ̃) = (θ,φ) |
| 算法1第5-12行 | `MetaTrainer.inner_loop_adapt()` | 内循环τ步适配 |
| 算法1第13行 | `MetaTrainer.meta_train_step()` | 查询集元损失L_qry |
| 算法1第14-16行 | `MetaTrainer.meta_train_step()` | 仅更新φ的外循环元更新 |

## 消融实验配置
| 模式 | gem_mode值 | 描述 | 论文对应 |
|------|-----------|------|---------|
| 基线 | `fixed_single` | 固定单参数GeM(p=3.0) | 论文4.3节消融1 |
| 对照 | `fixed_multi` | 固定多参数GeM(8组p=3.0) | 论文4.3节消融2 |
| 本文方法 | `dynamic` | 动态多参数GeM(LSTM控制) | 论文4.3节消融3 |

## 超参数(论文4.1节严格对齐)
- GEM分组数K=8
- 内循环步数τ=20
- 元训练总步数500
- 元批次大小B=2
- 训练批次大小4
- 外循环学习率β=1e-4
- 随机种子42
- GeM幂次范围[1.0, 10.0]
- LSTM隐藏维度128
- 控制器参数量≈70K

## 入口脚本
- **元训练**: `python meta_train.py --datasets_folder /path/to/datasets --dataset_name msls --aggregation dynamic_gem`
- **元评估**: `python meta_eval.py --datasets_folder /path/to/datasets --checkpoint logs/meta_learning/best_controller.pth`
- **消融实验**: `python meta_train.py --gem_mode fixed_single` (基线) / `--gem_mode fixed_multi` (对照) / `--gem_mode dynamic` (本文)

## 依赖
- Python >= 3.8
- PyTorch >= 1.12
- torchvision
- transformers (DeiT模型)
- faiss-cpu / faiss-gpu
- numpy, tqdm, Pillow
