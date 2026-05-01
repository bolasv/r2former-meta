# meta/msls_meta_dataset.py
"""
MSLS数据集城市级元任务构建模块 (对应论文3.1节、4.1节)

核心功能:
1. 基于MSLS数据集天然城市划分, 将每个城市定义为独立元任务T_c
2. 对每个城市任务, 自动划分支持集D_c^sup和查询集D_c^qry
3. 兼容R2Former官方三元组数据加载逻辑 (1正+5负)
4. 严格对齐论文的城市划分实验设置
"""
import os
import logging
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from os.path import join
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# 图像预处理: 缩放+中心裁剪+归一化
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    """将图像路径转换为PIL RGB图像"""
    return Image.open(path).convert("RGB")


class CityTask:
    """单个城市的元任务定义 (对应论文3.1节)

    每个城市T_c包含:
    - 支持集 D_c^sup: 用于内循环任务适配
    - 查询集 D_c^qry: 用于外循环元损失计算
    """

    def __init__(self, city_name: str, support_indices: List[int],
                 query_indices: List[int], all_indices: List[int]):
        self.city_name = city_name
        self.support_indices = support_indices   # 支持集索引
        self.query_indices = query_indices       # 查询集索引
        self.all_indices = all_indices           # 该城市所有样本索引

    def __repr__(self):
        return (f"CityTask({self.city_name}: "
                f"support={len(self.support_indices)}, "
                f"query={len(self.query_indices)})")


class MSLSMetaDataset:
    """MSLS城市级元任务数据集 (对应论文3.1节)

    基于MSLS数据集的天然城市划分构造元任务:
    - 训练集5个城市 → 元训练任务分布p(T)
    - 测试集5-6个未见城市 → 元测试评估

    每个元任务T_c:
    - 支持集D_c^sup: 用于内循环适配 (论文算法1第8-12行)
    - 查询集D_c^qry: 用于外循环元损失 (论文算法1第13行)
    """

    def __init__(self, config):
        """
        Args:
            config: MetaConfig实例
        """
        self.config = config
        self.datasets_folder = config.datasets_folder
        self.dataset_name = config.dataset_name
        self.resize = config.resize
        self.negs_num = config.negs_num_per_query

        # 元训练/测试城市列表
        self.meta_train_cities = config.meta_train_cities
        self.meta_test_cities = config.meta_test_cities

        # 每个城市的数据索引缓存
        self.city_data = {}  # city_name -> dict with 'db_paths', 'q_paths', 'pIdx', 'nonNegIdx'

        # 初始化各城市数据
        self._load_city_data()

    def _load_city_data(self):
        """加载所有城市的MSLS数据索引"""
        from mapillary_sls_main.mapillary_sls.datasets.msls import MSLS

        msls_root = join(self.datasets_folder, self.dataset_name)

        for city in self.meta_train_cities + self.meta_test_cities:
            try:
                # 尝试加载单个城市的数据
                msls_dataset = MSLS(
                    root_dir=msls_root,
                    save=True,
                    cities=city,
                    mode='train' if city in self.meta_train_cities else 'val',
                    nNeg=self.negs_num,
                    posDistThr=25,
                )
                self.city_data[city] = {
                    'db_paths': msls_dataset.dbImages,
                    'q_paths': msls_dataset.qImages,
                    'pIdx': msls_dataset.pIdx,
                    'nonNegIdx': msls_dataset.nonNegIdx,
                    'qIdx': msls_dataset.qIdx,
                }
                logger.info(f"Loaded city {city}: "
                            f"{len(self.city_data[city]['db_paths'])} db, "
                            f"{len(self.city_data[city]['q_paths'])} queries")
            except Exception as e:
                logger.warning(f"Could not load city {city}: {e}")
                # 创建空数据占位
                self.city_data[city] = {
                    'db_paths': np.array([]),
                    'q_paths': np.array([]),
                    'pIdx': [],
                    'nonNegIdx': [],
                    'qIdx': np.array([]),
                }

    def sample_meta_batch(self, meta_batch_size: int, split: str = "train") -> List[CityTask]:
        """采样一个元批次的城市任务 (对应论文算法1第3行)

        从p(T)采样B个城市任务, 每个城市自动划分支持集和查询集

        Args:
            meta_batch_size: 元批次大小B (论文4.1节: B=2)
            split: "train" 或 "test"

        Returns:
            List[CityTask]: 采样的城市任务列表
        """
        cities = self.meta_train_cities if split == "train" else self.meta_test_cities
        # 随机采样B个城市 (可重复采样)
        sampled_cities = random.choices(cities, k=meta_batch_size)

        tasks = []
        for city in sampled_cities:
            task = self._create_city_task(city)
            if task is not None:
                tasks.append(task)

        return tasks

    def _create_city_task(self, city_name: str) -> Optional[CityTask]:
        """为单个城市创建元任务, 划分支持集和查询集

        对每个城市任务T_c, 将查询样本按比例划分为:
        - 支持集D_c^sup: 用于内循环适配
        - 查询集D_c^qry: 用于外循环元损失计算

        Args:
            city_name: 城市名称

        Returns:
            CityTask实例, 或None(如果城市数据不足)
        """
        city_info = self.city_data.get(city_name)
        if city_info is None:
            return None

        num_queries = len(city_info['q_paths'])
        num_db = len(city_info['db_paths'])

        if num_queries < 4 or num_db < 4:
            logger.warning(f"City {city_name} has insufficient data "
                           f"({num_queries} queries, {num_db} db), skipping")
            return None

        # 将查询样本划分为支持集和查询集
        query_indices = list(range(num_queries))
        random.shuffle(query_indices)

        support_size = max(2, int(num_queries * self.config.support_ratio))
        support_indices = query_indices[:support_size]
        query_set_indices = query_indices[support_size:]

        # 如果查询集为空, 至少保留2个
        if len(query_set_indices) < 2:
            query_set_indices = query_indices[max(0, support_size - 2):]
            support_indices = query_indices[:max(2, support_size - 2)]

        all_indices = list(range(num_queries + num_db))

        return CityTask(
            city_name=city_name,
            support_indices=support_indices,
            query_indices=query_set_indices,
            all_indices=all_indices,
        )

    def get_triplet_batch(self, city_task: CityTask, batch_size: int,
                          split: str = "support") -> Dict[str, torch.Tensor]:
        """获取一个城市任务的三元组批次 (兼容R2Former官方数据加载)

        每个查询样本对应1个正样本和negs_num个负样本,
        三元组索引张量维度对齐论文定义

        Args:
            city_task: 城市元任务
            batch_size: 批次大小 (论文4.1节: 4)
            split: "support" 或 "query"

        Returns:
            Dict包含:
            - images: [batch_size*(1+1+negs_num), 3, H, W]
            - triplets: [batch_size*negs_num, 3] 三元组索引
            - features_stats_input: 特征统计量输入占位
        """
        city_info = self.city_data.get(city_task.city_name)
        if city_info is None:
            return {}

        # 选择支持集或查询集的索引
        if split == "support":
            query_indices = city_task.support_indices
        else:
            query_indices = city_task.query_indices

        if len(query_indices) == 0:
            return {}

        # 采样batch_size个查询
        sampled_queries = random.choices(query_indices, k=min(batch_size, len(query_indices)))

        images_list = []
        triplets_list = []
        img_offset = 0

        for q_idx in sampled_queries:
            # 获取查询图像路径
            q_path = city_info['q_paths'][q_idx]

            # 获取正样本
            positives = city_info['pIdx'][q_idx] if q_idx < len(city_info['pIdx']) else []
            if len(positives) == 0:
                continue
            pos_idx = random.choice(positives) if isinstance(positives, list) else positives
            pos_path = city_info['db_paths'][pos_idx]

            # 获取负样本
            non_negs = city_info['nonNegIdx'][q_idx] if q_idx < len(city_info['nonNegIdx']) else []
            all_db_indices = list(range(len(city_info['db_paths'])))
            neg_candidates = [i for i in all_db_indices if i not in non_negs] if len(non_negs) > 0 else all_db_indices

            if len(neg_candidates) < self.negs_num:
                neg_candidates = all_db_indices

            neg_indices = random.sample(neg_candidates, min(self.negs_num, len(neg_candidates)))
            neg_paths = [city_info['db_paths'][i] for i in neg_indices]

            # 加载和预处理图像
            try:
                q_img = base_transform(path_to_pil_img(q_path))
                p_img = base_transform(path_to_pil_img(pos_path))
                n_imgs = torch.stack([base_transform(path_to_pil_img(p)) for p in neg_paths])
                batch_imgs = torch.cat([q_img.unsqueeze(0), p_img.unsqueeze(0), n_imgs], dim=0)
                images_list.append(batch_imgs)

                # 构造三元组索引
                for n_i in range(len(neg_indices)):
                    triplets_list.append([img_offset, img_offset + 1, img_offset + 2 + n_i])
                img_offset += len(batch_imgs)
            except Exception as e:
                logger.warning(f"Error loading images for city {city_task.city_name}: {e}")
                continue

        if len(images_list) == 0:
            return {}

        images = torch.cat(images_list, dim=0)
        triplets = torch.tensor(triplets_list, dtype=torch.long)

        return {
            'images': images,
            'triplets': triplets,
            'city_name': city_task.city_name,
        }
