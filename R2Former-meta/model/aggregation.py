import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler

import model.functional as LF
import model.normalization as normalization

class MAC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return LF.mac(x)
    def __repr__(self):
        return self.__class__.__name__ + '()'

class SPoC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return LF.spoc(x)
    def __repr__(self):
        return self.__class__.__name__ + '()'

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class DynamicGeMController(nn.Module):
    """LSTM-GeM控制器模块 (对应论文3.3节, 公式(7)-(13))

    严格实现论文公式:
    - 输入为5维特征统计向量s_t (均值、标准差、最大值、最小值、二范数)
    - LSTM隐藏维度128
    - 输出K=8维池化幂次向量p_t, 映射到[1.0, 10.0]区间
    - 参数量约70K (LSTMCell(5,128): 68608 + Linear(128,8): 1032 ≈ 70K)

    支持两种输入模式:
    - paper_mode=True (默认): 输入5维全局统计向量, 严格对齐论文
    - paper_mode=False: 输入groups*5维分组统计向量, 更灵活但参数量更大
    """

    def __init__(self, groups=8, hidden_dim=128, paper_mode=True):
        super().__init__()
        self.groups = groups
        self.hidden_dim = hidden_dim
        self.paper_mode = paper_mode
        # 论文公式(7)-(12): LSTM输入维度
        # paper_mode=True: s_t ∈ R^5 (论文3.3节)
        # paper_mode=False: groups*5 (每组的5个统计量拼接)
        self.input_dim = 5 if paper_mode else groups * 5
        # 公式(7)-(12): LSTMCell
        self.lstm = nn.LSTMCell(self.input_dim, hidden_dim)
        # 公式(13): 线性投影 W_p, b_p
        self.proj = nn.Linear(hidden_dim, groups)

    def init_state(self, batch_size, device, dtype):
        """初始化LSTM隐藏状态 (h_0, c_0) 为零向量"""
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def forward(self, stats, state=None):
        """前向传播: 根据统计向量预测池化幂次

        公式(7)-(13):
        i_t = σ(W_i·s_t + U_i·h_{t-1} + b_i)          (7)
        f_t = σ(W_f·s_t + U_f·h_{t-1} + b_f)          (8)
        o_t = σ(W_o·s_t + U_o·h_{t-1} + b_o)          (9)
        c̃_t = tanh(W_c·s_t + U_c·h_{t-1} + b_c)      (10)
        c_t = f_t⊙c_{t-1} + i_t⊙c̃_t                   (11)
        h_t = o_t⊙tanh(c_t)                             (12)
        p_t = 1_K + 9·σ(W_p·h_t + b_p)                 (13)

        Args:
            stats: 统计向量, paper_mode时为[B, 5], 否则为[B, groups*5]
            state: LSTM状态(h, c), 可选

        Returns:
            p: 池化幂次向量 [B, K], 范围[1.0, 10.0]
            state: 更新后的LSTM状态(h, c)
        """
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)
        if state is None:
            state = self.init_state(stats.shape[0], stats.device, stats.dtype)
        # LSTM更新 (公式7-12)
        h, c = self.lstm(stats, state)
        # 公式(13): p_t = 1 + 9 * sigmoid(W_p·h_t + b_p)
        # Sigmoid输出[0,1], 映射到[1.0, 10.0]
        p = 1.0 + 9.0 * torch.sigmoid(self.proj(h))
        return p, (h, c)


class DynamicMultiGeM(nn.Module):
    """动态多参数GeM池化模块 (对应论文3.2节, 公式(3)-(6))

    严格实现论文公式:
    - 将384个通道均匀划分为K=8组, 每组48个通道
    - 公式(4): z_j = (∑_{u∈Ω} X_{j,u}^{p_k})^{1/p_k}, j∈G_k, p_k>0
    - 公式(5): α_k = exp(p_k) / ∑_{l=1}^{K} exp(p_l)  (自适应融合权重)
    - 公式(6): z_j = α_k · z_j, j∈G_k (加权后拼接输出384维描述子)

    支持三种模式(用于消融实验, 对应论文4.3节):
    - "dynamic": 动态多参数GeM (本文方法)
    - "fixed_multi": 固定多参数GeM (8组p值均固定, 如p=3.0)
    - "fixed_single": 固定单参数GeM (传统方式, p=3.0)
    """

    def __init__(self, p=3, eps=1e-6, groups=8, hidden_dim=128,
                 work_with_tokens=False, paper_mode=True, gem_mode="dynamic"):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.work_with_tokens = work_with_tokens
        self.paper_mode = paper_mode
        self.gem_mode = gem_mode  # "dynamic", "fixed_multi", "fixed_single"

        # 公式(4): 基础池化幂次参数 (每组一个, 初始化为p=3.0)
        self.base_p = Parameter(torch.ones(groups) * p)

        if gem_mode == "dynamic":
            # 公式(7)-(13): LSTM-GeM控制器
            self.controller = DynamicGeMController(
                groups=groups, hidden_dim=hidden_dim, paper_mode=paper_mode
            )
        else:
            self.controller = None

        # 缓存机制: 支持内循环中覆写p值
        self.cached_p = None
        self.cached_state = None

    def reset_task_state(self):
        self.cached_p = None
        self.cached_state = None

    def set_override_p(self, p, state=None):
        self.cached_p = p
        self.cached_state = state

    def clear_override_p(self):
        self.cached_p = None
        self.cached_state = None

    def _to_token_sequence(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            return x.view(b, c, h * w).permute(0, 2, 1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Unsupported feature shape for DynamicMultiGeM: {tuple(x.shape)}")

    def _resolve_p(self, batch_size, device, dtype, p_override=None):
        if p_override is None:
            p_override = self.cached_p
        if p_override is None:
            p_override = self.base_p.unsqueeze(0)
        if p_override.dim() == 1:
            p_override = p_override.unsqueeze(0)
        if p_override.shape[0] == 1 and batch_size > 1:
            p_override = p_override.expand(batch_size, -1)
        return p_override.to(device=device, dtype=dtype).clamp(min=1.0, max=10.0)

    def extract_statistics(self, x, reduce_batch=False):
        """提取特征统计量 (对应论文3.3节, 公式(7)的输入s_t)

        paper_mode=True: 计算全局5维统计向量 [均值, 标准差, 最大值, 最小值, 二范数]
        paper_mode=False: 计算每组的5维统计向量, 拼接为groups*5维

        Args:
            x: 特征图 [B, C, H, W] 或token序列 [B, N, C]
            reduce_batch: 是否在batch维度取均值 (用于元学习内循环)

        Returns:
            stats: paper_mode时[B, 5], 否则[B, groups*5]
        """
        tokens = self._to_token_sequence(x)
        if self.paper_mode:
            # 论文模式: 全局统计向量 s_t ∈ R^5
            mean_v = tokens.mean(dim=(1, 2))
            std_v = tokens.std(dim=(1, 2), unbiased=False)
            max_v = tokens.amax(dim=(1, 2))
            min_v = tokens.amin(dim=(1, 2))
            l2_v = tokens.pow(2).mean(dim=(1, 2)).sqrt()
            stats = torch.stack([mean_v, std_v, max_v, min_v, l2_v], dim=1)
        else:
            # 分组模式: 每组5维统计量
            chunks = torch.chunk(tokens, self.groups, dim=2)
            stat_list = []
            for chunk in chunks:
                mean_v = chunk.mean(dim=(1, 2))
                std_v = chunk.std(dim=(1, 2), unbiased=False)
                max_v = chunk.amax(dim=(1, 2))
                min_v = chunk.amin(dim=(1, 2))
                l2_v = chunk.pow(2).mean(dim=(1, 2)).sqrt()
                stat_list.extend([mean_v, std_v, max_v, min_v, l2_v])
            stats = torch.stack(stat_list, dim=1)
        if reduce_batch:
            stats = stats.mean(dim=0, keepdim=True)
        return stats

    def predict_p(self, x_or_stats, state=None, reduce_batch=True):
        if x_or_stats.dim() >= 3:
            stats = self.extract_statistics(x_or_stats, reduce_batch=reduce_batch)
        else:
            stats = x_or_stats
        p, state = self.controller(stats, state=state)
        return p, state

    def pool_tokens(self, x, p_override=None):
        """动态多参数GeM池化 (对应论文3.2节, 公式(3)-(6))

        公式(4): z_j = (∑_{u∈Ω} X_{j,u}^{p_k})^{1/p_k}, j∈G_k
        公式(5): α_k = exp(p_k) / ∑_{l=1}^{K} exp(p_l)  (softmax over p values)
        公式(6): z_j = α_k · z_j, j∈G_k

        Args:
            x: token序列 [B, N, C] 或特征图 [B, C, H, W]
            p_override: 覆盖的p值 [B, K] 或 [K], 可选

        Returns:
            全局描述子 [B, C] (C=384, 8组×48=384)
        """
        tokens = self._to_token_sequence(x)
        batch_size = tokens.shape[0]
        p_values = self._resolve_p(batch_size, tokens.device, tokens.dtype, p_override)

        # 公式(4): 分组GeM池化
        group_descriptors = []
        for group_idx, chunk in enumerate(torch.chunk(tokens, self.groups, dim=2)):
            group_p = p_values[:, group_idx].view(batch_size, 1, 1)
            # 数值稳定性: 对输入做非负约束, 避免幂次计算错误
            pooled = chunk.clamp(min=self.eps).pow(group_p).mean(dim=1)
            pooled = pooled.pow(1.0 / group_p.squeeze(2))
            group_descriptors.append(pooled)

        # 公式(5): α_k = exp(p_k) / ∑_l exp(p_l) = softmax(p)
        # 论文严格公式: 直接对p值做softmax, 不经过额外线性层
        fusion_weights = torch.softmax(p_values, dim=1)  # [B, K]

        # 公式(6): z_j = α_k · z_j, j∈G_k
        weighted_groups = [
            descriptor * fusion_weights[:, idx].unsqueeze(1)
            for idx, descriptor in enumerate(group_descriptors)
        ]
        # 拼接输出384维描述子
        return torch.cat(weighted_groups, dim=1)

    def forward(self, x, p_override=None):
        pooled = self.pool_tokens(x, p_override=p_override)
        return pooled.unsqueeze(2).unsqueeze(3)

class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super().__init__()
        self.L = L
        self.eps = eps
    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class RRM(nn.Module):
    """Residual Retrieval Module as described in the paper 
    `Leveraging EfficientNet and Contrastive Learning for AccurateGlobal-scale 
    Location Estimation <https://arxiv.org/pdf/2105.07645.pdf>`
    """
    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)
        self.ln2 = nn.LayerNorm(normalized_shape=dim)
        self.l2 = normalization.L2Norm()
    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.ln1(x)
        identity = x
        out = self.fc2(self.relu(self.fc1(x)))
        out += identity
        out = self.l2(self.ln2(out))
        return out


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(args.features_dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)


class CRNModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Downsample pooling
        self.downsample_pool = nn.AvgPool2d(kernel_size=3, stride=(2, 2),
                                            padding=0, ceil_mode=True)
        
        # Multiscale Context Filters
        self.filter_3_3 = nn.Conv2d(in_channels=dim, out_channels=32,
                                    kernel_size=(3, 3), padding=1)
        self.filter_5_5 = nn.Conv2d(in_channels=dim, out_channels=32,
                                    kernel_size=(5, 5), padding=2)
        self.filter_7_7 = nn.Conv2d(in_channels=dim, out_channels=20,
                                    kernel_size=(7, 7), padding=3)
        
        # Accumulation weight
        self.acc_w = nn.Conv2d(in_channels=84, out_channels=1, kernel_size=(1, 1))
        # Upsampling
        self.upsample = F.interpolate
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize Context Filters
        torch.nn.init.xavier_normal_(self.filter_3_3.weight)
        torch.nn.init.constant_(self.filter_3_3.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_5_5.weight)
        torch.nn.init.constant_(self.filter_5_5.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_7_7.weight)
        torch.nn.init.constant_(self.filter_7_7.bias, 0.0)
        
        torch.nn.init.constant_(self.acc_w.weight, 1.0)
        torch.nn.init.constant_(self.acc_w.bias, 0.0)
        self.acc_w.weight.requires_grad = False
        self.acc_w.bias.requires_grad = False
    
    def forward(self, x):
        # Contextual Reweighting Network
        x_crn = self.downsample_pool(x)
        
        # Compute multiscale context filters g_n
        g_3 = self.filter_3_3(x_crn)
        g_5 = self.filter_5_5(x_crn)
        g_7 = self.filter_7_7(x_crn)
        g = torch.cat((g_3, g_5, g_7), dim=1)
        g = F.relu(g)
        
        w = F.relu(self.acc_w(g))  # Accumulation weight
        mask = self.upsample(w, scale_factor=2, mode='bilinear')  # Reweighting Mask
        
        return mask


class CRN(NetVLAD):
    def __init__(self, clusters_num=64, dim=128, normalize_input=True):
        super().__init__(clusters_num, dim, normalize_input)
        self.crn = CRNModule(dim)
    
    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        
        mask = self.crn(x)
        
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # Weight soft_assign using CRN's mask
        soft_assign = soft_assign * mask.view(N, 1, H * W)
        
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[D:D + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D:D + 1, :].unsqueeze(2)
            vlad[:, D:D + 1, :] = residual.sum(dim=-1)
        
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

