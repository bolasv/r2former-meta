
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel
from googledrivedownloader import download_file_from_google_drive
from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock

from model.R2Former import R2Former
from functools import partial
import numpy as np
import torch.nn.functional as F
# from matplotlib.patches import Circle
from model.Deit import DistilledVisionTransformer, deit_small_distilled_patch16_224, deit_base_distilled_patch16_384




# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places': '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places': '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places': '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places': '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2': '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2': '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2': '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2': '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. 可学习的亮度/对比度调整层
class LearnableBrightnessContrast(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 初始化参数：contrast接近1，brightness接近0（无变换）
        self.contrast = nn.Parameter(torch.ones(in_channels))
        self.brightness = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        # x: [B, C, H, W]，224x224的图像输入
        # 对每个通道独立调整对比度和亮度
        x = x * self.contrast.view(1, -1, 1, 1) + self.brightness.view(1, -1, 1, 1)
        # 限制输出范围在[0,1]（适用于归一化后的图像）
        return torch.clamp(x, 0, 1)


# 2. 可学习的仿射变换层（平移、旋转、缩放）
class LearnableAffineTransform(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.img_size = img_size
        # 初始化仿射矩阵参数（单位矩阵，无变换）
        self.theta = nn.Parameter(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32))

    def forward(self, x):
        # 生成网格并应用仿射变换
        grid = F.affine_grid(self.theta.unsqueeze(0).repeat(x.shape[0], 1, 1),
                             x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


# 3. 可学习的图像滤波层（边缘增强/模糊等）
class LearnableFilterLayer(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # 初始化卷积核（接近单位滤波器，无变换）
        self.kernel = nn.Parameter(torch.zeros(in_channels, 1, kernel_size, kernel_size))
        # 中心元素设为1，其余为0（单位滤波）
        nn.init.constant_(self.kernel[:, :, kernel_size // 2, kernel_size // 2], 1.0)

    def forward(self, x):
        # 应用可学习滤波，保持尺寸不变
        padding = self.kernel_size // 2
        x = F.conv2d(x, self.kernel, padding=padding, groups=x.shape[1])
        return x


class ImageEnhancementNet(nn.Module):
    def __init__(self, img_size=224, in_channels=3):
        super().__init__()
        self.brightness_contrast = LearnableBrightnessContrast(in_channels)
        self.filter = LearnableFilterLayer(in_channels)
        self.affine = LearnableAffineTransform(img_size)

    def forward(self, x):
        x = self.brightness_contrast(x)
        x = self.filter(x)
        # 可选：仿射变换计算成本较高，按需使用
        # x = self.affine(x)
        return x

class AttnModule(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads  # 头数：6（384/64=6，单头维度64）
        self.head_dim = embed_dim // num_heads  # 单头维度：64
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # qkv合并投影层
        self.scale = self.head_dim ** -0.5  # 缩放因子：1/√64=0.125

    def forward(self, x):
        # 仅作为参数容器，forward逻辑在主模型中手动实现（保持你的原有逻辑）
        return x


# 2. 顶层blk模块：包含 norm1 和 attn 子模块
class CustomBlk(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  # blk直接包含norm1
        self.attn = AttnModule(embed_dim, num_heads)  # blk直接包含attn子模块


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载DINOv2-ViT-L/14（全局特征维度1024）
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', force_reload=True)
        # 冻结教师模型权重（关键：蒸馏时教师不更新）
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()  # 推理模式，关闭 dropout/BN 更新
        self.multi_out =256
        self.Reranker = R2Former(decoder_depth=3, decoder_num_heads=4,
                                 decoder_embed_dim=32, decoder_mlp_ratio=4,
                                 decoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 num_classes=2, num_patches=2 * self.multi_out,
                                 input_dim=384, num_corr=5)
        self.small_cnn = nn.Sequential(
            # 第一层：3->64，下采样到112x112
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 第二层：64->128，下采样到56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 第三层：128->256，下采样到28x28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第四层：256->384，下采样到16x16（与DINOv2的patch数量一致）
            nn.Conv2d(256, 384, kernel_size=2, stride=2, padding=2),

            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(1024, 384),  # 拼接后投影回384维度
            nn.LayerNorm(384)
        )
        self.blk = CustomBlk(
            embed_dim=384,
            num_heads=6
        )
        self.img_enhancement = ImageEnhancementNet()
        self.single = False

    def forward(self, x):
        x = self.img_enhancement(x)
        B, C, H, W = x.shape
        blk = self.blk
        # 1. 小型CNN处理输入图像x，提取空间特征
        #cnn_feat = self.small_cnn(x)  # (B, 384, 16, 16)（16=224/14，与DINOv2的patch尺寸匹配）
        # CNN特征转为序列形式：(B, 384, 16, 16) -> (B, 16*16, 384) = (B, 256, 384)
        #cnn_feat_seq = cnn_feat.flatten(2).transpose(1, 2)  # 展平空间维度，转成序列
        feature = self.model(x)
        feature = self.feature_fusion(feature)
        output = self.model.get_intermediate_layers(x, n=1, return_class_token=False)[0]
        output = self.feature_fusion(output)
        #combined_output = self.feature_fusion(torch.cat([output, cnn_feat_seq], dim=-1))  # (B, 256, 384)

        y = blk.norm1(output)  # 归一化
        B, N, C = y.shape
        qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1,
                                                                                                        4)  # 划分多头，B, N, 3, num_heads, d_k]，各维度含义：[批量, 序列长度, q/k/v, 头数, 单头维度]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        att = (q @ k.transpose(-2, -1)) * blk.attn.scale
        att = att.softmax(dim=-1)
        last_map = (att[:, :, :, :].detach()).sum(dim=1).sum(dim=1)  # [B, dim-2]
        order = torch.argsort(last_map, dim=1, descending=True)
        multi_out = np.minimum(order.shape[1], self.multi_out)
        local_features = torch.gather(input=output,
                                          index=order[:, :multi_out].unsqueeze(2).repeat(1, 1, output.shape[2]),
                                          dim=1)
        # compute attention and coordinates
        HW = max(H, W)
        x_xy = torch.cat([(order[:, :multi_out].unsqueeze(2) % np.ceil(W / 14).astype(int) * 14 + 7) / 1. / HW,
                          (order[:, :multi_out].unsqueeze(2) // np.ceil(W / 14).astype(int) * 14 + 7) / 1. / HW],
                         dim=2)
        x_attention = torch.sort(last_map, dim=1, descending=True)[0][:, :multi_out]
        x_attention = (x_attention / torch.max(x_attention, dim=1, keepdim=True)[0]).reshape(x_xy.shape[0],
                                                                                                 x_xy.shape[1], 1)
        if self.single:
            return  feature
        else:
            return  feature, torch.cat([x_xy, x_attention, local_features],dim=2)

class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """

    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
        self.self_att = False

        if args.aggregation in ["gem", "spoc", "mac", "rmac", "dynamic_gem"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())

        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim
        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                            channel_inner=args.channel_bottleneck)] * args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True
        self.single = True

    def forward(self, x):
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            return x
        x = self.aggregation(x)
        return x


class GeoLocalizationNetRerank(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.arch_name = args.backbone
        self.out_dim = args.local_dim
        if args.backbone.startswith("deit"):
            if args.backbone == 'deitBase':
                self.backbone = deit_base_distilled_patch16_384(img_size=args.resize, num_classes=args.fc_output_dim,
                                                                embed_layer=AnySizePatchEmbed)
            else:
                self.backbone = deit_small_distilled_patch16_224(img_size=args.resize, num_classes=args.fc_output_dim,
                                                                 embed_layer=AnySizePatchEmbed)
            args.features_dim = args.fc_output_dim
            if args.hypercolumn:
                self.hyper_s = args.hypercolumn // 100
                self.hyper_e = args.hypercolumn % 100
                self.local_head = nn.Linear(self.backbone.embed_dim * (self.hyper_e - self.hyper_s), self.out_dim,
                                            bias=True)
            else:
                self.local_head = nn.Linear(self.backbone.embed_dim, self.out_dim, bias=True)
            self.dynamic_aggregation = None
            if args.aggregation == "dynamic_gem":
                self.dynamic_aggregation = aggregation.DynamicMultiGeM(
                    p=args.dynamic_gem_init_p,
                    eps=args.dynamic_gem_eps,
                    groups=args.dynamic_gem_groups,
                    hidden_dim=args.dynamic_gem_hidden_dim,
                    work_with_tokens=True,
                    paper_mode=getattr(args, 'paper_mode', True),
                    gem_mode=getattr(args, 'gem_mode', 'dynamic'),
                )
        else:
            self.backbone = get_backbone(args)
            self.aggregation = get_aggregation(args)
            self.self_att = False

            if args.aggregation in ["gem", "spoc", "mac", "rmac", "dynamic_gem"]:
                if args.l2 == "before_pool":
                    self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
                elif args.l2 == "after_pool":
                    self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
                elif args.l2 == "none":
                    self.aggregation = nn.Sequential(self.aggregation, Flatten())

            if args.fc_output_dim != None:
                # Concatenate fully connected layer to the aggregation layer
                self.aggregation = nn.Sequential(self.aggregation,
                                                 nn.Linear(args.features_dim, args.fc_output_dim),
                                                 L2Norm())
                args.features_dim = args.fc_output_dim
            if args.non_local:
                non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                                channel_inner=args.channel_bottleneck)] * args.num_non_local
                self.non_local = nn.Sequential(*non_local_list)
                self.self_att = True
            if args.hypercolumn:
                self.local_head = nn.Linear(1856, self.out_dim, bias=True)
            else:
                self.local_head = nn.Linear(1024, self.out_dim, bias=True)
        # ==================================================================
        self.local_head.weight.data.normal_(mean=0.0, std=0.01)
        self.local_head.bias.data.zero_()
        self.multi_out = args.num_local
        self.single = False
        if args.rerank_model == 'r2former':
            self.Reranker = R2Former(decoder_depth=6, decoder_num_heads=4,
                                     decoder_embed_dim=32, decoder_mlp_ratio=4,
                                     decoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                     num_classes=2, num_patches=2 * self.multi_out,
                                     input_dim=args.fc_output_dim, num_corr=5)
        else:
            print('rerank_model not implemented!')
            raise Exception

    def reset_dynamic_gem_state(self):
        if hasattr(self, "dynamic_aggregation") and self.dynamic_aggregation is not None:
            self.dynamic_aggregation.reset_task_state()

    def set_dynamic_gem_p(self, p_values, state=None):
        if hasattr(self, "dynamic_aggregation") and self.dynamic_aggregation is not None:
            self.dynamic_aggregation.set_override_p(p_values, state=state)

    def clear_dynamic_gem_p(self):
        if hasattr(self, "dynamic_aggregation") and self.dynamic_aggregation is not None:
            self.dynamic_aggregation.clear_override_p()

    def predict_dynamic_gem(self, tokens, state=None, reduce_batch=True):
        if not hasattr(self, "dynamic_aggregation") or self.dynamic_aggregation is None:
            raise RuntimeError("Dynamic GeM is only available when --aggregation dynamic_gem is enabled.")
        return self.dynamic_aggregation.predict_p(tokens, state=state, reduce_batch=reduce_batch)

    def _pool_deit_global_descriptor(self, tokens):
        if hasattr(self, "dynamic_aggregation") and self.dynamic_aggregation is not None:
            projected_tokens = self.backbone.head(tokens[:, 2:])
            descriptor = self.dynamic_aggregation.pool_tokens(projected_tokens)
            return self.backbone.l2_norm(descriptor), projected_tokens
        x_cls = self.backbone.head(tokens[:, 0])
        x_dist = self.backbone.head_dist(tokens[:, 1])
        return self.backbone.l2_norm((x_cls + x_dist) / 2), tokens[:, 2:]

    def forward_ori(self, x):
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            x=nn.Linear(768,256)(x)
            return x
        x = self.aggregation(x)
        return x

    def res_forward(self, x):
        # print(self.backbone)
        # raise Exception
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        x0 = x * 1  #.detach()

        x = self.backbone[4](x)
        x1 = x * 1  #.detach()
        x = self.backbone[5](x)
        x2 = x * 1  #.detach()
        x = self.backbone[6](x)
        x3 = x * 1
        x = self.backbone[7](x)
        x4 = x * 1  #.detach()

        if self.args.hypercolumn:
            B, C, H, W = x3.shape
            local_feature = torch.cat([
                F.interpolate(x0, size=(H, W), mode='bicubic'),  # 64
                F.interpolate(x1, size=(H, W), mode='bicubic'),  # 256
                F.interpolate(x2, size=(H, W), mode='bicubic'),  # 512
                x3,  # 1024
                # F.interpolate(x4, size=(H, W), mode='bicubic'),
            ], dim=1)
        else:
            local_feature = x3

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x, local_feature, x4

    def forward_cnn(self, x):
        # with torch.no_grad():
        B, _, H, W = x.shape
        query_img = x.clone()
        x, feature, feature_last = self.res_forward(x)
        x = self.aggregation(x)

        _, C, f_H, f_W = feature.shape
        assert f_H == np.ceil(H / 16).astype(int) and f_W == np.ceil(W / 16).astype(int)
        feature_reshape = feature.permute((0, 2, 3, 1)).reshape(B, f_H * f_W, C)
        # print(feature_last.shape, feature_reshape.shape, query_img.shape, H//32, W//32)
        feature_last_reshape = feature_last.permute((0, 2, 3, 1)).reshape(B, np.ceil(H / 32).astype(int) * np.ceil(
            W / 32).astype(int), 2048)
        feature_last_reshape = F.normalize(feature_last_reshape, p=2, dim=2)
        # print(self.aggregation)
        fc_weight = self.aggregation[1].weight.t()
        # fc_weight = torch.eye(2048, dtype=torch.float32).cuda()
        sim = torch.matmul(feature_last_reshape.clamp(min=1e-6), fc_weight)
        last_map = (sim.clamp(min=1e-6)).sum(dim=2)  # /sim.max(dim=1,keepdim=True)[0]
        last_map_reshape = F.interpolate(
            last_map.reshape([B, 1, np.ceil(H / 32).astype(int), np.ceil(W / 32).astype(int)]),
            size=(np.ceil(H / 16).astype(int), np.ceil(W / 16).astype(int)), mode='bicubic')
        last_map = last_map_reshape.reshape(B, np.ceil(H / 16).astype(int) * np.ceil(W / 16).astype(int))
        # print(query_img.shape, x.shape, feature.shape, feature_reshape.shape)
        # print(sim.shape, last_map.shape, last_map_reshape.shape)

        order = torch.argsort(last_map, dim=1)
        multi_out = np.minimum(order.shape[1], self.multi_out)
        if order.shape[1] < self.multi_out:
            print(order.shape, last_map.shape, last_map)
        local_features = torch.gather(input=feature_reshape,
                                      index=order[:, -multi_out:].unsqueeze(2).repeat(1, 1, feature_reshape.shape[2]),
                                      dim=1)

        HW = max(H, W)
        # HW = 512.
        x_xy = torch.cat([(order[:, -multi_out:].unsqueeze(2) % np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW,
                          (order[:, -multi_out:].unsqueeze(2) // np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW],
                         dim=2)
        x_attention = torch.sort(last_map, dim=1)[0][:, -multi_out:]
        x_attention = (x_attention / torch.max(x_attention, dim=1, keepdim=True)[0]).reshape(x_xy.shape[0],
                                                                                             x_xy.shape[1], 1)
        if self.args.finetune:
            local_features = self.local_head(local_features.reshape(B * multi_out, C)).reshape(B, multi_out,
                                                                                               self.out_dim)
        else:
            local_features = self.local_head(local_features.detach().reshape(B * multi_out, C)).reshape(B, multi_out,
                                                                                                        self.out_dim)
        if self.single:
            return x
        else:
            return x, torch.flip(torch.cat([x_xy, x_attention, local_features], dim=2), dims=(1,))

    def forward_deit(self, x):
        # with torch.no_grad():
        B, _, H, W = x.shape
        x_ori = x.detach()
        x = self.backbone.patch_embed(x)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.backbone.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        if H != self.backbone.patch_embed.img_size[0] or W != self.backbone.patch_embed.img_size[1]:
            grid_size = [self.backbone.patch_embed.img_size[0] // 16, self.backbone.patch_embed.img_size[1] // 16]
            matrix = self.backbone.pos_embed[:, 2:].reshape(
                (1, grid_size[0], grid_size[1], self.backbone.embed_dim)).permute((0, 3, 1, 2))
            new_size = max(H // 16, W // 16)
            if grid_size[0] >= new_size and grid_size[1] >= new_size:
                re_matrix = matrix[:, :,
                            (grid_size[0] // 2 - new_size // 2):(grid_size[0] // 2 - new_size // 2 + new_size),
                            (grid_size[1] // 2 - new_size // 2):(grid_size[1] // 2 - new_size // 2 + new_size)]
            else:
                re_matrix = pos_resize(matrix, (new_size, new_size))
            if H >= W:
                new_matrix = re_matrix[:, :, :,
                             (new_size // 2 - W // 16 // 2):(new_size // 2 - W // 16 // 2 + W // 16)].permute(0, 2, 3,
                                                                                                              1).reshape(
                    [1, -1, self.backbone.pos_embed.shape[-1]])
            else:
                new_matrix = re_matrix[:, :, (new_size // 2 - H // 16 // 2):(new_size // 2 - H // 16 // 2 + H // 16),
                             :].permute(0, 2, 3, 1).reshape([1, -1, self.backbone.pos_embed.shape[-1]])
            # print(new_matrix.shape,H//16, W//16,new_size)
            new_pos_embed = torch.cat([self.backbone.pos_embed[:, :2], new_matrix], dim=1)
            x = x + new_pos_embed
        else:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        output_list = []

        for i, blk in enumerate(self.backbone.blocks):
            if (not self.single) and i == (len(self.backbone.blocks) - 1):  # len(self.blocks)-1:判断是否为最后一层
                output = x * 1  #保留结果
                y = blk.norm1(x)  #归一化
                B, N, C = y.shape
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)  #划分多头，B, N, 3, num_heads, d_k]，各维度含义：[批量, 序列长度, q/k/v, 头数, 单头维度]
                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                last_map = (att[:, :, :2, 2:].detach()).sum(dim=1).sum(dim=1)  #[B, dim-2]
            x = blk(x)

            # to support hypercolumn, not used, can be removed.
            if (not self.single) and self.args.hypercolumn:
                if self.hyper_s <= i < self.hyper_e:
                    output_list.append(x * 1.)  # .detach()

        x = self.backbone.norm(x)

        x_cls = self.backbone.head(x[:, 0])
        x_dist = self.backbone.head_dist(x[:, 1])

        if self.single:
            return self.backbone.l2_norm((x_cls + x_dist) / 2)
        else:
            if self.args.hypercolumn:
                output = torch.cat(output_list, dim=2)
            order = torch.argsort(last_map, dim=1, descending=True)
            multi_out = np.minimum(order.shape[1], self.multi_out)
            local_features = torch.gather(input=output,
                                          index=order[:, :multi_out].unsqueeze(2).repeat(1, 1, output.shape[2]),
                                          dim=1)
            # compute attention and coordinates
            HW = max(H, W)
            x_xy = torch.cat([(order[:, :multi_out].unsqueeze(2) % np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW,
                              (order[:, :multi_out].unsqueeze(2) // np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW],
                             dim=2)
            x_attention = torch.sort(last_map, dim=1, descending=True)[0][:, :multi_out]
            x_attention = (x_attention / torch.max(x_attention, dim=1, keepdim=True)[0]).reshape(x_xy.shape[0],
                                                                                                 x_xy.shape[1], 1)
            if self.args.finetune:
                local_features = self.local_head(local_features.reshape(B * multi_out, -1)). \
                    reshape(B, multi_out, self.out_dim)
            else:
                local_features = self.local_head(local_features.detach().reshape(B * multi_out, -1)). \
                    reshape(B, multi_out, self.out_dim)
            return self.backbone.l2_norm((x_cls + x_dist) / 2), torch.cat([x_xy, x_attention, local_features],dim=2),output[:,2:,:]  #x_xy：归一化的关键 patch 坐标（[B, multi_out, 2]）；
                                                                                                                             #x_attention：归一化的注意力权重（[B, multi_out, 1]）；
                                                                                                                            #local_features：映射后的局部特征（[B, multi_out, self.out_dim]）

    def forward(self, x):
        if self.args.backbone.startswith("deit"):
            return self.forward_deit(x)
        elif self.args.backbone.startswith("vit"):
            return self.forward_ori(x)
        else:
            return self.forward_cnn(x)


def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "dynamic_gem":
        return aggregation.DynamicMultiGeM(p=3, eps=1e-6, groups=8, hidden_dim=128,
                                           work_with_tokens=getattr(args, 'work_with_tokens', False),
                                           paper_mode=True, gem_mode=getattr(args, 'gem_mode', 'dynamic'))
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation == 'none' \
            or args.aggregation == 'cls' \
            or args.aggregation == 'seqpool':
        return nn.Identity()


'''def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name + ".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name], dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)

        if args.backbone.endswith("conv4"):
            for name, child in backbone.named_children():
                # Freeze layers before conv_3
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            for name, child in backbone.named_children():
                # Freeze layers before conv_3
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
        else:
            logging.debug(
                f"Train all layers of the resnet{args.backbone.split('conv')[0]}")
            layers = list(backbone.children())[:-2]

    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        else:
            raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone'''


def get_pretrained_model(args):
    if args.pretrain in ['places', 'gldv2']:
        if args.pretrain == 'places':
            num_classes = 365
        elif args.pretrain == 'gldv2':
            num_classes = 512

        if args.backbone.startswith("resnet18"):
            model = torchvision.models.resnet18(num_classes=num_classes)
        elif args.backbone.startswith("resnet50"):
            model = torchvision.models.resnet50(num_classes=num_classes)
        elif args.backbone.startswith("resnet101"):
            model = torchvision.models.resnet101(num_classes=num_classes)
        elif args.backbone.startswith("vgg16"):
            model = torchvision.models.vgg16(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported backbone {args.backbone} for pretrain {args.pretrain}")

    elif args.backbone.startswith("dinov2"):
        from transformers import Dinov2Model
        # DINOv2模型变体映射（名称对应HuggingFace模型库）
        dinov2_variants = {
            "dinov2_vitb14": "meta/dinov2-base",  # 基础版，768维特征
            "dinov2_vitl14": "meta/dinov2-large",  # 大型版，1024维特征
            "dinov2_vitg14": "meta/dinov2-giant",  # 巨型版，1536维特征
            "dinov2_vits14": "meta/dinov2-small"  # 小型版，384维特征
        }
        if args.backbone not in dinov2_variants:
            raise ValueError(f"Unsupported DINOv2 variant: {args.backbone}, choose from {list(dinov2_variants.keys())}")
        model = Dinov2Model.from_pretrained(dinov2_variants[args.backbone])
        return model

    else:
        raise ValueError(f"Unsupported pretrain type: {args.pretrain}")
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name + ".pth")

    if not os.path.exists(file_path):
        download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name], dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_backbone(args):
    # 标记是否使用token特征（CCT/ViT/DINOv2均基于Transformer，使用token）
    args.work_with_tokens = (args.backbone.startswith('cct')
                             or args.backbone.startswith('vit')
                             or args.backbone.startswith('dinov2'))  # 新增DINOv2

    # 原有逻辑：ResNet处理
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)

        if args.backbone.endswith("conv4"):
            for name, child in backbone.named_children():
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]}, freeze previous")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            for name, child in backbone.named_children():
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train conv4_x and conv5_x of resnet{args.backbone.split('conv')[0]}, freeze previous")
            layers = list(backbone.children())[:-2]
        else:
            logging.debug(f"Train all layers of resnet{args.backbone.split('conv')[0]}")
            layers = list(backbone.children())[:-2]
        return torch.nn.Sequential(*layers)


    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of vgg16, freeze previous")
        return torch.nn.Sequential(*layers)

    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of alexnet, freeze previous")
        return torch.nn.Sequential(*layers)

    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze layers up to transformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone

    elif args.backbone.startswith("vit"):
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        else:
            raise ValueError('Image size for ViT must be 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze layers up to transformer encoder {args.freeze_te + 1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone


    elif args.backbone.startswith("dinov2"):
        # 加载DINOv2预训练模型（通过get_pretrained_model）
        backbone = get_pretrained_model(args)

        # DINOv2特征维度映射（根据模型变体）
        dinov2_feat_dims = {
            "dinov2_vits14": 384,  # 小型版
            "dinov2_vitb14": 768,  # 基础版
            "dinov2_vitl14": 1024,  # 大型版
            "dinov2_vitg14": 1536  # 巨型版
        }
        args.features_dim = dinov2_feat_dims[args.backbone]

        if args.trunc_te:
            if args.trunc_te < 1 or args.trunc_te > len(backbone.encoder.layer):
                raise ValueError(f"trunc_te must be between 1 and {len(backbone.encoder.layer)} for {args.backbone}")
            logging.debug(f"Truncate DINOv2 at transformer encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]

        # 冻结部分Transformer层（冻结前N层，训练后面的层）
        if args.freeze_te is not None:
            if args.freeze_te < 0 or args.freeze_te >= len(backbone.encoder.layer):
                raise ValueError(
                    f"freeze_te must be between -1 and {len(backbone.encoder.layer) - 1} for {args.backbone}")
            logging.debug(f"Freeze DINOv2 layers up to transformer encoder {args.freeze_te}")
            # 先冻结所有层
            for p in backbone.parameters():
                p.requires_grad = False
            # 解冻指定层之后的层
            for layer_idx, layer in enumerate(backbone.encoder.layer):
                if layer_idx > args.freeze_te:
                    for p in layer.parameters():
                        p.requires_grad = True
            # 解冻嵌入层（可选，根据需求调整）
            # for p in backbone.embeddings.parameters():
            #     p.requires_grad = True

        return backbone

    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]


def pos_resize(matrix, size):
    B, C, H, W = matrix.shape
    new_matrix = F.interpolate(matrix, size=size, mode='bicubic')
    ori_dis = ((matrix[:, :, 0, W // 2] - matrix[:, :, H // 2, W // 2]) ** 2).sum(dim=1)
    new_dis = ((new_matrix[:, :, size[0] // 2 - H // 2, size[1] // 2] - new_matrix[:, :, size[0] // 2,
                                                                        size[1] // 2]) ** 2).sum(dim=1)
    ratio = torch.sqrt(ori_dis / new_dis)
    center = new_matrix[:, :, size[0] // 2, size[1] // 2].unsqueeze(2).unsqueeze(3)
    new_matrix = (new_matrix - center) * ratio + center
    # print(ratio,matrix.shape, new_matrix.shape,((matrix[:,:,H//2, W//2]-new_matrix[:,:,size[0]//2,size[1]//2])**2).sum(dim=1))
    # print(((matrix[:,:,0,W//2]-matrix[:,:,H//2,W//2])**2).sum(dim=1),((new_matrix[:,:,0, size[1]//2]-new_matrix[:,:,size[0]//2,size[1]//2])**2).sum(dim=1))
    # print(((matrix[:, :, H-1, W // 2] - matrix[:, :, H // 2, W // 2]) ** 2).sum(dim=1),
    #       ((new_matrix[:, :, size[0]-1, size[1] // 2] - new_matrix[:, :, size[0] // 2, size[1] // 2]) ** 2).sum(dim=1))
    # raise Exception
    return new_matrix


# ===========================================================================
# for resolution change of ViT
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class AnySizePatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
