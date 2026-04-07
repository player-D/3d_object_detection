import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .tdr_lifting import TDRLifting
from .m_rope import MRoPE

def rotate_half(x):
    """将特征向量的一半特征与另一半交换并取反，用于 RoPE"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)


class CrossAttention(nn.Module):
    """
    Cross-Attention 模块
    用于 3D Query 与多视角 2D 图像特征之间的交互
    """
    def __init__(self, embed_dims):
        super(CrossAttention, self).__init__()
        self.embed_dims = embed_dims
        
        # 查询和值投影
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, reference_points, key_padding_mask, cam_intrinsics, cam_extrinsics, mlvl_feats):
        B, Num_Query, _ = query.shape
        N_cam = mlvl_feats[0].shape[1]
        H, W = mlvl_feats[0].shape[3], mlvl_feats[0].shape[4]

        # ================== 1. 【核心 Bug 修复】反归一化 ==================
        # 必须在乘外参矩阵前，将 [0, 1] 的坐标转回真实的物理米数 (-51.2 ~ 51.2)
        min_range, max_range = -51.2, 51.2
        ref_points = reference_points.clone() * (max_range - min_range) + min_range

        # ================== 2. 幽灵 Query 隔离 ==================
        if key_padding_mask is not None:
            # 将 padding 产生的废弃查询点推到宇宙边缘，彻底阻断特征污染
            ref_points = ref_points.clone()
            ref_points[key_padding_mask] = torch.tensor([-1000.0, -1000.0, -1000.0],
                                                       device=ref_points.device, dtype=ref_points.dtype)

        q = self.q_proj(query)

        sampled_feats_list = []
        for cam_idx in range(N_cam):
            curr_intrinsics = cam_intrinsics[:, cam_idx]
            if curr_intrinsics.shape[-2:] == (4, 4):
                curr_intrinsics = curr_intrinsics[..., :3, :3]
            curr_extrinsics = cam_extrinsics[:, cam_idx]

            reference_points_hom = torch.cat([
                ref_points, torch.ones(B, Num_Query, 1, device=ref_points.device, dtype=ref_points.dtype)
            ], dim=-1)

            # 反外参：世界/Ego → Camera
            inv_extrinsics = torch.inverse(curr_extrinsics)
            inv_extrinsics = torch.nan_to_num(inv_extrinsics, nan=0.0, posinf=1e6, neginf=-1e6)
            points_cam_hom = torch.matmul(reference_points_hom, inv_extrinsics.transpose(-1, -2))
            
            # ================== 【核心防护】提取深度，处理背后点和极近点 ==================
            depth = points_cam_hom[..., 2:3]
            depth = torch.nan_to_num(depth, nan=10.0, posinf=100.0, neginf=-100.0)
            
            # 【关键加强】把无效深度阈值提高到 1.5 米
            invalid_mask = (depth.squeeze(-1) < 1.5)  # [B, Num_Query]
            # 最小 1.5 米，彻底防止透视投影时的除零爆炸
            depth_safe = torch.clamp(depth, min=1.5)
            
            # 归一化到相机平面 (X/Z, Y/Z, 1)
            points_cam = points_cam_hom[..., :3] / depth_safe
            
            # 乘内参得到图像像素坐标
            if curr_intrinsics.shape[-2:] == (4, 4):
                curr_intrinsics = curr_intrinsics[..., :3, :3]
            
            # 【核心修复】：直接使用 matmul，[B, Num_Query, 3] x [B, 3, 3]
            points_img_hom = torch.matmul(points_cam, curr_intrinsics.transpose(-1, -2))
            
            # 由于 points_cam 的 Z 已经是 1，此时 points_img_hom 的 XY 即为像素坐标
            points_img = points_img_hom[..., :2]
            # 【最后一道保险】强力像素坐标裁剪
            points_img = torch.clamp(points_img, -3000.0, 3000.0)
            
            orig_W, orig_H = 800.0, 448.0
            
            cam_sampled_levels = []
            # 遍历多尺度特征
            for feat_level in mlvl_feats:
                curr_feat = feat_level[:, cam_idx]  # [B, C, H_feat, W_feat]
                H_feat, W_feat = curr_feat.shape[2], curr_feat.shape[3]
                
                scale_x = W_feat / orig_W
                scale_y = H_feat / orig_H
                feat_x = points_img[..., 0] * scale_x
                feat_y = points_img[..., 1] * scale_y
                
                # 归一化到 [-1, 1] 供 grid_sample 使用
                grid_x = torch.clamp((feat_x / (W_feat - 1.0)) * 2.0 - 1.0, -10.0, 10.0)
                grid_y = torch.clamp((feat_y / (H_feat - 1.0)) * 2.0 - 1.0, -10.0, 10.0)
                grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, Num_Query, 2]
                
                # 踢出无效点
                grid[invalid_mask] = -100.0
                grid = grid.unsqueeze(2)  # [B, Num_Query, 1, 2]
                
                # 采样当前层特征
                sampled = F.grid_sample(
                    curr_feat, grid,
                    mode='bilinear', padding_mode='zeros', align_corners=True
                ).squeeze(-1).transpose(1, 2)
                
                sampled = torch.nan_to_num(sampled, nan=0.0)
                cam_sampled_levels.append(sampled)
            
            # 多尺度特征融合：对 4 个尺度的特征取平均
            multi_scale_feat = torch.stack(cam_sampled_levels, dim=1).mean(dim=1)
            # 通过 V 投影
            multi_scale_feat = self.v_proj(multi_scale_feat)
            sampled_feats_list.append(multi_scale_feat)

        # ================== 4. 【多视角特征聚合】改为 max ==================
        # 使用 max 代替 sum，完美提取对应相机的有效特征，过滤其它相机的 0
        sampled_feats = torch.stack(sampled_feats_list, dim=1).max(dim=1)[0]

        fused = q + sampled_feats
        fused = F.relu(fused) + q
        updated_query = self.out_proj(fused)

        # 双重保险：在输出端再次清零 padding query 的特征
        if key_padding_mask is not None:
            updated_query = updated_query * (~key_padding_mask).unsqueeze(-1).float()

        return updated_query


class DecoderLayer(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
        self.cross_attn = CrossAttention(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims)
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, reference_points, key_padding_mask, cam_intrinsics, cam_extrinsics, mlvl_feats):
        # 自注意力（使用 padding mask 屏蔽幽灵 Query）
        q = self.norm1(query)
        self_attn_output, _ = self.self_attn(q, q, q, key_padding_mask=key_padding_mask)
        query = query + self_attn_output

        # 交叉注意力
        q = self.norm2(query)
        cross_attn_output = self.cross_attn(q, reference_points, key_padding_mask, cam_intrinsics, cam_extrinsics, mlvl_feats)
        query = query + cross_attn_output

        # FFN
        q = self.norm3(query)
        ffn_output = self.ffn(q)
        query = query + ffn_output

        # 强制清零 padding 位置的特征，阻断幽灵梯度
        if key_padding_mask is not None:
            query = query * (~key_padding_mask).unsqueeze(-1).float()
        return query


class TDRHead(nn.Module):
    """
    TDR-QAF 系统的检测头
    集成 TDR-Lifting、M-RoPE 和 Transformer Decoder
    """
    def __init__(self, num_classes=10, in_channels=256, embed_dims=256 ,
                 num_decoder_layers=6, num_depth_dense=48, max_queries=400):
        super(TDRHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_decoder_layers = num_decoder_layers
        
        # TDRLifting（会返回 reference_points 和 key_padding_mask）
        self.lifting = TDRLifting(
            num_depth_dense=num_depth_dense,      # 使用传入参数
            num_depth_local=8 ,       
            depth_range=[1.0, 60.0 ],
            iou_threshold=0.12 ,      
            space_range=[-51.2, 51.2 ],
            max_queries=max_queries               # 使用传入参数
        )
        
        # 多模态旋转位置编码
        self.m_rope = MRoPE(embed_dims)
        
        # Transformer Decoder Layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dims) for _ in range(num_decoder_layers) # 使用传入参数
        ])
        
        # Query 初始化
        self.query_embed = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )
        
        # 分类和回归分支
        self.cls_branches = nn.Linear(embed_dims, num_classes)      # 10 类
        self.reg_branches = nn.Linear(embed_dims, 10)
        
        # ================== Focal Loss 偏置初始化 ==================
        prior_prob = 0.15
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_branches.bias, bias_value)
        
        # 回归分支偏置初始化为 0
        nn.init.constant_(self.reg_branches.bias, 0.0)
        # =========================================================

    def forward(self, mlvl_feats, boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior=None):
        """
        mlvl_feats: list of [B, N_cam, C, H, W]
        """
        # 1. Lifting 生成参考点和 padding mask
        reference_points, key_padding_mask = self.lifting(
            boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior
        )
        B, Num_Query, _ = reference_points.shape

        # 2. MRoPE 位置编码
        pos_sin, pos_cos = self.m_rope(reference_points)

        # 3. 初始化 Query 并施加旋转位置编码
        query = self.query_embed(reference_points)
        query = query * pos_cos + rotate_half(query) * pos_sin

        # 4. Decoder 层（已传入 key_padding_mask）
        for layer in self.decoder_layers:
            query = layer(query, reference_points, key_padding_mask, 
                          cam_intrinsics, cam_extrinsics, mlvl_feats)

        # 5. 分类和回归预测
        cls_scores = self.cls_branches(query)
        bbox_preds = self.reg_branches(query)

        # 后处理：坐标偏移 + 尺寸 exp 处理
        xyz_offset = bbox_preds[..., 0:3]
        min_range, max_range = -51.2, 51.2
        reference_points_denorm = reference_points * (max_range - min_range) + min_range
        
        xyz = reference_points_denorm + xyz_offset
        wlh = torch.exp(torch.clamp(bbox_preds[..., 3:6], min=-5.0, max=5.0))
        rest = bbox_preds[..., 6:]
        bbox_preds = torch.cat([xyz, wlh, rest], dim=-1)

        return cls_scores, bbox_preds