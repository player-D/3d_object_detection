import torch
import torch.nn as nn
import torch.nn.functional as F

class TDRLifting(nn.Module):
    def __init__(self, num_depth_dense=48, num_depth_local=8,   # 将 32 改为 48
                 depth_range=[1.0, 60.0], iou_threshold=0.05,   # 将 0.06/0.10 降为 0.05
                 space_range=[-51.2, 51.2],
                 max_queries=400):  # 新增参数，默认给全量配置
        super().__init__()
        self.num_depth_dense = num_depth_dense
        self.num_depth_local = num_depth_local
        self.min_depth = depth_range[0]
        self.max_depth = depth_range[1]
        self.iou_threshold = iou_threshold
        self.space_range = space_range
        self.max_queries = max_queries  # 移除硬编码，使用传入的参数
        self.depth_std_base = nn.Parameter(torch.tensor(1.0))

    def _get_center_points(self, boxes_2d):
        x1, y1, x2, y2 = boxes_2d[..., 0], boxes_2d[..., 1], boxes_2d[..., 2], boxes_2d[..., 3]
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2], dim=-1)

    def _compute_box_area(self, boxes_2d):
        return (boxes_2d[..., 2] - boxes_2d[..., 0]) * (boxes_2d[..., 3] - boxes_2d[..., 1])

    def _sample_depths(self, boxes_2d, temporal_depth_prior):
        B, N_cam, Num_boxes, _ = boxes_2d.shape
        device = boxes_2d.device
        if temporal_depth_prior is None:
            # 采用 linspace 线性均匀采样，改善远距离目标的特征捕获
            depths = torch.linspace(self.min_depth, self.max_depth, self.num_depth_dense, device=device)
            return depths.view(1, 1, 1, -1).repeat(B, N_cam, Num_boxes, 1)
        else:
            box_areas = self._compute_box_area(boxes_2d)
            rel_areas = (box_areas - box_areas.min()) / (box_areas.max() - box_areas.min() + 1e-6)
            std = self.depth_std_base * (1.0 + (1.0 - rel_areas) * 3.0)
            mean = temporal_depth_prior.unsqueeze(-1)
            depths = torch.normal(mean=mean, std=std.unsqueeze(-1).repeat(1, 1, 1, self.num_depth_local))
            return torch.clamp(depths, self.min_depth, self.max_depth)

    def _back_project(self, center_points, depths, cam_intrinsics, cam_extrinsics):
        # 请保留你当前文件中已有的 _back_project 实现（它已经很完整）
        B, N_cam, Num_boxes, _ = center_points.shape
        D = depths.shape[-1]
        points_2d = center_points.unsqueeze(3).repeat(1, 1, 1, D, 1)
        points_2d = points_2d * depths.unsqueeze(-1)
        points_cam = torch.cat([points_2d, depths.unsqueeze(-1)], dim=-1)
        
        inv_intrinsics = torch.inverse(cam_intrinsics[..., :3, :3])
        inv_intrinsics = torch.nan_to_num(inv_intrinsics, nan=0.0, posinf=1e6, neginf=-1e6)
        inv_intrinsics_exp = inv_intrinsics.unsqueeze(2).unsqueeze(3)
        points_cam_unsq = points_cam.unsqueeze(-2)
        points_cam = torch.matmul(points_cam_unsq, inv_intrinsics_exp.transpose(-1, -2)).squeeze(-2)
        
        points_cam_hom = torch.cat([points_cam, torch.ones_like(points_cam[..., :1])], dim=-1)
        
        cam_extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)
        points_cam_hom_unsq = points_cam_hom.unsqueeze(-2)
        points_ego_hom = torch.matmul(points_cam_hom_unsq, cam_extrinsics_exp.transpose(-1, -2)).squeeze(-2)
        return points_ego_hom[..., :3]

    def _compute_iou(self, box1, box2):
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = area1 + area2 - intersection + 1e-6
        return intersection / union

    def _project_consistency_filter(self, points_3d, boxes_2d, cam_intrinsics, cam_extrinsics):
        B, N_cam, Num_boxes, D, _ = points_3d.shape
        device = points_3d.device
        
        areas = self._compute_box_area(boxes_2d)
        # 加强过滤：去掉极小噪点和过大异常遮挡框
        valid_box_mask = (areas > 5.0) & (areas < 800 * 448 * 0.55 )
        
        # 反投影一致性
        points_3d_hom = torch.cat([points_3d, torch.ones(B, N_cam, Num_boxes, D, 1, device=device)], dim=-1)
        inv_extrinsics = torch.inverse(cam_extrinsics)
        inv_extrinsics = torch.nan_to_num(inv_extrinsics, nan=0.0, posinf=1e6, neginf=-1e6)
        points_cam_hom = torch.matmul(points_3d_hom.unsqueeze(-2), 
                                      inv_extrinsics.unsqueeze(2).unsqueeze(3).transpose(-1, -2)).squeeze(-2)
        points_cam = points_cam_hom[..., :3] / (points_cam_hom[..., 3:4] + 1e-6)
        
        intrinsics = cam_intrinsics[..., :3, :3] if cam_intrinsics.shape[-2:] == (4, 4) else cam_intrinsics
        points_img_hom = torch.matmul(points_cam.unsqueeze(-2), 
                                      intrinsics.unsqueeze(2).unsqueeze(3).transpose(-1, -2)).squeeze(-2)
        points_img = points_img_hom[..., :2] / (points_img_hom[..., 2:3] + 1e-6)
        
        depth_scale = points_cam[..., 2:3]
        # 将原来的 30 改为 40，适度放大 2D 感受野
        box_size = 40 * (10.0 / (depth_scale + 1e-6))
        # 将 clamp 的下限从 5 改为 8，上限从 150 改为 200
        box_size = torch.clamp(box_size, 8, 200)
        pseudo_boxes = torch.cat([points_img - box_size/2, points_img + box_size/2], dim=-1)
        
        boxes_2d_exp = boxes_2d.unsqueeze(-2).repeat(1, 1, 1, D, 1)
        iou = self._compute_iou(pseudo_boxes, boxes_2d_exp)
        consistency_mask = iou > self.iou_threshold
        
        final_mask = consistency_mask & valid_box_mask.unsqueeze(-1)
        
        valid_points_list = []
        padding_masks = []
        
        for b in range(B):
            batch_points = points_3d[b].flatten(0, 2)
            batch_mask = final_mask[b].flatten()
            batch_areas = areas[b].unsqueeze(-1).repeat(1, 1, D).flatten()[batch_mask]
            
            valid_pts = batch_points[batch_mask]
            
            if valid_pts.shape[0] == 0:
                valid_pts = torch.zeros(1, 3, device=device)
                padding_masks.append(torch.ones(1, dtype=torch.bool, device=device))
            else:
                k = min(self.max_queries, valid_pts.shape[0])
                if valid_pts.shape[0] > self.max_queries:
                    _, topk_idx = torch.topk(batch_areas, k=k)
                    valid_pts = valid_pts[topk_idx]
                padding_masks.append(torch.zeros(valid_pts.shape[0], dtype=torch.bool, device=device))
            
            valid_points_list.append(valid_pts)
            
        padded_points = torch.nn.utils.rnn.pad_sequence(valid_points_list, batch_first=True, padding_value=0.0)
        key_padding_mask = torch.nn.utils.rnn.pad_sequence(padding_masks, batch_first=True, padding_value=True)
        return padded_points, key_padding_mask

    def _normalize_points(self, points_3d):
        min_r, max_r = self.space_range
        normalized = (points_3d - min_r) / (max_r - min_r + 1e-6)
        return torch.clamp(normalized, 0.0, 1.0)

    def forward(self, boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior=None):
        cam_intrinsics = cam_intrinsics.to(boxes_2d.dtype)
        cam_extrinsics = cam_extrinsics.to(boxes_2d.dtype)
        
        if boxes_2d.shape[2] == 0:
            B = boxes_2d.shape[0]
            return torch.zeros(B, 1, 3, device=boxes_2d.device), torch.ones(B, 1, dtype=torch.bool, device=boxes_2d.device)
            
        center_points = self._get_center_points(boxes_2d)
        depths = self._sample_depths(boxes_2d, temporal_depth_prior)
        points_3d = self._back_project(center_points, depths, cam_intrinsics, cam_extrinsics)
        
        valid_points, key_padding_mask = self._project_consistency_filter(
            points_3d, boxes_2d, cam_intrinsics, cam_extrinsics
        )
        reference_points = self._normalize_points(valid_points)
        return reference_points, key_padding_mask