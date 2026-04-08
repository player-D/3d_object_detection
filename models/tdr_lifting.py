import torch
import torch.nn as nn
import torch.nn.functional as F

class TDRLifting(nn.Module):
    def __init__(self, num_depth_dense=48, num_depth_local=8,
                 depth_range=[2.0, 55.0], iou_threshold=0.20,
                 space_range=[-51.2, 51.2],
                 max_queries=400):
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
        
        # 【新增：物理有效性深度约束】
        # 1. 深度必须大于 1.5 米且小于 60 米
        physical_valid_mask = (depth_scale.squeeze(-1) > 1.5) & (depth_scale.squeeze(-1) < 60.0)
        # 2. pseudo_boxes 尺寸不能大得离谱（防止近处除零或透视畸变导致占据整个屏幕）
        box_areas = (pseudo_boxes[..., 2] - pseudo_boxes[..., 0]) * (pseudo_boxes[..., 3] - pseudo_boxes[..., 1])
        reasonable_size_mask = box_areas < (800 * 448 * 0.8) # 伪框不能超过画面的 80%
        
        # 合并物理有效性 Mask
        strict_valid_mask = physical_valid_mask & reasonable_size_mask
        
        boxes_2d_exp = boxes_2d.unsqueeze(-2).repeat(1, 1, 1, D, 1)
        iou = self._compute_iou(pseudo_boxes, boxes_2d_exp)
        # 必须同时满足 IoU 阈值和绝对物理有效性
        consistency_mask = (iou > self.iou_threshold) & strict_valid_mask
        
        final_mask = consistency_mask & valid_box_mask.unsqueeze(-1)
        
        valid_points_list = []
        padding_masks = []

        for b in range(B):
            batch_points = points_3d[b].flatten(0, 2)  # [N_cam*Num_boxes*D, 3]
            batch_mask = final_mask[b].flatten()  # [N_cam*Num_boxes*D]

            # batch_areas需要与batch_mask的长度匹配
            # areas[b]是[N_cam, Num_boxes]，需要扩展到[N_cam, Num_boxes, D]
            batch_areas = areas[b].unsqueeze(-1).repeat(1, 1, D).flatten()  # [N_cam*Num_boxes*D]

            valid_pts = batch_points[batch_mask]  # 只保留mask为True的点
            batch_areas = batch_areas[batch_mask]  # 只保留mask为True的areas

            # 物理过滤：确保点有效且深度在合理范围内
            finite_mask = torch.isfinite(valid_pts).all(dim=-1)
            valid_pts = valid_pts[finite_mask]
            batch_areas = batch_areas[finite_mask]  # 同步过滤areas

            depth_mask = (valid_pts[..., 2] > 1.0) & (valid_pts[..., 2] < 70.0)
            valid_pts = valid_pts[depth_mask]
            batch_areas = batch_areas[depth_mask]  # 同步过滤areas

            if valid_pts.shape[0] == 0:
                valid_pts = torch.zeros(1, 3, device=device)
                padding_masks.append(torch.ones(1, dtype=torch.bool, device=device))
            else:
                # 【关键修复】跨相机点去重 - 使用距离阈值去除重复点
                distance_threshold = 0.5  # 0.5米内的点视为重复

                if valid_pts.shape[0] > 1:
                    # 使用更高效的基于距离矩阵的去重方法
                    # 计算所有点对之间的距离矩阵
                    diff = valid_pts.unsqueeze(1) - valid_pts.unsqueeze(0)  # [N, N, 3]
                    dist_matrix = torch.norm(diff, dim=-1)  # [N, N]

                    # 创建上三角掩码（避免重复计算和自身比较）
                    mask = torch.triu(torch.ones_like(dist_matrix, dtype=torch.bool), diagonal=1)
                    dist_matrix = dist_matrix.masked_fill(~mask, float('inf'))

                    # 找到距离小于阈值的点对
                    duplicate_pairs = dist_matrix < distance_threshold

                    # 使用并查集思想进行去重
                    keep_mask = torch.ones(valid_pts.shape[0], dtype=torch.bool, device=device)
                    for i in range(valid_pts.shape[0]):
                        if not keep_mask[i]:
                            continue
                        # 找到所有与点i重复的点（不包括i自己）
                        duplicates = duplicate_pairs[i].nonzero(as_tuple=True)[0]
                        if len(duplicates) > 0:
                            # 标记这些点为不保留
                            keep_mask[duplicates] = False

                    valid_pts = valid_pts[keep_mask]
                    batch_areas = batch_areas[keep_mask]

                # 去重后，如果仍然超过max_queries，按面积选择
                k = min(self.max_queries, valid_pts.shape[0])
                if valid_pts.shape[0] > self.max_queries:
                    _, topk_idx = torch.topk(batch_areas, k=k)
                    valid_pts = valid_pts[topk_idx]
                # padding_mask的长度必须与valid_pts完全匹配
                padding_masks.append(torch.zeros(valid_pts.shape[0], dtype=torch.bool, device=device))

            valid_points_list.append(valid_pts)

        # 手动padding到固定长度
        max_len = max(len(pts) for pts in valid_points_list)

        padded_points = torch.zeros(B, max_len, 3, device=device, dtype=points_3d.dtype)
        key_padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for b in range(B):
            num_pts = valid_points_list[b].shape[0]
            padded_points[b, :num_pts] = valid_points_list[b]
            key_padding_mask[b, :num_pts] = False  # False表示有效点

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