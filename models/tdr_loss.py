import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器 - 加强数值稳定性
    """
    def __init__(self, cost_cls=2.0, cost_bbox=0.25):
        super(HungarianMatcher, self).__init__()
        self.cost_cls = cost_cls
        self.cost_bbox = cost_bbox
    
    def forward(self, cls_scores, bbox_preds, gt_labels, gt_bboxes):
        num_queries = cls_scores.shape[0]
        num_gt = gt_labels.shape[0]
        
        if num_gt == 0:
            return torch.tensor([], dtype=torch.int64, device=cls_scores.device), \
                   torch.tensor([], dtype=torch.int64, device=cls_scores.device)
        
        # ================== 强数值防护 ==================
        cls_scores = torch.nan_to_num(cls_scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        bbox_preds = torch.nan_to_num(bbox_preds, nan=0.0, posinf=100.0, neginf=-100.0)
        gt_bboxes = torch.nan_to_num(gt_bboxes, nan=0.0, posinf=100.0, neginf=-100.0)
        
        with torch.no_grad():
            # 分类成本 - 使用 sigmoid 统一规则
            cls_scores_clamped = torch.clamp(cls_scores, min=-20.0, max=20.0)
            cls_prob = torch.sigmoid(cls_scores_clamped)  # 修复：改为 sigmoid
            cost_cls = -cls_prob[:, gt_labels]
            
            # 回归成本归一化 + 裁剪
            bbox_norm = torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1],
                                   device=bbox_preds.device, dtype=torch.float32)
            
            cost_bbox = torch.cdist(
                torch.clamp(bbox_preds / bbox_norm, -100, 100),
                torch.clamp(gt_bboxes / bbox_norm, -100, 100),
                p=1
            )
            cost_bbox = torch.nan_to_num(cost_bbox, nan=1e8, posinf=1e8, neginf=1e8)
            
            C = self.cost_cls * cost_cls + self.cost_bbox * cost_bbox
            C = torch.nan_to_num(C, nan=1e9, posinf=1e9, neginf=1e9)
        
        C_np = C.detach().cpu().numpy()
        
        try:
            pred_indices, gt_indices = linear_sum_assignment(C_np)
            return (torch.tensor(pred_indices, dtype=torch.int64, device=cls_scores.device),
                    torch.tensor(gt_indices, dtype=torch.int64, device=cls_scores.device))
        except Exception as e:
            print(f"[WARNING] linear_sum_assignment failed: {e}. Batch has {num_gt} GTs.")
            return torch.tensor([], dtype=torch.int64, device=cls_scores.device), \
                   torch.tensor([], dtype=torch.int64, device=cls_scores.device)


class TDRLoss(nn.Module):
    """
    TDR-QAF 系统的损失函数
    """
    def __init__(self, num_classes=10, cost_cls=2.0, cost_bbox=0.25, alpha=0.25, gamma=2.0):
        super(TDRLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_cls, cost_bbox)
        self.alpha = alpha
        self.gamma = gamma
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(self, cls_scores, bbox_preds, gt_labels_list, gt_bboxes_list):
        gt_bboxes_list = [gt.float() for gt in gt_bboxes_list]
        
        B = cls_scores.shape[0]
        loss_cls = 0.0
        loss_bbox = 0.0
        
        metric_matched_count = 0
        metric_pos_acc = 0.0
        metric_xyz_err = 0.0
        
        for b in range(B):
            cls_score = cls_scores[b]
            bbox_pred = bbox_preds[b]
            gt_labels = gt_labels_list[b]
            gt_bboxes = gt_bboxes_list[b]
            
            # 替换循环内的旧打印语句，使用更清晰的格式
            # print(f"Batch {b} | cls_score max: {cls_score.max().item():.2f} | "
            #       f"bbox_max: {bbox_pred.max().item():.2f} | GT: {gt_labels.shape[0]}")
            
            num_gt = gt_labels.shape[0]
            num_query = cls_score.shape[0]
            
            # ================== 增强数值检查 ==================
            if (torch.isnan(cls_score).any() or torch.isinf(cls_score).any() or
                torch.isnan(bbox_pred).any() or torch.isinf(bbox_pred).any()):
                print(f"[WARNING] Batch {b} has NaN/Inf, skipping matching.")
                pred_indices = torch.tensor([], dtype=torch.int64, device=cls_score.device)
                gt_indices = torch.tensor([], dtype=torch.int64, device=cls_score.device)
            else:
                pred_indices, gt_indices = self.matcher(cls_score, bbox_pred, gt_labels, gt_bboxes)
            
            # 准备分类目标（未匹配的视为背景）
            target_labels_onehot = torch.zeros([num_query, self.num_classes],
                                             dtype=cls_score.dtype, device=cls_score.device)
            if num_gt > 0:
                target_labels_onehot[pred_indices, gt_labels[gt_indices]] = 1.0
            
            # Sigmoid Focal Loss
            prob = cls_score.sigmoid()
            ce_loss = F.binary_cross_entropy_with_logits(cls_score, target_labels_onehot, reduction="none")
            p_t = prob * target_labels_onehot + (1 - prob) * (1 - target_labels_onehot)
            focal_weight = (self.alpha * target_labels_onehot + (1 - self.alpha) * (1 - target_labels_onehot)) * ((1 - p_t) ** self.gamma)
            
            loss_cls_batch = (ce_loss * focal_weight).sum() / max(1, num_gt)
            loss_cls += loss_cls_batch
            
            # 必须同时确保有 GT 且 匹配成功数大于 0，防止空张量求 .mean() 产生 NaN
            if num_gt > 0 and len(pred_indices) > 0:
                batch_matched_count = len(pred_indices)
                matched_preds = bbox_pred[pred_indices]
                matched_gts = gt_bboxes[gt_indices]
                
                bbox_norm = torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1],
                                       device=matched_preds.device, dtype=torch.float32)
                loss_bbox_batch = (self.l1_loss(matched_preds / bbox_norm,
                                               matched_gts / bbox_norm)).mean()
                loss_bbox += loss_bbox_batch
                
                matched_logits = cls_score[pred_indices]
                matched_pred_classes = matched_logits.argmax(dim=-1)
                batch_pos_acc = (matched_pred_classes == gt_labels[gt_indices]).float().mean().item()
                batch_xyz_err = F.l1_loss(matched_preds[..., 0:3], matched_gts[..., 0:3]).item()
                
            else:
                # 有 GT 但匹配失败，或无 GT 时，使用显式 0.0 防止隐式 NaN 污染梯度，并保持计算图连通
                loss_bbox_batch = torch.tensor(0.0, device=cls_score.device, requires_grad=True)
                loss_bbox += loss_bbox_batch * bbox_pred.sum() * 0.0
                batch_pos_acc = 0.0
                batch_xyz_err = 0.0
                
            # 将当前批次的统计指标保存下来
            metric_matched_count += batch_matched_count
            metric_pos_acc += batch_pos_acc
            metric_xyz_err += batch_xyz_err
        
        # 计算平均损失
        loss_cls = loss_cls / B
        loss_bbox = loss_bbox / B
        
        # 将新增的指标打包进返回字典
        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'matched_queries': metric_matched_count / B,    # 平均每张图抓到了几个目标
            'pos_acc': metric_pos_acc / max(1, B),          # 目标类别猜对的百分比
            'xyz_err_m': metric_xyz_err / max(1, B)         # 三维定位误差(米)
        }