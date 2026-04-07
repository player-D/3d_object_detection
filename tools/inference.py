import os
import sys
import copy
import random
import time
import math
import numpy as np
import torch
import cv2

# 把项目根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.nuscenes_dataset import NuScenesDataset
from models.tdr_detector import TDRDetector
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# 相机名称列表
CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]

# 类别名称映射
CLASS_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'trailer',
    4: 'construction_vehicle',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle',
    8: 'traffic_cone',
    9: 'barrier'
}

# 类别颜色映射
CLASS_COLORS = {
    0: (0, 255, 0),  # 绿色 - GT
    1: (0, 0, 255),  # 红色 - Pred
    2: (255, 0, 0),  # 蓝色
    3: (255, 255, 0),  # 黄色
    4: (0, 255, 255),  # 青色
    5: (255, 0, 255),  # 洋红
    6: (128, 0, 128),  # 紫色
    7: (0, 128, 128),  # 蓝绿色
    8: (128, 128, 0),  # 橄榄色
    9: (128, 128, 128)  # 灰色
}

def create_output_dir():
    """创建输出目录"""
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"infer_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_ego_pose(nusc, sample_token):
    """获取自车姿态"""
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    sd = nusc.get('sample_data', cam_token)
    return nusc.get('ego_pose', sd['ego_pose_token'])

def global_to_ego(box, ego_pose):
    """将全局坐标转换为自车坐标"""
    box_ego = copy.deepcopy(box)
    # 平移
    box_ego.translate([-x for x in ego_pose['translation']])
    # 旋转
    box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
    return box_ego



def tensor_to_bgr_image(img_tensor):
    """将张量转换为 BGR 图像"""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def load_model(device, is_overfit=False, checkpoint_path='./saved_models/04_07_13-37/tdr_qaf_epoch_20.pth'):
    """加载模型
    
    Args:
        device: 运行设备
        is_overfit: 是否为本地过拟合模式的权重
        checkpoint_path: 权重文件路径
    
    Returns:
        加载好的模型
    """
    # 打印绝对路径
    abs_path = os.path.abspath(checkpoint_path)
    print(f'=== DEBUG: 加载权重文件路径: {abs_path} ===')
    
    if is_overfit:
        decoder_layers = 2
        depth_dense = 24
        max_q = 200
    else:
        decoder_layers = 6
        depth_dense = 48
        max_q = 400
        
    model = TDRDetector(
        num_classes=10 ,
        embed_dims=256 ,
        num_decoder_layers=decoder_layers,
        num_depth_dense=depth_dense,
        max_queries=max_q
    ).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'权重文件不存在：{checkpoint_path}')
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'成功加载模型权重：{checkpoint_path}')
    print(f'missing keys：{len(missing)}, unexpected keys：{len(unexpected)}')
    if missing:
        print('  缺少键：', missing[:10])
    if unexpected:
        print('  意外键：', unexpected[:10])
    
    model.eval()
    return model

def get_sample(dataset, index, device):
    """获取样本数据
    
    Args:
        dataset: 数据集
        index: 样本索引
        device: 运行设备
    
    Returns:
        样本数据，如果样本损坏则返回 None
    """
    sample = dataset[index]
    if sample is None:
        return None
    
    images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels = sample
    
    images = images.unsqueeze(0).to(device)
    boxes_2d = boxes_2d.unsqueeze(0).to(device)
    cam_intrinsics = cam_intrinsics.unsqueeze(0).to(device)
    cam_extrinsics = cam_extrinsics.unsqueeze(0).to(device)
    
    return images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels

def run_model(model, images, boxes_2d, cam_intrinsics, cam_extrinsics):
    """运行模型推理
    
    Args:
        model: 模型
        images: 图像
        boxes_2d: 2D 框
        cam_intrinsics: 相机内参
        cam_extrinsics: 相机外参
    
    Returns:
        分类得分和边界框预测
    """
    with torch.no_grad():
        cls_scores, bbox_preds = model(
            images,
            boxes_2d,
            cam_intrinsics,
            cam_extrinsics,
            temporal_depth_prior=None
        )
    

    

    
    return cls_scores, bbox_preds

def decode_bbox(cls_scores, bbox_preds, threshold=0.05, topk=50, nms_dist=1.5):
    """
    返回: pred_scores, pred_labels, pred_bboxes
    """
    if cls_scores.dim() == 3:
        cls_scores = cls_scores[0]
    if bbox_preds.dim() == 3:
        bbox_preds = bbox_preds[0]
    
    # 对 logits 做 sigmoid
    prob = torch.sigmoid(cls_scores)                    # [Num_Query, 11]
    
    # ================== 📊 分类分支成熟度体检 ==================
    max_score = prob[..., :10].max().item()
    mean_score = prob[..., :10].mean().item()
    num_query = prob.shape[0]
    
    print(f"\n--- 📊 [Decode Debug] 分类置信度体检 ---")
    print(f"Query 总数          : {num_query}")
    print(f"前景最高得分        : {max_score:.4f}")
    print(f"前景平均得分        : {mean_score:.4f}")
    print(f"得分 > 0.25 的框数 : {(prob[..., :10].max(dim=-1)[0] > 0.25).sum().item()}")
    print(f"得分 > 0.35 的框数 : {(prob[..., :10].max(dim=-1)[0] > 0.35).sum().item()}")
    print(f"得分 > 0.45 的框数 : {(prob[..., :10].max(dim=-1)[0] > 0.45).sum().item()}")
    print(f"当前过滤阈值        : {threshold}")
    print(f"------------------------------------------\n")
    # =========================================================
    
    # 取前10类最大得分和对应标签
    pred_scores, pred_labels = torch.max(prob[..., :10], dim=-1)   # [Num_Query]

    valid_mask = torch.isfinite(pred_scores)
    valid_mask &= torch.isfinite(bbox_preds).all(dim=-1)
    valid_mask &= pred_scores >= threshold

    pred_scores = pred_scores[valid_mask]
    pred_labels = pred_labels[valid_mask]
    pred_bboxes = bbox_preds[valid_mask]

    if pred_scores.numel() == 0:
        print(f'最终保留的预测框数量：0\n')
        return pred_scores, pred_labels, pred_bboxes

    # 先做尺寸/深度硬过滤，防止极端值
    pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2], 0.0, 80.0)
    pred_bboxes[:, 3:6] = torch.clamp(pred_bboxes[:, 3:6], 0.2, 20.0)

    # topk
    if pred_scores.shape[0] > topk:
        idx = torch.topk(pred_scores, topk).indices
        pred_scores = pred_scores[idx]
        pred_labels = pred_labels[idx]
        pred_bboxes = pred_bboxes[idx]

    # class-aware BEV center-distance NMS
    order = torch.argsort(pred_scores, descending=True)
    keep = []
    centers = pred_bboxes[:, :2]

    for idx in order.tolist():
        ok = True
        for j in keep:
            if pred_labels[idx] != pred_labels[j]:
                continue

            dist = torch.norm(centers[idx] - centers[j])
            size_i = torch.sqrt(pred_bboxes[idx, 3] * pred_bboxes[idx, 4])
            size_j = torch.sqrt(pred_bboxes[j, 3] * pred_bboxes[j, 4])
            thr = 0.7 * torch.max(size_i, size_j) + nms_dist

            if dist < thr:
                ok = False
                break

        if ok:
            keep.append(idx)

    keep = torch.tensor(keep, device=pred_scores.device, dtype=torch.long)
    print(f'最终保留的预测框数量：{len(keep)}\n')
    return pred_scores[keep], pred_labels[keep], pred_bboxes[keep]

def build_box_from_vec(bbox_vec):
    """从向量构建 Box 对象
    
    Args:
        bbox_vec: 边界框向量 [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy]
    
    Returns:
        Box 对象或 None（如果无效）
    """
    vec = np.asarray(bbox_vec, dtype=np.float32)

    if not np.all(np.isfinite(vec)):
        return None

    x, y, z, w, l, h, sin_yaw, cos_yaw = vec[:8]

    # 硬过滤：深度、尺寸都必须合理
    if z < 0.5 or z > 80.0:
        return None
    if min(w, l, h) < 0.2:
        return None
    if max(w, l, h) > 20.0:
        return None

    yaw = np.arctan2(sin_yaw, cos_yaw)
    return Box(
        center=[float(x), float(y), float(z)],
        size=[float(w), float(l), float(h)],
        orientation=Quaternion(axis=[0, 0, 1], radians=float(yaw))
    )

def project_box_cam(box_cam, K):
    """将相机坐标系下的框投影到图像平面
    
    Args:
        box_cam: 相机坐标系下的 Box 对象
        K: 相机内参矩阵
    
    Returns:
        2D 点和深度
    """
    corners = box_cam.corners()  # (3, 8)
    points_2d = view_points(corners, K, normalize=True)  # (2, 8)
    depth = corners[2, :]  # (8,)
    return points_2d, depth

def draw_box(img, points_2d, depth, color, thickness=1):
    """在图像上绘制 3D 框
    
    Args:
        img: 图像
        points_2d: 2D 点
        depth: 深度
        color: 颜色
        thickness: 线宽
    
    Returns:
        img: 绘制后的图像
        text_pos: 标签位置
        success: 是否成功绘制
    """
    img_h, img_w = img.shape[:2]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    
    # 过滤掉深度为负的点
    valid_mask = depth > 0.1
    if not np.any(valid_mask):
        return img, None, False
    
    # 检查是否有 NaN
    if np.any(np.isnan(points_2d)):
        return img, None, False
    
    # 严格的坐标边界检查和截断
    # 丢弃坐标绝对值大于 img_w * 5 或 img_h * 5 的极端离谱的预测点
    max_coord = max(img_w * 5, img_h * 5)
    for i in range(points_2d.shape[1]):
        if abs(points_2d[0, i]) > max_coord or abs(points_2d[1, i]) > max_coord:
            valid_mask[i] = False
    
    # 再次检查是否有有效点
    if not np.any(valid_mask):
        return img, None, False
    
    # 临时放宽过滤，便于看到早期预测结果
    valid_x = points_2d[0, valid_mask]
    valid_y = points_2d[1, valid_mask]
    if len(valid_x) < 3:   # 改小一点
        return img, None, False
        
    span_x = np.max(valid_x) - np.min(valid_x)
    span_y = np.max(valid_y) - np.min(valid_y)
    
    if span_x < 3 and span_y < 3 or span_x > img_w * 3 or span_y > img_h * 3:
        return img, None, False
    
    # 绘制边
    lines_drawn = False
    for edge in edges:
        i, j = edge
        if valid_mask[i] and valid_mask[j]:
            # 强转之前进行严格的边界截断
            x1 = int(np.clip(points_2d[0, i], 0, img_w-1))
            y1 = int(np.clip(points_2d[1, i], 0, img_h-1))
            x2 = int(np.clip(points_2d[0, j], 0, img_w-1))
            y2 = int(np.clip(points_2d[1, j], 0, img_h-1))
            # 再次检查坐标是否在合理范围内
            if x1 >= 0 and x1 < img_w and y1 >= 0 and y1 < img_h and x2 >= 0 and x2 < img_w and y2 >= 0 and y2 < img_h:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                lines_drawn = True
    
    # 计算 2D 投影点的边界框，取 [min_x, min_y] 作为文字的左上角起点
    text_pos = None
    if np.any(valid_mask):
        valid_x = points_2d[0, valid_mask]
        valid_y = points_2d[1, valid_mask]
        min_x = int(np.min(valid_x))
        min_y = int(np.min(valid_y))
        if 0 <= min_x < img_w and 0 <= min_y < img_h:
            text_pos = (min_x, max(10, min_y - 10))
    
    # 只有绘制了边且找到了标签位置才算成功
    success = lines_drawn and text_pos is not None
    
    return img, text_pos, success

def draw_text(img, text, pos, color, font_scale=0.4, thickness=1):
    """在图像上绘制文字
    
    Args:
        img: 图像
        text: 文字
        pos: 位置
        color: 颜色
        font_scale: 字体大小
        thickness: 线宽
    """
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (pos[0]-2, pos[1]-h-2), (pos[0]+w+2, pos[1]+2), (255, 255, 255), -1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def project_3d_box_to_image(box, nusc, sample_token, cam_name, K, is_global=True):
    """将 3D 框投影到图像平面
    
    Args:
        box: Box 对象
        nusc: nuScenes 实例
        sample_token: 样本 token
        cam_name: 相机名称
        K: 相机内参矩阵
        is_global: 是否为全局坐标系（True 用于 GT 框，False 用于 Pred 框）
    
    Returns:
        points_2d: 2D 投影点
        depth: 深度
        box_cam: 相机坐标系下的 Box 对象
    """
    # 获取相机传感器数据
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_name]
    sd = nusc.get('sample_data', cam_token)
    
    # 坐标变换：Global -> Ego（仅当 is_global 为 True 时）
    if is_global:
        # Global -> Ego
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        box_ego = copy.deepcopy(box)
        box_ego.translate([-x for x in pose['translation']])
        box_ego.rotate(Quaternion(pose['rotation']).inverse)
        # 打印转换后的自车坐标
        # print(f'    center (ego): {box_ego.center}')
    else:
        # 直接使用传入的框作为自车坐标系
        box_ego = copy.deepcopy(box)
        # print(f'    center (ego, direct): {box_ego.center}')
    
    # Ego -> Camera
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    box_cam = copy.deepcopy(box_ego)
    box_cam.translate([-x for x in cs['translation']])
    box_cam.rotate(Quaternion(cs['rotation']).inverse)
    
    # 投影到图像平面
    points_2d, depth = project_box_cam(box_cam, K)
    
    return points_2d, depth, box_cam

def merge_6_cams(path_dict):
    # 横向拼接前排
    front_combined = cv2.hconcat([path_dict["CAM_FRONT_LEFT"], path_dict["CAM_FRONT"], path_dict["CAM_FRONT_RIGHT"]])
    # 横向拼接后排（调整为左、中、右顺序，并废弃 flip 翻转防止文字镜像）
    back_combined = cv2.hconcat([path_dict["CAM_BACK_LEFT"], path_dict["CAM_BACK"], path_dict["CAM_BACK_RIGHT"]])
    # 上下拼接
    img_combined = cv2.vconcat([front_combined, back_combined])
    return img_combined

def visualize(images, cam_intrinsics, gt_bboxes, gt_labels, pred_scores, pred_labels, pred_bboxes, sample_token, output_dir, nusc):
    """可视化结果
    
    Args:
        images: 图像
        cam_intrinsics: 相机内参
        gt_bboxes: GT 边界框
        gt_labels: GT 标签
        pred_scores: 预测得分
        pred_labels: 预测标签
        pred_bboxes: 预测边界框
        sample_token: 样本 token
        output_dir: 输出目录（可选，为 None 时不保存文件）
        nusc: nuScenes 实例
    
    Returns:
        canvas_combined: 拼接好的 NumPy 图像数组（GT + Pred）
        canvas_pred_only: 拼接好的 NumPy 图像数组（仅 Pred）
        bev_img: BEV 鸟瞰图
        stats: 统计信息
    """
    # 获取图像尺寸
    img = tensor_to_bgr_image(images[0, 0])
    img_h, img_w = img.shape[:2]
    print(f'图像尺寸：{img_w}x{img_h}')
    
    # 存储绘制后的图像
    drawn_combined = {}
    drawn_pred_only = {}
    
    # 统计信息
    gt_stats = {cam: 0 for cam in CAMERA_NAMES}
    pred_stats = {cam: 0 for cam in CAMERA_NAMES}
    pred_class_stats = {}
    gt_class_stats = {}
    
    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        # 获取图像两份
        img_combined = tensor_to_bgr_image(images[0, cam_idx])
        img_pred_only = tensor_to_bgr_image(images[0, cam_idx])
        cv2.putText(img_combined, cam_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_pred_only, cam_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 获取相机内参（直接使用，不再缩放）
        K = cam_intrinsics[0, cam_idx].detach().cpu().numpy()[:3, :3]
        
        # 直接从 nuScenes API 获取 GT 框（使用全局坐标系）
        sample = nusc.get('sample', sample_token)
        for i, ann_token in enumerate(sample['anns']):
            # 获取标注信息
            annotation = nusc.get('sample_annotation', ann_token)
            cat_name = annotation['category_name']
            
            # 从官方 API 获取 Box 对象（全局坐标系）
            box = nusc.get_box(ann_token)
            
            # GT 框使用全局坐标系，执行完整的 Global -> Ego -> Camera 转换
            points_2d, depth, box_cam = project_3d_box_to_image(box, nusc, sample_token, cam_name, K, is_global=True)
            
            # 跳过无效投影
            if np.all(depth <= 0.1):
                continue
            
            # 绘制框
            color = (0, 255, 0)  # GT 用纯绿色
            img_combined, text_pos, success = draw_box(img_combined, points_2d, depth, color, thickness=2)
            
            # 只有画框成功后才绘制标签
            if success:
                class_name = cat_name
                # 去除前缀，只显示类名
                draw_text(img_combined, f'{class_name}', text_pos, color, font_scale=0.4)
                gt_stats[cam_name] += 1
                # 统计 GT 类别
                if class_name not in gt_class_stats:
                    gt_class_stats[class_name] = 0
                gt_class_stats[class_name] += 1
        
        # 绘制预测框
        if len(pred_bboxes) > 0:
            # 只处理前10个预测框，加快运行速度
            for i, (score, label, bbox) in enumerate(zip(pred_scores, pred_labels, pred_bboxes)):
                if i >= 10:
                    break
                # 解包边界框向量
                bbox_np = bbox.detach().cpu().numpy()
                x, y, z, w, l, h = bbox_np[:6]
                
                box = build_box_from_vec(bbox_np)
                # 预测框使用自车坐标系，仅执行 Ego -> Camera 转换
                points_2d, depth, box_cam = project_3d_box_to_image(box, nusc, sample_token, cam_name, K, is_global=False)
                
                # 绘制框到 combined 画布
                color = (0, 0, 255)  # Pred 用纯红色
                img_combined, text_pos, success = draw_box(img_combined, points_2d, depth, color, thickness=2)
                
                # 只有画框成功后才绘制标签
                if success:
                    class_name = CLASS_NAMES.get(label.item(), str(label.item()))
                    # 去除前缀，只显示类名和分数
                    draw_text(img_combined, f'{class_name} {score.item():.2f}', text_pos, color, font_scale=0.4)
                    pred_stats[cam_name] += 1
                    # 统计 Pred 类别
                    if class_name not in pred_class_stats:
                        pred_class_stats[class_name] = 0
                    pred_class_stats[class_name] += 1
                
                # 绘制框到 pred_only 画布
                img_pred_only, text_pos, success = draw_box(img_pred_only, points_2d, depth, color, thickness=2)
                
                # 只有画框成功后才绘制标签
                if success:
                    class_name = CLASS_NAMES.get(label.item(), str(label.item()))
                    draw_text(img_pred_only, f'{class_name} {score.item():.2f}', text_pos, color, font_scale=0.4)
        
        # 保存绘制后的图像
        drawn_combined[cam_name] = img_combined
        drawn_pred_only[cam_name] = img_pred_only
    
    # 拼接图像
    canvas_combined = merge_6_cams(drawn_combined)
    canvas_pred_only = merge_6_cams(drawn_pred_only)
    
    # 打印统计信息
    print('\n统计信息：')
    for cam in CAMERA_NAMES:
        print(f'{cam}: GT={gt_stats[cam]}, Pred={pred_stats[cam]}')
    
    # 检查是否有预测框
    total_pred = sum(pred_stats.values())
    if total_pred == 0:
        print('\n⚠️  警告：没有预测框被绘制，请检查模型输出和坐标系')
    
    # 构建统计信息
    total_gt = sum(gt_stats.values())
    total_pred = sum(pred_stats.values())
    
    stats = {
        'pred_total': total_pred,
        'pred_details': pred_class_stats,
        'gt_total': total_gt,
        'gt_details': gt_class_stats
    }
    
    # === 纯视觉 BEV 绘制 (白底+网格+红绿框) ===
    import math
    bev_size = 800
    pixels_per_meter = 10
    bev_img = np.ones((bev_size, bev_size, 3), dtype=np.uint8) * 255 # 纯白底
    center_px = bev_size // 2
    
    # 1. 画浅灰色十字网格
    for i in range(0, bev_size, 100):
        cv2.line(bev_img, (i, 0), (i, bev_size), (230, 230, 230), 1)
        cv2.line(bev_img, (0, i), (bev_size, i), (230, 230, 230), 1)
    
    # 2. 画中心自车 (深灰色)
    cv2.rectangle(bev_img, (center_px - 9, center_px - 20), (center_px + 9, center_px + 20), (100, 100, 100), -1)
    
    # 画框内部函数
    def draw_bev_box(box_ego, color):
        x, y, z = box_ego.center
        w, l, h = box_ego.wlh
        yaw = box_ego.orientation.yaw_pitch_roll[0]
        # 计算四个角
        corners = np.array([[-w/2, l/2], [w/2, l/2], [w/2, -l/2], [-w/2, -l/2]])
        rot_mat = np.array([[math.cos(-yaw), -math.sin(-yaw)], [math.sin(-yaw), math.cos(-yaw)]])
        corners_rot = corners.dot(rot_mat.T)
        
        corners_img = np.zeros((4, 2), dtype=np.int32)
        for j in range(4):
            # 坐标转换：Ego(X右,Y前) -> Image(X右,Y下)
            img_x = int(center_px + (x + corners_rot[j, 0]) * pixels_per_meter)
            img_y = int(center_px - (y + corners_rot[j, 1]) * pixels_per_meter)
            corners_img[j] = [img_x, img_y]
        cv2.polylines(bev_img, [corners_img], isClosed=True, color=color, thickness=2)

    # 获取 ego_pose (用于将 GT 转到自车坐标系)
    sample = nusc.get('sample', sample_token)
    sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])

    # 3. 画 GT 绿框
    # 直接从 nuScenes API 获取 GT 框
    sample = nusc.get('sample', sample_token)
    for ann_token in sample['anns']:
        box = nusc.get_box(ann_token)
        # 转换到 Ego 坐标系
        box_ego = copy.deepcopy(box)
        box_ego.translate([-px for px in ego_pose['translation']])
        box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
        draw_bev_box(box_ego, (0, 255, 0)) # BGR: Green
        
    # 4. 画 Pred 红框
    if len(pred_bboxes) > 0:
        for bbox in pred_bboxes:
            bbox_np = bbox.detach().cpu().numpy()
            box = build_box_from_vec(bbox_np)
            if box is None:
                continue
            draw_bev_box(box, (0, 0, 255)) # BGR: Red
            
    # 保存结果（如果指定了输出目录）
    if output_dir is not None:
        output_path = os.path.join(output_dir, f'pred_{sample_token}.jpg')
        cv2.imwrite(output_path, canvas_combined)
        print(f'\n结果已保存到：{output_path}')
        
        pred_output_path = os.path.join(output_dir, f'pred_only_{sample_token}.jpg')
        cv2.imwrite(pred_output_path, canvas_pred_only)
        print(f'仅预测结果已保存到：{pred_output_path}')
        
        bev_output_path = os.path.join(output_dir, f'bev_{sample_token}.jpg')
        cv2.imwrite(bev_output_path, bev_img)
        print(f'BEV 图像已保存到：{bev_output_path}')
    
    # 打印统计信息
    print('\n分类统计信息：')
    print(f'GT 总数：{total_gt}')
    print(f'Pred 总数：{total_pred}')
    print(f'Pred 类别明细：{pred_class_stats}')
    
    # 结构化输出预测框数据
    pred_data = []
    if len(pred_bboxes) > 0:
        for i, (score, label, bbox) in enumerate(zip(pred_scores, pred_labels, pred_bboxes)):
            bbox_np = bbox.detach().cpu().numpy()
            pred_data.append({
                'id': i,
                'score': score.item(),
                'label': label.item(),
                'center': bbox_np[:3].tolist(),
                'size': bbox_np[3:6].tolist(),
                'sin_yaw': bbox_np[6],
                'cos_yaw': bbox_np[7],
                'velocity': bbox_np[8:10].tolist()
            })
    
    # 确保返回三个图像
    return canvas_combined, canvas_pred_only, bev_img, stats

def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='TDR-QAF 3D Object Detection Inference')
    parser.add_argument('--confidence', type=float, default=0.05, help='Confidence threshold for filtering predictions')
    parser.add_argument('--topk', type=int, default=50, help='Maximum number of predictions to keep')
    parser.add_argument('--overfit', action='store_true', help='是否为本地过拟合模式的权重' )
    args = parser.parse_args()
    
    # 1. 环境准备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')
    print(f'Confidence threshold: {args.confidence}')
    print(f'TopK: {args.topk}')
    
    # 2. 创建输出目录
    output_dir = create_output_dir()
    
    # 3. 加载模型
    try:
        model = load_model(device, is_overfit=args.overfit)
    except Exception as e:
        print(f'加载模型失败：{e}')
        return
    
    # 4. 加载数据集
    try:
        dataset = NuScenesDataset(
            root='./dataset',
            debug_mode=False
        )
        print(f'数据集加载成功，共 {len(dataset)} 个样本')
    except Exception as e:
        print(f'加载数据集失败：{e}')
        return
    
    # 5. 连续抽取并测试 2 个有效样本
    random.seed(time.time())
    
    for test_idx in range(2):
        print(f"\n{'='*20} 开始处理第 {test_idx + 1}/2 个样本 {'='*20}")
        
        # 5.1 自动跳过损坏样本的抽签循环
        while True:
            index = random.randint(0, len(dataset) - 1)
            print(f'尝试处理样本索引：{index}')
            try:
                sample_data = get_sample(dataset, index, device)
                if sample_data is None:
                    print(f"⚠️ 样本 {index} 缺失图片，正在重新抽取...")
                    continue
                    
                images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels = sample_data
                sample_token = dataset.samples[index]['token']
                print(f'✅ 成功获取样本 Token：{sample_token}')
                break # 抽到完美数据，跳出循环
            except Exception as e:
                print(f'获取样本数据失败：{e}，正在重新抽取...')
                continue
        
        # 6. 运行模型推理
        try:
            images = images.float()
            boxes_2d = boxes_2d.float()
            cam_intrinsics = cam_intrinsics.float()
            cam_extrinsics = cam_extrinsics.float()
            
            # 诊断日志：检查 2D 框的有效性
            print(f"[Debug] boxes_2d shape: {boxes_2d.shape}, sum: {boxes_2d.sum().item():.2f}, "
                  f"non-zero boxes: {(boxes_2d.abs().sum(dim=-1) > 1e-3).sum().item()}")
            
            cls_scores, bbox_preds = run_model(model, images, boxes_2d, cam_intrinsics, cam_extrinsics)
        except Exception as e:
            print(f'❌ 模型推理失败：{e}，跳过当前样本')
            continue
        
        # 7. 解码边界框
        try:
            # 使用命令行参数中的置信度阈值和topk
            pred_scores, pred_labels, pred_bboxes = decode_bbox(cls_scores, bbox_preds, threshold=args.confidence, topk=args.topk)
        except Exception as e:
            print(f'❌ 解码边界框失败：{e}')
            continue
            
        # 8. 可视化并保存
        try:
            # 调用原有的可视化函数，确保每张图名字唯一
            visualize(images, cam_intrinsics, gt_bboxes, gt_labels, pred_scores, pred_labels, pred_bboxes, sample_token, output_dir, dataset.nusc)
            print(f"🎉 第 {test_idx + 1} 个样本处理完成")
        except Exception as e:
            print(f'❌ 可视化失败：{e}')
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
    # 确保结果保存到 results 目录
    import os
    os.makedirs('results', exist_ok=True)
    print(f"✅ 推理完成，结果已保存至: {os.path.abspath('results')}")