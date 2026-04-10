import os
import sys
import copy
import glob
import random
import time
import math
import traceback
import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.scene_schema import build_scene_stream
from dataloaders.nuscenes_dataset import NuScenesDataset
from models.tdr_detector import TDRDetector
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

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
    9: 'barrier',
}


def create_output_dir():
    import datetime
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', f'infer_{current_time}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_ego_pose(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    sd = nusc.get('sample_data', cam_token)
    return nusc.get('ego_pose', sd['ego_pose_token'])


def global_to_ego(box, ego_pose):
    box_ego = copy.deepcopy(box)
    box_ego.translate([-x for x in ego_pose['translation']])
    box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
    return box_ego


def tensor_to_bgr_image(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resolve_checkpoint_path(checkpoint_path=None):
    explicit_candidates = []
    env_checkpoint = os.environ.get('TDR_QAF_CHECKPOINT')
    if checkpoint_path:
        explicit_candidates.append(checkpoint_path)
    if env_checkpoint:
        explicit_candidates.append(env_checkpoint)

    for candidate in explicit_candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate

    auto_candidates = []
    search_patterns = [
        os.path.join('saved_models', '**', 'best_model.pth'),
        os.path.join('saved_models', '**', 'tdr_qaf_epoch_*.pth'),
    ]
    for pattern in search_patterns:
        auto_candidates.extend(glob.glob(pattern, recursive=True))

    auto_candidates = [p for p in auto_candidates if os.path.isfile(p)]
    if auto_candidates:
        auto_candidates.sort(key=os.path.getmtime, reverse=True)
        return os.path.abspath(auto_candidates[0])

    searched = explicit_candidates or ['./saved_models/**/best_model.pth', './saved_models/04_08_17_18/tdr_qaf_epoch_50.pth']
    raise FileNotFoundError(f'No checkpoint found. Searched: {searched}')


def load_model(device, is_overfit=False, checkpoint_path=None):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    print(f'=== DEBUG: checkpoint path: {os.path.abspath(checkpoint_path)} ===')

    if is_overfit:
        decoder_layers, depth_dense, max_q = 2, 24, 200
    else:
        decoder_layers, depth_dense, max_q = 6, 48, 400

    model = TDRDetector(
        num_classes=10,
        embed_dims=256,
        num_decoder_layers=decoder_layers,
        num_depth_dense=depth_dense,
        max_queries=max_q,
    ).to(device)

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
    print(f'Checkpoint loaded: {checkpoint_path}')
    print(f'missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
    model.eval()
    return model


def get_sample(dataset, index, device):
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
    with torch.no_grad():
        cls_scores, bbox_preds = model(
            images,
            boxes_2d,
            cam_intrinsics,
            cam_extrinsics,
            temporal_depth_prior=None,
        )
    return cls_scores, bbox_preds


def decode_bbox(cls_scores, bbox_preds, threshold=0.05, topk=50, nms_dist=1.5):
    if cls_scores.dim() == 3:
        cls_scores = cls_scores[0]
    if bbox_preds.dim() == 3:
        bbox_preds = bbox_preds[0]

    prob = torch.sigmoid(cls_scores)
    pred_scores, pred_labels = torch.max(prob[..., :10], dim=-1)

    valid_mask = torch.isfinite(pred_scores)
    valid_mask &= torch.isfinite(bbox_preds).all(dim=-1)
    valid_mask &= pred_scores >= threshold
    pred_scores = pred_scores[valid_mask]
    pred_labels = pred_labels[valid_mask]
    pred_bboxes = bbox_preds[valid_mask]
    if pred_scores.numel() == 0:
        return pred_scores, pred_labels, pred_bboxes

    pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2], 0.0, 80.0)
    pred_bboxes[:, 3:6] = torch.clamp(pred_bboxes[:, 3:6], 0.2, 20.0)

    if pred_scores.shape[0] > topk:
        idx = torch.topk(pred_scores, topk).indices
        pred_scores = pred_scores[idx]
        pred_labels = pred_labels[idx]
        pred_bboxes = pred_bboxes[idx]

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
    return pred_scores[keep], pred_labels[keep], pred_bboxes[keep]


def build_box_from_vec(bbox_vec):
    vec = np.asarray(bbox_vec, dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        return None
    x, y, z, w, l, h, sin_yaw, cos_yaw = vec[:8]
    if z < 0.5 or z > 80.0:
        return None
    if min(w, l, h) < 0.2 or max(w, l, h) > 20.0:
        return None
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return Box(
        center=[float(x), float(y), float(z)],
        size=[float(w), float(l), float(h)],
        orientation=Quaternion(axis=[0, 0, 1], radians=float(yaw)),
    )

def project_box_cam(box_cam, k):
    corners = box_cam.corners()
    points_2d = view_points(corners, k, normalize=True)
    depth = corners[2, :]
    return points_2d, depth


def draw_box(img, points_2d, depth, color, thickness=1):
    img_h, img_w = img.shape[:2]
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    valid_mask = depth > 0.1
    if not np.any(valid_mask):
        return img, None, False
    if np.any(np.isnan(points_2d)) or np.any(np.isinf(points_2d)):
        return img, None, False
    if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
        return img, None, False

    max_coord = max(img_w * 5, img_h * 5)
    for i in range(points_2d.shape[1]):
        if abs(points_2d[0, i]) > max_coord or abs(points_2d[1, i]) > max_coord:
            valid_mask[i] = False
    if not np.any(valid_mask):
        return img, None, False

    lines_drawn = False
    for i, j in edges:
        if valid_mask[i] and valid_mask[j]:
            x1 = int(np.clip(points_2d[0, i], 0, img_w - 1))
            y1 = int(np.clip(points_2d[1, i], 0, img_h - 1))
            x2 = int(np.clip(points_2d[0, j], 0, img_w - 1))
            y2 = int(np.clip(points_2d[1, j], 0, img_h - 1))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            lines_drawn = True

    text_pos = None
    valid_x = points_2d[0, valid_mask]
    valid_y = points_2d[1, valid_mask]
    if len(valid_x) >= 3:
        min_x = int(np.min(valid_x))
        min_y = int(np.min(valid_y))
        if 0 <= min_x < img_w and 0 <= min_y < img_h:
            text_pos = (min_x, max(10, min_y - 10))
    return img, text_pos, bool(lines_drawn and text_pos is not None)


def draw_text(img, text, pos, color, font_scale=0.4, thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (pos[0] - 2, pos[1] - h - 2), (pos[0] + w + 2, pos[1] + 2), (255, 255, 255), -1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def project_3d_box_to_image(box, nusc, sample_token, cam_name, k, is_global=True):
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_name]
    sd = nusc.get('sample_data', cam_token)

    if is_global:
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        box_ego = copy.deepcopy(box)
        box_ego.translate([-x for x in pose['translation']])
        box_ego.rotate(Quaternion(pose['rotation']).inverse)
    else:
        box_ego = copy.deepcopy(box)

    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    if cs is None:
        return None, None, None
    box_cam = copy.deepcopy(box_ego)
    box_cam.translate([-x for x in cs['translation']])
    box_cam.rotate(Quaternion(cs['rotation']).inverse)
    points_2d, depth = project_box_cam(box_cam, k)
    return points_2d, depth, box_cam


def _prepare_canvas(img, target_size):
    if img is None:
        w, h = target_size
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    tw, th = target_size
    if img.shape[1] != tw or img.shape[0] != th:
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    return img


def merge_6_cams(path_dict):
    ref = path_dict.get('CAM_FRONT')
    if ref is None:
        raise ValueError('Missing CAM_FRONT image, cannot merge 6-camera canvas.')
    target_size = (ref.shape[1], ref.shape[0])
    prepared = {cam: _prepare_canvas(path_dict.get(cam), target_size) for cam in CAMERA_NAMES}
    gap = 10
    gap_color = (50, 50, 50)
    gap_h = np.full((target_size[1], gap, 3), gap_color, dtype=np.uint8)
    front = cv2.hconcat([prepared['CAM_FRONT_LEFT'], gap_h, prepared['CAM_FRONT'], gap_h, prepared['CAM_FRONT_RIGHT']])
    back = cv2.hconcat([prepared['CAM_BACK_LEFT'], gap_h, prepared['CAM_BACK'], gap_h, prepared['CAM_BACK_RIGHT']])
    gap_v = np.full((gap, front.shape[1], 3), gap_color, dtype=np.uint8)
    return cv2.vconcat([front, gap_v, back])


def simplify_class_name(name):
    table = {
        'vehicle.car': 'car',
        'vehicle.truck': 'truck',
        'vehicle.bus': 'bus',
        'vehicle.trailer': 'trailer',
        'vehicle.construction': 'const',
        'human.pedestrian': 'ped',
        'vehicle.motorcycle': 'moto',
        'vehicle.bicycle': 'bike',
        'movable_object.trafficcone': 'cone',
        'movable_object.barrier': 'barrier',
        'construction_vehicle': 'const',
        'traffic_cone': 'cone',
    }
    for key, short in table.items():
        if name.startswith(key):
            return short
    return name.split('.')[-1]


def map_gt_category_to_class(cat_name):
    if cat_name.startswith('vehicle.car'):
        return 0
    if cat_name.startswith('vehicle.truck'):
        return 1
    if cat_name.startswith('vehicle.bus'):
        return 2
    if cat_name.startswith('vehicle.trailer'):
        return 3
    if cat_name.startswith('vehicle.construction'):
        return 4
    if cat_name.startswith('human.pedestrian'):
        return 5
    if cat_name.startswith('vehicle.motorcycle'):
        return 6
    if cat_name.startswith('vehicle.bicycle'):
        return 7
    if cat_name.startswith('movable_object.trafficcone'):
        return 8
    if cat_name.startswith('movable_object.barrier'):
        return 9
    return None


def projected_rect(points_2d, depth, img_shape):
    if points_2d is None or depth is None:
        return None
    valid = np.isfinite(depth) & (depth > 0.1)
    valid &= np.isfinite(points_2d[0]) & np.isfinite(points_2d[1])
    if valid.sum() < 4:
        return None
    h, w = img_shape[:2]
    x = points_2d[0, valid]
    y = points_2d[1, valid]
    x1 = int(np.clip(np.min(x), 0, w - 1))
    y1 = int(np.clip(np.min(y), 0, h - 1))
    x2 = int(np.clip(np.max(x), 0, w - 1))
    y2 = int(np.clip(np.max(y), 0, h - 1))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2, y2


def zoom_crop(img, rect, pad_ratio=0.35):
    if rect is None:
        return img.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_w = bw // 2 + int(bw * pad_ratio)
    half_h = bh // 2 + int(bh * pad_ratio)
    sx1, sx2 = max(0, cx - half_w), min(w, cx + half_w)
    sy1, sy2 = max(0, cy - half_h), min(h, cy + half_h)
    if sx2 - sx1 < 10 or sy2 - sy1 < 10:
        return img.copy()
    crop = img[sy1:sy2, sx1:sx2]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def clip_focus_rect(rect, img_shape, min_area_ratio=0.00035, max_area_ratio=0.72):
    if rect is None:
        return None
    h, w = img_shape[:2]
    x1, y1, x2, y2 = rect
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return None
    return x1, y1, x2, y2


def rect_center(rect):
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)


def resize_with_pad(img, target_size, pad_color=(25, 25, 25)):
    tw, th = target_size
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((th, tw, 3), pad_color, dtype=np.uint8)
    scale = min(tw / float(w), th / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    off_x = (tw - new_w) // 2
    off_y = (th - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def gentle_focus_crop(img, rect, target_size, pad_ratio=0.7, min_crop_ratio=0.48):
    h, w = img.shape[:2]
    if rect is None:
        return resize_with_pad(img, target_size)
    x1, y1, x2, y2 = rect
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    crop_w = max(int(bw * (1.0 + pad_ratio)), int(w * min_crop_ratio))
    crop_h = max(int(bh * (1.0 + pad_ratio)), int(h * min_crop_ratio))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sx1 = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
    sy1 = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
    sx2 = sx1 + crop_w
    sy2 = sy1 + crop_h
    crop = img[sy1:sy2, sx1:sx2]
    return resize_with_pad(crop, target_size)


def draw_rect_with_label(img, rect, color, text):
    if rect is None:
        return
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    y_text = max(14, y1 - 8)
    draw_text(img, text, (x1 + 2, y_text), color, font_scale=0.5, thickness=1)


def short_pred_name(name):
    table = {
        'car': 'car',
        'truck': 'trk',
        'bus': 'bus',
        'trailer': 'trl',
        'construction_vehicle': 'const',
        'pedestrian': 'ped',
        'motorcycle': 'mot',
        'bicycle': 'bic',
        'traffic_cone': 'cone',
        'barrier': 'bar',
    }
    if name in table:
        return table[name]
    return name


def full_pred_name(name):
    table = {
        'car': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'trailer': 'trailer',
        'construction_vehicle': 'construction_vehicle',
        'pedestrian': 'pedestrian',
        'motorcycle': 'motorcycle',
        'bicycle': 'bicycle',
        'traffic_cone': 'traffic_cone',
        'barrier': 'barrier',
    }
    return table.get(name, name)


def full_gt_name(cat_name):
    cls_id = map_gt_category_to_class(cat_name)
    if cls_id is not None:
        return CLASS_NAMES.get(cls_id, cat_name)
    return cat_name.split('.')[-1]

def visualize(images, cam_intrinsics, gt_bboxes, gt_labels, pred_scores, pred_labels, pred_bboxes, sample_token, output_dir, nusc):
    _ = gt_bboxes, gt_labels
    img_ref = tensor_to_bgr_image(images[0, 0])
    img_h, img_w = img_ref.shape[:2]
    sample = nusc.get('sample', sample_token)
    ego_pose = get_ego_pose(nusc, sample_token)

    gt_items = []
    gt_class_stats = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        cat_name = ann['category_name']
        gt_box_global = nusc.get_box(ann_token)
        gt_box_ego = global_to_ego(gt_box_global, ego_pose)
        class_id = map_gt_category_to_class(cat_name)
        class_name = CLASS_NAMES.get(class_id, simplify_class_name(cat_name)) if class_id is not None else simplify_class_name(cat_name)
        if class_id is not None:
            cat_abbr = short_pred_name(CLASS_NAMES.get(class_id, 'car'))
        else:
            cat_abbr = short_pred_name(simplify_class_name(cat_name))
        gt_items.append({
            'id': f'gt-{ann_token}',
            'cat_name': cat_name,
            'cat_short': simplify_class_name(cat_name),
            'cat_abbr': cat_abbr,
            'class_id': class_id,
            'class_name': class_name,
            'label': f'GT {cat_abbr}',
            'box_global': gt_box_global,
            'box_ego': gt_box_ego,
        })
        gt_class_stats[cat_name] = gt_class_stats.get(cat_name, 0) + 1

    pred_items = []
    pred_class_stats = {}
    for score, label, bbox in zip(pred_scores, pred_labels, pred_bboxes):
        cls_id = int(label.item())
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        cls_short = short_pred_name(cls_name)
        box = build_box_from_vec(bbox.detach().cpu().numpy())
        if box is None:
            continue
        pred_items.append({
            'id': f'pred-{len(pred_items)}-{cls_name}',
            'score': float(score.item()),
            'label_id': cls_id,
            'class_name': cls_name,
            'label': f'Pred {cls_short}',
            'box': box,
        })
        pred_class_stats[cls_short] = pred_class_stats.get(cls_short, 0) + 1

    drawn_combined = {}
    gt_per_cam = {cam: 0 for cam in CAMERA_NAMES}
    pred_per_cam = {cam: 0 for cam in CAMERA_NAMES}
    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        img_combined = tensor_to_bgr_image(images[0, cam_idx])
        cv2.putText(img_combined, cam_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        k = cam_intrinsics[0, cam_idx].detach().cpu().numpy()[:3, :3]
        for gt in gt_items:
            points_2d, depth, _ = project_3d_box_to_image(gt['box_global'], nusc, sample_token, cam_name, k, is_global=True)
            if points_2d is None or depth is None or np.all(depth <= 0.1):
                continue
            img_combined, text_pos, ok = draw_box(img_combined, points_2d, depth, (0, 255, 0), thickness=2)
            if ok:
                draw_text(img_combined, gt['cat_abbr'], text_pos, (0, 255, 0), font_scale=0.42)
                gt_per_cam[cam_name] += 1
        for pred in pred_items:
            points_2d, depth, _ = project_3d_box_to_image(pred['box'], nusc, sample_token, cam_name, k, is_global=False)
            if points_2d is None or depth is None:
                continue
            img_combined, text_pos, ok = draw_box(img_combined, points_2d, depth, (0, 0, 255), thickness=2)
            if ok:
                draw_text(img_combined, f"{short_pred_name(pred['class_name'])} {pred['score']:.2f}", text_pos, (0, 0, 255), font_scale=0.42)
                pred_per_cam[cam_name] += 1
        drawn_combined[cam_name] = img_combined

    canvas_combined = merge_6_cams(drawn_combined)

    # second image: choose the camera where the highest-score prediction is best visible,
    # then compose top (GT/Pred focus) + bottom (original image) with guide lines.
    best_pred, best_gt, best_cam = None, None, 'CAM_FRONT'
    best_focus_candidate = None
    for pred in pred_items:
        for cam_idx, cam_name in enumerate(CAMERA_NAMES):
            k = cam_intrinsics[0, cam_idx].detach().cpu().numpy()[:3, :3]
            pts_p, dep_p, _ = project_3d_box_to_image(pred['box'], nusc, sample_token, cam_name, k, is_global=False)
            rect_p = clip_focus_rect(projected_rect(pts_p, dep_p, img_ref.shape), img_ref.shape)
            if rect_p is None:
                continue
            area = (rect_p[2] - rect_p[0]) * (rect_p[3] - rect_p[1])
            candidate = {'pred': pred, 'cam_name': cam_name, 'rect_pred': rect_p, 'area': area}
            if best_focus_candidate is None:
                best_focus_candidate = candidate
            else:
                if pred['score'] > best_focus_candidate['pred']['score'] + 1e-6:
                    best_focus_candidate = candidate
                elif abs(pred['score'] - best_focus_candidate['pred']['score']) <= 1e-6 and area > best_focus_candidate['area']:
                    best_focus_candidate = candidate

    if best_focus_candidate is not None:
        best_pred = best_focus_candidate['pred']
        best_cam = best_focus_candidate['cam_name']
    elif pred_items:
        best_pred = max(pred_items, key=lambda x: x['score'])

    if best_pred is not None:
        pred_xy = np.array(best_pred['box'].center[:2], dtype=np.float32)
        same_cls = [g for g in gt_items if g['class_id'] is not None and g['class_id'] == best_pred['label_id']]
        pool = same_cls if same_cls else gt_items
        if pool:
            best_gt = min(pool, key=lambda g: np.linalg.norm(np.array(g['box_ego'].center[:2], dtype=np.float32) - pred_xy))

    cam_idx = CAMERA_NAMES.index(best_cam)
    focus_base = tensor_to_bgr_image(images[0, cam_idx])
    k_focus = cam_intrinsics[0, cam_idx].detach().cpu().numpy()[:3, :3]

    rect_gt, rect_pred = None, None
    if best_gt is not None:
        pts_gt, dep_gt, _ = project_3d_box_to_image(best_gt['box_global'], nusc, sample_token, best_cam, k_focus, is_global=True)
        rect_gt = clip_focus_rect(projected_rect(pts_gt, dep_gt, focus_base.shape), focus_base.shape, max_area_ratio=0.9)
    if best_pred is not None:
        pts_pred, dep_pred, _ = project_3d_box_to_image(best_pred['box'], nusc, sample_token, best_cam, k_focus, is_global=False)
        rect_pred = clip_focus_rect(projected_rect(pts_pred, dep_pred, focus_base.shape), focus_base.shape)
    if rect_pred is None and best_focus_candidate is not None and best_focus_candidate['cam_name'] == best_cam:
        rect_pred = best_focus_candidate['rect_pred']

    bottom_img = focus_base.copy()
    if best_gt is not None:
        gt_label = f"GT {best_gt['cat_abbr']}"
        draw_rect_with_label(bottom_img, rect_gt, (30, 220, 80), gt_label)
    if best_pred is not None:
        pred_label = f"Pred {short_pred_name(best_pred['class_name'])} {best_pred['score']:.2f}"
        draw_rect_with_label(bottom_img, rect_pred, (40, 80, 235), pred_label)

    gap = max(8, img_w // 80)
    header_h = 34
    half_h = img_h
    panel_w = max(120, (img_w - gap * 3) // 2)
    top_y = header_h
    left_x = gap
    right_x = left_x + panel_w + gap
    bottom_y = top_y + half_h + gap
    canvas_h = header_h + half_h + gap + half_h + gap
    canvas_top_pair = np.full((canvas_h, img_w, 3), (23, 26, 30), dtype=np.uint8)

    left_focus_src = bottom_img.copy()
    right_focus_src = bottom_img.copy()
    left_focus = gentle_focus_crop(left_focus_src, rect_gt if rect_gt is not None else rect_pred, (panel_w, half_h))
    right_focus = gentle_focus_crop(right_focus_src, rect_pred if rect_pred is not None else rect_gt, (panel_w, half_h))
    bottom_view = resize_with_pad(bottom_img, (img_w, half_h), pad_color=(23, 26, 30))

    canvas_top_pair[top_y:top_y + half_h, left_x:left_x + panel_w] = left_focus
    canvas_top_pair[top_y:top_y + half_h, right_x:right_x + panel_w] = right_focus
    canvas_top_pair[bottom_y:bottom_y + half_h, :img_w] = bottom_view

    cv2.rectangle(canvas_top_pair, (left_x, top_y), (left_x + panel_w, top_y + half_h), (70, 90, 95), 1)
    cv2.rectangle(canvas_top_pair, (right_x, top_y), (right_x + panel_w, top_y + half_h), (70, 90, 95), 1)
    cv2.rectangle(canvas_top_pair, (0, bottom_y), (img_w - 1, bottom_y + half_h - 1), (65, 80, 90), 1)

    cv2.putText(canvas_top_pair, f"Focus Camera: {best_cam}", (gap, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 232, 236), 2)
    gt_title = "GT Focus"
    if best_gt is not None:
        gt_title = f"GT Focus | {full_gt_name(best_gt['cat_name'])}"
    pred_title = "Pred Focus"
    if best_pred is not None:
        pred_title = f"Pred Focus | {full_pred_name(best_pred['class_name'])} {best_pred['score']:.2f}"
    cv2.putText(canvas_top_pair, gt_title, (left_x + 8, top_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 225, 95), 2)
    cv2.putText(canvas_top_pair, pred_title, (right_x + 8, top_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (45, 90, 235), 2)
    cv2.putText(canvas_top_pair, "Original", (10, bottom_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (226, 232, 238), 2)

    gt_center = rect_center(rect_gt) or rect_center(rect_pred) or (img_w // 2, img_h // 2)
    pred_center = rect_center(rect_pred) or gt_center
    gt_anchor_top = (left_x + panel_w // 2, top_y + half_h - 4)
    pred_anchor_top = (right_x + panel_w // 2, top_y + half_h - 4)
    gt_anchor_bottom = (int(gt_center[0]), int(bottom_y + gt_center[1]))
    pred_anchor_bottom = (int(pred_center[0]), int(bottom_y + pred_center[1]))
    cv2.line(canvas_top_pair, gt_anchor_top, gt_anchor_bottom, (45, 220, 100), 2, cv2.LINE_AA)
    cv2.line(canvas_top_pair, pred_anchor_top, pred_anchor_bottom, (50, 90, 235), 2, cv2.LINE_AA)
    cv2.circle(canvas_top_pair, gt_anchor_bottom, 4, (45, 220, 100), -1)
    cv2.circle(canvas_top_pair, pred_anchor_bottom, 4, (50, 90, 235), -1)

    if best_pred is None:
        cv2.putText(canvas_top_pair, 'No prediction found', (30, bottom_y + half_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 简化版：只输出 2D BEV 鸟瞰图和 JSON 数据
    def draw_simple_bev(items, title="BEV"):
        bev_size, ppm = 900, 10.0
        center_x, center_y = bev_size // 2, bev_size // 2  # 中心点在图片中心
        bev_img = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        
        # 背景色：传统俯视图使用浅灰色背景
        bev_img[:, :] = (240, 240, 240)

        # 绘制网格线（传统俯视图）
        for meters in [10, 20, 30, 40, 50]:
            r = int(meters * ppm)
            cv2.circle(bev_img, (center_x, center_y), r, (180, 180, 180), 1)
        
        # 绘制十字线
        cv2.line(bev_img, (center_x, 0), (center_x, bev_size), (180, 180, 180), 1)
        cv2.line(bev_img, (0, center_y), (bev_size, center_y), (180, 180, 180), 1)

        # 坐标转换：自车前方为 X 轴正方向，左侧为 Y 轴正方向
        def ego_to_pixel(x_forward, y_left):
            # 传统俯视图：X 轴向上，Y 轴向右
            pixel_x = center_x + int(y_left * ppm)
            pixel_y = center_y - int(x_forward * ppm)
            return pixel_x, pixel_y

        def corners_xy(box_ego):
            x, y, _ = box_ego.center
            w, l, _ = box_ego.wlh
            yaw = box_ego.orientation.yaw_pitch_roll[0]
            local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]], dtype=np.float32)
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=np.float32)
            c = local @ rot.T
            c[:, 0] += x
            c[:, 1] += y
            return c

        def draw_box_bev(box_ego, color):
            corners = corners_xy(box_ego)
            pts = np.array([ego_to_pixel(x, y) for x, y in corners], dtype=np.int32)
            cv2.fillPoly(bev_img, [pts], color)
            cv2.polylines(bev_img, [pts], True, (0, 0, 0), 2)

        # 绘制自车（在原点）
        ego_corners = np.array([
            [2.25, 0.9], [2.25, -0.9], [-2.25, -0.9], [-2.25, 0.9]
        ], dtype=np.float32)
        ego_pts = np.array([ego_to_pixel(x, y) for x, y in ego_corners], dtype=np.int32)
        cv2.fillPoly(bev_img, [ego_pts], (100, 100, 100))
        cv2.polylines(bev_img, [ego_pts], True, (0, 0, 0), 2)

        # 绘制预测框
        for pred in pred_items:
            draw_box_bev(pred['box'], (70, 130, 180))

        return bev_img

    bev_img = draw_simple_bev(pred_items, "BEV")

    # 恢复 SR 参考图的实现（3D matplotlib 绘图）
    x_range = (-20.0, 65.0)
    y_range = (-24.0, 24.0)
    z_range = (0.0, 7.5)

    gt_scene_items = []
    for gt in gt_items:
        cls_name = CLASS_NAMES.get(gt['class_id'], 'car') if gt['class_id'] is not None else gt['cat_short']
        gt_scene_items.append({'box': gt['box_ego'], 'class_name': cls_name, 'label': short_pred_name(cls_name), 'score': None})

    pred_scene_items = []
    for pred in pred_items:
        pred_scene_items.append({'box': pred['box'], 'class_name': pred['class_name'], 'label': short_pred_name(pred['class_name']), 'score': pred['score']})

    def in_range(box_ego):
        x_f, y_l, _ = box_ego.center
        return (x_range[0] - 2.0) <= x_f <= (x_range[1] + 2.0) and (y_range[0] - 2.0) <= y_l <= (y_range[1] + 2.0)

    def calc_cuboid_corners(center, size_lwh, yaw):
        cx, cy, cz = center
        l, w, h = size_lwh
        local = np.array([
            [ l * 0.5,  w * 0.5, -h * 0.5],
            [ l * 0.5, -w * 0.5, -h * 0.5],
            [-l * 0.5, -w * 0.5, -h * 0.5],
            [-l * 0.5,  w * 0.5, -h * 0.5],
            [ l * 0.5,  w * 0.5,  h * 0.5],
            [ l * 0.5, -w * 0.5,  h * 0.5],
            [-l * 0.5, -w * 0.5,  h * 0.5],
            [-l * 0.5,  w * 0.5,  h * 0.5],
        ], dtype=np.float32)
        rot = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw),  math.cos(yaw), 0.0],
            [0.0,            0.0,           1.0],
        ], dtype=np.float32)
        verts = local @ rot.T
        verts[:, 0] += cx
        verts[:, 1] += cy
        verts[:, 2] += cz
        return verts

    def draw_3d_cuboid(ax, corners_3d, facecolor, edgecolor, alpha=0.7):
        faces = [
            [corners_3d[0], corners_3d[1], corners_3d[2], corners_3d[3]],
            [corners_3d[4], corners_3d[5], corners_3d[6], corners_3d[7]],
            [corners_3d[0], corners_3d[1], corners_3d[5], corners_3d[4]],
            [corners_3d[2], corners_3d[3], corners_3d[7], corners_3d[6]],
            [corners_3d[1], corners_3d[2], corners_3d[6], corners_3d[5]],
            [corners_3d[4], corners_3d[7], corners_3d[3], corners_3d[0]],
        ]
        poly3d = Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor, alpha=alpha, linewidths=0.8)
        ax.add_collection3d(poly3d)

    def add_ground_shadow(ax, corners_3d, scale=1.08, alpha=0.18):
        base = np.array(corners_3d[:4], dtype=np.float32)
        center_xy = np.mean(base[:, :2], axis=0)
        base[:, :2] = center_xy + (base[:, :2] - center_xy) * scale
        base[:, 2] = 0.02
        poly = Poly3DCollection([base], facecolors=(0.24, 0.27, 0.32, alpha), edgecolors='none')
        ax.add_collection3d(poly)

    def setup_scene(ax):
        ax.set_facecolor('#F0F5FA')
        x = np.linspace(x_range[0], x_range[1], 2)
        y = np.linspace(y_range[0], y_range[1], 2)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        ground = np.zeros(xx.shape + (4,), dtype=np.float32)
        ground[..., 0] = 240.0 / 255.0
        ground[..., 1] = 245.0 / 255.0
        ground[..., 2] = 250.0 / 255.0
        ground[..., 3] = 1.0
        ax.plot_surface(xx, yy, zz, facecolors=ground, linewidth=0, antialiased=False, shade=False)

        lane_offsets = [-7.2, -3.6, 0.0, 3.6, 7.2]
        lane_x = np.linspace(-8.0, 62.0, 180)
        for offset in lane_offsets:
            lane_y = np.full_like(lane_x, offset)
            lane_z = np.full_like(lane_x, 0.03)
            ax.plot(lane_x, lane_y, lane_z, color=(1, 1, 1, 0.92), linewidth=1.0, linestyle=(0, (6, 6)))

        for i in range(24):
            x0 = i * 2.0
            x1 = x0 + 2.1
            w0 = 1.5 + i * 0.14
            w1 = 1.7 + i * 0.14
            alpha = max(0.02, 0.30 * (1.0 - i / 24.0))
            quad = np.array([
                [x0, -w0, 0.02],
                [x0,  w0, 0.02],
                [x1,  w1, 0.02],
                [x1, -w1, 0.02],
            ], dtype=np.float32)
            ax.add_collection3d(
                Poly3DCollection([quad], facecolors=(0.16, 0.33, 0.86, alpha), edgecolors='none')
            )

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_range[0], z_range[1])
        ax.view_init(elev=35, azim=-120)
        ax.set_proj_type('persp')
        ax.set_axis_off()
        ax.set_box_aspect([1.0, 1.0, 0.3])

    def render_semantic_scene(items, scene_title, is_pred):
        fig = plt.figure(figsize=(16, 9), dpi=100)
        fig.patch.set_facecolor('#F0F5FA')
        ax = fig.add_subplot(111, projection='3d')
        setup_scene(ax)

        ego_corners = calc_cuboid_corners(center=(0.0, 0.0, 0.75), size_lwh=(4.5, 1.8, 1.5), yaw=0.0)
        add_ground_shadow(ax, ego_corners, scale=1.12, alpha=0.20)
        draw_3d_cuboid(ax, ego_corners, facecolor='#2F343E', edgecolor='#1F242C', alpha=0.95)

        label_color = (0.22, 0.27, 0.33, 0.95)
        sorted_items = sorted(items, key=lambda obj: float(np.linalg.norm(np.asarray(obj['box'].center[:2], dtype=np.float32))))

        label_budget = 20
        for obj in sorted_items:
            box = obj['box']
            if not in_range(box):
                continue

            x, y, z = [float(v) for v in box.center]
            w, l, h = [float(v) for v in box.wlh]
            yaw = float(box.orientation.yaw_pitch_roll[0])
            cls_name = obj['class_name']

            if cls_name in ('car', 'truck', 'bus', 'trailer', 'construction_vehicle'):
                facecolor = '#D3D9DF'
                edgecolor = '#808A9F'
                alpha = 0.80
                l = float(np.clip(l, 1.8, 13.0))
                w = float(np.clip(w, 0.8, 3.4))
                h = float(np.clip(h, 1.0, 4.2))
            elif cls_name == 'pedestrian':
                facecolor = '#9EDCEB'
                edgecolor = '#4E9CB5'
                alpha = 0.82
                l = 0.45
                w = 0.45
                h = 1.72
                z = max(z, h * 0.5 + 0.02)
            elif cls_name in ('traffic_cone', 'barrier'):
                facecolor = '#F29A55'
                edgecolor = '#A7642D'
                alpha = 0.86
                if cls_name == 'traffic_cone':
                    l = 0.55
                    w = 0.55
                    h = 0.92
                    z = max(z, h * 0.5 + 0.02)
                else:
                    l = 1.6
                    w = 0.55
                    h = 0.8
                    z = max(z, h * 0.5 + 0.02)
            else:
                facecolor = '#D3D9DF'
                edgecolor = '#808A9F'
                alpha = 0.80

            corners = calc_cuboid_corners(center=(x, y, z), size_lwh=(l, w, h), yaw=yaw)
            add_ground_shadow(ax, corners, scale=1.08, alpha=0.17)
            draw_3d_cuboid(ax, corners, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
            z_text = float(np.max(corners[:, 2]) + 0.18)

            if label_budget > 0:
                text = obj['label']
                if is_pred and obj['score'] is not None:
                    text = f"{text} {obj['score']:.2f}"
                ax.text(x, y, z_text, text, fontsize=7, fontname='Arial', color=label_color, ha='center', va='bottom')
                label_budget -= 1

        ax.text2D(
            0.03,
            0.95,
            scene_title,
            transform=ax.transAxes,
            fontsize=12,
            fontname='Arial',
            color=(0.22, 0.27, 0.33, 1.0),
            weight='bold',
        )

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return img

    scene_gt_img = render_semantic_scene(gt_scene_items, "Surrounding Reality | Ground Truth", is_pred=False)
    scene_pred_img = render_semantic_scene(pred_scene_items, "Surrounding Reality | Prediction", is_pred=True)
    scene_stream = build_scene_stream(
        sample_token=sample_token,
        sample_timestamp=sample['timestamp'],
        nusc=nusc,
        gt_items=gt_items,
        pred_items=pred_items,
    )

    stats = {'pred_total': len(pred_items), 'pred_details': pred_class_stats, 'gt_total': len(gt_items), 'gt_details': gt_class_stats}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f'pred_{sample_token}.jpg'), canvas_combined)
        cv2.imwrite(os.path.join(output_dir, f'top_pair_{sample_token}.jpg'), canvas_top_pair)
        cv2.imwrite(os.path.join(output_dir, f'bev_{sample_token}.jpg'), bev_img)
        cv2.imwrite(os.path.join(output_dir, f'sr_gt_{sample_token}.jpg'), scene_gt_img)
        cv2.imwrite(os.path.join(output_dir, f'sr_pred_{sample_token}.jpg'), scene_pred_img)
        
        json_path = os.path.join(output_dir, f'scene_stream_{sample_token}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scene_stream, f, indent=2, ensure_ascii=False)
        print(f"Detection results saved to: {json_path}")
    
    return canvas_combined, canvas_top_pair, bev_img, scene_gt_img, scene_pred_img, stats, scene_stream

def main():
    # 绠€鍖栫増锛氬啓姝绘ā鍨嬭矾寰勫拰鏍锋湰绱㈠紩璺緞
    checkpoint_path = './saved_models/04_08_17-18/tdr_qaf_epoch_50.pth'
    sample_indices_path = './saved_models/04_08_17-18/sample_indices.json'
    confidence = 0.05
    topk = 50
    num_samples = 2  # 榛樿鎺ㄧ悊2涓牱鏈?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Confidence threshold: {confidence}')
    print(f'TopK: {topk}')
    print(f'Checkpoint path: {checkpoint_path}')
    print(f'Sample indices path: {sample_indices_path}')
    print(f'Number of samples to inference: {num_samples}')

    output_dir = create_output_dir()

    try:
        model = load_model(device, is_overfit=False, checkpoint_path=checkpoint_path)
    except Exception as e:
        print(f'Failed to load model: {e}')
        return

    try:
        dataset = NuScenesDataset(
            root='./dataset',
            debug_mode=False,
            max_samples=None,
            version='v1.0-mini',
        )
        print(f'Dataset loaded. Total samples: {len(dataset)}')

        indices_data = NuScenesDataset.load_sample_indices(sample_indices_path)
        dataset.set_sample_indices(indices_data['indices'])
        print(f'Loaded train indices. Inference sample pool: {len(indices_data["indices"])}')
    except Exception as e:
        print(f'Failed to load dataset: {e}')
        return

    random.seed(time.time())
    sample_pool = list(range(len(dataset)))
    random.shuffle(sample_pool)

    for test_idx in range(max(1, num_samples)):
        print(f"\n{'=' * 20} Processing sample {test_idx + 1}/{max(1, num_samples)} {'=' * 20}")

        sample_data = None
        sample_token = None
        while sample_pool:
            index = sample_pool.pop()
            print(f'Trying sample index: {index}')
            try:
                sample_data = get_sample(dataset, index, device)
                if sample_data is None:
                    print(f'Sample {index} is incomplete, retrying...')
                    continue
                sample_token = dataset.get_sample_token(index)
                print(f'Loaded sample token: {sample_token}')
                break
            except Exception as e:
                print(f'Failed reading sample {index}: {e}, retrying...')
                continue
        else:
            print('No more valid samples in sample pool, stop early.')
            break

        images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels = sample_data
        try:
            images = images.float()
            boxes_2d = boxes_2d.float()
            cam_intrinsics = cam_intrinsics.float()
            cam_extrinsics = cam_extrinsics.float()
            cls_scores, bbox_preds = run_model(model, images, boxes_2d, cam_intrinsics, cam_extrinsics)
            pred_scores, pred_labels, pred_bboxes = decode_bbox(
                cls_scores,
                bbox_preds,
                threshold=confidence,
                topk=topk,
            )
            visualize(
                images,
                cam_intrinsics,
                gt_bboxes,
                gt_labels,
                pred_scores,
                pred_labels,
                pred_bboxes,
                sample_token,
                output_dir,
                dataset.nusc,
            )
            print(f'Sample {test_idx + 1} done.')
        except Exception as e:
            print(f'Inference failed: {e}')
            traceback.print_exc()


if __name__ == '__main__':
    main()

