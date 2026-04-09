import os
import sys
import copy
import glob
import random
import time
import math
import traceback
import numpy as np
import torch
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    searched = explicit_candidates or ['saved_models/**/best_model.pth', 'saved_models/**/tdr_qaf_epoch_*.pth']
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
        gt_items.append({
            'cat_name': cat_name,
            'cat_short': simplify_class_name(cat_name),
            'class_id': map_gt_category_to_class(cat_name),
            'box_global': gt_box_global,
            'box_ego': gt_box_ego,
        })
        gt_class_stats[cat_name] = gt_class_stats.get(cat_name, 0) + 1

    pred_items = []
    pred_class_stats = {}
    for score, label, bbox in zip(pred_scores, pred_labels, pred_bboxes):
        cls_id = int(label.item())
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        box = build_box_from_vec(bbox.detach().cpu().numpy())
        if box is None:
            continue
        pred_items.append({'score': float(score.item()), 'label': cls_id, 'class_name': cls_name, 'box': box})
        pred_class_stats[cls_name] = pred_class_stats.get(cls_name, 0) + 1

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
                draw_text(img_combined, gt['cat_short'], text_pos, (0, 255, 0), font_scale=0.42)
                gt_per_cam[cam_name] += 1
        for pred in pred_items:
            points_2d, depth, _ = project_3d_box_to_image(pred['box'], nusc, sample_token, cam_name, k, is_global=False)
            if points_2d is None or depth is None:
                continue
            img_combined, text_pos, ok = draw_box(img_combined, points_2d, depth, (0, 0, 255), thickness=2)
            if ok:
                draw_text(img_combined, f"{pred['class_name']} {pred['score']:.2f}", text_pos, (0, 0, 255), font_scale=0.42)
                pred_per_cam[cam_name] += 1
        drawn_combined[cam_name] = img_combined

    canvas_combined = merge_6_cams(drawn_combined)

    # second image: top-score prediction + nearest GT, split view
    best_pred, best_gt, best_cam = None, None, 'CAM_FRONT'
    if pred_items:
        best_pred = max(pred_items, key=lambda x: x['score'])
        pred_xy = np.array(best_pred['box'].center[:2], dtype=np.float32)
        same_cls = [g for g in gt_items if g['class_id'] is not None and g['class_id'] == best_pred['label']]
        pool = same_cls if same_cls else gt_items
        if pool:
            best_gt = min(pool, key=lambda g: np.linalg.norm(np.array(g['box_ego'].center[:2], dtype=np.float32) - pred_xy))
        best_area = -1
        for cam_idx, cam_name in enumerate(CAMERA_NAMES):
            k = cam_intrinsics[0, cam_idx].detach().cpu().numpy()[:3, :3]
            pts_p, dep_p, _ = project_3d_box_to_image(best_pred['box'], nusc, sample_token, cam_name, k, is_global=False)
            rect_p = projected_rect(pts_p, dep_p, img_ref.shape)
            if rect_p is None:
                continue
            area = (rect_p[2] - rect_p[0]) * (rect_p[3] - rect_p[1])
            if area > best_area:
                best_area, best_cam = area, cam_name

    cam_idx = CAMERA_NAMES.index(best_cam)
    left_img = tensor_to_bgr_image(images[0, cam_idx])
    right_img = tensor_to_bgr_image(images[0, cam_idx])
    k_focus = cam_intrinsics[0, cam_idx].detach().cpu().numpy()[:3, :3]
    rect_gt, rect_pred = None, None
    if best_gt is not None:
        pts_gt, dep_gt, _ = project_3d_box_to_image(best_gt['box_global'], nusc, sample_token, best_cam, k_focus, is_global=True)
        left_img, text_gt, ok_gt = draw_box(left_img, pts_gt, dep_gt, (0, 255, 0), thickness=3)
        rect_gt = projected_rect(pts_gt, dep_gt, left_img.shape)
        if ok_gt:
            draw_text(left_img, f"GT {best_gt['cat_short']}", text_gt, (0, 255, 0), font_scale=0.55, thickness=2)
    if best_pred is not None:
        pts_pred, dep_pred, _ = project_3d_box_to_image(best_pred['box'], nusc, sample_token, best_cam, k_focus, is_global=False)
        right_img, text_pred, ok_pred = draw_box(right_img, pts_pred, dep_pred, (0, 0, 255), thickness=3)
        rect_pred = projected_rect(pts_pred, dep_pred, right_img.shape)
        if ok_pred:
            draw_text(
                right_img,
                f"Pred {best_pred['class_name']} {best_pred['score']:.2f}",
                text_pred,
                (0, 0, 255),
                font_scale=0.55,
                thickness=2,
            )

    left_zoom = zoom_crop(left_img, rect_gt if rect_gt is not None else rect_pred)
    right_zoom = zoom_crop(right_img, rect_pred if rect_pred is not None else rect_gt)
    cv2.putText(left_zoom, f"GT Focus | {best_cam}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 240, 20), 2)
    cv2.putText(right_zoom, f"Pred Focus | {best_cam}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 240), 2)
    gap_focus = np.full((img_h, 8, 3), (35, 35, 35), dtype=np.uint8)
    canvas_top_pair = cv2.hconcat([left_zoom, gap_focus, right_zoom])
    if best_pred is None:
        cv2.putText(canvas_top_pair, 'No prediction found', (20, img_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # third image: pseudo 3D BEV
    bev_size, ppm = 900, 10.0
    center_x, center_y = bev_size // 2, int(bev_size * 0.78)
    bev_img = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    for y in range(bev_size):
        t = y / float(bev_size - 1)
        v = int(22 + 40 * (1.0 - t))
        bev_img[y, :, :] = (v, v + 6, v + 12)

    for meters in [10, 20, 30, 40]:
        r = int(meters * ppm)
        cv2.circle(bev_img, (center_x, center_y), r, (70, 78, 90), 1)
    cv2.line(bev_img, (center_x, 0), (center_x, bev_size), (60, 68, 82), 1)
    cv2.line(bev_img, (0, center_y), (bev_size, center_y), (60, 68, 82), 1)

    def ego_to_pixel(x_forward, y_left):
        return int(center_x + y_left * ppm), int(center_y - x_forward * ppm)

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

    def draw_prism(box_ego, color):
        overlay = bev_img.copy()
        bottom = np.array([ego_to_pixel(a, b) for a, b in corners_xy(box_ego)], dtype=np.int32)
        z_pix = int(2 + float(np.clip(box_ego.wlh[2], 0.5, 4.0)) * 3.2)
        top = bottom.copy()
        top[:, 1] = np.clip(top[:, 1] - z_pix, 0, bev_size - 1)
        for i in range(4):
            j = (i + 1) % 4
            face = np.array([bottom[i], bottom[j], top[j], top[i]], dtype=np.int32)
            cv2.fillPoly(overlay, [face], tuple(int(c * 0.55) for c in color))
        cv2.fillPoly(overlay, [bottom], tuple(int(c * 0.35) for c in color))
        cv2.fillPoly(overlay, [top], tuple(int(min(255, c * 1.15)) for c in color))
        cv2.addWeighted(overlay, 0.55, bev_img, 0.45, 0, bev_img)
        cv2.polylines(bev_img, [bottom], True, color, 2)
        cv2.polylines(bev_img, [top], True, color, 2)

    for gt in gt_items:
        draw_prism(gt['box_ego'], (60, 220, 110))
    for pred in pred_items:
        draw_prism(pred['box'], (70, 80, 245))

    stats = {'pred_total': len(pred_items), 'pred_details': pred_class_stats, 'gt_total': len(gt_items), 'gt_details': gt_class_stats}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f'pred_{sample_token}.jpg'), canvas_combined)
        cv2.imwrite(os.path.join(output_dir, f'top_pair_{sample_token}.jpg'), canvas_top_pair)
        cv2.imwrite(os.path.join(output_dir, f'bev_{sample_token}.jpg'), bev_img)
    return canvas_combined, canvas_top_pair, bev_img, stats

def main():
    import argparse

    parser = argparse.ArgumentParser(description='TDR-QAF 3D Object Detection Inference')
    parser.add_argument('--confidence', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--topk', type=int, default=50, help='Maximum number of predictions')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Dataset root path')
    parser.add_argument('--nuscenes_version', type=str, default='v1.0-mini', help='NuScenes version')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--max_samples', type=int, default=None, help='Cap dataset sample pool')
    parser.add_argument('--load_indices', type=str, default='', help='Load sample_indices.json from training')
    parser.add_argument('--num_samples', type=int, default=1, help='How many samples to run per command')
    parser.add_argument('--overfit', action='store_true', help='Use overfit model config')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Confidence threshold: {args.confidence}')
    print(f'TopK: {args.topk}')

    output_dir = create_output_dir()

    try:
        model = load_model(device, is_overfit=args.overfit, checkpoint_path=args.checkpoint)
    except Exception as e:
        print(f'Failed to load model: {e}')
        return

    try:
        dataset = NuScenesDataset(
            root=args.data_root,
            debug_mode=False,
            max_samples=args.max_samples,
            version=args.nuscenes_version,
        )
        print(f'Dataset loaded. Total samples: {len(dataset)}')

        if args.load_indices:
            indices_data = NuScenesDataset.load_sample_indices(args.load_indices)
            dataset.set_sample_indices(indices_data['indices'])
            print(f'Loaded train indices. Inference sample pool: {len(indices_data["indices"])}')
        elif args.max_samples is not None:
            print(f'No --load_indices. Random sampling from capped pool: {len(dataset)}')
        else:
            print(f'No --load_indices. Random sampling from full dataset pool: {len(dataset)}')
    except Exception as e:
        print(f'Failed to load dataset: {e}')
        return

    random.seed(time.time())
    sample_pool = list(range(len(dataset)))
    random.shuffle(sample_pool)

    for test_idx in range(max(1, args.num_samples)):
        print(f"\n{'=' * 20} Processing sample {test_idx + 1}/{max(1, args.num_samples)} {'=' * 20}")

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
                threshold=args.confidence,
                topk=args.topk,
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
