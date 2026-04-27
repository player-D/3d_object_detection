import os
import sys
import copy
import glob
import random
import time
import math
import traceback
import json
import argparse
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
from tools.runtime_config import (
    resolve_checkpoint_path,
    resolve_output_root,
    resolve_sample_indices_path,
)
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))

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
    output_dir = os.path.join(resolve_output_root('output'), current_time)
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


def enhance_visual_detail(img_bgr, detail_level='balanced'):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    level = {
        'mild': {'clip': 1.18, 'sigma_s': 4, 'sigma_r': 0.04, 'sharp': 0.04, 'saturation': 1.00},
        'balanced': {'clip': 1.35, 'sigma_s': 6, 'sigma_r': 0.06, 'sharp': 0.06, 'saturation': 1.00},
        'focus': {'clip': 1.55, 'sigma_s': 8, 'sigma_r': 0.08, 'sharp': 0.08, 'saturation': 1.01},
    }.get(detail_level, {'clip': 1.35, 'sigma_s': 6, 'sigma_r': 0.06, 'sharp': 0.06, 'saturation': 1.00})

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=level['clip'], tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    enhanced = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    enhanced = cv2.detailEnhance(enhanced, sigma_s=level['sigma_s'], sigma_r=level['sigma_r'])
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.0 + level['sharp'], blurred, -level['sharp'], 0)
    sharpened = cv2.addWeighted(sharpened, 0.62, img_bgr, 0.38, 0)

    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * level['saturation'], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.01, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def tensor_to_bgr_image(img_tensor, detail_level='balanced'):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return enhance_visual_detail(img_bgr, detail_level=detail_level)


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


def merge_6_cams(path_dict, bottom_panel=None, gap=8):
    ref = path_dict.get('CAM_FRONT')
    if ref is None:
        raise ValueError('Missing CAM_FRONT image, cannot merge 6-camera canvas.')
    target_size = (ref.shape[1], ref.shape[0])
    prepared = {cam: _prepare_canvas(path_dict.get(cam), target_size) for cam in CAMERA_NAMES}
    gap_color = (50, 50, 50)
    gap_h = np.full((target_size[1], gap, 3), gap_color, dtype=np.uint8)
    front = cv2.hconcat([prepared['CAM_FRONT_LEFT'], gap_h, prepared['CAM_FRONT'], gap_h, prepared['CAM_FRONT_RIGHT']])
    back = cv2.hconcat([prepared['CAM_BACK_LEFT'], gap_h, prepared['CAM_BACK'], gap_h, prepared['CAM_BACK_RIGHT']])
    gap_v = np.full((gap, front.shape[1], 3), gap_color, dtype=np.uint8)
    mosaic = cv2.vconcat([front, gap_v, back])
    if bottom_panel is None:
        return mosaic

    bottom = resize_with_pad(bottom_panel, (mosaic.shape[1], target_size[1]), pad_color=(23, 26, 30))
    return cv2.vconcat([mosaic, gap_v, bottom])


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


def _line_bottom_x(line, bottom_y):
    x1, y1, x2, y2 = [float(v) for v in line]
    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) * 0.5
    ratio = (bottom_y - y1) / (y2 - y1)
    return x1 + (x2 - x1) * ratio


def infer_lane_profile_from_image(front_img):
    if front_img is None or front_img.size == 0:
        return None

    img_h, img_w = front_img.shape[:2]
    roi_start = int(img_h * 0.48)
    roi = front_img[roi_start:, :]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)

    mask = np.zeros_like(edges)
    polygon = np.array(
        [[
            (int(img_w * 0.08), roi.shape[0] - 1),
            (int(img_w * 0.34), int(roi.shape[0] * 0.1)),
            (int(img_w * 0.66), int(roi.shape[0] * 0.1)),
            (int(img_w * 0.92), roi.shape[0] - 1),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=36,
        minLineLength=max(24, int(img_w * 0.05)),
        maxLineGap=40,
    )

    samples = []
    if lines is not None:
        bottom_y = roi.shape[0] - 1
        for line in lines[:, 0]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 4 or abs(dy) < 12:
                continue
            slope = dy / float(dx)
            if abs(slope) < 0.35 or abs(slope) > 5.5:
                continue
            length = math.hypot(dx, dy)
            bottom_x = _line_bottom_x(line, bottom_y)
            samples.append(
                {
                    'bottom_x': float(np.clip(bottom_x, 0, img_w - 1)),
                    'slope': slope,
                    'length': length,
                }
            )

    if not samples:
        return None

    left_samples = sorted(
        [sample for sample in samples if sample['bottom_x'] < img_w * 0.5 and sample['slope'] < 0],
        key=lambda sample: (sample['bottom_x'], sample['length']),
        reverse=True,
    )
    right_samples = sorted(
        [sample for sample in samples if sample['bottom_x'] > img_w * 0.5 and sample['slope'] > 0],
        key=lambda sample: (sample['bottom_x'], sample['length']),
    )

    if not left_samples and not right_samples:
        return None

    px_to_lateral = lambda px: clamp(-(float(px) - img_w * 0.5) / max(img_w * 0.18, 1.0) * 3.6, -10.0, 10.0)

    inner_left_px = left_samples[0]['bottom_x'] if left_samples else img_w * 0.36
    inner_right_px = right_samples[0]['bottom_x'] if right_samples else img_w * 0.64
    lane_width = clamp(abs(px_to_lateral(inner_left_px) - px_to_lateral(inner_right_px)), 3.2, 4.2)

    left_outer_px = left_samples[-1]['bottom_x'] if len(left_samples) > 1 else inner_left_px - (inner_right_px - inner_left_px) * 0.45
    right_outer_px = right_samples[-1]['bottom_x'] if len(right_samples) > 1 else inner_right_px + (inner_right_px - inner_left_px) * 0.45

    left_boundary_y = max(px_to_lateral(left_outer_px), px_to_lateral(inner_left_px) + lane_width * 0.55)
    right_boundary_y = min(px_to_lateral(right_outer_px), px_to_lateral(inner_right_px) - lane_width * 0.55)
    center_offset = clamp((px_to_lateral(inner_left_px) + px_to_lateral(inner_right_px)) * 0.5, -2.4, 2.4)

    lane_positions = [center_offset]
    if left_boundary_y - center_offset > lane_width * 0.72:
        lane_positions.append(center_offset + lane_width)
    if center_offset - right_boundary_y > lane_width * 0.72:
        lane_positions.append(center_offset - lane_width)

    left_boundary_kind = 'curb' if left_outer_px < img_w * 0.16 or len(left_samples) <= 1 else 'boundary'
    right_boundary_kind = 'curb' if right_outer_px > img_w * 0.84 or len(right_samples) <= 1 else 'boundary'

    asymmetry = ((inner_left_px + inner_right_px) * 0.5 - img_w * 0.5) / max(img_w * 0.5, 1.0)
    curvature = clamp(-asymmetry * 0.9, -0.35, 0.35)

    return {
        'source': 'vision_heuristic',
        'lane_width': lane_width,
        'center_offset': center_offset,
        'curvature': curvature,
        'lane_positions': lane_positions,
        'left_boundary_y': left_boundary_y,
        'right_boundary_y': right_boundary_y,
        'left_boundary_kind': left_boundary_kind,
        'right_boundary_kind': right_boundary_kind,
        'left_shoulder': 1.4 if left_boundary_kind == 'curb' else 0.9,
        'right_shoulder': 1.4 if right_boundary_kind == 'curb' else 0.9,
    }


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


def resize_to_cover(img, target_size):
    tw, th = target_size
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = max(tw / float(w), th / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    off_x = max(0, (new_w - tw) // 2)
    off_y = max(0, (new_h - th) // 2)
    return resized[off_y:off_y + th, off_x:off_x + tw]


def gentle_focus_crop(img, rect, target_size, pad_ratio=0.7, min_crop_ratio=0.48):
    h, w = img.shape[:2]
    if rect is None:
        return enhance_visual_detail(resize_to_cover(img, target_size), detail_level='mild')
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
    focused = resize_to_cover(crop, target_size)
    return enhance_visual_detail(focused, detail_level='focus')


def auto_crop_scene_image(img, bg_bgr=(250, 245, 240), tolerance=12, pad=36):
    if img is None or img.size == 0:
        return img
    bg = np.array(bg_bgr, dtype=np.int16).reshape(1, 1, 3)
    diff = np.abs(img.astype(np.int16) - bg).max(axis=2)
    mask = diff > tolerance
    if not np.any(mask):
        return img

    ys, xs = np.where(mask)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(img.shape[0], int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(img.shape[1], int(xs.max()) + pad)
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cv2.resize(cropped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


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


def focus_class_priority(class_name):
    order = {
        'pedestrian': 12,
        'car': 11,
        'motorcycle': 10,
        'bicycle': 9,
        'bus': 8,
        'truck': 7,
        'trailer': 6,
        'construction_vehicle': 5,
        'barrier': 4,
        'traffic_cone': 3,
    }
    return order.get(class_name, 1)


def add_overlay_panel(img, top_left, bottom_right, color, alpha=0.32):
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1 = max(0, min(int(x1), img.shape[1] - 1))
    y1 = max(0, min(int(y1), img.shape[0] - 1))
    x2 = max(0, min(int(x2), img.shape[1]))
    y2 = max(0, min(int(y2), img.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    img[y1:y2, x1:x2] = cv2.addWeighted(
        overlay[y1:y2, x1:x2],
        alpha,
        img[y1:y2, x1:x2],
        1.0 - alpha,
        0.0,
    )


def item_distance_meters(item, is_pred):
    box = item['box'] if is_pred else item['box_ego']
    center_xy = np.asarray(box.center[:2], dtype=np.float32)
    return float(np.linalg.norm(center_xy))


def item_display_name(item, is_pred):
    if is_pred:
        return full_pred_name(item['class_name'])
    return full_gt_name(item['cat_name'])


def build_visual_focus_candidates(items, is_pred, camera_images, cam_intrinsics, nusc, sample_token):
    candidates = []
    for item in items:
        class_name = item['class_name'] if is_pred else CLASS_NAMES.get(item['class_id'], item['cat_short'])
        distance = item_distance_meters(item, is_pred=is_pred)
        score = float(item.get('score', 0.0) or 0.0)
        box_ref = item['box'] if is_pred else item['box_global']
        display_name = item_display_name(item, is_pred=is_pred)
        label_text = f"{display_name} {score:.2f}" if is_pred else display_name

        views = []
        for cam_idx, cam_name in enumerate(CAMERA_NAMES):
            img = camera_images[cam_name]
            img_h, img_w = img.shape[:2]
            k = cam_intrinsics[0, cam_idx].detach().cpu().numpy()[:3, :3]
            points_2d, depth, _ = project_3d_box_to_image(
                box_ref,
                nusc,
                sample_token,
                cam_name,
                k,
                is_global=not is_pred,
            )
            rect = clip_focus_rect(
                projected_rect(points_2d, depth, img.shape),
                img.shape,
                min_area_ratio=0.00018,
                max_area_ratio=0.88,
            )
            if rect is None:
                continue

            x1, y1, x2, y2 = rect
            rect_area = float((x2 - x1) * (y2 - y1))
            area_ratio = rect_area / float(max(1, img_w * img_h))
            cx, cy = rect_center(rect)
            center_dx = abs(cx - img_w * 0.5) / max(img_w * 0.5, 1.0)
            center_dy = abs(cy - img_h * 0.58) / max(img_h * 0.58, 1.0)
            center_score = max(0.2, 1.25 - center_dx * 0.45 - center_dy * 0.30)
            view_score = area_ratio * 10000.0 * center_score + max(0.0, 30.0 - distance)

            views.append(
                {
                    'cam_name': cam_name,
                    'rect': rect,
                    'points_2d': points_2d,
                    'depth': depth,
                    'view_score': view_score,
                    'area_ratio': area_ratio,
                }
            )

        if not views:
            continue

        views.sort(key=lambda view: view['view_score'], reverse=True)
        priority = focus_class_priority(class_name)
        rank_score = priority * 1000.0 + max(0.0, 80.0 - distance) * 8.0 + score * 120.0 + views[0]['view_score']
        candidates.append(
            {
                'item': item,
                'class_name': class_name,
                'display_name': display_name,
                'label_text': label_text,
                'distance': distance,
                'score': score,
                'priority': priority,
                'rank_score': rank_score,
                'best_view': views[0],
                'views': views,
            }
        )

    candidates.sort(key=lambda candidate: candidate['rank_score'], reverse=True)
    return candidates


def draw_candidate_on_image(img, candidate, view, color):
    draw_box(img, view['points_2d'], view['depth'], color, thickness=2)
    x1, y1, x2, y2 = view['rect']
    label_pos = (x1 + 2, max(16, y1 - 8))
    label = candidate['display_name']
    if candidate['score'] > 0:
        label = f"{label} {candidate['score']:.2f}"
    draw_text(img, label, label_pos, color, font_scale=0.5, thickness=1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def build_sr_detail_board(title, subtitle, items, is_pred, camera_images, cam_intrinsics, nusc, sample_token):
    card_color = (52, 119, 214) if is_pred else (39, 167, 92)
    accent_color = (236, 240, 245)
    board_w, board_h = 1600, 920
    board = np.full((board_h, board_w, 3), (247, 249, 252), dtype=np.uint8)

    candidates = build_visual_focus_candidates(
        items,
        is_pred=is_pred,
        camera_images=camera_images,
        cam_intrinsics=cam_intrinsics,
        nusc=nusc,
        sample_token=sample_token,
    )

    main_cam = candidates[0]['best_view']['cam_name'] if candidates else 'CAM_FRONT'
    main_img = camera_images.get(main_cam, camera_images['CAM_FRONT']).copy()
    visible_in_main = []
    for candidate in candidates:
        for view in candidate['views']:
            if view['cam_name'] == main_cam:
                visible_in_main.append((candidate, view))
                break
    visible_in_main.sort(key=lambda pair: pair[0]['rank_score'], reverse=True)

    main_annotation_count = 4 if is_pred else 3
    for candidate, view in visible_in_main[:main_annotation_count]:
        draw_candidate_on_image(main_img, candidate, view, card_color)

    header_h = 84
    outer_pad = 28
    right_w = 392
    gap = 22
    main_w = board_w - outer_pad * 2 - right_w - gap
    main_h = 690
    crop_gap = 16
    summary_h = 110
    crop_h = 188
    crop_w = right_w - 24

    main_panel = resize_with_pad(main_img, (main_w, main_h), pad_color=(243, 246, 250))
    main_x = outer_pad
    main_y = header_h + 10
    board[main_y:main_y + main_h, main_x:main_x + main_w] = main_panel
    cv2.rectangle(board, (main_x, main_y), (main_x + main_w, main_y + main_h), (202, 210, 220), 1, cv2.LINE_AA)

    cv2.putText(board, title, (outer_pad, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (46, 55, 66), 2, cv2.LINE_AA)
    cv2.putText(board, subtitle, (outer_pad, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (104, 117, 132), 1, cv2.LINE_AA)
    cv2.putText(board, f"Main camera: {main_cam}", (main_x + 12, main_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (52, 64, 78), 2, cv2.LINE_AA)

    right_x = main_x + main_w + gap
    right_y = header_h + 10
    cv2.rectangle(board, (right_x, right_y), (right_x + right_w, right_y + summary_h), (214, 220, 228), 1, cv2.LINE_AA)
    add_overlay_panel(board, (right_x, right_y), (right_x + right_w, right_y + summary_h), accent_color, alpha=0.85)
    cv2.putText(board, "Focus Summary", (right_x + 14, right_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (50, 58, 70), 2, cv2.LINE_AA)
    cv2.putText(board, f"Main annotations: {min(main_annotation_count, len(visible_in_main))}", (right_x + 14, right_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (92, 102, 116), 1, cv2.LINE_AA)
    cv2.putText(board, f"Focus crops: {min(2, len(candidates))}", (right_x + 14, right_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (92, 102, 116), 1, cv2.LINE_AA)
    cv2.putText(board, "Paper-ready view keeps only key objects and local details.", (right_x + 14, right_y + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (116, 126, 138), 1, cv2.LINE_AA)

    crop_start_y = right_y + summary_h + 18
    for slot in range(2):
        card_y = crop_start_y + slot * (crop_h + crop_gap)
        add_overlay_panel(board, (right_x, card_y), (right_x + right_w, card_y + crop_h), accent_color, alpha=0.9)
        cv2.rectangle(board, (right_x, card_y), (right_x + right_w, card_y + crop_h), (214, 220, 228), 1, cv2.LINE_AA)

        if slot >= len(candidates):
            cv2.putText(board, "No additional key target", (right_x + 18, card_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (72, 82, 94), 2, cv2.LINE_AA)
            cv2.putText(board, "The remaining scene is intentionally kept clean.", (right_x + 18, card_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 130, 142), 1, cv2.LINE_AA)
            continue

        candidate = candidates[slot]
        view = candidate['best_view']
        crop_source = camera_images[view['cam_name']].copy()
        draw_candidate_on_image(crop_source, candidate, view, card_color)
        crop_target_h = max(112, crop_h - 62)
        crop_img = gentle_focus_crop(crop_source, view['rect'], (crop_w, crop_target_h), pad_ratio=0.75, min_crop_ratio=0.32)

        card_img_y = card_y + 52
        board[card_img_y:card_img_y + crop_img.shape[0], right_x + 12:right_x + 12 + crop_img.shape[1]] = crop_img
        cv2.rectangle(
            board,
            (right_x + 12, card_img_y),
            (right_x + 12 + crop_img.shape[1], card_img_y + crop_img.shape[0]),
            (204, 211, 220),
            1,
            cv2.LINE_AA,
        )

        header_text = candidate['display_name']
        if is_pred:
            header_text = f"{header_text} {candidate['score']:.2f}"
        cv2.putText(board, header_text, (right_x + 14, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (48, 56, 68), 2, cv2.LINE_AA)
        cv2.putText(
            board,
            f"{view['cam_name']} | {candidate['distance']:.1f}m",
            (right_x + 14, card_y + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            card_color,
            1,
            cv2.LINE_AA,
        )

    figure_note = "Suggested for thesis: clean camera evidence with limited target annotation."
    cv2.putText(board, figure_note, (outer_pad, board_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (108, 118, 130), 1, cv2.LINE_AA)

    return board

def visualize(images, cam_intrinsics, gt_bboxes, gt_labels, pred_scores, pred_labels, pred_bboxes, sample_token, output_dir, nusc):
    _ = gt_bboxes, gt_labels
    img_ref = tensor_to_bgr_image(images[0, 0], detail_level='balanced')
    img_h, img_w = img_ref.shape[:2]
    sample = nusc.get('sample', sample_token)
    ego_pose = get_ego_pose(nusc, sample_token)
    lane_profile = infer_lane_profile_from_image(img_ref)
    camera_images = {
        cam_name: tensor_to_bgr_image(
            images[0, cam_idx],
            detail_level='focus' if cam_name == 'CAM_FRONT' else 'balanced',
        )
        for cam_idx, cam_name in enumerate(CAMERA_NAMES)
    }

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
        img_combined = camera_images[cam_name].copy()
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

    canvas_combined = None

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
    focus_base = tensor_to_bgr_image(images[0, cam_idx], detail_level='focus')
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

    # 左边放大图只显示 GT 框
    left_focus_src = focus_base.copy()
    if best_gt is not None:
        gt_label = f"GT {best_gt['cat_abbr']}"
        draw_rect_with_label(left_focus_src, rect_gt, (30, 220, 80), gt_label)
    left_focus = gentle_focus_crop(left_focus_src, rect_gt if rect_gt is not None else rect_pred, (panel_w, half_h))

    # 右边放大图只显示预测框
    right_focus_src = focus_base.copy()
    if best_pred is not None:
        pred_label = f"Pred {short_pred_name(best_pred['class_name'])} {best_pred['score']:.2f}"
        draw_rect_with_label(right_focus_src, rect_pred, (40, 80, 235), pred_label)
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
    
    # 改进连接线：使用虚线和更柔和的颜色
    # GT 连接线（绿色虚线）
    cv2.line(canvas_top_pair, gt_anchor_top, gt_anchor_bottom, (60, 200, 100), 2, cv2.LINE_AA)
    # 添加箭头效果
    arrow_length = 15
    arrow_angle = math.pi / 6
    gt_dx = gt_anchor_bottom[0] - gt_anchor_top[0]
    gt_dy = gt_anchor_bottom[1] - gt_anchor_top[1]
    gt_angle = math.atan2(gt_dy, gt_dx)
    gt_arrow1 = (
        int(gt_anchor_bottom[0] - arrow_length * math.cos(gt_angle - arrow_angle)),
        int(gt_anchor_bottom[1] - arrow_length * math.sin(gt_angle - arrow_angle))
    )
    gt_arrow2 = (
        int(gt_anchor_bottom[0] - arrow_length * math.cos(gt_angle + arrow_angle)),
        int(gt_anchor_bottom[1] - arrow_length * math.sin(gt_angle + arrow_angle))
    )
    cv2.line(canvas_top_pair, gt_anchor_bottom, gt_arrow1, (60, 200, 100), 2, cv2.LINE_AA)
    cv2.line(canvas_top_pair, gt_anchor_bottom, gt_arrow2, (60, 200, 100), 2, cv2.LINE_AA)
    cv2.circle(canvas_top_pair, gt_anchor_bottom, 5, (60, 200, 100), 2)
    
    # Pred 连接线（蓝色虚线）
    cv2.line(canvas_top_pair, pred_anchor_top, pred_anchor_bottom, (70, 120, 220), 2, cv2.LINE_AA)
    # 添加箭头效果
    pred_dx = pred_anchor_bottom[0] - pred_anchor_top[0]
    pred_dy = pred_anchor_bottom[1] - pred_anchor_top[1]
    pred_angle = math.atan2(pred_dy, pred_dx)
    pred_arrow1 = (
        int(pred_anchor_bottom[0] - arrow_length * math.cos(pred_angle - arrow_angle)),
        int(pred_anchor_bottom[1] - arrow_length * math.sin(pred_angle - arrow_angle))
    )
    pred_arrow2 = (
        int(pred_anchor_bottom[0] - arrow_length * math.cos(pred_angle + arrow_angle)),
        int(pred_anchor_bottom[1] - arrow_length * math.sin(pred_angle + arrow_angle))
    )
    cv2.line(canvas_top_pair, pred_anchor_bottom, pred_arrow1, (70, 120, 220), 2, cv2.LINE_AA)
    cv2.line(canvas_top_pair, pred_anchor_bottom, pred_arrow2, (70, 120, 220), 2, cv2.LINE_AA)
    cv2.circle(canvas_top_pair, pred_anchor_bottom, 5, (70, 120, 220), 2)

    if best_pred is None:
        cv2.putText(canvas_top_pair, 'No prediction found', (30, bottom_y + half_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    def draw_simple_bev(items, title="BEV"):
        bev_w, bev_h = 1280, 920
        ppm = 12.0
        center_x = int(bev_w * 0.46)
        center_y = int(bev_h * 0.78)
        bev_img = np.full((bev_h, bev_w, 3), (245, 247, 250), dtype=np.uint8)

        cv2.rectangle(bev_img, (34, 34), (bev_w - 34, bev_h - 34), (220, 226, 233), 2, cv2.LINE_AA)
        cv2.putText(bev_img, title, (54, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (47, 58, 70), 2, cv2.LINE_AA)
        cv2.putText(bev_img, "Clean BEV for thesis: lane structure, ego vehicle and key predictions.", (54, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (108, 118, 130), 1, cv2.LINE_AA)

        def ego_to_pixel(x_forward, y_left):
            pixel_x = center_x + int(y_left * ppm)
            pixel_y = center_y - int(x_forward * ppm)
            return pixel_x, pixel_y

        lane_positions = [0.0, 3.6, -3.6]
        left_boundary_y = 7.2
        right_boundary_y = -7.2
        left_boundary_kind = 'boundary'
        right_boundary_kind = 'boundary'
        curvature = 0.0
        if lane_profile:
            lane_positions = [float(value) for value in lane_profile.get('lane_positions', lane_positions)]
            left_boundary_y = float(lane_profile.get('left_boundary_y', left_boundary_y))
            right_boundary_y = float(lane_profile.get('right_boundary_y', right_boundary_y))
            left_boundary_kind = lane_profile.get('left_boundary_kind', left_boundary_kind)
            right_boundary_kind = lane_profile.get('right_boundary_kind', right_boundary_kind)
            curvature = float(lane_profile.get('curvature', 0.0))

        def road_shift(x_forward):
            blend = min(1.0, max(0.0, (x_forward + 6.0) / 66.0))
            return curvature * (blend ** 2) * 3.8

        for meters in [10, 20, 30, 40, 50]:
            left = ego_to_pixel(meters, -11.5)
            right = ego_to_pixel(meters, 11.5)
            cv2.line(bev_img, left, right, (229, 233, 238), 1, cv2.LINE_AA)
            cv2.putText(bev_img, f"{meters} m", (right[0] + 10, right[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (124, 132, 143), 1, cv2.LINE_AA)

        road_left = []
        road_right = []
        shoulder_left = []
        shoulder_right = []
        for x_forward in np.linspace(-4.0, 54.0, 56):
            shift = road_shift(float(x_forward))
            road_left.append(ego_to_pixel(float(x_forward), left_boundary_y + shift))
            road_right.append(ego_to_pixel(float(x_forward), right_boundary_y + shift))
            shoulder_left.append(ego_to_pixel(float(x_forward), left_boundary_y + shift + 1.2))
            shoulder_right.append(ego_to_pixel(float(x_forward), right_boundary_y + shift - 1.2))

        shoulder_polygon = np.array(shoulder_left + shoulder_right[::-1], dtype=np.int32)
        road_polygon = np.array(road_left + road_right[::-1], dtype=np.int32)
        cv2.fillPoly(bev_img, [shoulder_polygon], (227, 230, 234))
        cv2.fillPoly(bev_img, [road_polygon], (84, 88, 94))

        left_pts = np.array(road_left, dtype=np.int32)
        right_pts = np.array(road_right, dtype=np.int32)
        boundary_color = (244, 246, 248)
        boundary_width_left = 4 if left_boundary_kind == 'curb' else 2
        boundary_width_right = 4 if right_boundary_kind == 'curb' else 2
        cv2.polylines(bev_img, [left_pts], False, boundary_color, boundary_width_left, cv2.LINE_AA)
        cv2.polylines(bev_img, [right_pts], False, boundary_color, boundary_width_right, cv2.LINE_AA)

        for offset in lane_positions:
            lane_points = []
            for x_forward in np.linspace(-2.0, 52.0, 34):
                lane_points.append(ego_to_pixel(float(x_forward), offset + road_shift(float(x_forward))))
            lane_points = np.array(lane_points, dtype=np.int32)
            for index in range(0, len(lane_points) - 1, 2):
                cv2.line(
                    bev_img,
                    tuple(lane_points[index]),
                    tuple(lane_points[min(index + 1, len(lane_points) - 1)]),
                    (225, 228, 232),
                    2,
                    cv2.LINE_AA,
                )

        def corners_xy(box_ego):
            x, y, _ = box_ego.center
            w, l, _ = box_ego.wlh
            yaw = box_ego.orientation.yaw_pitch_roll[0]
            local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]], dtype=np.float32)
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=np.float32)
            corners = local @ rot.T
            corners[:, 0] += x
            corners[:, 1] += y
            return corners

        def draw_box_bev(box_ego, fill_color, edge_color, heading_color):
            corners = corners_xy(box_ego)
            pts = np.array([ego_to_pixel(x, y) for x, y in corners], dtype=np.int32)
            center = ego_to_pixel(float(box_ego.center[0]), float(box_ego.center[1]))
            head = ego_to_pixel(
                float(box_ego.center[0] + math.cos(box_ego.orientation.yaw_pitch_roll[0]) * box_ego.wlh[1] * 0.35),
                float(box_ego.center[1] + math.sin(box_ego.orientation.yaw_pitch_roll[0]) * box_ego.wlh[1] * 0.35),
            )
            cv2.fillPoly(bev_img, [pts], fill_color)
            cv2.polylines(bev_img, [pts], True, edge_color, 2, cv2.LINE_AA)
            cv2.line(bev_img, center, head, heading_color, 2, cv2.LINE_AA)
            return center

        ego_corners = np.array([
            [2.25, 0.9], [2.25, -0.9], [-2.25, -0.9], [-2.25, 0.9]
        ], dtype=np.float32)
        ego_pts = np.array([ego_to_pixel(x, y) for x, y in ego_corners], dtype=np.int32)
        cv2.fillPoly(bev_img, [ego_pts], (211, 221, 233))
        cv2.polylines(bev_img, [ego_pts], True, (250, 252, 255), 2, cv2.LINE_AA)
        cv2.putText(bev_img, "Ego", (ego_pts[3][0] - 12, ego_pts[3][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 67, 80), 2, cv2.LINE_AA)

        def item_priority(pred):
            class_weight = focus_class_priority(pred['class_name'])
            distance = float(np.linalg.norm(np.asarray(pred['box'].center[:2], dtype=np.float32)))
            score = float(pred.get('score', 0.0))
            return class_weight * 100.0 + max(0.0, 40.0 - distance) * 4.0 + score * 40.0

        ranked_items = sorted(
            [pred for pred in items if float(np.linalg.norm(np.asarray(pred['box'].center[:2], dtype=np.float32))) <= 38.0],
            key=item_priority,
            reverse=True,
        )[:8]

        label_items = ranked_items[:4]
        for pred in ranked_items:
            x, y, _ = pred['box'].center
            distance = math.sqrt(x * x + y * y)
            threat_color = (92, 187, 255)
            edge_color = (255, 255, 255)
            if distance <= 14.0:
                threat_color = (98, 110, 236)
            elif distance <= 24.0:
                threat_color = (109, 163, 255)
            center = draw_box_bev(pred['box'], threat_color, edge_color, (243, 247, 252))
            if pred in label_items:
                label = f"{short_pred_name(pred['class_name'])} {pred['score']:.2f}"
                text_origin = (center[0] + 10, center[1] - 10)
                cv2.rectangle(
                    bev_img,
                    (text_origin[0] - 4, text_origin[1] - 18),
                    (text_origin[0] + 94, text_origin[1] + 5),
                    (251, 252, 254),
                    -1,
                )
                cv2.rectangle(
                    bev_img,
                    (text_origin[0] - 4, text_origin[1] - 18),
                    (text_origin[0] + 94, text_origin[1] + 5),
                    (217, 223, 230),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(bev_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.46, (62, 72, 84), 1, cv2.LINE_AA)

        summary_x1, summary_y1 = 880, 156
        summary_x2, summary_y2 = bev_w - 58, 420
        cv2.rectangle(bev_img, (summary_x1, summary_y1), (summary_x2, summary_y2), (229, 233, 239), -1)
        cv2.rectangle(bev_img, (summary_x1, summary_y1), (summary_x2, summary_y2), (214, 221, 228), 1, cv2.LINE_AA)
        cv2.putText(bev_img, "Figure Notes", (summary_x1 + 20, summary_y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (54, 63, 76), 2, cv2.LINE_AA)
        cv2.putText(bev_img, f"Key predictions: {len(ranked_items)}", (summary_x1 + 20, summary_y1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (87, 98, 110), 1, cv2.LINE_AA)
        cv2.putText(bev_img, "Only near-range targets are retained.", (summary_x1 + 20, summary_y1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (109, 118, 129), 1, cv2.LINE_AA)
        cv2.putText(bev_img, "Road geometry is drawn from the inferred lane profile.", (summary_x1 + 20, summary_y1 + 156), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (109, 118, 129), 1, cv2.LINE_AA)
        cv2.putText(bev_img, "Suitable for thesis comparison with SR detail figures.", (summary_x1 + 20, summary_y1 + 192), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (109, 118, 129), 1, cv2.LINE_AA)

        cv2.putText(bev_img, "Legend", (summary_x1 + 20, summary_y1 + 236), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (63, 73, 85), 2, cv2.LINE_AA)
        cv2.rectangle(bev_img, (summary_x1 + 22, summary_y1 + 256), (summary_x1 + 44, summary_y1 + 278), (211, 221, 233), -1)
        cv2.putText(bev_img, "ego vehicle", (summary_x1 + 56, summary_y1 + 273), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (97, 107, 118), 1, cv2.LINE_AA)
        cv2.rectangle(bev_img, (summary_x1 + 22, summary_y1 + 292), (summary_x1 + 44, summary_y1 + 314), (109, 163, 255), -1)
        cv2.putText(bev_img, "predicted object", (summary_x1 + 56, summary_y1 + 309), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (97, 107, 118), 1, cv2.LINE_AA)
        cv2.line(bev_img, (summary_x1 + 22, summary_y1 + 328), (summary_x1 + 44, summary_y1 + 328), (225, 228, 232), 2, cv2.LINE_AA)
        cv2.putText(bev_img, "lane marking", (summary_x1 + 56, summary_y1 + 334), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (97, 107, 118), 1, cv2.LINE_AA)

        return bev_img

    bev_img = draw_simple_bev(pred_items, "BEV")

    # 恢复 SR 参考图的实现（3D matplotlib 绘图）
    default_lane_positions = lane_profile.get('lane_positions', [0.0, 3.6, -3.6]) if lane_profile else [0.0, 3.6, -3.6]
    default_left_boundary = float(lane_profile.get('left_boundary_y', 7.2)) if lane_profile else 7.2
    default_right_boundary = float(lane_profile.get('right_boundary_y', -7.2)) if lane_profile else -7.2

    gt_scene_items = []
    for gt in gt_items:
        cls_name = CLASS_NAMES.get(gt['class_id'], 'car') if gt['class_id'] is not None else gt['cat_short']
        gt_scene_items.append({'box': gt['box_ego'], 'class_name': cls_name, 'label': short_pred_name(cls_name), 'score': None})

    pred_scene_items = []
    for pred in pred_items:
        pred_scene_items.append({'box': pred['box'], 'class_name': pred['class_name'], 'label': short_pred_name(pred['class_name']), 'score': pred['score']})

    def compute_scene_extents(items, reference_items=None):
        combined_items = list(items or [])
        if reference_items:
            combined_items.extend(reference_items)

        x_values = [0.0, 8.0, 18.0]
        y_values = [0.0]
        z_max = 2.0

        for obj in combined_items:
            box = obj['box']
            x_val = clamp(float(box.center[0]), -8.0, 60.0)
            y_val = clamp(float(box.center[1]), -18.0, 18.0)
            x_values.append(x_val)
            y_values.append(y_val)
            z_max = max(z_max, float(box.center[2] + box.wlh[2] * 0.9))

        x_array = np.array(x_values, dtype=np.float32)
        y_array = np.array(y_values, dtype=np.float32)
        y_center = float(np.median(y_array))
        y_radius = float(np.percentile(np.abs(y_array - y_center), 80) + 3.6)

        x_min = clamp(float(np.min(x_array)) - 6.0, -6.0, 10.0)
        x_max = clamp(float(np.percentile(x_array, 90)) + 12.0, 18.0, 44.0)
        if x_max - x_min < 18.0:
            x_max = min(48.0, x_min + 18.0)

        y_radius = clamp(y_radius, 5.5, 11.0)
        y_min = clamp(y_center - y_radius, -16.0, 8.0)
        y_max = clamp(y_center + y_radius, -8.0, 16.0)
        if y_max - y_min < 9.0:
            mid = (y_min + y_max) * 0.5
            y_min = max(-16.0, mid - 4.5)
            y_max = min(16.0, mid + 4.5)

        return (x_min, x_max), (y_min, y_max), (0.0, clamp(z_max + 1.1, 4.6, 7.0))

    def in_range(box_ego, ranges):
        (x_min, x_max), (y_min, y_max), _ = ranges
        x_f, y_l, _ = box_ego.center
        return (x_min - 2.0) <= x_f <= (x_max + 2.0) and (y_min - 2.0) <= y_l <= (y_max + 2.0)

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

    def setup_scene(ax, ranges):
        x_range, y_range, z_range = ranges
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

        lane_x = np.linspace(max(-8.0, x_range[0]), min(68.0, x_range[1]), 180)
        road_shift = np.array([
            float(lane_profile.get('center_offset', 0.0) if lane_profile else 0.0) + float(lane_profile.get('curvature', 0.0) if lane_profile else 0.0) * (min(1.0, max(0.0, (x_val + 8.0) / 72.0)) ** 2) * 4.0
            for x_val in lane_x
        ], dtype=np.float32)
        left_boundary_line = np.full_like(lane_x, default_left_boundary) + road_shift
        right_boundary_line = np.full_like(lane_x, default_right_boundary) + road_shift

        road_poly = np.column_stack([lane_x, left_boundary_line, np.full_like(lane_x, 0.015)])
        road_poly = np.vstack([
            road_poly,
            np.column_stack([lane_x[::-1], right_boundary_line[::-1], np.full_like(lane_x, 0.015)]),
        ])
        ax.add_collection3d(
            Poly3DCollection([road_poly], facecolors=(0.18, 0.18, 0.19, 0.92), edgecolors='none')
        )

        if lane_profile and lane_profile.get('left_boundary_kind') == 'curb':
            ax.plot(lane_x, left_boundary_line, np.full_like(lane_x, 0.06), color=(0.97, 0.97, 0.97, 0.96), linewidth=1.8)
        if lane_profile and lane_profile.get('right_boundary_kind') == 'curb':
            ax.plot(lane_x, right_boundary_line, np.full_like(lane_x, 0.06), color=(0.97, 0.97, 0.97, 0.96), linewidth=1.8)

        for offset in default_lane_positions:
            lane_y = np.full_like(lane_x, offset) + road_shift
            lane_z = np.full_like(lane_x, 0.03)
            ax.plot(lane_x, lane_y, lane_z, color=(1, 1, 1, 0.92), linewidth=1.0, linestyle=(0, (6, 6)))

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_range[0], z_range[1])
        ax.view_init(elev=28, azim=-126)
        ax.set_proj_type('persp')
        ax.set_axis_off()
        ax.set_box_aspect([1.22, 1.0, 0.34])

    def render_semantic_scene(items, scene_title, is_pred, reference_items=None):
        scene_ranges = compute_scene_extents(items, reference_items=reference_items)
        fig = plt.figure(figsize=(16, 9), dpi=100)
        fig.patch.set_facecolor('#F0F5FA')
        ax = fig.add_subplot(111, projection='3d')
        setup_scene(ax, scene_ranges)

        ego_corners = calc_cuboid_corners(center=(0.0, 0.0, 0.75), size_lwh=(4.5, 1.8, 1.5), yaw=0.0)
        add_ground_shadow(ax, ego_corners, scale=1.12, alpha=0.20)
        draw_3d_cuboid(ax, ego_corners, facecolor='#2F343E', edgecolor='#1F242C', alpha=0.95)

        label_color = (0.22, 0.27, 0.33, 0.95)
        sorted_items = sorted(items, key=lambda obj: float(np.linalg.norm(np.asarray(obj['box'].center[:2], dtype=np.float32))))

        label_budget = 10
        rendered_count = 0
        for obj in sorted_items:
            box = obj['box']
            if not in_range(box, scene_ranges):
                continue
            rendered_count += 1

            x, y, z = [float(v) for v in box.center]
            w, l, h = [float(v) for v in box.wlh]
            yaw = float(box.orientation.yaw_pitch_roll[0])
            cls_name = obj['class_name']

            if cls_name in ('car', 'truck', 'bus', 'trailer', 'construction_vehicle'):
                facecolor = '#D3D9DF'
                edgecolor = '#808A9F'
                alpha = 0.80
                
                # 小米 SU7 流线型设计参数
                if cls_name == 'car':
                    # SU7 特征：更长、更宽、更低，流线型车身
                    l = float(np.clip(l, 4.8, 5.2))  # 车长约 5.0 米
                    w = float(np.clip(w, 1.9, 2.1))  # 车宽约 2.0 米
                    h = float(np.clip(h, 1.4, 1.5))  # 车高约 1.45 米
                    facecolor = '#E8E8E8'  # 更亮的颜色
                    edgecolor = '#607080'
                else:
                    l = float(np.clip(l, 1.8, 13.0))
                    w = float(np.clip(w, 0.8, 3.4))
                    h = float(np.clip(h, 1.0, 4.2))
                
                # 绘制主车身
                corners = calc_cuboid_corners(center=(x, y, z), size_lwh=(l, w, h), yaw=yaw)
                add_ground_shadow(ax, corners, scale=1.08, alpha=0.17)
                draw_3d_cuboid(ax, corners, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
                
                if cls_name == 'car':
                    # 小米 SU7 流线型车顶设计
                    # 车头较长，车尾溜背
                    hood_l = l * 0.35  # 引擎盖长度
                    hood_h = h * 0.15
                    hood_z = z + h * 0.5
                    hood_offset = l * 0.15  # 向前偏移
                    
                    # 引擎盖
                    hood_center_x = x + hood_offset * math.cos(yaw)
                    hood_center_y = y + hood_offset * math.sin(yaw)
                    hood_corners = calc_cuboid_corners(center=(hood_center_x, hood_center_y, hood_z), size_lwh=(hood_l, w * 0.95, hood_h), yaw=yaw)
                    draw_3d_cuboid(ax, hood_corners, facecolor='#D8D8D8', edgecolor='#506070', alpha=0.75)
                    
                    # 流线型车顶（用多个小长方体模拟）
                    roof_l = l * 0.45
                    roof_w = w * 0.9
                    roof_h = h * 0.25
                    roof_z = z + h * 0.65
                    roof_offset = -l * 0.05  # 稍微向后偏移
                    
                    roof_center_x = x + roof_offset * math.cos(yaw)
                    roof_center_y = y + roof_offset * math.sin(yaw)
                    roof_corners = calc_cuboid_corners(center=(roof_center_x, roof_center_y, roof_z), size_lwh=(roof_l, roof_w, roof_h), yaw=yaw)
                    draw_3d_cuboid(ax, roof_corners, facecolor='#C8D0D8', edgecolor='#708090', alpha=0.70)
                    
                    # 溜背车尾
                    trunk_l = l * 0.25
                    trunk_h = h * 0.2
                    trunk_z = z + h * 0.55
                    trunk_offset = -l * 0.35  # 向后偏移
                    
                    trunk_center_x = x + trunk_offset * math.cos(yaw)
                    trunk_center_y = y + trunk_offset * math.sin(yaw)
                    trunk_corners = calc_cuboid_corners(center=(trunk_center_x, trunk_center_y, trunk_z), size_lwh=(trunk_l, w * 0.85, trunk_h), yaw=yaw)
                    draw_3d_cuboid(ax, trunk_corners, facecolor='#D0D8E0', edgecolor='#607080', alpha=0.68)
                else:
                    # 其他车辆类型的车顶细节
                    roof_h = h * 0.3
                    roof_l = l * 0.6
                    roof_w = w * 0.9
                    roof_z = z + h * 0.6
                    roof_corners = calc_cuboid_corners(center=(x, y, roof_z), size_lwh=(roof_l, roof_w, roof_h), yaw=yaw)
                    draw_3d_cuboid(ax, roof_corners, facecolor='#C8D0D8', edgecolor='#708090', alpha=0.75)
                
            elif cls_name == 'pedestrian':
                facecolor = '#9EDCEB'
                edgecolor = '#4E9CB5'
                alpha = 0.82
                l = 0.45
                w = 0.45
                h = 1.72
                z = max(z, h * 0.5 + 0.02)
                
                # 为行人添加更详细的形状（头部和身体）
                corners = calc_cuboid_corners(center=(x, y, z), size_lwh=(l, w, h * 0.7), yaw=yaw)
                add_ground_shadow(ax, corners, scale=1.08, alpha=0.17)
                draw_3d_cuboid(ax, corners, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
                
                # 添加头部（球体）
                head_radius = 0.12
                head_z = z + h * 0.75
                u = np.linspace(0, 2 * np.pi, 16)
                v = np.linspace(0, np.pi, 8)
                head_x = head_radius * np.outer(np.cos(u), np.sin(v)) + x
                head_y = head_radius * np.outer(np.sin(u), np.sin(v)) + y
                head_z = head_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + head_z
                # 应用旋转
                rot_yaw = yaw
                head_x_rot = (head_x - x) * np.cos(rot_yaw) - (head_y - y) * np.sin(rot_yaw) + x
                head_y_rot = (head_x - x) * np.sin(rot_yaw) + (head_y - y) * np.cos(rot_yaw) + y
                ax.plot_surface(head_x_rot, head_y_rot, head_z, color='#7CB9E8', alpha=0.85, shade=True)
                
            elif cls_name in ('traffic_cone', 'barrier'):
                facecolor = '#F29A55'
                edgecolor = '#A7642D'
                alpha = 0.86
                if cls_name == 'traffic_cone':
                    l = 0.55
                    w = 0.55
                    h = 0.92
                    z = max(z, h * 0.5 + 0.02)
                    
                    # 为交通锥添加锥形形状
                    corners = calc_cuboid_corners(center=(x, y, z), size_lwh=(l, w, h), yaw=yaw)
                    add_ground_shadow(ax, corners, scale=1.08, alpha=0.17)
                    draw_3d_cuboid(ax, corners, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
                else:
                    l = 1.6
                    w = 0.55
                    h = 0.8
                    z = max(z, h * 0.5 + 0.02)
                    
                    # 为路障添加更详细的形状
                    corners = calc_cuboid_corners(center=(x, y, z), size_lwh=(l, w, h), yaw=yaw)
                    add_ground_shadow(ax, corners, scale=1.08, alpha=0.17)
                    draw_3d_cuboid(ax, corners, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
                    # 添加条纹细节
                    stripe_h = h * 0.2
                    for i in range(3):
                        stripe_z = z + i * stripe_h * 1.5
                        stripe_corners = calc_cuboid_corners(center=(x, y, stripe_z), size_lwh=(l, w * 1.05, stripe_h), yaw=yaw)
                        if i % 2 == 0:
                            draw_3d_cuboid(ax, stripe_corners, facecolor='#FFFFFF', edgecolor='none', alpha=0.9)
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
        ax.text2D(
            0.03,
            0.90,
            f'Objects in focus: {rendered_count}',
            transform=ax.transAxes,
            fontsize=9,
            fontname='Arial',
            color=(0.35, 0.41, 0.48, 1.0),
        )

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return auto_crop_scene_image(img)

    scene_gt_img = build_sr_detail_board(
        "Surrounding Reality | Ground Truth",
        "Real camera board with high-detail crops and class-focused review.",
        gt_items,
        is_pred=False,
        camera_images=camera_images,
        cam_intrinsics=cam_intrinsics,
        nusc=nusc,
        sample_token=sample_token,
    )
    scene_pred_img = build_sr_detail_board(
        "Surrounding Reality | Prediction",
        "Real camera board with high-detail crops and confidence-aware review.",
        pred_items,
        is_pred=True,
        camera_images=camera_images,
        cam_intrinsics=cam_intrinsics,
        nusc=nusc,
        sample_token=sample_token,
    )
    scene_stream = build_scene_stream(
        sample_token=sample_token,
        sample_timestamp=sample['timestamp'],
        nusc=nusc,
        gt_items=gt_items,
        pred_items=pred_items,
        lane_profile=lane_profile,
    )

    stats = {'pred_total': len(pred_items), 'pred_details': pred_class_stats, 'gt_total': len(gt_items), 'gt_details': gt_class_stats}
    
    # 添加平面图原图（前视相机）
    front_img = camera_images['CAM_FRONT'] if images is not None else None
    canvas_combined = merge_6_cams(drawn_combined, bottom_panel=bottom_img)
    
    return canvas_combined, canvas_top_pair, bev_img, scene_gt_img, scene_pred_img, front_img, stats, scene_stream

def main():
    parser = argparse.ArgumentParser(description='TDR-QAF inference and visualization')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to best_model.pth')
    parser.add_argument('--sample_indices', type=str, default='', help='Optional path to sample_indices.json')
    parser.add_argument('--data_root', type=str, default='./dataset', help='NuScenes dataset root')
    parser.add_argument('--nuscenes_version', type=str, default='v1.0-mini', help='NuScenes version')
    parser.add_argument('--max_samples', type=int, default=None, help='Optional dataset cap for quick checks')
    parser.add_argument('--confidence', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--topk', type=int, default=50, help='Maximum predictions kept after decode')
    parser.add_argument('--num_samples', type=int, default=2, help='Number of samples to visualize')
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint or None)
    sample_indices_path = resolve_sample_indices_path(checkpoint_path, args.sample_indices or None)
    confidence = args.confidence
    topk = args.topk
    num_samples = args.num_samples

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Confidence threshold: {confidence}')
    print(f'TopK: {topk}')
    print(f'Checkpoint path: {checkpoint_path}')
    print(f'Sample indices path: {sample_indices_path or "None"}')
    print(f'Number of samples to inference: {num_samples}')

    output_dir = create_output_dir()

    try:
        model = load_model(device, is_overfit=False, checkpoint_path=checkpoint_path)
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

        if sample_indices_path:
            indices_data = NuScenesDataset.load_sample_indices(sample_indices_path)
            dataset.set_sample_indices(indices_data['indices'])
            print(f'Loaded train indices. Inference sample pool: {len(indices_data["indices"])}')
        else:
            print('No sample indices provided. Using the current dataset sample pool.')
    except Exception as e:
        print(f'Failed to load dataset: {e}')
        return

    random.seed(time.time())
    sample_pool = list(range(len(dataset)))
    random.shuffle(sample_pool)

    # 收集所有样本的结果
    all_results = []

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
            canvas_combined, canvas_top_pair, bev_img, scene_gt_img, scene_pred_img, front_img, stats, scene_stream = visualize(
                images,
                cam_intrinsics,
                gt_bboxes,
                gt_labels,
                pred_scores,
                pred_labels,
                pred_bboxes,
                sample_token,
                None,
                dataset.nusc,
            )
            
            # 收集结果
            all_results.append({
                'sample_token': sample_token,
                'canvas_combined': canvas_combined,
                'canvas_top_pair': canvas_top_pair,
                'bev_img': bev_img,
                'scene_gt_img': scene_gt_img,
                'scene_pred_img': scene_pred_img,
                'front_img': front_img,
                'scene_stream': scene_stream,
                'stats': stats
            })
            
            print(f'Sample {test_idx + 1} done.')
        except Exception as e:
            print(f'Inference failed: {e}')
            traceback.print_exc()

    # 统一保存所有图片到时间命名的文件夹
    print(f"\nSaving {len(all_results)} samples to {output_dir}")
    for idx, result in enumerate(all_results):
        sample_token = result['sample_token']
        # 保存 BEV 图、SR 图和平面图原图
        cv2.imwrite(os.path.join(output_dir, f'{idx:02d}_bev_{sample_token}.jpg'), result['bev_img'])
        cv2.imwrite(os.path.join(output_dir, f'{idx:02d}_sr_gt_{sample_token}.jpg'), result['scene_gt_img'])
        cv2.imwrite(os.path.join(output_dir, f'{idx:02d}_sr_pred_{sample_token}.jpg'), result['scene_pred_img'])
        if result['front_img'] is not None:
            cv2.imwrite(os.path.join(output_dir, f'{idx:02d}_front_{sample_token}.jpg'), result['front_img'])
        
        json_path = os.path.join(output_dir, f'{idx:02d}_scene_stream_{sample_token}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result['scene_stream'], f, indent=2, ensure_ascii=False)
        print(f"Saved sample {idx + 1}: {sample_token}")
    
    print(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()

