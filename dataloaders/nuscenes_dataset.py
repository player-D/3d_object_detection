import os
import cv2
import numpy as np
import torch
import copy
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image


class NuScenesDataset(Dataset):
    def __init__(self, root="./dataset", debug_mode=False):
        self.root = root
        self.debug_mode = debug_mode

        if not debug_mode:
            self.nusc = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)
            self.samples = self.nusc.sample

            # 官方合并字典逻辑
            self.official_mapping = {
                'vehicle.car': 0,
                'vehicle.truck': 1,
                'vehicle.bus.bendy': 2,
                'vehicle.bus.rigid': 2,
                'vehicle.trailer': 3,
                'vehicle.construction': 4,
                'human.pedestrian.adult': 5,
                'human.pedestrian.child': 5,
                'human.pedestrian.construction_worker': 5,
                'human.pedestrian.police_officer': 5,
                'vehicle.motorcycle': 6,
                'vehicle.bicycle': 7,
                'movable_object.trafficcone': 8,
                'movable_object.barrier': 9
            }

            self.camera_names = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT'
            ]
        else:
            self._generate_fake_data()

    def _generate_fake_data(self):
        self.samples = [{} for _ in range(100)]

    def __len__(self):
        return min(50, len(self.samples))

    def __getitem__(self, index):
        if self.debug_mode:
            return self._get_fake_item(index)
        else:
            return self._get_real_item(index)

    def _get_fake_item(self, index):
        images = torch.randn(6, 3, 256, 704)
        boxes_2d = torch.zeros(6, 10, 4)
        cam_intrinsics = torch.eye(4).repeat(6, 1, 1)
        cam_extrinsics = torch.eye(4).repeat(6, 1, 1)
        gt_bboxes = torch.empty((0, 10), dtype=torch.float32)
        gt_labels = torch.empty((0,), dtype=torch.long)
        return images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels

    def _get_real_item(self, index):
        sample = self.samples[index]

        # 获取当前 sample 的 ego_pose
        cam_data = sample['data']['CAM_FRONT']
        cam_sample_data = self.nusc.get('sample_data', cam_data)
        ego_pose = self.nusc.get('ego_pose', cam_sample_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        ego_rotation_inv = ego_rotation.inverse

        # 准备数据容器
        images = []
        boxes_2d = []
        cam_intrinsics = []
        cam_extrinsics = []
        all_gt_bboxes = []
        all_gt_labels = []

        valid_sample = True

        # 遍历所有相机
        for cam_name in self.camera_names:
            try:
                cam_data = sample['data'][cam_name]
                cam_sample_data = self.nusc.get('sample_data', cam_data)

                # 读取图像 --- 增加容错
                img_path = os.path.join(self.root, cam_sample_data['filename'])
                if not os.path.exists(img_path):
                    print(f"⚠️  警告: 图片不存在，跳过样本 {index} | {cam_name}: {img_path}")
                    valid_sample = False
                    break

                img = Image.open(img_path).convert('RGB')
                # ================== 关键修改点 ==================
                # 原图 resize 提升分辨率（32的整数倍）
                img = img.resize((800, 448))          # ← 修改这里
                img = np.array(img, dtype=np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)
                images.append(img)

                # 读取相机参数（内参、外参）
                calibrated_sensor = self.nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])
                camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

                orig_size = (1600, 900)
                new_size = (800, 448)                  # ← 必须同步修改！
                scale_w = new_size[0] / orig_size[0]
                scale_h = new_size[1] / orig_size[1]

                camera_intrinsic[0, 0] *= scale_w
                camera_intrinsic[0, 2] *= scale_w
                camera_intrinsic[1, 1] *= scale_h
                camera_intrinsic[1, 2] *= scale_h

                cam_intrinsic_4x4 = np.eye(4)
                cam_intrinsic_4x4[:3, :3] = camera_intrinsic
                cam_intrinsics.append(torch.from_numpy(cam_intrinsic_4x4).float())

                # 外参
                rotation = Quaternion(calibrated_sensor['rotation'])
                translation = np.array(calibrated_sensor['translation'])
                cam_extrinsic = np.eye(4)
                cam_extrinsic[:3, :3] = rotation.rotation_matrix
                cam_extrinsic[:3, 3] = translation
                cam_extrinsics.append(torch.from_numpy(cam_extrinsic).float())

                # 计算当前相机的 2D proposals
                cam_boxes_2d = []
                for ann_token in sample['anns']:
                    annotation = self.nusc.get('sample_annotation', ann_token)
                    cat_name = annotation['category_name']
                    if cat_name not in self.official_mapping:
                        continue

                    box = self.nusc.get_box(ann_token)
                    box_ego = copy.deepcopy(box)
                    box_ego.translate([-x for x in ego_pose['translation']])
                    box_ego.rotate(ego_rotation_inv)

                    corners_3d = self._get_box_corners(
                        box_ego.center.tolist(),
                        annotation['size'],
                        box_ego.orientation
                    )
                    corners_2d = self._project_3d_to_2d(corners_3d, cam_intrinsic_4x4, cam_extrinsic)
                    # 新增：如果是幽灵框，直接跳过当前框
                    if corners_2d is None:
                        continue
                    box_2d = self._get_2d_bbox(corners_2d)

                    if self._is_box_in_image(box_2d, (800, 448)):   # ← 修改这里
                        cam_boxes_2d.append(box_2d)

                padded_boxes_2d = self._pad_boxes_2d(cam_boxes_2d, 30)
                boxes_2d.append(padded_boxes_2d)

            except Exception as e:
                print(f"⚠️  样本 {index} 处理相机 {cam_name} 时出错: {e}")
                valid_sample = False
                break

        # 【定位 A】彻底杜绝脏数据
        if (not valid_sample) or (len(images) < 6):
            print(f"⏭️  跳过不完整或损坏的样本: {index}")
            return None  # 拒绝返回假数据

        # 读取真实的标注信息 (GT)
        for ann_token in sample['anns']:
            annotation = self.nusc.get('sample_annotation', ann_token)
            cat_name = annotation['category_name']
            if cat_name not in self.official_mapping:
                continue

            box = self.nusc.get_box(ann_token)
            box_ego = copy.deepcopy(box)
            box_ego.translate([-x for x in ego_pose['translation']])
            box_ego.rotate(ego_rotation_inv)

            yaw = box_ego.orientation.yaw_pitch_roll[0]
            gt_box = np.array([
                box_ego.center[0], box_ego.center[1], box_ego.center[2],
                box_ego.wlh[0], box_ego.wlh[1], box_ego.wlh[2],
                np.sin(yaw),
                np.cos(yaw),
                0.0,
                0.0
            ], dtype=np.float32)

            all_gt_bboxes.append(gt_box)
            all_gt_labels.append(self.official_mapping[cat_name])

        # 最后返回
        images = torch.stack(images)
        boxes_2d = torch.stack(boxes_2d)
        cam_intrinsics = torch.stack(cam_intrinsics).float()
        cam_extrinsics = torch.stack(cam_extrinsics).float()

        if all_gt_bboxes:
            gt_bboxes = torch.tensor(all_gt_bboxes, dtype=torch.float32)
            gt_labels = torch.tensor(all_gt_labels, dtype=torch.long)
        else:
            gt_bboxes = torch.empty((0, 10), dtype=torch.float32)
            gt_labels = torch.empty((0,), dtype=torch.long)

        return images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels

    def _get_box_corners(self, translation, size, rotation):
        """
        计算3D边界框的8个角点坐标
        """
        w, l, h = size[0], size[1], size[2]

        # 生成标准的 8 个角点坐标 (3 x 8)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # 旋转
        rotation_matrix = rotation.rotation_matrix
        corners = np.dot(rotation_matrix, corners)

        # 平移
        translation = np.array(translation).reshape(3, 1)
        corners = corners + translation

        return corners.T

    def _project_3d_to_2d(self, corners_3d, cam_intrinsics, cam_extrinsics):
        """投影 3D 角点到 2D，并严格过滤相机背后的点"""
        corners_3d_hom = np.hstack([corners_3d, np.ones((8, 1))])
        corners_cam = np.dot(np.linalg.inv(cam_extrinsics), corners_3d_hom.T).T
        
        # 【关键防御】：如果角点在相机背后或极近处（深度 <= 0.1米），直接判定为无效投影
        depths = corners_cam[:, 2]
        if np.any(depths <= 0.1):
            return None  
            
        corners_img_hom = np.dot(cam_intrinsics, corners_cam.T).T
        # 防止除以极小值产生 Inf
        z = corners_img_hom[:, 2:3].copy()
        z[z < 1e-6] = 1e-6
        corners_2d = corners_img_hom[:, :2] / z
        return corners_2d

    def _get_2d_bbox(self, corners_2d):
        min_x = np.min(corners_2d[:, 0])
        min_y = np.min(corners_2d[:, 1])
        max_x = np.max(corners_2d[:, 0])
        max_y = np.max(corners_2d[:, 1])
        return [min_x, min_y, max_x, max_y]

    def _is_box_in_image(self, box_2d, img_size):
        img_w, img_h = img_size
        min_x, min_y, max_x, max_y = box_2d
        area = (max_x - min_x) * (max_y - min_y)
        # 新增：太小（小于4）或太大（大于画面的80%）的异常框直接抛弃
        if area < 4 or area > img_w * img_h * 0.8:   
            return False
        return not (max_x < 0 or min_x > img_w or max_y < 0 or min_y > img_h)

    def _pad_boxes_2d(self, boxes_2d, max_boxes):
        padded = []
        for box in boxes_2d[:max_boxes]:
            padded.append(box)
        while len(padded) < max_boxes:
            padded.append([0.0, 0.0, 0.0, 0.0])
        return torch.tensor(padded, dtype=torch.float32)


def collate_fn(batch):
    # 过滤掉所有返回 None 的损坏样本
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        # 如果整个 batch 都损坏，返回空张量防崩
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), [], []
    
    images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels = zip(*batch)
    
    images = torch.stack(images)
    boxes_2d = torch.stack(boxes_2d)
    cam_intrinsics = torch.stack(cam_intrinsics)
    cam_extrinsics = torch.stack(cam_extrinsics)

    return images, boxes_2d, cam_intrinsics, cam_extrinsics, list(gt_bboxes), list(gt_labels)