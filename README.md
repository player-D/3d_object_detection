# TDR-QAF (3d_object_detection)

本目录是当前可维护版本（只改这里，不改 `winsurf`）。  
目标是稳定完成从训练到推理、后端 API、前端可视化的完整链路。

## 这次重点修复

1. 推理样本抽取逻辑修复
- 不加 `--load_indices` 时：在当前数据集池随机抽取（默认完整数据集）。
- 加 `--load_indices` 时：只在训练保存的索引集合中抽取。
- `tools/inference.py` 默认 `--max_samples=None`，不会再默认卡死 50 条。

2. 可视化稳定性修复
- 六视角拼接前统一尺寸/通道/类型，避免 `cv2.hconcat` 报错。
- 推理输出统一为三张图（每个样本）：
  - `pred_<token>.jpg`：6 相机 GT + Pred 全量叠加
  - `top_pair_<token>.jpg`：最高分 Pred 与匹配 GT 的左右分栏放大图
  - `bev_<token>.jpg`：伪 3D 风格 BEV（带障碍物形体效果）

3. 全量 NuScenes 训练/推理支持
- 数据集版本不再写死 `v1.0-mini`，支持参数化：
  - `--nuscenes_version v1.0-mini`
  - `--nuscenes_version v1.0-trainval`
- 后端也支持环境变量切换路径和版本。

## 环境准备

```bash
pip install -r requirements_gpu.txt
# 或
pip install -r requirements_cpu.txt
```

## 数据目录

默认数据根目录是 `./dataset`，示例结构：

```text
dataset/
  samples/
  sweeps/
  maps/
  v1.0-mini/
  v1.0-trainval/
```

## 训练

### 基础训练

```bash
python tools/train.py --data_root ./dataset --nuscenes_version v1.0-mini
```

### 全量 NuScenes（服务器）

```bash
python tools/train.py \
  --data_root /path/to/nuscenes \
  --nuscenes_version v1.0-trainval \
  --batch_size 4 \
  --epochs 50
```

### 复用训练样本索引

```bash
python tools/train.py --load_indices saved_models/<run_id>/sample_indices.json
```

> 每次训练会在对应 `saved_models/<run_id>/` 下保存 `sample_indices.json`。

## 推理（CLI）

### 默认随机抽样（完整数据池）

```bash
python tools/inference.py --data_root ./dataset --nuscenes_version v1.0-mini
```

### 仅在训练样本上推理

```bash
python tools/inference.py \
  --load_indices saved_models/<run_id>/sample_indices.json
```

### 关键参数

- `--confidence`：置信度阈值（默认 `0.05`）
- `--topk`：最多保留预测框数（默认 `50`）
- `--max_samples`：限制可抽样数据池（默认 `None`，即不限制）
- `--num_samples`：本次命令推理多少个样本（默认 `1`）
- `--data_root`：数据集根路径
- `--nuscenes_version`：NuScenes 版本（`v1.0-mini`/`v1.0-trainval`）

## 后端与前端

### 启动后端

```bash
cd backend
python main.py
```

后端可通过环境变量切换数据集：

- `NUSCENES_ROOT`（默认 `./dataset`）
- `NUSCENES_VERSION`（默认 `v1.0-mini`）
- `NUSCENES_MAX_SAMPLES`（默认空，不限）

示例：

```bash
set NUSCENES_ROOT=D:\datasets\nuscenes
set NUSCENES_VERSION=v1.0-trainval
python backend/main.py
```

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

## 输出说明

单次推理（单样本）固定输出三张图：

1. 全量 GT + Pred 六视角拼接图  
2. 最高分 Pred 与匹配 GT 的左右放大对比图  
3. 伪 3D BEV 图（障碍物有体感，不是纯平面线框）

## 迁移说明（相对 winsurf）

已保留并增强的点：
- 训练索引保存/加载链路
- 三图输出思路
- 推理与可视化的容错处理

已修复/替换的问题：
- 默认限制 50 样本导致“非全量随机”的问题
- “最佳 Pred 对应 GT”匹配逻辑不稳的问题
- 六视角拼接尺寸/类型不一致导致的 OpenCV 崩溃
