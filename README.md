# TDR-QAF (3d_object_detection)

本目录是当前维护版本。只修改本目录，不修改 `winsurf/`。

## 1. 当前行为说明（重点）

### 推理采样逻辑
- 不加 `--load_indices`：从当前数据集池随机抽样（默认是完整数据集）。
- 加 `--load_indices path/to/sample_indices.json`：只在训练保存的索引集合内抽样。
- `--max_samples` 仅用于限制数据池（调试用），默认 `None`。

### 每次推理固定输出三张图
- `pred_<sample_token>.jpg`：6 相机拼图，叠加所有 GT + Pred 3D 框。
- `top_pair_<sample_token>.jpg`：最高分 Pred 与匹配 GT 的左右分屏放大图。
- `bev_<sample_token>.jpg`：伪 3D BEV（带障碍物体感，不是纯平面线框）。

### 可视化稳定性
- 6 相机拼接前会统一图像尺寸/通道/类型，避免 `cv2.hconcat` 崩溃。

## 2. 环境安装

```bash
pip install -r requirements_gpu.txt
# 或
pip install -r requirements_cpu.txt
```

前端：

```bash
cd frontend
npm install
```

## 3. 数据目录

默认根目录是 `./dataset`，示例：

```text
dataset/
  samples/
  sweeps/
  maps/
  v1.0-mini/
  v1.0-trainval/
```

## 4. 训练

### mini 训练（快速验证）

```bash
python tools/train.py \
  --data_root ./dataset \
  --nuscenes_version v1.0-mini \
  --batch_size 2 \
  --num_workers 0 \
  --epochs 20
```

### 全量 trainval（服务器）

```bash
python tools/train.py \
  --data_root /path/to/nuscenes \
  --nuscenes_version v1.0-trainval \
  --batch_size 1 \
  --num_workers 4 \
  --lr 1e-4 \
  --epochs 80
```

### 复用训练样本索引

```bash
python tools/train.py --load_indices saved_models/<run_id>/sample_indices.json
```

每次训练会在 `saved_models/<run_id>/sample_indices.json` 自动保存本次使用索引。

## 5. 推理（CLI）

### 默认随机（完整数据池）

```bash
python tools/inference.py \
  --data_root ./dataset \
  --nuscenes_version v1.0-mini \
  --num_samples 2
```

### 只在训练索引池推理

```bash
python tools/inference.py \
  --load_indices saved_models/<run_id>/sample_indices.json \
  --num_samples 2
```

### 常用参数
- `--confidence`：置信度阈值（默认 `0.05`）。
- `--topk`：最多保留预测框数（默认 `50`）。
- `--max_samples`：限制数据池大小（默认 `None`）。
- `--num_samples`：本次命令推理样本数。
- `--data_root`：NuScenes 根目录。
- `--nuscenes_version`：`v1.0-mini` 或 `v1.0-trainval`。

## 6. 后端与前端

### 启动后端

```bash
cd backend
python main.py
```

后端支持环境变量：
- `NUSCENES_ROOT`（默认 `./dataset`）
- `NUSCENES_VERSION`（默认 `v1.0-mini`）
- `NUSCENES_MAX_SAMPLES`（默认空，即不限制）

示例：

```bash
set NUSCENES_ROOT=D:\datasets\nuscenes
set NUSCENES_VERSION=v1.0-trainval
python backend/main.py
```

### 启动前端

```bash
cd frontend
npm run dev
```

## 7. 60G 数据集训练建议（你当前机器选择）

你给的两个卡里，优先用 **V100 16GB**（PyTorch/CUDA 兼容与稳定性通常更好）。

### V100 16GB（推荐）
- `--batch_size 1`（稳定起步，先跑通）
- `--num_workers 4`
- `--lr 1e-4`（若 batch=2 可试 `2e-4`）
- `--epochs 80`（至少 50，建议 80 起）
- `--nuscenes_version v1.0-trainval`
- 不设置 `--max_samples`（全量训练）

### 国产 32GB 卡（可选）
- 显存更大，可尝试 `--batch_size 2` 或 `4`
- 学习率按 batch 线性放大（如 batch=4 可尝试 `4e-4`）
- 前提是驱动与 PyTorch 适配稳定，否则优先 V100

## 8. 已知注意事项

- 当前仓库有未完成 merge 状态（`UU`），这是 Git 索引状态问题，不等于代码不可运行。
- 如果你只看运行效果，优先按本 README 的训练/推理命令执行。
