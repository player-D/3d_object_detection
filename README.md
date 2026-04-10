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
pip install -r requirements.txt
```

PyTorch 会根据系统自动选择 CPU 或 CUDA 版本。

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

### 简化版推理（推荐）

```bash
python tools/inference.py
```

默认配置：
- 模型路径：`./saved_models/04_08_17-18/tdr_qaf_epoch_50.pth`
- 样本索引：`./saved_models/04_08_17-18/sample_indices.json`
- 推理样本数：2
- 置信度阈值：0.05
- 最大预测框数：50

如需修改路径，请编辑 `tools/inference.py` 第592-596行。

### 自定义推理（旧版参数）

如需自定义参数，请恢复 `tools/inference.py` 中的 argparse 参数部分。

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

## 9. Mini-50 Training (v1.0-mini only 50 samples)

如果你只想在 `v1.0-mini` 上快速训练 50 个样本，直接用下面命令：

```bash
python tools/train.py \
  --data_root ./dataset \
  --nuscenes_version v1.0-mini \
  --max_samples 50 \
  --batch_size 2 \
  --num_workers 0 \
  --epochs 20
```

关键点：
- `--max_samples 50` 会把训练样本池限制为 50。
- 训练会自动保存这 50 个样本的索引到 `saved_models/<run_id>/sample_indices.json`。
- 若要复现实验，后续训练可加：`--load_indices saved_models/<run_id>/sample_indices.json`。

## Visualization Outputs

Each sample now saves 3 images:
- `combined_<sample_token>.jpg`: GT + Pred 3D projection across six cameras.
- `top_pair_<sample_token>.jpg`: split focus panel (top: GT/Pred focus, bottom: original image).
- `bev_<sample_token>.jpg`: stylized BEV image (ego + lane carpet + GT/Pred objects).

Backend API fields:
- `image_combined`
- `image_pred`
- `image_bev`
