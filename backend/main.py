from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import time
import random
import sys
import os
from contextlib import asynccontextmanager

# 类别名称映射
CLASS_NAMES = {
    0: 'car',
    1: 'pedestrian',
    2: 'cyclist'
}

# 把项目根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.inference import load_model, get_sample, run_model, decode_bbox, visualize
from dataloaders.nuscenes_dataset import NuScenesDataset
import torch

# 定义生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    print("正在加载模型和数据集...")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    try:
        model = load_model(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)
    
    # 加载数据集
    try:
        nusc_root = os.environ.get('NUSCENES_ROOT', './dataset')
        nusc_version = os.environ.get('NUSCENES_VERSION', 'v1.0-mini')
        nusc_max_samples = os.environ.get('NUSCENES_MAX_SAMPLES', '').strip()
        max_samples = int(nusc_max_samples) if nusc_max_samples else None

        dataset = NuScenesDataset(
            root=nusc_root,
            debug_mode=False,
            max_samples=max_samples,
            version=nusc_version
        )
        print(f"数据集加载成功，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        sys.exit(1)
    
    # 存储到 app.state
    app.state.model = model
    app.state.dataset = dataset
    app.state.device = device
    app.state.dataset_size = len(dataset)
    
    print("模型和数据集加载完成！")
    
    yield
    
    # 关闭时
    print("正在关闭服务...")

# 创建 FastAPI 应用
app = FastAPI(
    title="TDR-QAF 多视角 3D 目标检测系统",
    description="基于多视角图像的 3D 目标检测 API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 定义请求模型
class PredictRequest(BaseModel):
    mode: str = "random"
    confidence_threshold: float = 0.3
    sample_index: int = -1

# 定义响应模型
class PredictResponse(BaseModel):
    status: str
    image_combined: str
    image_pred: str
    image_bev: str
    stats: dict

# 核心预测接口
@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # 从 app.state 获取模型和数据集
    model = app.state.model
    dataset = app.state.dataset
    device = app.state.device
    
    if model is None or dataset is None:
        raise HTTPException(status_code=500, detail="模型或数据集未加载")
    
    start_time = time.time()
    
    # 选择样本
    if request.mode == "specific":
        if request.sample_index < 0 or request.sample_index >= len(dataset):
            raise HTTPException(status_code=400, detail=f"样本索引超出范围: {request.sample_index}")
        candidate_indices = [request.sample_index]
    else:
        candidate_indices = list(range(len(dataset)))
        random.shuffle(candidate_indices)

    images = boxes_2d = cam_intrinsics = cam_extrinsics = gt_bboxes = gt_labels = None
    sample_token = None
    index = None
    last_error = None

    for candidate_index in candidate_indices:
        print(f"处理样本索引: {candidate_index}")
        try:
            sample_data = get_sample(dataset, candidate_index, device)
            if sample_data is None:
                last_error = f"样本 {candidate_index} 数据不完整"
                if request.mode == "specific":
                    break
                continue

            images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels = sample_data
            sample_token = dataset.get_sample_token(candidate_index)
            index = candidate_index
            print(f"样本 Token: {sample_token}")
            break
        except Exception as e:
            last_error = str(e)
            if request.mode == "specific":
                break

    if sample_token is None:
        detail = last_error or "未找到可用样本"
        raise HTTPException(status_code=500, detail=f"获取样本数据失败: {detail}")
    
    # 运行模型推理
    try:
        cls_scores, bbox_preds = run_model(model, images, boxes_2d, cam_intrinsics, cam_extrinsics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")
    
    # 解码边界框
    try:
        pred_scores, pred_labels, pred_bboxes = decode_bbox(cls_scores, bbox_preds, threshold=request.confidence_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解码边界框失败: {str(e)}")
    
    # 可视化结果
    try:
        # 调用修改后的 visualize 函数，获取三个 NumPy 图像数组和统计信息
        canvas_combined, canvas_top_pair, bev_img, stats = visualize(
            images,
            cam_intrinsics,
            gt_bboxes,
            gt_labels,
            pred_scores,
            pred_labels,
            pred_bboxes,
            sample_token,
            None,
            dataset.nusc
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"可视化失败: {str(e)}")
    
    # 计算推理时间
    inference_time = time.time() - start_time
    
    # 构建统计信息
    response_stats = {
        "pred_total": stats.get('pred_total', 0),
        "pred_details": stats.get('pred_details', {}),
        "gt_total": stats.get('gt_total', 0),
        "latency": f"{round(inference_time * 1000, 2)} ms",
        "latency_ms": round(inference_time * 1000, 2),
        "sample_token": sample_token,
        "sample_index": index
    }
    
    # 编码三张图
    _, buffer_combined = cv2.imencode('.jpg', canvas_combined)
    img_combined_b64 = base64.b64encode(buffer_combined).decode('utf-8')

    _, buffer_pred = cv2.imencode('.jpg', canvas_top_pair)
    img_pred_b64 = base64.b64encode(buffer_pred).decode('utf-8')

    _, buffer_bev = cv2.imencode('.jpg', bev_img)
    img_bev_b64 = base64.b64encode(buffer_bev).decode('utf-8')

    # 保存结果到 output 目录
    output_dir = "output"
    merged_images_dir = os.path.join(output_dir, "merged_images")
    bev_dir = os.path.join(output_dir, "bev")
    metrics_dir = os.path.join(output_dir, "metrics")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(merged_images_dir, exist_ok=True)
    os.makedirs(bev_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 保存拼接图
    combined_output_path = os.path.join(merged_images_dir, f'combined_{sample_token}.jpg')
    cv2.imwrite(combined_output_path, canvas_combined)
    
    # 保存预测图
    pred_output_path = os.path.join(merged_images_dir, f'top_pair_{sample_token}.jpg')
    cv2.imwrite(pred_output_path, canvas_top_pair)
    
    # 保存 BEV 图
    bev_output_path = os.path.join(bev_dir, f'bev_{sample_token}.jpg')
    cv2.imwrite(bev_output_path, bev_img)

    # 返回最终 JSON 响应
    return {
        "status": "success",
        "image_combined": f"data:image/jpeg;base64,{img_combined_b64}",
        "image_pred": f"data:image/jpeg;base64,{img_pred_b64}",
        "image_bev": f"data:image/jpeg;base64,{img_bev_b64}",
        "stats": response_stats
    }

# 根路径
@app.get("/")
async def root():
    return {"message": "TDR-QAF 多视角 3D 目标检测系统 API"}

# 健康检查
@app.get("/health")
async def health_check():
    model = app.state.model
    dataset = app.state.dataset
    device = app.state.device
    if model is not None and dataset is not None:
        return {"status": "healthy", "model_loaded": True, "dataset_size": len(dataset), "device": str(device)}
    else:
        return {"status": "unhealthy", "model_loaded": model is not None, "dataset_loaded": dataset is not None, "device": str(device)}

# API 健康检查（前端调用）
@app.get("/api/health")
async def api_health_check():
    return await health_check()

# 入口启动代码
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
