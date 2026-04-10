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
        # 使用硬编码路径，与 inference.py 保持一致
        checkpoint_path = './saved_models/04_10_10-21/best_model.pth'
        # 自动从 checkpoint 目录读取 sample_indices.json
        checkpoint_dir = os.path.dirname(checkpoint_path)
        sample_indices_path = os.path.join(checkpoint_dir, 'sample_indices.json')
        
        model = load_model(device, checkpoint_path=checkpoint_path)
        
        # 加载样本索引
        try:
            from dataloaders.nuscenes_dataset import NuScenesDataset
            indices_data = NuScenesDataset.load_sample_indices(sample_indices_path)
            app.state.sample_indices = indices_data['indices']
            app.state.sample_pool_size = len(indices_data['indices'])
            print(f"Loaded sample indices from: {sample_indices_path}")
            print(f"Sample pool size: {app.state.sample_pool_size}")
        except Exception as e:
            print(f"Warning: Failed to load sample indices: {e}")
            app.state.sample_indices = None
            app.state.sample_pool_size = None
    except Exception as e:
        print(f"加载模型失败: {e}")
        print(f"提示：请确保 {checkpoint_path} 存在，或修改main.py中的checkpoint路径")
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
    app.state.sample_pool_size = app.state.sample_pool_size if getattr(app.state, 'sample_pool_size', None) else len(dataset)
    
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
    image_sr_gt: str
    image_sr_pred: str
    image_front: str
    scene_stream: dict
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
    sample_indices = app.state.sample_indices if hasattr(app.state, 'sample_indices') and app.state.sample_indices else list(range(len(dataset)))
    sample_pool_size = app.state.sample_pool_size if hasattr(app.state, 'sample_pool_size') else len(dataset)
    
    if request.mode == "specific":
        if request.sample_index < 0 or request.sample_index >= sample_pool_size:
            raise HTTPException(status_code=400, detail=f"样本索引超出范围: {request.sample_index} (可用范围: 0-{sample_pool_size-1})")
        actual_index = sample_indices[request.sample_index] if request.sample_index < len(sample_indices) else request.sample_index
        candidate_indices = [actual_index]
    else:
        candidate_indices = sample_indices
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
        # 调用修改后的 visualize 函数，获取五个 NumPy 图像数组和统计信息
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
    
    # 编码五张图
    _, buffer_combined = cv2.imencode('.jpg', canvas_combined)
    img_combined_b64 = base64.b64encode(buffer_combined).decode('utf-8')

    _, buffer_pred = cv2.imencode('.jpg', canvas_top_pair)
    img_pred_b64 = base64.b64encode(buffer_pred).decode('utf-8')

    _, buffer_bev = cv2.imencode('.jpg', bev_img)
    img_bev_b64 = base64.b64encode(buffer_bev).decode('utf-8')

    _, buffer_sr_gt = cv2.imencode('.jpg', scene_gt_img)
    img_sr_gt_b64 = base64.b64encode(buffer_sr_gt).decode('utf-8')

    _, buffer_sr_pred = cv2.imencode('.jpg', scene_pred_img)
    img_sr_pred_b64 = base64.b64encode(buffer_sr_pred).decode('utf-8')

    _, buffer_front = cv2.imencode('.jpg', front_img)
    img_front_b64 = base64.b64encode(buffer_front).decode('utf-8')

    # 保存结果到时间命名的文件夹
    import datetime
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("output", current_time)
    os.makedirs(output_dir, exist_ok=True)
    save_debug_visuals = os.environ.get("SAVE_DEBUG_VISUALS", "0").strip().lower() in {"1", "true", "yes", "on"}
    
    # 保存所有图片到同一个文件夹
    combined_output_path = os.path.join(output_dir, f'combined_{sample_token}.jpg')
    cv2.imwrite(combined_output_path, canvas_combined)
    
    pred_output_path = os.path.join(output_dir, f'top_pair_{sample_token}.jpg')
    cv2.imwrite(pred_output_path, canvas_top_pair)
    
    if save_debug_visuals:
        bev_output_path = os.path.join(output_dir, f'bev_{sample_token}.jpg')
        cv2.imwrite(bev_output_path, bev_img)

        sr_gt_output_path = os.path.join(output_dir, f'sr_gt_{sample_token}.jpg')
        cv2.imwrite(sr_gt_output_path, scene_gt_img)
        sr_pred_output_path = os.path.join(output_dir, f'sr_pred_{sample_token}.jpg')
        cv2.imwrite(sr_pred_output_path, scene_pred_img)
        
        front_output_path = os.path.join(output_dir, f'front_{sample_token}.jpg')
        cv2.imwrite(front_output_path, front_img)

    # 返回最终 JSON 响应
    return {
        "status": "success",
        "image_combined": f"data:image/jpeg;base64,{img_combined_b64}",
        "image_pred": f"data:image/jpeg;base64,{img_pred_b64}",
        "image_bev": f"data:image/jpeg;base64,{img_bev_b64}",
        "image_sr_gt": f"data:image/jpeg;base64,{img_sr_gt_b64}",
        "image_sr_pred": f"data:image/jpeg;base64,{img_sr_pred_b64}",
        "image_front": f"data:image/jpeg;base64,{img_front_b64}",
        "scene_stream": scene_stream,
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
    sample_pool_size = app.state.sample_pool_size if hasattr(app.state, 'sample_pool_size') else len(dataset)
    if model is not None and dataset is not None:
        return {"status": "healthy", "model_loaded": True, "dataset_size": sample_pool_size, "device": str(device)}
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
