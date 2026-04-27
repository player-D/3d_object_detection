import base64
import datetime
import os
import random
import sys
import time
from contextlib import asynccontextmanager

import cv2
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.nuscenes_dataset import NuScenesDataset
from tools.inference import decode_bbox, get_sample, load_model, run_model, visualize
from tools.runtime_config import (
    resolve_checkpoint_path,
    resolve_output_root,
    resolve_sample_indices_path,
)


class PredictRequest(BaseModel):
    mode: str = "random"
    confidence_threshold: float = 0.3
    sample_index: int = -1


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


def _encode_image(image):
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode visualization image.")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def _save_visuals(output_root, sample_token, payload, save_debug_visuals):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, current_time)
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f"combined_{sample_token}.jpg"), payload["canvas_combined"])
    cv2.imwrite(os.path.join(output_dir, f"top_pair_{sample_token}.jpg"), payload["canvas_top_pair"])

    if save_debug_visuals:
        cv2.imwrite(os.path.join(output_dir, f"bev_{sample_token}.jpg"), payload["bev_img"])
        cv2.imwrite(os.path.join(output_dir, f"sr_gt_{sample_token}.jpg"), payload["scene_gt_img"])
        cv2.imwrite(os.path.join(output_dir, f"sr_pred_{sample_token}.jpg"), payload["scene_pred_img"])
        cv2.imwrite(os.path.join(output_dir, f"front_{sample_token}.jpg"), payload["front_img"])


def _resolve_candidate_indices(dataset, sample_indices, request_mode, sample_index):
    sample_pool = list(sample_indices) if sample_indices else list(range(len(dataset)))
    sample_pool_size = len(sample_pool)

    if request_mode == "specific":
        if sample_index < 0 or sample_index >= sample_pool_size:
            raise HTTPException(
                status_code=400,
                detail=f"Sample index out of range: {sample_index} (valid: 0-{max(0, sample_pool_size - 1)})",
            )
        return [sample_pool[sample_index]], sample_pool_size

    random.shuffle(sample_pool)
    return sample_pool, sample_pool_size


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = resolve_checkpoint_path()
    sample_indices_path = resolve_sample_indices_path(checkpoint_path)
    output_root = resolve_output_root(os.path.join("output", "api"))

    model = load_model(device, checkpoint_path=checkpoint_path)

    nusc_root = os.environ.get("NUSCENES_ROOT", "./dataset")
    nusc_version = os.environ.get("NUSCENES_VERSION", "v1.0-mini")
    nusc_max_samples = os.environ.get("NUSCENES_MAX_SAMPLES", "").strip()
    max_samples = int(nusc_max_samples) if nusc_max_samples else None

    dataset = NuScenesDataset(
        root=nusc_root,
        debug_mode=False,
        max_samples=max_samples,
        version=nusc_version,
    )

    sample_indices = None
    sample_pool_size = len(dataset)
    if sample_indices_path:
        indices_data = NuScenesDataset.load_sample_indices(sample_indices_path)
        sample_indices = indices_data["indices"]
        sample_pool_size = len(sample_indices)
        print(f"Loaded sample indices from: {sample_indices_path}")
        print(f"Sample pool size: {sample_pool_size}")
    else:
        print("No sample_indices.json found for the active checkpoint. Using the full dataset.")

    app.state.model = model
    app.state.dataset = dataset
    app.state.device = device
    app.state.dataset_size = len(dataset)
    app.state.sample_indices = sample_indices
    app.state.sample_pool_size = sample_pool_size
    app.state.checkpoint_path = checkpoint_path
    app.state.output_root = output_root

    print("Model and dataset are ready.")
    yield
    print("Shutting down service...")


app = FastAPI(
    title="TDR-QAF 3D Detection API",
    description="Backend service for training-time assets, inference, and visualization.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model = app.state.model
    dataset = app.state.dataset
    device = app.state.device

    if model is None or dataset is None:
        raise HTTPException(status_code=500, detail="Model or dataset is not loaded.")

    start_time = time.time()
    candidate_indices, sample_pool_size = _resolve_candidate_indices(
        dataset=dataset,
        sample_indices=app.state.sample_indices,
        request_mode=request.mode,
        sample_index=request.sample_index,
    )

    sample_data = None
    sample_token = None
    resolved_index = None
    last_error = None

    for candidate_index in candidate_indices:
        try:
            sample_data = get_sample(dataset, candidate_index, device)
            if sample_data is None:
                last_error = f"Sample {candidate_index} is incomplete."
                if request.mode == "specific":
                    break
                continue

            sample_token = dataset.get_sample_token(candidate_index)
            resolved_index = candidate_index
            break
        except Exception as error:
            last_error = str(error)
            if request.mode == "specific":
                break

    if sample_data is None or sample_token is None:
        detail = last_error or "No valid sample was found."
        raise HTTPException(status_code=500, detail=f"Failed to read sample data: {detail}")

    images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels = sample_data

    try:
        cls_scores, bbox_preds = run_model(model, images, boxes_2d, cam_intrinsics, cam_extrinsics)
        pred_scores, pred_labels, pred_bboxes = decode_bbox(
            cls_scores,
            bbox_preds,
            threshold=request.confidence_threshold,
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
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference or visualization failed: {error}") from error

    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    payload = {
        "canvas_combined": canvas_combined,
        "canvas_top_pair": canvas_top_pair,
        "bev_img": bev_img,
        "scene_gt_img": scene_gt_img,
        "scene_pred_img": scene_pred_img,
        "front_img": front_img,
    }

    save_debug_visuals = os.environ.get("SAVE_DEBUG_VISUALS", "0").strip().lower() in {"1", "true", "yes", "on"}
    _save_visuals(app.state.output_root, sample_token, payload, save_debug_visuals)

    return {
        "status": "success",
        "image_combined": _encode_image(canvas_combined),
        "image_pred": _encode_image(canvas_top_pair),
        "image_bev": _encode_image(bev_img),
        "image_sr_gt": _encode_image(scene_gt_img),
        "image_sr_pred": _encode_image(scene_pred_img),
        "image_front": _encode_image(front_img),
        "scene_stream": scene_stream,
        "stats": {
            "pred_total": stats.get("pred_total", 0),
            "pred_details": stats.get("pred_details", {}),
            "gt_total": stats.get("gt_total", 0),
            "latency": f"{inference_time_ms} ms",
            "latency_ms": inference_time_ms,
            "sample_token": sample_token,
            "sample_index": resolved_index,
            "sample_pool_size": sample_pool_size,
            "checkpoint_path": app.state.checkpoint_path,
        },
    }


@app.get("/")
async def root():
    return {"message": "TDR-QAF 3D Detection API"}


@app.get("/health")
async def health_check():
    model = getattr(app.state, "model", None)
    dataset = getattr(app.state, "dataset", None)
    device = getattr(app.state, "device", "cpu")
    sample_pool_size = getattr(app.state, "sample_pool_size", 0)

    if model is not None and dataset is not None:
        return {
            "status": "healthy",
            "model_loaded": True,
            "dataset_size": sample_pool_size,
            "device": str(device),
            "checkpoint_path": getattr(app.state, "checkpoint_path", ""),
        }

    return {
        "status": "unhealthy",
        "model_loaded": model is not None,
        "dataset_loaded": dataset is not None,
        "device": str(device),
    }


@app.get("/api/health")
async def api_health_check():
    return await health_check()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
