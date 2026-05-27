import os
from pathlib import Path

# 设置模型缓存目录（必须在导入 timm/anomalib 之前）
PROJECT_ROOT = Path(__file__).parent
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
HUB_DIR = PRETRAINED_DIR / "hub"
os.environ["TIMM_HOME"] = str(PRETRAINED_DIR)
os.environ["HF_HOME"] = str(PRETRAINED_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(PRETRAINED_DIR / "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUB_DIR)  # 指向 hub 子目录
# 使用 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 允许联网下载
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
# PyTorch 2.6+ 兼容性：允许加载包含 numpy 的模型文件
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from fastapi import FastAPI, Header, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import defaultdict
import asyncio
import json
import time
import logging
import cv2
import numpy as np
import base64
import shutil
import tempfile
import random
import platform
import warnings
import uuid as uuid_lib
import uvicorn
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

# 屏蔽框架无关紧要的日志和警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*ccache.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from utils.trainer import ModelTrainer
from utils.deployer import ModelDeployer
from utils.config import path_config, get_output_dir, get_product_dir
from dataclasses import asdict

logger = logging.getLogger(__name__)

# 初始化模型训练器（使用环境变量配置的output路径）
trainer = ModelTrainer(output_dir=str(get_output_dir()), max_concurrent=3)

app = FastAPI(title="One2All Paddle API")

# 配置 CORS
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.getcwd()), name="static")

# 注册相机路由
from routers.camera import router as camera_router
app.include_router(camera_router)


@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("Application started.")


class COCOAnnotation(BaseModel):
    id: Optional[Any] = None
    image_id: Optional[Any] = None
    category_id: Optional[Any] = 0
    bbox: Optional[List[float]] = None
    rbbox: Optional[List[float]] = None
    points: Optional[List[float]] = None
    segmentation: Optional[List[List[float]]] = None
    area: Optional[float] = None
    iscrowd: Optional[int] = 0
    label: Optional[str] = None
    type: Optional[str] = None
    angle: Optional[float] = None
    horizontal_flip: Optional[bool] = None
    vertical_flip: Optional[bool] = None
    pos_id: Optional[Any] = None


class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: Optional[str] = None


class COCOImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    angle: Optional[float] = 0


class COCOData(BaseModel):
    images: Optional[List[COCOImage]] = None
    annotations: List[COCOAnnotation]
    categories: Optional[List[COCOCategory]] = None




class TrainRequest(BaseModel):
    images: List[str]
    coco_data: COCOData
    base_path: str
    project_id: str
    label_names: Optional[List[str]] = None
    parallel_train: bool = False
    train_mode: str = "by_pos_id"  # "by_pos_id" | "by_category"

    # Dinomaly 参数
    encoder_name: str = "dinov2_vit_base_14"
    decoder_depth: int = 8
    bottleneck_dropout: float = 0.2
    epochs: int = 12
    batch_size: int = 2
    freeze_encoder: bool = True

    # 通用参数
    augment: bool = True
    num_augmentations: int = 1
    augmentation_config: Optional[str] = None  # 数据增强配置文件路径
    normalize_brightness: bool = False
    normalize_contrast: bool = False
    threshold_buffer: float = 1.0
    save_images: bool = True
    max_concurrent: int = 3



@app.post("/train/anomaly")
def train_anomaly(request: TrainRequest):
    """
    接收 COCO 数据，按照位置+类型组合保存裁剪信息并启动训练
    层级结构: {base_path}/product/{project_id}/train/{uuid}/
    输出路径: output/{project_id}/{uuid}/
    """
    task_uuid = uuid_lib.uuid4().hex[:8]

    cat_map = (
        {cat.id: cat.name for cat in request.coco_data.categories}
        if request.coco_data.categories
        else {}
    )

    # 使用环境变量配置的产品数据目录
    storage_base = path_config.get_project_product_path(request.project_id, task_uuid)

    try:
        # ── 1. 解码所有图片 ────────────────────────────────────────────────
        decoded_images: Dict[Any, np.ndarray] = {}

        for idx, img_b64 in enumerate(request.images):
            try:
                img_data = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    if request.coco_data.images and idx < len(request.coco_data.images):
                        image_id = request.coco_data.images[idx].id
                    else:
                        image_id = idx + 1
                    decoded_images[image_id] = img
            except Exception as e:
                logger.error(f"Failed to decode image[{idx}]: {e}")

        # ── 2. 保存原始完整图像 ────────────────────────────────────────────
        raw_images_dir = storage_base / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)

        annotations_data: Dict[str, Any] = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        if request.coco_data.categories:
            for cat in request.coco_data.categories:
                annotations_data["categories"].append({
                    "id": cat.id,
                    "name": cat.name,
                    "supercategory": cat.supercategory or "",
                })

        for img_id, img in decoded_images.items():
            coco_img_info = None
            if request.coco_data.images:
                for coco_img in request.coco_data.images:
                    if coco_img.id == img_id:
                        coco_img_info = coco_img
                        break

            raw_filename = (
                f"raw_{coco_img_info.file_name}" if coco_img_info else f"raw_{img_id}.jpg"
            )
            cv2.imwrite(str(raw_images_dir / raw_filename), img)

            h, w = img.shape[:2]
            annotations_data["images"].append({
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": raw_filename,
            })

        # ── 3. 构建 annotations.json（不裁剪，裁剪在训练时完成）──────────────
        crop_count = 0
        labels_processed: set = set()

        for ann in request.coco_data.annotations:
            base_label = ann.label or cat_map.get(ann.category_id, f"class_{ann.category_id}")

            if request.label_names and base_label not in request.label_names:
                continue

            image_id = ann.image_id
            if image_id not in decoded_images:
                continue

            img = decoded_images[image_id]
            h, w = img.shape[:2]

            label_name = base_label
            labels_processed.add(label_name)

            # 使用原始 bbox（不进行裁剪）
            crop_bbox = None
            if ann.bbox:
                crop_bbox = [int(x) for x in ann.bbox]
            elif ann.rbbox and len(ann.rbbox) >= 5:
                # 从 rbbox 计算 bbox
                if len(ann.rbbox) >= 8:
                    xs = ann.rbbox[0::2]
                    ys = ann.rbbox[1::2]
                    crop_bbox = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
                else:
                    cx, cy, bw, bh, angle = ann.rbbox
                    angle_rad = np.deg2rad(angle)
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    hw, hh = bw / 2, bh / 2
                    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
                    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    rotated = np.dot(corners, rot_matrix.T) + np.array([cx, cy])
                    xs, ys = rotated[:, 0], rotated[:, 1]
                    crop_bbox = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
            elif ann.segmentation and len(ann.segmentation) > 0:
                # 从 segmentation 计算 bbox
                seg = ann.segmentation[0]
                xs = seg[0::2]
                ys = seg[1::2]
                crop_bbox = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]

            if crop_bbox is None:
                crop_bbox = [0, 0, w, h]

            ann_data: Dict[str, Any] = {
                "id": len(annotations_data["annotations"]) + 1,
                "image_id": image_id,
                "category_id": ann.category_id or 0,
                "bbox": crop_bbox,
                "area": crop_bbox[2] * crop_bbox[3],
                "label": label_name,
            }
            if ann.pos_id is not None:
                ann_data["pos_id"] = ann.pos_id
            if ann.rbbox is not None:
                ann_data["rbbox"] = ann.rbbox
            if ann.segmentation is not None:
                ann_data["segmentation"] = ann.segmentation
            if ann.angle is not None:
                ann_data["angle"] = ann.angle
            if ann.horizontal_flip is not None:
                ann_data["horizontal_flip"] = ann.horizontal_flip
            if ann.vertical_flip is not None:
                ann_data["vertical_flip"] = ann.vertical_flip

            annotations_data["annotations"].append(ann_data)
            crop_count += 1

        if crop_count == 0:
            raise HTTPException(status_code=400, detail="No valid annotations found")

        # ── 4. 保存 annotations.json ───────────────────────────────────────
        annotations_path = storage_base / "annotations.json"
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations_data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Saved annotations.json: {len(annotations_data['images'])} images, "
            f"{len(annotations_data['annotations'])} annotations"
        )

        # ── 5. 保存 labels.txt ────────────────────────────────────────────
        with open(storage_base / "labels.txt", "w") as f:
            for label in sorted(labels_processed):
                f.write(f"{label}\n")

        # ── 6. 确定 pos_id 分组 ───────────────────────────────────────────
        train_mode = request.train_mode

        pos_ids_in_request: set = set()
        for ann in request.coco_data.annotations:
            if ann.pos_id is not None:
                pos_ids_in_request.add(ann.pos_id)
        use_pos_id = len(pos_ids_in_request) > 0

        group_id = f"group_{int(time.time())}_{request.project_id}"

        groups_for_trainer: Dict[Any, List[Dict]] = defaultdict(list)
        for ann in annotations_data["annotations"]:
            if train_mode == "by_category":
                key = ann.get("label", "unknown")
            elif use_pos_id:
                key = ann.get("pos_id", ann.get("label", "unknown"))
            else:
                key = ann.get("label", "unknown")
            groups_for_trainer[key].append(ann)

        # 构建训练配置（仅支持 Dinomaly）
        base_train_config = {
            "model_name": "Dinomaly",
            "use_pos_id": use_pos_id,
            "project_id": request.project_id,
            "task_uuid": task_uuid,
            "parallel_train": request.parallel_train,
            "train_mode": train_mode,
            "augment": request.augment,
            "num_augmentations": request.num_augmentations,
            "augmentation_config": request.augmentation_config,
            "normalize_brightness": request.normalize_brightness,
            "normalize_contrast": request.normalize_contrast,
            "threshold_buffer": request.threshold_buffer,
            "save_images": request.save_images,
            "max_concurrent": request.max_concurrent,
            "encoder_name": request.encoder_name,
            "decoder_depth": request.decoder_depth,
            "bottleneck_dropout": request.bottleneck_dropout,
            "epochs": request.epochs,
            "batch_size":  request.batch_size,
            "freeze_encoder": request.freeze_encoder,
        }

        t0 = time.time()
        all_task_ids, filtered_keys = trainer.run_batch_training_async(
            str(storage_base),
            base_train_config,
            dict(groups_for_trainer),
            group_id=group_id,
        )
        logger.info(
            f"[TrainAnomaly] {len(all_task_ids)} tasks launched in {time.time() - t0:.3f}s"
        )

        task_results = []
        for pid, tid in zip(filtered_keys, all_task_ids):
            group_annotations = groups_for_trainer.get(int(pid) if pid.isdigit() else pid, [])
            first_ann = group_annotations[0] if group_annotations else {}
            label = first_ann.get("label", str(pid))
            task_results.append({
                "pos_id": pid,
                "task_id": tid,
                "label": label
            })

        pos_ids_processed = filtered_keys

        return {
            "status": "success",
            "project_id": request.project_id,
            "task_uuid": task_uuid,
            "group_id": group_id,
            "storage_path": str(storage_base),
            "total_crops": crop_count,
            "labels": sorted(list(labels_processed)),
            "pos_ids": [str(p) for p in pos_ids_processed],
            "use_pos_id": use_pos_id,
            "tasks": task_results,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 训练状态 / 控制接口
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/train/status/{task_id}")
async def get_train_status(task_id: str):
    """查询单个训练任务状态"""
    status = trainer.get_training_status(task_id)
    if status is None or status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@app.get("/train/status/group/{group_id}")
async def get_group_train_status(group_id: str):
    """查询任务组状态"""
    return trainer.get_task_group_status(group_id)


@app.post("/train/stop/{task_id}")
async def stop_train_task(task_id: str):
    """停止单个训练任务"""
    return trainer.stop_task(task_id)


@app.post("/train/stop/group/{group_id}")
async def stop_group_train(group_id: str):
    """停止整个任务组训练"""
    return trainer.stop_group(group_id)


@app.post("/train/resume/{task_id}")
async def resume_train_task(
    task_id: str,
    resume_path: Optional[str] = None,
    resume_mode: Optional[str] = None,
):
    """显式恢复某个训练任务"""
    result = trainer.resume_task(task_id, resume_path=resume_path, resume_mode=resume_mode)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/train/events/{task_id}")
async def train_events(task_id: str, include_history: bool = True):
    """SSE 实时推送训练进度和日志
    
    Args:
        task_id: 任务ID
        include_history: 是否包含历史日志（首次连接时设为true）
    """
    async def event_generator():
        last_log_idx = 0
        first_send = True
        while True:
            status = trainer.get_training_status(task_id)
            if status is None:
                status = {"status": "not_found", "progress": 0, "logs": [], "metrics": [], "eval_metrics": []}

            data = {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "task_type": status.get("task_type"),
                "label": status.get("label"),
                "stage": status.get("stage"),
                "threshold": status.get("threshold"),
                "num_samples": status.get("num_samples"),
                "new_logs": [],
                "metrics": status.get("metrics", []),
                "eval_metrics": status.get("eval_metrics", []),
            }

            logs = status.get("logs", [])
            if first_send and include_history:
                # 首次发送时返回所有日志
                data["new_logs"] = logs
                last_log_idx = len(logs)
                first_send = False
            elif len(logs) > last_log_idx:
                data["new_logs"] = logs[last_log_idx:]
                last_log_idx = len(logs)

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if status.get("status") in {"completed", "failed", "not_found", "cancelled", "interrupted"}:
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/train/checkpoints/{task_id}")
async def get_task_checkpoints(task_id: str):
    """获取某个任务下可用的检查点列表"""
    status = trainer.get_training_status(task_id)
    if status is None or status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")

    save_dir = status.get("save_dir")
    if not save_dir or not os.path.exists(save_dir):
        return {"checkpoints": []}

    checkpoints = []
    for root, _, files in os.walk(save_dir):
        if "model.pdparams" in files:
            rel_path = os.path.relpath(root, save_dir)
            pdparams_path = os.path.join(root, "model.pdparams")
            mtime = os.path.getmtime(pdparams_path)
            checkpoints.append({
                "name": rel_path if rel_path != "." else "latest",
                "path": pdparams_path,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
            })

    checkpoints.sort(key=lambda x: x["time"], reverse=True)
    return {"task_id": task_id, "checkpoints": checkpoints}


@app.get("/train/history/{task_id}")
async def get_task_history(task_id: str):
    """获取任务的历史日志"""
    status = trainer.get_training_status(task_id)
    if status is None or status.get("status") == "not_found":
        return {"task_id": task_id, "logs": [], "error": "Task not found"}

    return {
        "task_id": task_id,
        "logs": status.get("logs", []),
        "metrics": status.get("metrics", []),
        "eval_metrics": status.get("eval_metrics", []),
    }


@app.get("/train/data/{task_id}")
async def get_train_data(task_id: str):
    """获取某个训练任务所使用的图片列表及访问 URL"""
    status = trainer.get_training_status(task_id)
    if status is None or status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")

    dataset_dir = status.get("dataset_dir")
    if not dataset_dir or not os.path.exists(dataset_dir):
        raise HTTPException(status_code=404, detail="Dataset directory not found")

    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        return {"task_id": task_id, "images": []}

    image_files = [
        f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    cwd = os.getcwd()
    try:
        rel_dataset_path = os.path.relpath(images_dir, cwd)
    except ValueError:
        raise HTTPException(status_code=500, detail="Cannot calculate relative path for images")

    result = [
        {"name": f, "url": f"/static/{rel_dataset_path}/{f}".replace("\\", "/")}
        for f in image_files
    ]

    return {
        "task_id": task_id,
        "label": status.get("label"),
        "total": len(result),
        "images": result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 增量训练接口（同时接收正常样本和异常 ROI）
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackGroup(BaseModel):
    """
    单个 path_id 的反馈数据分组
    
    每个 path_id 对应一个位置/类别，包含该位置收集的 FP 和 FN 样本
    """
    path_id: str  # 位置/类别 ID
    false_positive_images: List[str] = []  # base64 编码的正常样本（被 Dinomaly 误检为异常）
    false_negative_images: List[str] = []  # base64 编码的异常样本 ROI（被漏检）
    false_negative_pos_ids: List[str] = []  # 每个 FN 图片对应的 pos_id，用于按位置组织 YOLO 训练
    yolo_false_positive_images: List[str] = []  # base64 编码的正常样本（被 YOLO 误检为异常，Dinomaly 判定正确）


class RetrainRequest(BaseModel):
    """
    增量训练请求
    
    前端在重新训练时发送：
    - feedback_groups: 按 path_id 分组的反馈数据列表
      每个分组包含该位置的 FP 样本（用于微调）和 FN 样本（用于原型库）
    """
    project_id: str
    base_task_uuid: str  # 基础模型的 task_uuid
    feedback_groups: List[FeedbackGroup]  # 按 path_id 分组的反馈数据
    # Dinomaly 训练参数（可选，默认继承基础模型配置）
    encoder_name: Optional[str] = None
    decoder_depth: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    freeze_encoder: Optional[bool] = None


@app.post("/train/anomaly/retrain")
async def incremental_retrain(request: RetrainRequest):
    """
    增量训练：处理多个 path_id 的用户反馈数据
    
    保证新版本模型完整：
    - 有反馈的 path_id：微调/更新原型库
    - 无反馈的 path_id：直接复制原模型文件
    """
    from PIL import Image
    import io
    import shutil

    if not request.feedback_groups:
        raise HTTPException(status_code=400, detail="No feedback groups provided")

    # 生成新的 task_uuid（所有 path_id 共享）
    new_task_uuid = uuid_lib.uuid4().hex[:8]
    
    # 获取基础模型的所有 path_id
    base_output_dir = path_config.get_project_output_path(request.project_id, request.base_task_uuid)
    if not base_output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Base model directory not found: {base_output_dir}")
    
    all_base_path_ids = [d.name for d in base_output_dir.iterdir() if d.is_dir() and d.name != "temp"]
    logger.info(f"Found {len(all_base_path_ids)} path_ids in base model: {all_base_path_ids}")
    
    # 有反馈的 path_id 集合
    feedback_path_ids = {g.path_id for g in request.feedback_groups}
    
    # 需要复制的 path_id（无反馈）
    copy_path_ids = set(all_base_path_ids) - feedback_path_ids
    logger.info(f"Path_ids to copy (no feedback): {copy_path_ids}")
    
    results = []
    all_task_ids = []
    total_fp = 0
    total_fn = 0
    total_yolo_fp = 0

    # ========== 第一步：复制无反馈的 path_id ==========
    for path_id in copy_path_ids:
        try:
            base_model_dir = base_output_dir / path_id
            new_model_dir = path_config.get_project_output_path(request.project_id, new_task_uuid) / path_id
            new_model_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制所有模型相关文件
            model_files = ["model.ckpt", "dinomaly_model.pt", "config.json"]
            copied_files = []
            for file_name in model_files:
                src_file = base_model_dir / file_name
                if src_file.exists():
                    shutil.copy2(str(src_file), str(new_model_dir / file_name))
                    copied_files.append(file_name)
            
            # 复制 template_variants 目录（推理服务模板变体）
            template_variants_src = base_model_dir / "template_variants"
            if template_variants_src.exists() and template_variants_src.is_dir():
                template_variants_dst = new_model_dir / "template_variants"
                if template_variants_dst.exists():
                    shutil.rmtree(str(template_variants_dst))
                shutil.copytree(str(template_variants_src), str(template_variants_dst))
                logger.info(f"path_id={path_id}: Copied template_variants directory")

            # 复制 yolo_model.pt（如果基础模型有 YOLO 分类器）
            yolo_src = base_model_dir / "yolo_model.pt"
            if yolo_src.exists():
                shutil.copy2(str(yolo_src), str(new_model_dir / "yolo_model.pt"))
                logger.info(f"path_id={path_id}: Copied yolo_model.pt from base version")

            # 如果没有找到任何模型文件，记录警告
            if not copied_files:
                logger.warning(f"path_id={path_id}: No model files found in {base_model_dir}")
            
            results.append({
                "path_id": path_id,
                "status": "copied",
                "message": f"Model copied from base version (files: {copied_files})"
            })
            logger.info(f"path_id={path_id}: Copied files {copied_files} from base version")
            
        except Exception as e:
            logger.error(f"Failed to copy path_id={path_id}: {e}")
            results.append({
                "path_id": path_id,
                "status": "error",
                "message": f"Failed to copy: {str(e)}"
            })

    # ========== 第二步：处理有反馈的 path_id ==========
    for group in request.feedback_groups:
        path_id = group.path_id
        logger.info(f"Processing feedback group for path_id={path_id}")

        try:
            # 定位该 path_id 的基础模型目录
            base_model_dir = base_output_dir / path_id
            if not base_model_dir.exists():
                logger.error(f"Base model directory not found: {base_model_dir}")
                results.append({
                    "path_id": path_id,
                    "status": "error",
                    "message": f"Base model directory not found: {base_model_dir}"
                })
                continue

            # 查找基础模型文件（支持多种格式）
            base_model_path = base_model_dir / "model.ckpt"
            if not base_model_path.exists():
                base_model_path = base_model_dir / "dinomaly_model.pt"
            base_config_path = base_model_dir / "config.json"
            
            if not base_model_path.exists():
                logger.error(f"Base model checkpoint not found in {base_model_dir}")
                results.append({
                    "path_id": path_id,
                    "status": "error",
                    "message": f"Base model checkpoint not found in {base_model_dir}"
                })
                continue

            # 读取基础模型配置
            base_config = {}
            if base_config_path.exists():
                with open(base_config_path, "r", encoding="utf-8") as f:
                    base_config = json.load(f)

            # 创建统一的训练目录（所有 path_id 共享）
            # 结构：train/{new_task_uuid}/raw_images/, train/{new_task_uuid}/roi/{path_id}/
            train_base_dir = path_config.get_project_product_path(request.project_id, new_task_uuid)
            train_base_dir.mkdir(parents=True, exist_ok=True)
            new_model_dir = path_config.get_project_output_path(request.project_id, new_task_uuid) / str(path_id)
            new_model_dir.mkdir(parents=True, exist_ok=True)

            # 复制父模型的 raw_images 和 annotations.json 到新的训练目录（只在第一次处理时复制）
            # 基础模型的文件在 product/{project_id}/train/{base_task_uuid}/ 下
            base_product_train_dir = path_config.get_project_product_path(request.project_id, request.base_task_uuid)
            base_raw_images_dir = base_product_train_dir / "raw_images"
            base_annotations_path = base_product_train_dir / "annotations.json"
            new_raw_images_dir = train_base_dir / "raw_images"
            new_annotations_path = train_base_dir / "annotations.json"
            
            if base_raw_images_dir.exists() and base_raw_images_dir.is_dir() and not new_raw_images_dir.exists():
                shutil.copytree(str(base_raw_images_dir), str(new_raw_images_dir), ignore=shutil.ignore_patterns("temp"))
                logger.info(f"Copied {len(list(base_raw_images_dir.glob('*.jpg')))} raw images from base model to {new_raw_images_dir}")
            elif not new_raw_images_dir.exists():
                new_raw_images_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"No raw_images found in {base_raw_images_dir}, creating empty directory")
            
            # 复制 annotations.json
            if base_annotations_path.exists() and not new_annotations_path.exists():
                shutil.copy2(str(base_annotations_path), str(new_annotations_path))
                logger.info(f"Copied annotations.json from base model to {new_annotations_path}")

            # 复制基础 ROI 目录（用于 YOLO 训练的 normal 类数据源）
            # 排除 fp_* 文件（这些是 FP 重训时加入的异常样本）
            base_roi_dir = base_product_train_dir / "roi" / str(path_id)
            new_roi_dir = train_base_dir / "roi" / str(path_id)
            if base_roi_dir.exists() and base_roi_dir.is_dir() and not new_roi_dir.exists():
                new_roi_dir.mkdir(parents=True, exist_ok=True)
                roi_copied = 0
                for f in base_roi_dir.iterdir():
                    if f.is_file() and not f.name.startswith("fp_"):
                        shutil.copy2(str(f), str(new_roi_dir / f.name))
                        roi_copied += 1
                logger.info(f"path_id={path_id}: Copied {roi_copied} normal ROI images from base to {new_roi_dir}")

            # 解码 False Positive 图片
            # FP 图片已经是 ROI，直接保存到 roi/{path_id}/ 目录
            fp_images: List[np.ndarray] = []
            if group.false_positive_images:
                # ROI 目录：roi/{path_id}/
                new_roi_dir = train_base_dir / "roi" / str(path_id)
                new_roi_dir.mkdir(parents=True, exist_ok=True)

                for idx, img_b64 in enumerate(group.false_positive_images):
                    try:
                        img_data = base64.b64decode(img_b64)
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            fp_images.append(img)
                            cv2.imwrite(str(new_roi_dir / f"fp_{idx:04d}.jpg"), img)
                            logger.info(f"Saved FP ROI to {new_roi_dir / f'fp_{idx:04d}.jpg'}")
                    except Exception as e:
                        logger.error(f"Failed to decode FP image[{idx}] for path_id={path_id}: {e}")

            # 解码 False Negative 图片，同时记录每个图片的 pos_id
            fn_images: List[Tuple[Image.Image, str]] = []  # (image, pos_id)
            fn_pos_ids = group.false_negative_pos_ids if group.false_negative_pos_ids else []
            for idx, img_b64 in enumerate(group.false_negative_images):
                try:
                    img_data = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    pos_id = fn_pos_ids[idx] if idx < len(fn_pos_ids) else "unknown"
                    fn_images.append((img, pos_id))
                except Exception as e:
                    logger.error(f"Failed to decode FN image[{idx}] for path_id={path_id}: {e}")

            # ========== 保存 FN 图像到模型目录（用于 YOLO 训练） ==========
            if fn_images:
                fn_images_dir = new_model_dir / "fn_images"
                batch_ts = int(time.time() * 1000) % 1000000  # 批次时间戳，避免多轮重训命名冲突
                for idx, (fn_img, fn_pos_id) in enumerate(fn_images):
                    pos_dir = fn_images_dir / str(fn_pos_id)
                    pos_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        fn_img.save(str(pos_dir / f"fn_{batch_ts}_{idx:04d}.jpg"), "JPEG", quality=95)
                    except Exception as e:
                        logger.error(f"Failed to save FN image for pos_id={fn_pos_id}: {e}")
                logger.info(f"path_id={path_id}: Saved {len(fn_images)} FN images to {fn_images_dir}")

            # ========== 保存 YOLO FP 图像到模型目录（用于 YOLO normal 类训练） ==========
            yolo_fp_images: List[np.ndarray] = []
            if group.yolo_false_positive_images:
                yolo_fp_dir = new_model_dir / "yolo_fp_images"
                batch_ts = int(time.time() * 1000) % 1000000
                for idx, img_b64 in enumerate(group.yolo_false_positive_images):
                    try:
                        img_data = base64.b64decode(img_b64)
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            yolo_fp_images.append(img)
                            yolo_fp_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(yolo_fp_dir / f"yolo_fp_{batch_ts}_{idx:04d}.jpg"), img)
                    except Exception as e:
                        logger.error(f"Failed to decode YOLO FP image[{idx}] for path_id={path_id}: {e}")
                logger.info(f"path_id={path_id}: Saved {len(yolo_fp_images)} YOLO FP images to {yolo_fp_dir}")

            # ========== 处理 False Positives：微调 Dinomaly ==========
            if fp_images:
                num_fp = len(fp_images)

                # FP 重训策略：基础 ROI 原样复制，FP 图做 30 张增强
                num_fp_augmentations = 30
                finetune_epochs = request.epochs or base_config.get("epochs", 12)
                logger.info(f"path_id={path_id}: Processing {num_fp} FP for retraining ({num_fp_augmentations}x augmentation, {finetune_epochs} epochs)")

                # 基础模型的 ROI 目录
                base_roi_dir = base_product_train_dir / "roi" / str(path_id)

                train_config = {
                    "model_name": "Dinomaly",
                    "project_id": request.project_id,
                    "task_uuid": new_task_uuid,
                    "path_id": path_id,
                    "train_mode": base_config.get("train_mode", "by_category"),
                    "category": base_config.get("category", ""),
                    "category_label": base_config.get("category_label", "unknown"),
                    "encoder_name": request.encoder_name or base_config.get("encoder_name", "dinov2_vit_base_14"),
                    "decoder_depth": request.decoder_depth or base_config.get("decoder_depth", 8),
                    "bottleneck_dropout": base_config.get("bottleneck_dropout", 0.2),
                    "epochs": request.epochs or base_config.get("epochs", 12),
                    "finetune_epochs": finetune_epochs,
                    "batch_size": request.batch_size or base_config.get("batch_size", 2),
                    "freeze_encoder": request.freeze_encoder if request.freeze_encoder is not None else base_config.get("freeze_encoder", True),
                    "normalize_brightness": base_config.get("normalize_brightness", False),
                    "normalize_contrast": base_config.get("normalize_contrast", False),
                    "checkpoint_path": str(base_model_path),
                    "base_roi_dir": str(base_roi_dir),
                    "num_fp_augmentations": num_fp_augmentations,
                    "augmentation_config": str(PROJECT_ROOT / "configs" / "augmentations.yaml"),
                }

                # 重训练时不需要group_annotations，ROI图片已直接保存
                path_group_id = f"retrain_{int(time.time())}_{path_id}"
                path_task_ids, _ = trainer.run_batch_training_async(
                    str(train_base_dir),
                    train_config,
                    {path_id: []},  # 空列表，因为ROI已直接保存
                    group_id=path_group_id,
                )
                all_task_ids.extend(path_task_ids)
                total_fp += num_fp
            elif fn_images or yolo_fp_images:
                # 没有 FP 数据，但存在 FN 或 YOLO FP 数据 → 创建纯 YOLO 训练任务
                logger.info(f"path_id={path_id}: No FP data, but {len(fn_images)} FN + {len(yolo_fp_images)} YOLO FP images found. Launching YOLO-only training task.")
                yolo_task_id = trainer._create_yolo_only_training_task(
                    dataset_dir=str(train_base_dir),
                    config={
                        "model_name": "Dinomaly",
                        "project_id": request.project_id,
                        "task_uuid": new_task_uuid,
                        "path_id": path_id,
                        "train_mode": base_config.get("train_mode", "by_category"),
                        "category": base_config.get("category", ""),
                        "category_label": base_config.get("category_label", "unknown"),
                        "yolo_batch": request.batch_size or base_config.get("batch_size", 8),
                    },
                    path_id=str(path_id),
                    base_model_dir=str(base_model_dir),
                )
                path_task_ids = [yolo_task_id]
                all_task_ids.append(yolo_task_id)
                path_group_id = f"yolo_only_{int(time.time())}_{path_id}"
            else:
                # 既没有 FP 也没有 FN，直接复制原模型
                logger.info(f"path_id={path_id}: No FP/FN data, copying model from base version")
                path_task_ids = []
                for file_name in ["model.ckpt", "dinomaly_model.pt", "config.json"]:
                    src_file = base_model_dir / file_name
                    if src_file.exists():
                        shutil.copy2(str(src_file), str(new_model_dir / file_name))

                # 复制 yolo_model.pt（如果基础模型有）
                yolo_src = base_model_dir / "yolo_model.pt"
                if yolo_src.exists():
                    shutil.copy2(str(yolo_src), str(new_model_dir / "yolo_model.pt"))

                # 复制 template_variants 目录
                template_variants_src = base_model_dir / "template_variants"
                if template_variants_src.exists() and template_variants_src.is_dir():
                    template_variants_dst = new_model_dir / "template_variants"
                    if template_variants_dst.exists():
                        shutil.rmtree(str(template_variants_dst))
                    shutil.copytree(str(template_variants_src), str(template_variants_dst))

            total_fn += len(fn_images)
            total_yolo_fp += len(yolo_fp_images)

            # 构建 group_ids 列表（用于前端轮询组状态）
            path_group_ids = []
            if path_task_ids:
                path_group_ids.append(path_group_id)

            results.append({
                "path_id": path_id,
                "status": "success",
                "num_fp": len(fp_images),
                "num_fn": len(fn_images),
                "num_yolo_fp": len(yolo_fp_images),
                "fp_task_ids": path_task_ids,
                "task_ids": path_task_ids,
                "group_ids": path_group_ids,
            })

        except Exception as e:
            logger.error(f"Failed to process path_id={path_id}: {e}", exc_info=True)
            results.append({
                "path_id": path_id,
                "status": "error",
                "message": str(e)
            })

    return {
        "status": "success",
        "project_id": request.project_id,
        "base_task_uuid": request.base_task_uuid,
        "new_task_uuid": new_task_uuid,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_yolo_fp": total_yolo_fp,
        "task_ids": all_task_ids,
        "results": results,
        "message": f"Retraining completed. Total path_ids: {len(results)}, FP: {total_fp}, FN: {total_fn}, YOLO_FP: {total_yolo_fp}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 项目数据集 / 模型接口
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/project/{project_id}/datasets")
async def get_project_datasets(project_id: str):
    """获取项目级历史训练数据列表"""
    base_dir = str(path_config.get_project_product_path(project_id))

    if not os.path.exists(base_dir):
        return {"project_id": project_id, "datasets": []}

    datasets = []

    for task_uuid in os.listdir(base_dir):
        uuid_path = os.path.join(base_dir, task_uuid)
        if not os.path.isdir(uuid_path):
            continue

        images_dir = os.path.join(uuid_path, "images")
        labels_file = os.path.join(uuid_path, "labels.txt")

        if os.path.exists(images_dir):
            image_files = [
                f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            labels = []
            if os.path.exists(labels_file):
                with open(labels_file) as f:
                    labels = [l.strip() for l in f if l.strip()]

            images_preview = []
            try:
                rel_path = os.path.relpath(images_dir, os.getcwd())
                for f in sorted(image_files)[:10]:
                    images_preview.append({"filename": f, "url": f"/static/{rel_path}/{f}"})
            except ValueError:
                pass

            datasets.append({
                "task_uuid": task_uuid,
                "label": labels[0] if len(labels) == 1 else "multiple",
                "labels": labels,
                "image_count": len(image_files),
                "dataset_path": uuid_path,
                "relative_path": os.path.relpath(uuid_path, os.getcwd()),
                "images": images_preview,
                "is_unified_structure": False,
            })
        else:
            for label in os.listdir(uuid_path):
                label_path = os.path.join(uuid_path, label)
                if not os.path.isdir(label_path):
                    continue
                images_dir = os.path.join(label_path, "images")
                if not os.path.exists(images_dir):
                    continue
                image_files = [
                    f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                images_preview = []
                try:
                    rel_path = os.path.relpath(images_dir, os.getcwd())
                    for f in sorted(image_files):
                        images_preview.append({"filename": f, "url": f"/static/{rel_path}/{f}"})
                except ValueError:
                    pass

                datasets.append({
                    "task_uuid": task_uuid,
                    "label": label,
                    "image_count": len(image_files),
                    "dataset_path": label_path,
                    "relative_path": os.path.relpath(label_path, os.getcwd()),
                    "images": images_preview,
                    "is_unified_structure": False,
                })

    return {"project_id": project_id, "datasets": datasets}


@app.delete("/project/{project_id}/datasets")
async def delete_project_dataset(
    project_id: str,
    task_uuid: str = Query(..., description="训练任务的UUID"),
    label: Optional[str] = Query(None, description="数据集标签（可选，不提供则删除整个任务）"),
):
    """删除项目下的数据集"""
    base_path = path_config.get_project_product_path(project_id, task_uuid)
    if label:
        dataset_path = str(base_path / label)
    else:
        dataset_path = str(base_path)

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        shutil.rmtree(dataset_path)
        if label:
            parent = os.path.dirname(dataset_path)
            if os.path.exists(parent) and not os.listdir(parent):
                shutil.rmtree(parent)
        return {
            "success": True,
            "message": f"Dataset {'label: ' + label if label else 'task: ' + task_uuid} deleted",
            "deleted_path": dataset_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}")


def _scan_model_files(label_path: str, rel_label_path: str) -> tuple:
    """扫描模型文件，返回 (model_files, has_best_model, latest_checkpoint, max_iter, model_type)"""
    model_files = []
    model_type = None

    # 检查新的 anomalib PatchCore 模型格式
    patchcore_model_path = os.path.join(label_path, "patchcore_model.pt")
    has_patchcore_model = os.path.exists(patchcore_model_path)

    # 检查旧的 memory_bank 格式（向后兼容）
    memory_bank_path = os.path.join(label_path, "memory_bank.npz")
    has_memory_bank = os.path.exists(memory_bank_path)

    # 检查 Dinomaly 模型格式
    dinomaly_model_path = os.path.join(label_path, "model.ckpt")
    has_dinomaly_model = os.path.exists(dinomaly_model_path)

    config_path = os.path.join(label_path, "config.json")
    has_config = os.path.exists(config_path)

    # 检查 Dinomaly 模型
    if has_dinomaly_model and has_config:
        model_type = "dinomaly"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_files.append({
                "name": "config.json",
                "url": f"/static/{rel_label_path}/config.json",
                "type": "config",
            })
            model_files.append({
                "name": "model.ckpt",
                "url": f"/static/{rel_label_path}/model.ckpt",
                "type": "model",
            })
            threshold_path = os.path.join(label_path, "threshold.json")
            if os.path.exists(threshold_path):
                model_files.append({
                    "name": "threshold.json",
                    "url": f"/static/{rel_label_path}/threshold.json",
                    "type": "threshold",
                })
        except Exception as e:
            logger.error(f"Error scanning Dinomaly model files: {e}")
        return model_files, True, None, -1, model_type

    if (has_patchcore_model or has_memory_bank) and has_config:
        model_type = "patchcore"
        try:
            config = {}
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_files.append({
                "name": "config.json",
                "url": f"/static/{rel_label_path}/config.json",
                "type": "config",
            })
            if has_patchcore_model:
                model_files.append({
                    "name": "patchcore_model.pt",
                    "url": f"/static/{rel_label_path}/patchcore_model.pt",
                    "type": "model",
                })
            if has_memory_bank:
                model_files.append({
                    "name": "memory_bank.npz",
                    "url": f"/static/{rel_label_path}/memory_bank.npz",
                    "type": "memory_bank",
                })
            train_data_dir = os.path.join(label_path, "training_data")
            if os.path.exists(train_data_dir):
                rel_td = os.path.join(rel_label_path, "training_data")
                for f in os.listdir(train_data_dir):
                    if os.path.isfile(os.path.join(train_data_dir, f)):
                        model_files.append({
                            "name": f"training_data/{f}",
                            "url": f"/static/{rel_td}/{f}",
                            "type": "training_data",
                        })
            train_log = os.path.join(label_path, "train.log")
            if os.path.exists(train_log):
                model_files.append({
                    "name": "train.log",
                    "url": f"/static/{rel_label_path}/train.log",
                    "type": "log",
                })
        except Exception as e:
            logger.error(f"Error scanning PatchCore model files: {e}")
        return model_files, True, None, -1, model_type

    flat_model_path = os.path.join(label_path, "model.pdparams")
    has_flat_model = os.path.exists(flat_model_path)

    best_model_path = os.path.join(label_path, "best_model")
    has_best_model_old = os.path.exists(best_model_path) and os.path.exists(
        os.path.join(best_model_path, "model.pdparams")
    )
    has_best_model = has_flat_model or has_best_model_old

    latest_checkpoint = None
    max_iter = -1
    if os.path.isdir(label_path):
        for item in os.listdir(label_path):
            if item.startswith("iter_") and os.path.isdir(os.path.join(label_path, item)):
                try:
                    it = int(item.split("_")[1])
                    if it > max_iter:
                        max_iter = it
                        latest_checkpoint = item
                except ValueError:
                    continue

    if has_best_model or latest_checkpoint:
        model_type = "stfpm"
        try:
            if has_flat_model:
                for f in os.listdir(label_path):
                    if os.path.isfile(os.path.join(label_path, f)) and f in (
                        "model.pdparams", "config.json"
                    ):
                        model_files.append({
                            "name": f,
                            "url": f"/static/{rel_label_path}/{f}",
                            "type": "flat_model",
                        })

            if has_best_model_old:
                rel_best = os.path.join(rel_label_path, "best_model")
                for f in os.listdir(best_model_path):
                    if os.path.isfile(os.path.join(best_model_path, f)):
                        model_files.append({
                            "name": f"best_model/{f}",
                            "url": f"/static/{rel_best}/{f}",
                            "type": "best_model",
                        })

            if latest_checkpoint:
                ckpt_dir = os.path.join(label_path, latest_checkpoint)
                rel_ckpt = os.path.join(rel_label_path, latest_checkpoint)
                for f in os.listdir(ckpt_dir):
                    if os.path.isfile(os.path.join(ckpt_dir, f)):
                        model_files.append({
                            "name": f"{latest_checkpoint}/{f}",
                            "url": f"/static/{rel_ckpt}/{f}",
                            "type": "checkpoint",
                        })

            train_log = os.path.join(label_path, "train.log")
            if os.path.exists(train_log):
                model_files.append({
                    "name": "train.log",
                    "url": f"/static/{rel_label_path}/train.log",
                    "type": "log",
                })

            vdl_dir = os.path.join(label_path, "vdl_log")
            if os.path.exists(vdl_dir):
                rel_vdl = os.path.join(rel_label_path, "vdl_log")
                for f in os.listdir(vdl_dir):
                    model_files.append({
                        "name": f"vdl_log/{f}",
                        "url": f"/static/{rel_vdl}/{f}",
                        "type": "vdl_log",
                    })
        except Exception as e:
            logger.error(f"Error scanning model files: {e}")

    return model_files, has_best_model, latest_checkpoint, max_iter, model_type


@app.get("/project/{project_id}/models")
async def get_project_models(project_id: str):
    """获取项目级历史模型列表"""
    output_base = str(path_config.get_project_output_path(project_id))

    if not os.path.exists(output_base):
        return {"project_id": project_id, "models": []}

    models = []

    for task_uuid in os.listdir(output_base):
        uuid_path = os.path.join(output_base, task_uuid)
        if not os.path.isdir(uuid_path):
            continue

        has_label_dirs = any(
            os.path.isdir(os.path.join(uuid_path, item)) for item in os.listdir(uuid_path)
        )

        if has_label_dirs:
            for label in os.listdir(uuid_path):
                label_path = os.path.join(uuid_path, label)
                if not os.path.isdir(label_path):
                    continue
                rel_label_path = os.path.relpath(label_path, os.getcwd())
                model_files, has_best, latest_ckpt, max_iter, model_type = _scan_model_files(
                    label_path, rel_label_path
                )
                if model_files:
                    models.append({
                        "task_uuid": task_uuid,
                        "label": label,
                        "model_type": model_type,
                        "has_best_model": has_best,
                        "latest_checkpoint": latest_ckpt,
                        "latest_iter": max_iter,
                        "model_path": label_path,
                        "relative_path": rel_label_path,
                        "files": model_files,
                        "is_unified_structure": False,
                    })
        else:
            rel_uuid_path = os.path.relpath(uuid_path, os.getcwd())
            model_files, has_best, latest_ckpt, max_iter, model_type = _scan_model_files(
                uuid_path, rel_uuid_path
            )
            if model_files:
                models.append({
                    "task_uuid": task_uuid,
                    "label": task_uuid,
                    "model_type": model_type,
                    "has_best_model": has_best,
                    "latest_checkpoint": latest_ckpt,
                    "latest_iter": max_iter,
                    "model_path": uuid_path,
                    "relative_path": rel_uuid_path,
                    "files": model_files,
                    "is_unified_structure": True,
                })

    return {"project_id": project_id, "models": models}


@app.delete("/project/{project_id}/models")
async def delete_project_model(
    project_id: str,
    task_uuid: str = Query(..., description="训练任务的UUID"),
    label: Optional[str] = Query(None, description="模型标签（可选，不提供则删除整个任务）"),
):
    """删除项目下的模型"""
    base_path = path_config.get_project_output_path(project_id, task_uuid)
    if label:
        model_path = str(base_path / label)
    else:
        model_path = str(base_path)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        shutil.rmtree(model_path)
        if label:
            parent = os.path.dirname(model_path)
            if os.path.exists(parent) and not os.listdir(parent):
                shutil.rmtree(parent)
        return {
            "success": True,
            "message": f"Model {'label: ' + label if label else 'task: ' + task_uuid} deleted",
            "deleted_path": model_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 部署接口
# ─────────────────────────────────────────────────────────────────────────────

http_deployer = ModelDeployer()  # 使用环境变量配置的output和scripts路径


class HTTPDeployRequest(BaseModel):
    project_id: str
    task_uuid: str
    device: str = "GPU"
    port: Optional[int] = None


class HTTPDeployResponse(BaseModel):
    success: bool
    service_id: Optional[str] = None
    port: Optional[int] = None
    http_url: Optional[str] = None
    message: str


@app.post("/deploy/http", response_model=HTTPDeployResponse)
async def deploy_http_service(request: HTTPDeployRequest, host: str = Header(None)):
    try:
        result = http_deployer.deploy_service(
            project_id=request.project_id,
            task_uuid=request.task_uuid,
            port=request.port,
        )
        if result.get("status") in {"already_exists", "success"}:
            server_host = host.split(":")[0] if host else "localhost"
            return HTTPDeployResponse(
                success=True,
                service_id=result.get("service_id"),
                port=result.get("port"),
                http_url=f"http://{server_host}:{result.get('port')}",
                message=result.get("message", "服务部署成功"),
            )
        error_msg = result.get("message") or result.get("error") or "部署失败"
        logger.error(f"服务部署失败: {error_msg}, result={result}")
        return HTTPDeployResponse(success=False, message=error_msg)
    except Exception as e:
        logger.error(f"服务部署异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/http/services")
async def list_http_services(project_id: Optional[str] = None, include_health: bool = False):
    try:
        services = http_deployer.list_services(project_id, include_health=include_health)
        return {"success": True, "services": services, "count": len(services)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/http/service/{service_id}/health")
async def check_service_health(service_id: str):
    try:
        return {"success": True, "health": http_deployer.check_service_health(service_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/http/service/{service_id}")
async def get_http_service(service_id: str):
    try:
        service = http_deployer.get_service(service_id)
        if service:
            return {"success": True, "service": service}
        raise HTTPException(status_code=404, detail="服务不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deploy/http/service/{service_id}/stop")
async def stop_http_service(service_id: str):
    try:
        result = http_deployer.stop_service(service_id)
        return {"success": True, "message": f"服务已停止: {service_id}", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/deploy/http/service/{service_id}")
async def delete_http_service(service_id: str):
    try:
        result = http_deployer.delete_service(service_id)
        return {"success": True, "message": f"服务已删除: {service_id}", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/http/models/{project_id}")
async def list_available_models(project_id: str):
    try:
        models = http_deployer.get_available_models(project_id)
        return {"success": True, "project_id": project_id, "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/http/service/{service_id}/logs")
async def get_service_logs(
    service_id: str,
    lines: int = Query(100, ge=1, le=5000),
    from_line: int = Query(-1, ge=-1),
):
    try:
        result = http_deployer.get_service_logs(service_id, lines=lines, from_line=from_line)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 根路由
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Welcome to One2All API",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)