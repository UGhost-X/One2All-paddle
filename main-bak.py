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
import os
import shutil
import tempfile
import random
import platform
import warnings
import uuid as uuid_lib
from pathlib import Path
import uvicorn

# 屏蔽框架无关紧要的日志和警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*ccache.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from utils.patchcore_trainer import PatchCoreTrainer
from utils.deployer import ModelDeployer
from dataclasses import asdict

logger = logging.getLogger(__name__)

# 初始化 PatchCore 训练器
trainer = PatchCoreTrainer(output_dir="output", max_concurrent=3)

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
    model_name: str = "PatchCore"
    label_names: Optional[List[str]] = None
    parallel_train: bool = False

    backbone: str = "resnet18"
    layers: List[str] = ["layer2", "layer3"]
    num_neighbors: int = 9
    augment: bool = True
    num_augmentations: int = 100
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

    if platform.system().lower() == "linux":
        normalized_base = os.getcwd()
    else:
        normalized_base = request.base_path.replace("\\", "/")

    storage_base = Path(normalized_base) / "product" / request.project_id / "train" / task_uuid

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
        pos_ids_in_request: set = set()
        for ann in request.coco_data.annotations:
            if ann.pos_id is not None:
                pos_ids_in_request.add(ann.pos_id)

        use_pos_id = len(pos_ids_in_request) > 0


        group_id = f"group_{int(time.time())}_{request.project_id}"

        groups_for_trainer: Dict[Any, List[Dict]] = defaultdict(list)
        for ann in annotations_data["annotations"]:
            key = ann.get("pos_id") if use_pos_id else ann.get("label", "unknown")
            groups_for_trainer[key].append(ann)

        # 过滤掉 label 为 "工件主体" 的 group
        excluded_label = "工件主体"
        filtered_groups = {}
        for grp_id, annotations in groups_for_trainer.items():
            # 检查该 group 是否全是 "工件主体"
            non_excluded = [ann for ann in annotations if ann.get("label") != excluded_label]
            if non_excluded:
                filtered_groups[grp_id] = non_excluded
            else:
                logger.info(f"Skipping training for excluded label '{excluded_label}' (group: {grp_id})")

        base_train_config = {
            "model_name": request.model_name,
            "use_pos_id": use_pos_id,
            "project_id": request.project_id,
            "task_uuid": task_uuid,
            "parallel_train": request.parallel_train,
            "backbone": request.backbone,
            "layers": request.layers,
            "num_neighbors": request.num_neighbors,
            "augment": request.augment,
            "num_augmentations": request.num_augmentations,
            "normalize_brightness": request.normalize_brightness,
            "normalize_contrast": request.normalize_contrast,
            "threshold_buffer": request.threshold_buffer,
            "save_images": request.save_images,
            "max_concurrent": request.max_concurrent,
        }

        t0 = time.time()
        all_task_ids = trainer.run_batch_training_async(
            str(storage_base),
            base_train_config,
            filtered_groups,
            group_id=group_id,
        )
        logger.info(
            f"[TrainAnomaly] {len(all_task_ids)} tasks launched in {time.time() - t0:.3f}s"
        )

        # 构建返回结果（保持 pos_id → task_id 的对应关系）
        sorted_keys = sorted(filtered_groups.keys(), key=str)
        task_results = []
        for pid, tid in zip(sorted_keys, all_task_ids):
            group_annotations = filtered_groups.get(pid, [])
            first_ann = group_annotations[0] if group_annotations else {}
            label = first_ann.get("label", str(pid))
            task_results.append({
                "pos_id": pid,
                "task_id": tid,
                "label": label
            })

        pos_ids_processed = sorted(filtered_groups.keys(), key=str)

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
async def train_events(task_id: str):
    """SSE 实时推送训练进度和日志"""
    async def event_generator():
        last_log_idx = 0
        while True:
            status = trainer.get_training_status(task_id)
            if status is None:
                status = {"status": "not_found", "progress": 0, "logs": [], "metrics": [], "eval_metrics": []}

            data = {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "label": status.get("label"),
                "stage": status.get("stage"),
                "threshold": status.get("threshold"),
                "num_samples": status.get("num_samples"),
                "new_logs": [],
                "metrics": status.get("metrics", []),
                "eval_metrics": status.get("eval_metrics", []),
            }

            logs = status.get("logs", [])
            if len(logs) > last_log_idx:
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
# 项目数据集 / 模型接口
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/project/{project_id}/datasets")
async def get_project_datasets(project_id: str):
    """获取项目级历史训练数据列表"""
    base_dir = os.path.join(os.getcwd(), "product", project_id, "train")

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
    dataset_path = os.path.join(
        os.getcwd(), "product", project_id, "train", task_uuid, *([label] if label else [])
    )

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

    config_path = os.path.join(label_path, "config.json")
    has_config = os.path.exists(config_path)

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
    output_base = os.path.join(os.getcwd(), "output", project_id)

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
    model_path = os.path.join(
        os.getcwd(), "output", project_id, task_uuid, *([label] if label else [])
    )

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

http_deployer = ModelDeployer(output_dir="output", scripts_dir="inference_services")


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