from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import time
import logging
import paddle
import cv2
import numpy as np
import base64
import os
import shutil
import tempfile
import random
import platform
import warnings
import logging
import uuid as uuid_lib
from pathlib import Path
import uvicorn

# 屏蔽框架无关紧要的日志和警告
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
warnings.filterwarnings("ignore", category=UserWarning, message=".*ccache.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("paddle").setLevel(logging.WARNING)
logging.getLogger("paddlex").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from utils.augmentation import DataAugmentor
from utils.trainer import trainer
from utils.onnx_converter import ONNXConverter, convert_paddle_to_onnx, check_paddle2onnx_available
from dataclasses import asdict

logger = logging.getLogger(__name__)

app = FastAPI(title="One2All Paddle API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有请求头
)

app.mount("/static", StaticFiles(directory=os.getcwd()), name="static")

# 应用启动时自动加载模板
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    try:
        from utils.template_matcher import template_matcher
        loaded_count = template_matcher.auto_load_templates()
        logger.info(f"Application started. Loaded {loaded_count} templates.")
    except Exception as e:
        logger.warning(f"Failed to auto-load templates: {e}")

class COCOAnnotation(BaseModel):
    id: Optional[Any] = None
    image_id: Optional[Any] = None
    category_id: Optional[Any] = 0
    bbox: Optional[List[float]] = None # [x, y, width, height]
    points: Optional[List[float]] = None # 兼容一些前端使用的 points 字段
    segmentation: Optional[List[List[float]]] = None
    area: Optional[float] = None
    iscrowd: Optional[int] = 0
    label: Optional[str] = None
    type: Optional[str] = None

class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: Optional[str] = None

class COCOImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str

class COCOData(BaseModel):
    images: Optional[List[COCOImage]] = None
    annotations: List[COCOAnnotation]
    categories: Optional[List[COCOCategory]] = None

class AugmentationConfig(BaseModel):
    horizontal_flip: Optional[Dict[str, Any]] = None
    vertical_flip: Optional[Dict[str, Any]] = None
    rotate: Optional[Dict[str, Any]] = None
    brightness: Optional[Dict[str, Any]] = None
    contrast: Optional[Dict[str, Any]] = None
    blur: Optional[Dict[str, Any]] = None
    pitch: Optional[Dict[str, Any]] = None # 俯视/仰视 (Pitch)
    yaw: Optional[Dict[str, Any]] = None # 侧视 (Yaw)

class AugmentRequest(BaseModel):
    image_base64: str  # 输入图片的 base64 编码
    coco_data: COCOData
    config: AugmentationConfig
    num_results: Optional[int] = 1 # 默认为 1，如果大于 1 则按梯度生成图片集

class TrainRequest(BaseModel):
    images: List[str]  # Base64 编码的图片列表
    coco_data: COCOData # 对应的 COCO 标注数据
    base_path: str # 基础路径，例如 "/data/projects"
    project_id: str # 项目 ID
    model_name: str = "STFPM"
    train_epochs: Optional[int] = None # 建议使用 train_iters，若提供则自动换算
    train_iters: Optional[int] = None  # 推荐：显式指定训练迭代次数
    batch_size: int = 8
    learning_rate: float = 0.01
    label_names: Optional[List[str]] = None
    resume_path: Optional[str] = None # 可选的恢复训练路径
    resume_mode: Optional[str] = "interrupted" # 续训模式: "interrupted" (中断续训) 或 "extended" (完结续训)
    parallel_train: bool = False # 是否开启多线程并行训练（默认为串行排队）
    template_name: Optional[str] = None # 模板名称，用于工件对齐（可选）
    template_image: Optional[str] = None # 模板图片base64，如果提供则保存为新模板

class TemplateRequest(BaseModel):
    name: str  # 模板名称
    image_base64: str  # 模板图片base64

class TemplateResponse(BaseModel):
    success: bool
    message: str
    template_name: Optional[str] = None
    template_info: Optional[Dict[str, Any]] = None

def generate_position_label(bbox: List[float], img_width: int, img_height: int, 
                            base_label: str, grid_size: int = 3) -> str:
    """
    根据 bbox 中心点位置生成位置感知的 label
    例如: pos_1_2_screw 表示在网格位置(1,2)的螺丝
    
    Args:
        bbox: [x, y, width, height]
        img_width: 图片宽度
        img_height: 图片高度
        base_label: 基础类别名称（如 screw, hole）
        grid_size: 网格大小（默认3x3）
    """
    x, y, bw, bh = bbox
    center_x = x + bw / 2
    center_y = y + bh / 2
    
    # 计算网格位置
    grid_x = min(int(center_x / img_width * grid_size), grid_size - 1)
    grid_y = min(int(center_y / img_height * grid_size), grid_size - 1)
    
    # 生成位置感知的 label
    return f"pos_{grid_x}_{grid_y}_{base_label}"


@app.post("/train/anomaly")
async def train_anomaly(request: TrainRequest):
    """
    接收 COCO 数据，按照位置+类型组合保存裁剪信息并启动训练
    层级结构: {base_path}/{project_id}/train/{uuid}/
    输出路径: output/{project_id}/{uuid}/
    
    修改：改为单模型多类别训练，类别包含位置信息（如 pos_0_1_screw）
    """
    task_uuid = uuid_lib.uuid4().hex[:8]
    
    cat_map = {cat.id: cat.name for cat in request.coco_data.categories} if request.coco_data.categories else {}
    
    if platform.system().lower() == "linux":
        normalized_base = os.getcwd()
    else:
        normalized_base = request.base_path.replace("\\", "/")
    
    # 修改：所有数据放在同一个目录下，按类别子目录组织
    storage_base = Path(normalized_base) / request.project_id / "train" / task_uuid
    
    try:
        # 导入模板匹配模块
        from utils.template_matcher import template_matcher

        # 2. 处理模板（如果提供了模板图片）
        template_name = request.template_name
        if request.template_image:
            # 保存新模板
            template_data = base64.b64decode(request.template_image)
            template_nparr = np.frombuffer(template_data, np.uint8)
            template_img = cv2.imdecode(template_nparr, cv2.IMREAD_COLOR)

            if template_img is None:
                raise HTTPException(status_code=400, detail="Invalid template image")

            # 使用提供的名称或生成默认名称
            template_name = template_name or f"template_{request.project_id}_{task_uuid}"

            # 保存模板
            if not template_matcher.save_template_to_disk(template_name, template_img):
                raise HTTPException(status_code=500, detail="Failed to save template")

            if not template_matcher.load_template(template_name, template_img):
                raise HTTPException(status_code=500, detail="Failed to load template")

            logger.info(f"Template '{template_name}' saved and loaded")

        # 3. 解码所有图片并进行模板匹配对齐
        decoded_images = {}
        image_sizes = {}  # 记录每张图片的尺寸
        alignment_results = {}  # 记录对齐结果

        for idx, img_b64 in enumerate(request.images):
            try:
                img_data = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # 优先从 images 列表中获取 id，如果没有则使用索引+1
                    if request.coco_data.images and idx < len(request.coco_data.images):
                        image_id = request.coco_data.images[idx].id
                    else:
                        image_id = idx + 1

                    # 如果指定了模板，进行对齐
                    if template_name and template_name in template_matcher.templates:
                        match_result = template_matcher.match_and_align(img, template_name)
                        if match_result.success:
                            decoded_images[image_id] = match_result.aligned_image
                            image_sizes[image_id] = (match_result.aligned_image.shape[1], match_result.aligned_image.shape[0])
                            alignment_results[image_id] = {
                                "success": True,
                                "match_score": match_result.match_score
                            }
                            logger.info(f"Image {image_id} aligned with template '{template_name}', score: {match_result.match_score:.3f}")
                        else:
                            # 对齐失败，使用原图但记录警告
                            decoded_images[image_id] = img
                            image_sizes[image_id] = (img.shape[1], img.shape[0])
                            alignment_results[image_id] = {
                                "success": False,
                                "error": match_result.error_message
                            }
                            logger.warning(f"Image {image_id} alignment failed: {match_result.error_message}")
                    else:
                        # 没有模板，使用原图
                        decoded_images[image_id] = img
                        image_sizes[image_id] = (img.shape[1], img.shape[0])

            except Exception as e:
                logger.error(f"Failed to decode/align image: {e}")

        # 3. 裁剪并按位置+类型组合分类存放
        crop_count = 0
        labels_processed = set()
        
        # 创建统一的数据集目录结构
        img_dir = storage_base / "images"
        mask_dir = storage_base / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 清理已存在的列表文件
        for f_name in ["train.txt", "val.txt", "labels.txt"]:
            f_path = storage_base / f_name
            if f_path.exists():
                f_path.unlink()
        
        for ann in request.coco_data.annotations:
            # 获取基础类别名称
            base_label = ann.label or cat_map.get(ann.category_id, f"class_{ann.category_id}")
            
            # 过滤标签：如果指定了 label_names 且当前标签不在其中，则跳过
            if request.label_names and base_label not in request.label_names:
                continue
                
            image_id = ann.image_id
            if image_id in decoded_images:
                img = decoded_images[image_id]
                img_w, img_h = image_sizes[image_id]
                h, w = img.shape[:2]
                
                bbox = ann.bbox or (ann.points[:4] if ann.points else None)
                if bbox:
                    x, y, bw, bh = map(int, bbox)
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                    
                    if x2 > x1 and y2 > y1:
                        roi = img[y1:y2, x1:x2]
                        
                        # 生成位置感知的 label
                        position_label = generate_position_label(
                            [x, y, bw, bh], img_w, img_h, base_label
                        )
                        
                        img_filename = f"{position_label}_{image_id}_{crop_count}.png"
                        mask_filename = f"{position_label}_{image_id}_{crop_count}_mask.png"
                        
                        img_path = img_dir / img_filename
                        mask_path = mask_dir / mask_filename
                        
                        # 保存图片
                        cv2.imwrite(str(img_path), roi)
                        
                        # 保存全黑掩码 (对于训练集的 "good" 样本，掩码全为 0)
                        mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                        cv2.imwrite(str(mask_path), mask)
                        
                        # 记录到列表
                        train_list_path = storage_base / "train.txt"
                        val_list_path = storage_base / "val.txt"
                        
                        # 统计当前已有的训练样本数
                        train_count = 0
                        if train_list_path.exists():
                            with open(train_list_path, "r") as f:
                                train_count = sum(1 for line in f if line.strip())
                        
                        if train_count < 2:
                            # 强制放入训练集
                            with open(train_list_path, "a") as f:
                                f.write(f"images/{img_filename} masks/{mask_filename}\n")
                        elif random.random() < 0.1:
                            # 放入验证集
                            with open(val_list_path, "a") as f:
                                f.write(f"images/{img_filename} masks/{mask_filename}\n")
                        else:
                            # 放入训练集
                            with open(train_list_path, "a") as f:
                                f.write(f"images/{img_filename} masks/{mask_filename}\n")
                            
                        crop_count += 1
                        labels_processed.add(position_label)

        if crop_count == 0:
            raise HTTPException(status_code=400, detail="No valid objects to crop")

        # 4. 后处理：确保验证集不为空
        val_list_path = storage_base / "val.txt"
        train_list_path = storage_base / "train.txt"
        if not val_list_path.exists() or os.path.getsize(val_list_path) == 0:
            if train_list_path.exists() and os.path.getsize(train_list_path) > 0:
                with open(train_list_path, "r") as f_train:
                    first_line = f_train.readline()
                if first_line:
                    with open(val_list_path, "w") as f_val:
                        f_val.write(first_line)
                    logger.info("Validation set was empty. Copied one sample from train set.")

        # 5. 保存类别列表
        labels_file = storage_base / "labels.txt"
        with open(labels_file, "w") as f:
            for label in sorted(labels_processed):
                f.write(f"{label}\n")

        # 6. 启动单个训练任务（所有类别一个模型）
        group_id = f"group_{int(time.time())}_{request.project_id}"
        
        train_config = {
            "model_name": request.model_name,
            "label_name": "multi_position",  # 统一使用一个名称
            "labels": sorted(list(labels_processed)),  # 传递所有类别
            "train_epochs": request.train_epochs,
            "train_iters": request.train_iters,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "project_id": request.project_id,
            "task_uuid": task_uuid,
            "resume_path": request.resume_path,
            "resume_mode": request.resume_mode,
            "parallel_train": request.parallel_train
        }
        
        task_id = trainer.run_training_async(str(storage_base), train_config, group_id=group_id)

        # 保存对齐结果
        if alignment_results:
            alignment_file = storage_base / "alignment_results.json"
            import json
            with open(alignment_file, "w") as f:
                json.dump(alignment_results, f, indent=2)

        return {
            "status": "success",
            "project_id": request.project_id,
            "task_uuid": task_uuid,
            "group_id": group_id,
            "storage_path": str(storage_base),
            "total_crops": crop_count,
            "labels": sorted(list(labels_processed)),
            "task_id": task_id,
            "template_name": template_name,
            "alignment_results": alignment_results
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/templates", response_model=TemplateResponse)
async def create_template(request: TemplateRequest):
    """
    创建/上传模板图片
    """
    try:
        from utils.template_matcher import template_matcher

        # 解码图片
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return TemplateResponse(
                success=False,
                message="Invalid image data",
                template_name=request.name
            )

        # 保存模板
        if not template_matcher.save_template_to_disk(request.name, image):
            return TemplateResponse(
                success=False,
                message="Failed to save template to disk",
                template_name=request.name
            )

        # 加载模板
        if not template_matcher.load_template(request.name, image):
            return TemplateResponse(
                success=False,
                message="Failed to load template (possibly not enough keypoints)",
                template_name=request.name
            )

        template_info = template_matcher.get_template_info(request.name)

        return TemplateResponse(
            success=True,
            message="Template created successfully",
            template_name=request.name,
            template_info=template_info
        )

    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        return TemplateResponse(
            success=False,
            message=str(e),
            template_name=request.name
        )


@app.get("/templates")
async def list_templates():
    """
    列出所有模板
    """
    try:
        from utils.template_matcher import template_matcher
        templates = template_matcher.list_templates()
        return {
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates/{template_name}")
async def get_template_info(template_name: str):
    """
    获取模板详细信息
    """
    try:
        from utils.template_matcher import template_matcher

        # 如果模板未加载，尝试从磁盘加载
        if template_name not in template_matcher.templates:
            if not template_matcher.load_template_from_disk(template_name):
                raise HTTPException(status_code=404, detail="Template not found")

        info = template_matcher.get_template_info(template_name)
        if info is None:
            raise HTTPException(status_code=404, detail="Template not found")

        return info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/templates/{template_name}")
async def delete_template(template_name: str):
    """
    删除模板
    """
    try:
        from utils.template_matcher import template_matcher

        if template_matcher.delete_template(template_name):
            return {
                "success": True,
                "message": f"Template '{template_name}' deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete template")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/templates/{template_name}/match")
async def match_template(template_name: str, image_base64: str = Body(..., embed=True)):
    """
    测试模板匹配
    """
    try:
        from utils.template_matcher import template_matcher

        # 如果模板未加载，尝试从磁盘加载
        if template_name not in template_matcher.templates:
            if not template_matcher.load_template_from_disk(template_name):
                raise HTTPException(status_code=404, detail="Template not found")

        # 解码图片
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 执行匹配
        result = template_matcher.match_and_align(image, template_name)

        if result.success:
            # 编码对齐后的图片
            _, buffer = cv2.imencode('.png', result.aligned_image)
            aligned_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "match_score": result.match_score,
                "aligned_image": aligned_image_base64
            }
        else:
            return {
                "success": False,
                "error": result.error_message
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/status/{task_id}")
async def get_train_status(task_id: str):
    """
    查询单个训练任务状态 (单个 label)
    """
    status = trainer.get_status(task_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@app.get("/project/{project_id}/datasets")
async def get_project_datasets(project_id: str):
    """
    获取项目级历史训练数据列表。
    返回该项目下所有已存在的训练数据快照（基于目录结构）。
    路径结构: {project_id}/train/{uuid}/ (新的统一存储结构)
    """
    base_dir = os.path.join(os.getcwd(), project_id, "train")

    if not os.path.exists(base_dir):
        return {"project_id": project_id, "datasets": []}

    datasets = []

    for task_uuid in os.listdir(base_dir):
        uuid_path = os.path.join(base_dir, task_uuid)
        if not os.path.isdir(uuid_path):
            continue

        # 新的统一存储结构: {uuid}/images/, {uuid}/masks/
        images_dir = os.path.join(uuid_path, "images")
        labels_file = os.path.join(uuid_path, "labels.txt")

        if os.path.exists(images_dir):
            # 新的统一结构
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_count = len(image_files)

            # 读取类别列表
            labels = []
            if os.path.exists(labels_file):
                with open(labels_file, "r") as f:
                    labels = [line.strip() for line in f if line.strip()]

            images_preview = []
            try:
                rel_images_path = os.path.relpath(images_dir, os.getcwd())
                for f in sorted(image_files)[:10]:  # 只预览前10张
                    images_preview.append({
                        "filename": f,
                        "url": f"/static/{rel_images_path}/{f}"
                    })
            except ValueError:
                pass

            datasets.append({
                "task_uuid": task_uuid,
                "label": "multi_position",  # 统一标记
                "labels": labels,  # 所有位置-类别组合
                "image_count": image_count,
                "dataset_path": uuid_path,
                "relative_path": os.path.relpath(uuid_path, os.getcwd()),
                "images": images_preview,
                "is_unified_structure": True
            })
        else:
            # 兼容旧结构: {uuid}/{label}/
            for label in os.listdir(uuid_path):
                label_path = os.path.join(uuid_path, label)
                if not os.path.isdir(label_path):
                    continue

                images_dir = os.path.join(label_path, "images")
                if not os.path.exists(images_dir):
                    continue

                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                image_count = len(image_files)

                images_preview = []
                try:
                    rel_images_path = os.path.relpath(images_dir, os.getcwd())
                    for f in sorted(image_files):
                        images_preview.append({
                            "filename": f,
                            "url": f"/static/{rel_images_path}/{f}"
                        })
                except ValueError:
                    pass

                datasets.append({
                    "task_uuid": task_uuid,
                    "label": label,
                    "image_count": image_count,
                    "dataset_path": label_path,
                    "relative_path": os.path.relpath(label_path, os.getcwd()),
                    "images": images_preview,
                    "is_unified_structure": False
                })

    return {"project_id": project_id, "datasets": datasets}

@app.delete("/project/{project_id}/datasets")
async def delete_project_dataset(
    project_id: str,
    task_uuid: str = Query(..., description="训练任务的UUID"),
    label: Optional[str] = Query(None, description="数据集标签（可选，不提供则删除整个任务）")
):
    """
    删除项目下的数据集。
    - 如果提供 task_uuid 和 label: 删除 {project_id}/train/{task_uuid}/{label}
    - 如果只提供 task_uuid: 删除 {project_id}/train/{task_uuid} 整个目录
    """
    if label:
        dataset_path = os.path.join(os.getcwd(), project_id, "train", task_uuid, label)
    else:
        dataset_path = os.path.join(os.getcwd(), project_id, "train", task_uuid)
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        shutil.rmtree(dataset_path)
        
        if label:
            parent_dir = os.path.dirname(dataset_path)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                shutil.rmtree(parent_dir)
        
        return {
            "success": True,
            "message": f"Dataset {'label: ' + label if label else 'task: ' + task_uuid} deleted successfully",
            "deleted_path": dataset_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

def _scan_model_files(label_path: str, rel_label_path: str) -> tuple:
    """扫描模型文件，返回 (model_files, has_best_model, latest_checkpoint, max_iter)"""
    model_files = []
    best_model_path = os.path.join(label_path, "best_model")
    has_best_model = os.path.exists(best_model_path) and os.path.exists(os.path.join(best_model_path, "model.pdparams"))

    latest_checkpoint = None
    max_iter = -1
    for item in os.listdir(label_path) if os.path.isdir(label_path) else []:
        if item.startswith("iter_") and os.path.isdir(os.path.join(label_path, item)):
            try:
                it = int(item.split("_")[1])
                if it > max_iter:
                    max_iter = it
                    latest_checkpoint = item
            except ValueError:
                continue

    if has_best_model or latest_checkpoint:
        try:
            if has_best_model:
                rel_best_path = os.path.join(rel_label_path, "best_model")
                for f in os.listdir(best_model_path):
                     if os.path.isfile(os.path.join(best_model_path, f)):
                        model_files.append({
                            "name": f"best_model/{f}",
                            "url": f"/static/{rel_best_path}/{f}",
                            "type": "best_model"
                        })

            if latest_checkpoint:
                 checkpoint_dir = os.path.join(label_path, latest_checkpoint)
                 rel_ckpt_path = os.path.join(rel_label_path, latest_checkpoint)
                 for f in os.listdir(checkpoint_dir):
                     if os.path.isfile(os.path.join(checkpoint_dir, f)):
                         model_files.append({
                             "name": f"{latest_checkpoint}/{f}",
                             "url": f"/static/{rel_ckpt_path}/{f}",
                             "type": "checkpoint"
                         })

            train_log = os.path.join(label_path, "train.log")
            if os.path.exists(train_log):
                model_files.append({
                    "name": "train.log",
                    "url": f"/static/{rel_label_path}/train.log",
                    "type": "log"
                })

            vdl_dir = os.path.join(label_path, "vdl_log")
            if os.path.exists(vdl_dir):
                 rel_vdl_path = os.path.join(rel_label_path, "vdl_log")
                 for f in os.listdir(vdl_dir):
                     model_files.append({
                         "name": f"vdl_log/{f}",
                         "url": f"/static/{rel_vdl_path}/{f}",
                         "type": "vdl_log"
                     })
        except ValueError:
            pass

    return model_files, has_best_model, latest_checkpoint, max_iter


@app.get("/project/{project_id}/models")
async def get_project_models(project_id: str):
    """
    获取项目级历史模型列表。
    遍历 output/project_id 下的所有任务，收集已完成的模型。
    路径结构: output/{project_id}/{uuid}/ (新的统一结构)
    """
    output_base = os.path.join(os.getcwd(), "output", project_id)

    if not os.path.exists(output_base):
        return {"project_id": project_id, "models": []}

    models = []

    for task_uuid in os.listdir(output_base):
        uuid_path = os.path.join(output_base, task_uuid)
        if not os.path.isdir(uuid_path):
            continue

        # 检查是否是新的统一结构
        labels_file = os.path.join(uuid_path, "labels.txt")
        is_unified = os.path.exists(labels_file)

        if is_unified:
            # 新的统一结构: 只有一个模型，但包含多个类别
            rel_uuid_path = os.path.relpath(uuid_path, os.getcwd())
            model_files, has_best_model, latest_checkpoint, max_iter = _scan_model_files(uuid_path, rel_uuid_path)

            labels = []
            with open(labels_file, "r") as f:
                labels = [line.strip() for line in f if line.strip()]

            if model_files:
                models.append({
                    "task_uuid": task_uuid,
                    "label": "multi_position",
                    "labels": labels,
                    "has_best_model": has_best_model,
                    "latest_checkpoint": latest_checkpoint,
                    "latest_iter": max_iter,
                    "model_path": uuid_path,
                    "relative_path": rel_uuid_path,
                    "files": model_files,
                    "is_unified_structure": True
                })
        else:
            # 旧结构: {uuid}/{label}/
            for label in os.listdir(uuid_path):
                label_path = os.path.join(uuid_path, label)
                if not os.path.isdir(label_path):
                    continue

                rel_label_path = os.path.relpath(label_path, os.getcwd())
                model_files, has_best_model, latest_checkpoint, max_iter = _scan_model_files(label_path, rel_label_path)

                if model_files:
                    models.append({
                        "task_uuid": task_uuid,
                        "label": label,
                        "has_best_model": has_best_model,
                        "latest_checkpoint": latest_checkpoint,
                        "latest_iter": max_iter,
                        "model_path": label_path,
                        "relative_path": rel_label_path,
                        "files": model_files,
                        "is_unified_structure": False
                    })

    return {"project_id": project_id, "models": models}

@app.delete("/project/{project_id}/models")
async def delete_project_model(
    project_id: str,
    task_uuid: str = Query(..., description="训练任务的UUID"),
    label: Optional[str] = Query(None, description="模型标签（可选，不提供则删除整个任务）")
):
    """
    删除项目下的模型。
    - 如果提供 task_uuid 和 label: 删除 output/{project_id}/{task_uuid}/{label}
    - 如果只提供 task_uuid: 删除 output/{project_id}/{task_uuid} 整个目录
    """
    if label:
        model_path = os.path.join(os.getcwd(), "output", project_id, task_uuid, label)
    else:
        model_path = os.path.join(os.getcwd(), "output", project_id, task_uuid)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        shutil.rmtree(model_path)
        
        if label:
            parent_dir = os.path.dirname(model_path)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                shutil.rmtree(parent_dir)
        
        return {
            "success": True,
            "message": f"Model {'label: ' + label if label else 'task: ' + task_uuid} deleted successfully",
            "deleted_path": model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@app.get("/train/data/{task_id}")
async def get_train_data(task_id: str):
    """
    获取某个训练任务所使用的图片列表及访问 URL
    """
    status = trainer.get_status(task_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    
    dataset_dir = status.get("dataset_dir")
    if not dataset_dir or not os.path.exists(dataset_dir):
        raise HTTPException(status_code=404, detail="Dataset directory not found")
    
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        return {"task_id": task_id, "images": []}
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 构建相对于静态目录的路径
    cwd = os.getcwd()
    try:
        rel_dataset_path = os.path.relpath(images_dir, cwd)
    except ValueError:
        # 如果不在同一个驱动器或无法计算相对路径
        raise HTTPException(status_code=500, detail="Cannot calculate relative path for images")

    result = []
    for f in image_files:
        result.append({
            "name": f,
            "url": f"/static/{rel_dataset_path}/{f}".replace("\\", "/")
        })
    
    return {
        "task_id": task_id,
        "label": status.get("label"),
        "total": len(result),
        "images": result
    }

@app.get("/train/status/group/{group_id}")
async def get_group_train_status(group_id: str):
    """
    查询任务组状态 (总进度)
    """
    status = trainer.get_group_status(group_id)
    return status

@app.post("/train/stop/{task_id}")
async def stop_train_task(task_id: str):
    """
    停止单个训练任务
    """
    result = trainer.stop_task(task_id)
    return result

@app.post("/train/stop/group/{group_id}")
async def stop_group_train(group_id: str):
    """
    停止整个任务组训练
    """
    result = trainer.stop_group(group_id)
    return result

@app.get("/train/events/{task_id}")
async def train_events(task_id: str):
    """
    SSE 实时推送训练进度和日志
    状态从 trainer 内存中直接获取，无需 ZeroMQ
    """
    async def event_generator():
        last_log_idx = 0
        while True:
            # 从 trainer 获取当前状态（内存）
            status = trainer.get_status(task_id)
            
            data = {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "label": status.get("label"),
                "new_logs": [],
                "metrics": status.get("metrics", []),
                "eval_metrics": status.get("eval_metrics", [])
            }
            
            logs = status.get("logs", [])
            if len(logs) > last_log_idx:
                data["new_logs"] = logs[last_log_idx:]
                last_log_idx = len(logs)
            
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            
            if status.get("status") in ["completed", "failed", "not_found", "cancelled", "interrupted"]:
                break
                
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/train/checkpoints/{task_id}")
async def get_task_checkpoints(task_id: str):
    """
    获取某个任务下可用的检查点列表
    """
    status = trainer.get_status(task_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    
    save_dir = status.get("save_dir")
    if not save_dir or not os.path.exists(save_dir):
        return {"checkpoints": []}
    
    checkpoints = []
    for root, _, files in os.walk(save_dir):
        if "model.pdparams" in files:
            # 获取相对于 save_dir tel 路径，方便前端展示
            rel_path = os.path.relpath(root, save_dir)
            pdparams_path = os.path.join(root, "model.pdparams")
            mtime = os.path.getmtime(pdparams_path)
            checkpoints.append({
                "name": rel_path if rel_path != "." else "latest",
                "path": pdparams_path,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            })
    
    # 按时间倒序排列
    checkpoints.sort(key=lambda x: x["time"], reverse=True)
    return {"task_id": task_id, "checkpoints": checkpoints}

@app.post("/train/resume/{task_id}")
async def resume_train_task(task_id: str, resume_path: Optional[str] = None, resume_mode: Optional[str] = None):
    """
    显式恢复某个训练任务
    :param task_id: 任务 ID
    :param resume_path: 可选，指定具体的检查点路径。如果不传，则自动寻找最新的。
    :param resume_mode: 可选，指定续训模式 ("interrupted" 或 "extended")。
    """
    result = trainer.resume_task(task_id, resume_path=resume_path, resume_mode=resume_mode)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result

@app.get("/train/history/{task_id}")
async def get_task_history(task_id: str):
    """
    获取任务的历史日志（从 trainer 内存中）
    """
    status = trainer.get_status(task_id)
    if status.get("status") == "not_found":
        return {"task_id": task_id, "logs": [], "error": "Task not found"}
    
    return {
        "task_id": task_id,
        "logs": status.get("logs", []),
        "metrics": status.get("metrics", []),
        "eval_metrics": status.get("eval_metrics", [])
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to One2All Paddle API",
        "paddle_version": paddle.__version__,
        "cuda_available": paddle.is_compiled_with_cuda()
    }

@app.post("/augment")
async def augment_data(request: AugmentRequest):
    try:
        # 1. 解码图片
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 2. 准备标注数据
        annotations = [ann.dict(exclude_none=True) for ann in request.coco_data.annotations]

        # 3. 初始化增强器
        augmentor = DataAugmentor(config=request.config.dict(exclude_none=True))
        
        results = []
        
        # 4. 判断是单张生成还是批量梯度生成
        if request.num_results > 1:
            # 批量梯度生成
            batch_results = augmentor.generate_batch(image, annotations, request.num_results)
            for item in batch_results:
                new_image = item["image"]
                new_annotations = item["annotations"]
                
                # 编码图片
                _, buffer = cv2.imencode('.jpg', new_image)
                new_image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 构建该张图片的 COCO 数据
                result_coco = request.coco_data.dict()
                result_coco["annotations"] = new_annotations
                if result_coco["images"]:
                    h, w = new_image.shape[:2]
                    for img in result_coco["images"]:
                        img["width"] = w
                        img["height"] = h
                
                results.append({
                    "image_base64": new_image_base64,
                    "coco_data": result_coco,
                    "params": item.get("params")
                })
        else:
            # 单张随机生成 (保持原有逻辑)
            new_image, new_annotations = augmentor.apply(image, annotations)
            _, buffer = cv2.imencode('.jpg', new_image)
            new_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result_coco = request.coco_data.dict()
            result_coco["annotations"] = new_annotations
            if result_coco["images"]:
                h, w = new_image.shape[:2]
                for img in result_coco["images"]:
                    img["width"] = w
                    img["height"] = h
            
            results.append({
                "image_base64": new_image_base64,
                "coco_data": result_coco
            })

        return {
            "total": len(results),
            "items": results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ConvertRequest(BaseModel):
    paddle_model_dir: str
    output_dir: Optional[str] = None
    opset_version: int = 11
    simplify: bool = True
    input_shape: Optional[Dict[str, List[int]]] = None


class ConvertResponse(BaseModel):
    success: bool
    message: str
    onnx_model_path: Optional[str] = None
    output_dir: Optional[str] = None
    opset_version: Optional[int] = None
    model_size: Optional[int] = None
    error: Optional[str] = None


@app.get("/convert/onnx/check")
async def check_onnx_converter():
    """检查 ONNX 转换器是否可用"""
    available = check_paddle2onnx_available()
    return {
        "available": available,
        "message": "paddle2onnx 已安装" if available else "paddle2onnx 未安装，请运行: pip install paddle2onnx"
    }


@app.post("/convert/onnx", response_model=ConvertResponse)
async def convert_model_to_onnx(request: ConvertRequest):
    """将 Paddle 模型转换为 ONNX 格式"""
    try:
        result = convert_paddle_to_onnx(
            paddle_model_dir=request.paddle_model_dir,
            output_dir=request.output_dir,
            opset_version=request.opset_version,
            simplify=request.simplify,
            input_shape=dict(request.input_shape) if request.input_shape else None
        )
        
        if result.get("success"):
            return ConvertResponse(
                success=True,
                message="模型转换成功",
                onnx_model_path=result.get("onnx_model_path"),
                output_dir=result.get("output_dir"),
                opset_version=result.get("opset_version"),
                model_size=result.get("model_size")
            )
        else:
            return ConvertResponse(
                success=False,
                message="模型转换失败",
                error=result.get("error", "未知错误")
            )
            
    except Exception as e:
        logger.error(f"ONNX 转换异常: {e}")
        return ConvertResponse(
            success=False,
            message="模型转换失败",
            error=str(e)
        )


@app.get("/convert/onnx/model/{project_id}/{task_uuid}/{label_name}")
async def get_model_info(project_id: str, task_uuid: str, label_name: str):
    """获取模型信息"""
    model_path = Path("output") / str(project_id) / task_uuid / label_name / "best_model"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")
    
    converter = ONNXConverter(str(model_path))
    info = converter.get_model_info()
    
    if not info.get("valid"):
        raise HTTPException(status_code=400, detail=info.get("error", "模型信息获取失败"))
    
    onnx_exists = (model_path / "model.onnx").exists()
    
    return {
        "model_info": info,
        "has_onnx": onnx_exists,
        "onnx_path": str(model_path / "model.onnx") if onnx_exists else None
    }


@app.post("/convert/onnx/convert/{project_id}/{task_uuid}/{label_name}")
async def convert_specific_model(
    project_id: str,
    task_uuid: str,
    label_name: str,
    output_dir: Optional[str] = None,
    opset_version: int = 11,
    simplify: bool = True,
    input_shape: Optional[Dict[str, List[int]]] = None
):
    """转换指定项目/任务/标签的模型"""
    model_path = Path("output") / str(project_id) / task_uuid / label_name / "best_model"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")
    
    if output_dir is None:
        output_dir = str(model_path)
    
    try:
        result = convert_paddle_to_onnx(
            paddle_model_dir=str(model_path),
            output_dir=output_dir,
            opset_version=opset_version,
            simplify=simplify,
            input_shape=dict(input_shape) if input_shape else None
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": "模型转换成功",
                "data": result
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "转换失败"))
            
    except Exception as e:
        logger.error(f"ONNX 转换异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)
