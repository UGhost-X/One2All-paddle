from fastapi import FastAPI, HTTPException, Query
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
from pathlib import Path

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

# 挂载静态文件目录，允许通过 HTTP 访问训练数据图片
# 假设所有训练数据都保存在当前目录下的项目文件夹中
app.mount("/static", StaticFiles(directory=os.getcwd()), name="static")

# 定义 COCO 数据模型
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
    version: str # 版本号
    data_version: str # 数据版本
    run_count: int = 1 # 运行次数
    model_name: str = "STFPM"
    train_epochs: Optional[int] = None # 建议使用 train_iters，若提供则自动换算
    train_iters: Optional[int] = None  # 推荐：显式指定训练迭代次数
    batch_size: int = 8
    learning_rate: float = 0.01
    label_names: Optional[List[str]] = None
    resume_path: Optional[str] = None # 可选的恢复训练路径
    resume_mode: Optional[str] = "interrupted" # 续训模式: "interrupted" (中断续训) 或 "extended" (完结续训)
    parallel_train: bool = False # 是否开启多线程并行训练（默认为串行排队）

@app.post("/train/anomaly")
async def train_anomaly(request: TrainRequest):
    """
    接收 COCO 数据，按照特定层级结构保存裁剪信息并启动训练
    层级结构: {base_path}/{project_id}/train/{data_version}/{run_count}/{label}/
    """
    # 1. 构建目录层级
    # 映射 category_id 到 label 名称
    cat_map = {cat.id: cat.name for cat in request.coco_data.categories} if request.coco_data.categories else {}
    
    if platform.system().lower() == "linux":
        normalized_base = os.getcwd()
    else:
        normalized_base = request.base_path.replace("\\", "/")
    
    storage_base = Path(normalized_base) / request.project_id / "train" / request.data_version / str(request.run_count)
    
    try:
        # 2. 解码所有图片
        decoded_images = {}
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
                    decoded_images[image_id] = img
            except Exception as e:
                logger.error(f"Failed to decode image: {e}")

        # 3. 裁剪并分类存放
        crop_count = 0
        labels_processed = set()
        label_val_count = {} # 记录每个 label 放入验证集的数量
        
        # 预先清理可能存在的列表文件
        for cat_id, cat_name in cat_map.items():
            label_dir = storage_base / cat_name
            if label_dir.exists():
                for f_name in ["train.txt", "val.txt"]:
                    f_path = label_dir / f_name
                    if f_path.exists():
                        f_path.unlink()
        
        for ann in request.coco_data.annotations:
            # 获取类别名称
            label_name = ann.label or cat_map.get(ann.category_id, f"class_{ann.category_id}")
            
            # 过滤标签：如果指定了 label_names 且当前标签不在其中，则跳过
            if request.label_names and label_name not in request.label_names:
                continue
                
            image_id = ann.image_id
            if image_id in decoded_images:
                img = decoded_images[image_id]
                h, w = img.shape[:2]
                
                bbox = ann.bbox or (ann.points[:4] if ann.points else None)
                if bbox:
                    x, y, bw, bh = map(int, bbox)
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                    
                    if x2 > x1 and y2 > y1:
                        roi = img[y1:y2, x1:x2]
                        
                        # 构建符合 PaddleX SegDataset 的目录结构
                        label_dir = storage_base / label_name
                        img_dir = label_dir / "images"
                        mask_dir = label_dir / "masks"
                        img_dir.mkdir(parents=True, exist_ok=True)
                        mask_dir.mkdir(parents=True, exist_ok=True)
                        
                        img_filename = f"crop_{image_id}_{crop_count}.png"
                        mask_filename = f"crop_{image_id}_{crop_count}_mask.png"
                        
                        img_path = img_dir / img_filename
                        mask_path = mask_dir / mask_filename
                        
                        # 保存图片
                        cv2.imwrite(str(img_path), roi)
                        
                        # 保存全黑掩码 (对于训练集的 "good" 样本，掩码全为 0)
                        mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                        cv2.imwrite(str(mask_path), mask)
                        
                        # 记录到列表
                        # 策略：前 2 个样本必入训练集，确保训练集不为空；后续样本 10% 概率入验证集
                        train_list_path = label_dir / "train.txt"
                        val_list_path = label_dir / "val.txt"
                        
                        # 统计当前 label 已有的训练样本数
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
                        labels_processed.add(label_name)

        if crop_count == 0:
            raise HTTPException(status_code=400, detail="No valid objects to crop")

        # 4. 后处理：确保每个 label 的验证集都不为空，避免 PaddleSeg 报错
        for label_name in labels_processed:
            label_dir = storage_base / label_name
            train_list_path = label_dir / "train.txt"
            val_list_path = label_dir / "val.txt"
            
            # 如果验证集为空，从训练集中复制一个样本过去
            if not val_list_path.exists() or os.path.getsize(val_list_path) == 0:
                if train_list_path.exists() and os.path.getsize(train_list_path) > 0:
                    with open(train_list_path, "r") as f_train:
                        first_line = f_train.readline()
                    if first_line:
                        with open(val_list_path, "w") as f_val:
                            f_val.write(first_line)
                        logger.info(f"Label '{label_name}' has empty validation set. Copied one sample from train set.")

        # 5. 启动多个后台训练任务（每个 Label 一个模型）
        task_results = []
        group_id = f"group_{int(time.time())}_{request.project_id}"
        
        # 检查是否有请求的类别未被处理（没有数据）
        if request.label_names:
            missing_labels = set(request.label_names) - labels_processed
            if missing_labels:
                logger.warning(f"The following requested labels have no valid annotations and will be skipped: {missing_labels}")
        
        for label_name in labels_processed:
            label_dataset_path = storage_base / label_name
            
            train_config = {
                "model_name": request.model_name,
                "label_name": label_name, 
                "train_epochs": request.train_epochs,
                "train_iters": request.train_iters,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "project_id": request.project_id,
                "version": request.version,
                "data_version": request.data_version,
                "run_count": request.run_count,
                "resume_path": request.resume_path,
                "resume_mode": request.resume_mode,
                "parallel_train": request.parallel_train
            }
            
            task_id = trainer.run_training_async(str(label_dataset_path), train_config, group_id=group_id)
            task_results.append({
                "label": label_name,
                "task_id": task_id
            })

        return {
            "status": "success",
            "project_id": request.project_id,
            "group_id": group_id,
            "data_version": request.data_version,
            "storage_path": str(storage_base),
            "total_crops": crop_count,
            "tasks": task_results
        }

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
async def train_events(task_id: str, auto_delete: bool = True):
    """
    SSE 实时推送训练进度和日志
    :param auto_delete: 是否在读取历史消息后自动删除数据库中的记录
    """
    async def event_generator():
        last_log_idx = 0
        while True:
            # 1. 从 trainer 获取当前状态
            status = trainer.get_status(task_id)
            
            # 2. 从消息队列获取历史消息（如果需要自动删除，可以通过 messenger.get_history）
            # 注意：这里的 status.logs 已经包含了内存中的日志
            # 如果前端是通过 SSE 持续订阅，我们通常不需要在这里删除
            
            data = {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "label": status.get("label"),
                "new_logs": [],
                "metrics": status.get("metrics", [])
            }
            
            logs = status.get("logs", [])
            if len(logs) > last_log_idx:
                data["new_logs"] = logs[last_log_idx:]
                last_log_idx = len(logs)
            
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            
            if status.get("status") in ["completed", "failed", "not_found"]:
                # 如果开启了自动删除，可以在任务结束时清理该 topic 的所有消息
                if auto_delete:
                    from utils.messaging import messenger
                    messenger.get_history(f"task_status_{task_id}", limit=9999, delete_after=True)
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
async def get_task_history(task_id: str, type: str = "log", since_ts: float = 0):
    """
    获取任务的历史消息（从持久化数据库中）
    :param type: 'log' 或 'status'
    :param since_ts: 获取该时间戳之后的消息
    """
    from utils.messaging import messenger
    topic = f"task_{type}_{task_id}"
    history = messenger.get_history(topic, since_ts=since_ts)
    return {"task_id": task_id, "type": type, "history": history}

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

if __name__ == "__main__":
    import uvicorn
    # 使用字符串导入方式 ("main:app") 才能开启 reload=True
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
