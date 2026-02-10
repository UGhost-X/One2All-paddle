from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import paddle
import cv2
import numpy as np
import base64
import os
import shutil
import tempfile
import random
import platform
from pathlib import Path
from utils.augmentation import DataAugmentor
from utils.trainer import trainer

app = FastAPI(title="One2All Paddle API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有请求头
)

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
    model_name: str = "STFPM" # 异常检测模型名称 (STFPM)
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.01
    label_names: Optional[List[str]] = None # 新增：指定要训练的 label 列表，不填则训练全部

@app.post("/train/anomaly")
async def train_anomaly(request: TrainRequest):
    """
    接收 COCO 数据，按照特定层级结构保存裁剪信息并启动训练
    层级结构: {base_path}/{project_id}/train/{data_version}/{run_count}/{label}/
    """
    # 1. 构建目录层级
    # 映射 category_id 到 label 名称
    cat_map = {cat.id: cat.name for cat in request.coco_data.categories} if request.coco_data.categories else {}
    
    # 基础存储路径
    # 1. 兼容 Windows 路径输入，将 \ 替换为 /
    # 2. 如果是 Linux 环境，base_path 强制改为当前项目目录
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
                print(f"Failed to decode image: {e}")

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
                        if label_name not in label_val_count or random.random() < 0.1:
                            # 放入验证集
                            val_list_path = label_dir / "val.txt"
                            with open(val_list_path, "a") as f:
                                f.write(f"images/{img_filename} masks/{mask_filename}\n")
                            label_val_count[label_name] = label_val_count.get(label_name, 0) + 1
                        else:
                            # 放入训练集
                            train_list_path = label_dir / "train.txt"
                            with open(train_list_path, "a") as f:
                                f.write(f"images/{img_filename} masks/{mask_filename}\n")
                            
                        crop_count += 1
                        labels_processed.add(label_name)

        if crop_count == 0:
            raise HTTPException(status_code=400, detail="No valid objects to crop")

        # 4. 启动多个后台训练任务（每个 Label 一个模型）
        task_results = []
        for label_name in labels_processed:
            label_dataset_path = storage_base / label_name
            
            train_config = {
                "model_name": request.model_name,
                "label_name": label_name, 
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "project_id": request.project_id,
                "version": request.version
            }
            
            task_id = trainer.run_training_async(str(label_dataset_path), train_config)
            task_results.append({
                "label": label_name,
                "task_id": task_id
            })

        return {
            "status": "success",
            "project_id": request.project_id,
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
    查询训练任务状态
    """
    status = trainer.get_status(task_id)
    return status

@app.get("/train/events/{task_id}")
async def train_events(task_id: str):
    """
    SSE 实时推送训练进度和日志
    """
    async def event_generator():
        last_log_idx = 0
        while True:
            status = trainer.get_status(task_id)
            
            # 准备要发送的数据
            data = {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "label": status.get("label"),
                "new_logs": [],
                "metrics": status.get("metrics", [])
            }
            
            # 获取新日志
            logs = status.get("logs", [])
            if len(logs) > last_log_idx:
                data["new_logs"] = logs[last_log_idx:]
                last_log_idx = len(logs)
            
            # 发送 SSE 格式数据
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            
            # 如果训练完成或失败，发送最后一条消息后退出
            if status.get("status") in ["completed", "failed", "not_found"]:
                break
                
            await asyncio.sleep(1) # 每隔一秒检查一次更新

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
