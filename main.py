from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import paddle
import cv2
import numpy as np
import base64
from utils.augmentation import DataAugmentor

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
