#!/usr/bin/env python3
"""
推理服务 - 自动生成
Service ID: svc_1771145762_4c450395
Project: 1
Task UUID: 4c450395
Labels: ['孔洞', '轴', '螺丝']
"""
import os
import sys
import signal
import logging
import json
import base64
import time
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2

app = FastAPI(
    title="Inference Service",
    description=f"Project: 1, UUID: 4c450395",
    version="1.0.0"
)

MODELS: Dict[str, Any] = {}
MODEL_PATHS = {
    "孔洞": "output/1/4c450395/孔洞/best_model",
    "轴": "output/1/4c450395/轴/best_model",
    "螺丝": "output/1/4c450395/螺丝/best_model"
}

class PredictRequest(BaseModel):
    image: str
    labels: List[str] = None

class PredictResponse(BaseModel):
    results: Dict[str, Dict[str, Any]]
    processing_time: float

def load_models():
    global MODELS
    import paddlex as pdx
    
    for label_name, model_path in MODEL_PATHS.items():
        try:
            logger.info(f"Loading model for [{label_name}] from: {model_path}")
            predictor = pdx.deploy.Predictor(model_path, device="GPU:0")
            MODELS[label_name] = predictor
            logger.info(f"Model [{label_name}] loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model [{label_name}]: {e}")
            MODELS[label_name] = None
    
    loaded_count = sum(1 for v in MODELS.values() if v is not None)
    logger.info(f"Loaded {loaded_count}/{len(MODEL_PATHS)} models")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
async def health_check():
    loaded = [k for k, v in MODELS.items() if v is not None]
    return {
        "status": "healthy",
        "service_id": "svc_1771145762_4c450395",
        "project_id": "1",
        "task_uuid": "4c450395",
        "loaded_models": loaded,
        "total_models": len(MODEL_PATHS)
    }

@app.get("/models")
async def list_models():
    return {
        "models": list(MODEL_PATHS.keys()),
        "loaded": [k for k, v in MODELS.items() if v is not None]
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    results = {}
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        for label_name, predictor in MODELS.items():
            if predictor is None:
                results[label_name] = {
                    "status": "error",
                    "message": "Model not loaded"
                }
                continue
            
            try:
                result = predictor.predict(image)
                
                if isinstance(result, dict):
                    results[label_name] = result
                elif isinstance(result, np.ndarray):
                    results[label_name] = {
                        "mask_shape": result.shape,
                        "mask_mean": float(np.mean(result)),
                        "mask_std": float(np.std(result))
                    }
                else:
                    results[label_name] = {"result": str(result)}
                    
            except Exception as e:
                results[label_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        processing_time = time.time() - start_time
        return PredictResponse(results=results, processing_time=processing_time)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_base64", response_model=PredictResponse)
async def predict_base64(request: PredictRequest):
    start_time = time.time()
    results = {}
    
    try:
        image_b64 = request.image
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        
        img_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        target_labels = request.labels if request.labels else list(MODELS.keys())
        
        for label_name in target_labels:
            if label_name not in MODELS:
                results[label_name] = {
                    "status": "error",
                    "message": f"Unknown label: {label_name}"
                }
                continue
            
            predictor = MODELS[label_name]
            if predictor is None:
                results[label_name] = {
                    "status": "error",
                    "message": "Model not loaded"
                }
                continue
            
            try:
                result = predictor.predict(image)
                
                if isinstance(result, dict):
                    results[label_name] = result
                elif isinstance(result, np.ndarray):
                    results[label_name] = {
                        "mask_shape": result.shape,
                        "mask_mean": float(np.mean(result)),
                        "mask_std": float(np.std(result))
                    }
                else:
                    results[label_name] = {"result": str(result)}
                    
            except Exception as e:
                results[label_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        processing_time = time.time() - start_time
        return PredictResponse(results=results, processing_time=processing_time)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def signal_handler(sig, frame):
    logger.info("Shutting down inference service...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    logger.info(f"Starting inference service on port 9528")
    uvicorn.run(app, host="0.0.0.0", port=9528)
