"""
推理服务部署管理模块
支持启动、停止、删除和查询推理服务
一个服务包含一个 task_uuid 下的所有模型
"""
import os
import json
import time
import subprocess
import signal
import threading
import random
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"

@dataclass
class DeployService:
    service_id: str
    project_id: str
    task_uuid: str
    port: int
    status: str
    labels: List[str] = field(default_factory=list)
    model_paths: Dict[str, str] = field(default_factory=dict)
    pid: Optional[int] = None
    created_at: float = 0.0
    error: Optional[str] = None
    inference_url: Optional[str] = None

class ModelDeployer:
    def __init__(self, output_dir: str = "output", scripts_dir: str = "inference_services"):
        self.output_dir = output_dir
        self.scripts_dir = scripts_dir
        self.services: Dict[str, DeployService] = {}
        self.port_index: Dict[int, str] = {}
        self.uuid_index: Dict[str, str] = {}
        self.state_file = str(Path(output_dir) / "_deploy_services.json")
        self._lock = threading.Lock()
        self.base_port = 9000
        self.max_port = 9999
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(scripts_dir).mkdir(parents=True, exist_ok=True)
        self._load_state()
    
    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, sdata in data.get("services", {}).items():
                service = DeployService(**sdata)
                self.services[sid] = service
                if service.port:
                    self.port_index[service.port] = sid
                if service.task_uuid:
                    self.uuid_index[service.task_uuid] = sid
                if service.pid:
                    if not self._is_process_alive(service.pid):
                        service.status = ServiceStatus.STOPPED.value
                        service.pid = None
        except Exception as e:
            logger.error(f"Failed to load deploy state: {e}")
    
    def _save_state(self):
        try:
            data = {
                "version": 2,
                "updated_at": time.time(),
                "services": {sid: asdict(s) for sid, s in self.services.items()}
            }
            tmp_path = f"{self.state_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save deploy state: {e}")
    
    def _is_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    
    def _allocate_port(self) -> int:
        for _ in range(100):
            port = random.randint(self.base_port, self.max_port)
            if port not in self.port_index:
                return port
        raise RuntimeError("No available port for deployment")
    
    def _find_models(self, project_id: str, task_uuid: str) -> Dict[str, str]:
        """
        查找模型路径
        支持新的统一结构: {output_dir}/{project_id}/{task_uuid}/best_model/
        兼容旧结构: {output_dir}/{project_id}/{task_uuid}/{label}/best_model/
        """
        models = {}
        base_path = Path(self.output_dir) / str(project_id) / task_uuid

        if not base_path.exists():
            return models

        # 检查是否是新的统一结构
        labels_file = base_path / "labels.txt"
        is_unified = labels_file.exists()

        if is_unified:
            # 新的统一结构: 只有一个模型
            best_model_path = base_path / "best_model"
            pdparams = best_model_path / "model.pdparams"
            if pdparams.exists():
                # 读取类别列表
                labels = []
                try:
                    with open(labels_file, "r") as f:
                        labels = [line.strip() for line in f if line.strip()]
                except:
                    pass
                # 使用统一的模型名称
                models["multi_position"] = str(best_model_path)
                models["_labels"] = labels  # 存储类别信息
        else:
            # 旧结构: 每个label一个模型
            for label_dir in base_path.iterdir():
                if not label_dir.is_dir():
                    continue

                best_model_path = label_dir / "best_model"
                pdparams = best_model_path / "model.pdparams"
                if pdparams.exists():
                    models[label_dir.name] = str(best_model_path)

        return models
    
    def _create_inference_service(self, service: DeployService) -> str:
        models_config = json.dumps(service.model_paths, ensure_ascii=False, indent=4)
        
        script_content = f'''#!/usr/bin/env python3
"""
推理服务 - 自动生成
Service ID: {service.service_id}
Project: {service.project_id}
Task UUID: {service.task_uuid}
Labels: {service.labels}
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
    description=f"Project: {service.project_id}, UUID: {service.task_uuid}",
    version="1.0.0"
)

MODELS: Dict[str, Any] = {{}}
MODEL_PATHS = {models_config}

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
            logger.info(f"Loading model for [{{label_name}}] from: {{model_path}}")
            predictor = pdx.deploy.Predictor(model_path, device="GPU:0")
            MODELS[label_name] = predictor
            logger.info(f"Model [{{label_name}}] loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model [{{label_name}}]: {{e}}")
            MODELS[label_name] = None
    
    loaded_count = sum(1 for v in MODELS.values() if v is not None)
    logger.info(f"Loaded {{loaded_count}}/{{len(MODEL_PATHS)}} models")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
async def health_check():
    loaded = [k for k, v in MODELS.items() if v is not None]
    return {{
        "status": "healthy",
        "service_id": "{service.service_id}",
        "project_id": "{service.project_id}",
        "task_uuid": "{service.task_uuid}",
        "loaded_models": loaded,
        "total_models": len(MODEL_PATHS)
    }}

@app.get("/models")
async def list_models():
    return {{
        "models": list(MODEL_PATHS.keys()),
        "loaded": [k for k, v in MODELS.items() if v is not None]
    }}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    results = {{}}
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        for label_name, predictor in MODELS.items():
            if predictor is None:
                results[label_name] = {{
                    "status": "error",
                    "message": "Model not loaded"
                }}
                continue
            
            try:
                result = predictor.predict(image)
                
                if isinstance(result, dict):
                    results[label_name] = result
                elif isinstance(result, np.ndarray):
                    results[label_name] = {{
                        "mask_shape": result.shape,
                        "mask_mean": float(np.mean(result)),
                        "mask_std": float(np.std(result))
                    }}
                else:
                    results[label_name] = {{"result": str(result)}}
                    
            except Exception as e:
                results[label_name] = {{
                    "status": "error",
                    "message": str(e)
                }}
        
        processing_time = time.time() - start_time
        return PredictResponse(results=results, processing_time=processing_time)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_base64", response_model=PredictResponse)
async def predict_base64(request: PredictRequest):
    start_time = time.time()
    results = {{}}
    
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
                results[label_name] = {{
                    "status": "error",
                    "message": f"Unknown label: {{label_name}}"
                }}
                continue
            
            predictor = MODELS[label_name]
            if predictor is None:
                results[label_name] = {{
                    "status": "error",
                    "message": "Model not loaded"
                }}
                continue
            
            try:
                result = predictor.predict(image)
                
                if isinstance(result, dict):
                    results[label_name] = result
                elif isinstance(result, np.ndarray):
                    results[label_name] = {{
                        "mask_shape": result.shape,
                        "mask_mean": float(np.mean(result)),
                        "mask_std": float(np.std(result))
                    }}
                else:
                    results[label_name] = {{"result": str(result)}}
                    
            except Exception as e:
                results[label_name] = {{
                    "status": "error",
                    "message": str(e)
                }}
        
        processing_time = time.time() - start_time
        return PredictResponse(results=results, processing_time=processing_time)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

def signal_handler(sig, frame):
    logger.info("Shutting down inference service...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    logger.info(f"Starting inference service on port {service.port}")
    uvicorn.run(app, host="0.0.0.0", port={service.port})
'''
        
        script_path = Path(self.scripts_dir) / f"service_{service.service_id}.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        return str(script_path)
    
    def deploy_service(
        self, 
        project_id: str, 
        task_uuid: str,
        port: Optional[int] = None
    ) -> Dict:
        with self._lock:
            if task_uuid in self.uuid_index:
                existing_sid = self.uuid_index[task_uuid]
                existing = self.services.get(existing_sid)
                if existing and existing.status == ServiceStatus.RUNNING.value:
                    return {
                        "status": "already_exists",
                        "service_id": existing_sid,
                        "port": existing.port,
                        "inference_url": existing.inference_url,
                        "labels": existing.labels,
                        "message": "Service already running for this task_uuid"
                    }
            
            model_paths = self._find_models(project_id, task_uuid)
            if not model_paths:
                return {
                    "status": "error",
                    "message": f"No models found: project={project_id}, uuid={task_uuid}"
                }
            
            service_id = f"svc_{int(time.time())}_{task_uuid}"
            
            if port:
                if port in self.port_index:
                    return {
                        "status": "error",
                        "message": f"Port {port} is already in use"
                    }
            else:
                port = self._allocate_port()
            
            service = DeployService(
                service_id=service_id,
                project_id=project_id,
                task_uuid=task_uuid,
                port=port,
                status=ServiceStatus.STARTING.value,
                labels=list(model_paths.keys()),
                model_paths=model_paths,
                created_at=time.time(),
                inference_url=f"http://localhost:{port}"
            )
            
            script_path = self._create_inference_service(service)
            
            try:
                import sys
                proc = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    cwd=os.getcwd()
                )
                service.pid = proc.pid
                service.status = ServiceStatus.RUNNING.value
                
                self.services[service_id] = service
                self.port_index[port] = service_id
                self.uuid_index[task_uuid] = service_id
                self._save_state()
                
                return {
                    "status": "success",
                    "service_id": service_id,
                    "port": port,
                    "inference_url": service.inference_url,
                    "pid": service.pid,
                    "labels": service.labels,
                    "model_count": len(model_paths)
                }
            except Exception as e:
                service.status = ServiceStatus.FAILED.value
                service.error = str(e)
                self.services[service_id] = service
                self._save_state()
                
                return {
                    "status": "error",
                    "message": f"Failed to start service: {str(e)}"
                }
    
    def stop_service(self, service_id: str) -> Dict:
        with self._lock:
            if service_id not in self.services:
                return {"status": "error", "message": "Service not found"}
            
            service = self.services[service_id]
            
            if service.status == ServiceStatus.STOPPED.value:
                return {"status": "success", "message": "Service already stopped"}
            
            if service.pid and self._is_process_alive(service.pid):
                try:
                    os.killpg(os.getpgid(service.pid), signal.SIGTERM)
                    time.sleep(2)
                    if self._is_process_alive(service.pid):
                        os.killpg(os.getpgid(service.pid), signal.SIGKILL)
                except Exception as e:
                    logger.warning(f"Error killing process: {e}")
            
            service.status = ServiceStatus.STOPPED.value
            service.pid = None
            
            if service.port in self.port_index:
                del self.port_index[service.port]
            
            self._save_state()
            
            return {"status": "success", "message": f"Service {service_id} stopped"}
    
    def delete_service(self, service_id: str) -> Dict:
        with self._lock:
            if service_id not in self.services:
                return {"status": "error", "message": "Service not found"}
            
            service = self.services[service_id]
            
            if service.pid and self._is_process_alive(service.pid):
                self.stop_service(service_id)
            
            script_path = Path(self.scripts_dir) / f"service_{service_id}.py"
            if script_path.exists():
                try:
                    script_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete script: {e}")
            
            if service.port in self.port_index:
                del self.port_index[service.port]
            if service.task_uuid in self.uuid_index:
                del self.uuid_index[service.task_uuid]
            
            del self.services[service_id]
            self._save_state()
            
            return {"status": "success", "message": f"Service {service_id} deleted"}
    
    def get_service(self, service_id: str) -> Dict:
        service = self.services.get(service_id)
        if not service:
            return {"status": "not_found", "message": "Service not found"}
        
        is_running = service.pid and self._is_process_alive(service.pid)
        
        return {
            "status": service.status if is_running else ServiceStatus.STOPPED.value,
            "service_id": service.service_id,
            "project_id": service.project_id,
            "task_uuid": service.task_uuid,
            "port": service.port,
            "inference_url": service.inference_url,
            "labels": service.labels,
            "model_paths": service.model_paths,
            "pid": service.pid,
            "created_at": service.created_at,
            "error": service.error
        }
    
    def get_service_by_uuid(self, task_uuid: str) -> Optional[DeployService]:
        service_id = self.uuid_index.get(task_uuid)
        if service_id:
            return self.services.get(service_id)
        return None
    
    def list_services(self, project_id: Optional[str] = None) -> List[Dict]:
        services = []
        for service in self.services.values():
            if project_id and service.project_id != project_id:
                continue
            
            is_running = service.pid and self._is_process_alive(service.pid)
            services.append({
                "service_id": service.service_id,
                "project_id": service.project_id,
                "task_uuid": service.task_uuid,
                "port": service.port,
                "status": service.status if is_running else ServiceStatus.STOPPED.value,
                "labels": service.labels,
                "created_at": service.created_at,
                "inference_url": service.inference_url
            })
        
        return services
    
    def get_available_models(self, project_id: str) -> Dict:
        project_path = Path(self.output_dir) / str(project_id)
        
        if not project_path.exists():
            return {}
        
        models_by_uuid = {}
        
        for task_dir in project_path.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_uuid = task_dir.name
            models = self._find_models(project_id, task_uuid)
            
            if models:
                models_by_uuid[task_uuid] = {
                    "labels": list(models.keys()),
                    "model_paths": models
                }
        
        return models_by_uuid


deployer = ModelDeployer()
