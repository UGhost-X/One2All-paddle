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

from jinja2 import Environment, FileSystemLoader

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
        支持扁平结构: {output_dir}/{project_id}/{task_uuid}/{label}/model.pdparams
        支持 Paddle 模型 (.pdparams) 和 ONNX 模型 (.onnx)
        """
        models = {}
        base_path = Path(self.output_dir) / str(project_id) / task_uuid

        if not base_path.exists():
            return models

        labels_file = base_path / "labels.txt"
        is_unified = labels_file.exists()

        if is_unified:
            best_model_path = base_path / "best_model"
            has_onnx = (best_model_path / "model.onnx").exists()
            has_pdparams = (best_model_path / "model.pdparams").exists()
            
            if has_onnx or has_pdparams:
                labels = []
                try:
                    with open(labels_file, "r") as f:
                        labels = [line.strip() for line in f if line.strip()]
                except:
                    pass
                
                model_type = "onnx" if has_onnx else "paddle"
                models["multi_position"] = str(best_model_path)
                models["_labels"] = labels
                models["_model_type"] = model_type
        else:
            for label_dir in base_path.iterdir():
                if not label_dir.is_dir():
                    continue

                # 尝试扁平结构: {label}/model.pdparams
                has_pdparams_flat = (label_dir / "model.pdparams").exists()
                has_config_flat = (label_dir / "config.json").exists()
                
                # 尝试旧结构: {label}/best_model/model.pdparams
                best_model_path = label_dir / "best_model"
                has_onnx = (best_model_path / "model.onnx").exists()
                has_pdparams = (best_model_path / "model.pdparams").exists()
                
                if has_pdparams_flat or (has_pdparams and has_config_flat):
                    # 扁平结构
                    model_type = "paddle"
                    models[label_dir.name] = str(label_dir)
                    models[f"{label_dir.name}_type"] = model_type
                elif has_onnx or has_pdparams:
                    # 旧结构
                    model_type = "onnx" if has_onnx else "paddle"
                    models[label_dir.name] = str(best_model_path)
                    models[f"{label_dir.name}_type"] = model_type

        return models
    
    def _create_inference_service(self, service: DeployService) -> str:
        """使用 Jinja2 模板生成推理服务脚本"""
        # 设置 Jinja2 环境
        template_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("inference_service.py.j2")
        
        # 构建 annotations.json 路径
        # 路径格式: {project_id}/train/{task_uuid}/annotations.json
        annotations_path = Path(str(service.project_id)) / "train" / service.task_uuid / "annotations.json"
        annotations_path_str = str(annotations_path.absolute()) if annotations_path.exists() else ""
        
        if annotations_path.exists():
            logger.info(f"Found annotations file: {annotations_path_str}")
        else:
            logger.warning(f"Annotations file not found: {annotations_path}")
        
        # 渲染模板
        script_content = template.render(
            service_id=service.service_id,
            project_id=service.project_id,
            task_uuid=service.task_uuid,
            labels=service.labels,
            port=service.port,
            models_config=service.model_paths,
            annotations_path=annotations_path_str
        )
        
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
            
            # 过滤掉特殊键，只保留实际的 label
            special_keys = {'_labels', '_model_type'}
            label_keys = [k for k in model_paths.keys() if not k.endswith('_type') and k not in special_keys]
            
            service=DeployService(
                service_id=service_id,
                project_id=project_id,
                task_uuid=task_uuid,
                port=port,
                status=ServiceStatus.STARTING.value,
                labels=label_keys,
                model_paths=model_paths,
                created_at=time.time(),
                inference_url=f"http://0.0.0.0:{port}"
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

    def check_service_health(self, service_id: str) -> Dict:
        """检查服务的健康状态，包括进程状态和服务响应状态"""
        service = self.services.get(service_id)
        if not service:
            return {
                "status": "not_found",
                "service_id": service_id,
                "healthy": False,
                "message": "服务不存在"
            }
        
        # 检查进程是否存活
        process_alive = service.pid and self._is_process_alive(service.pid)
        
        # 检查端口是否在监听
        port_listening = False
        if process_alive and service.port:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', service.port))
                port_listening = (result == 0)
                sock.close()
            except Exception:
                port_listening = False
        
        # 尝试调用健康检查接口
        service_healthy = False
        health_response = None
        if port_listening:
            try:
                import urllib.request
                import urllib.error
                url = f"http://localhost:{service.port}/health"
                req = urllib.request.Request(url, method='GET')
                req.add_header('Accept', 'application/json')
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        service_healthy = True
                        health_response = json.loads(response.read().decode('utf-8'))
            except Exception as e:
                service_healthy = False
                health_response = {"error": str(e)}
        
        # 确定最终状态
        if not process_alive:
            final_status = "stopped"
            healthy = False
        elif not port_listening:
            final_status = "starting"
            healthy = False
        elif not service_healthy:
            final_status = "unhealthy"
            healthy = False
        else:
            final_status = "healthy"
            healthy = True
        
        return {
            "status": final_status,
            "service_id": service_id,
            "healthy": healthy,
            "process_alive": process_alive,
            "port_listening": port_listening,
            "service_responsive": service_healthy,
            "project_id": service.project_id,
            "task_uuid": service.task_uuid,
            "port": service.port,
            "inference_url": service.inference_url,
            "pid": service.pid,
            "created_at": service.created_at,
            "health_response": health_response,
            "message": "服务运行正常" if healthy else f"服务状态异常: {final_status}"
        }

    def list_services(self, project_id: Optional[str] = None, include_health: bool = False) -> List[Dict]:
        services = []
        for service in self.services.values():
            if project_id and service.project_id != project_id:
                continue
            
            is_running = service.pid and self._is_process_alive(service.pid)
            service_info = {
                "service_id": service.service_id,
                "project_id": service.project_id,
                "task_uuid": service.task_uuid,
                "port": service.port,
                "status": service.status if is_running else ServiceStatus.STOPPED.value,
                "labels": service.labels,
                "created_at": service.created_at,
                "inference_url": service.inference_url
            }
            
            # 如果需要包含健康检查信息
            if include_health:
                health_info = self.check_service_health(service.service_id)
                service_info["health"] = {
                    "healthy": health_info.get("healthy"),
                    "process_alive": health_info.get("process_alive"),
                    "port_listening": health_info.get("port_listening"),
                    "service_responsive": health_info.get("service_responsive"),
                    "message": health_info.get("message")
                }
            
            services.append(service_info)
        
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
                # 过滤掉特殊键，只保留实际的 label
                special_keys = {'_labels', '_model_type'}
                label_keys = {k for k in models.keys() if not k.endswith('_type') and k not in special_keys}
                
                models_by_uuid[task_uuid] = {
                    "labels": list(label_keys),
                    "model_paths": models
                }
        
        return models_by_uuid


deployer = ModelDeployer()
