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
import socket
import psutil
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum

from jinja2 import Environment, FileSystemLoader
from utils.config import get_output_dir, get_inference_scripts_dir, path_config

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
    train_mode: str = "by_pos_id"

class ModelDeployer:
    def __init__(self, output_dir: str = None, scripts_dir: str = None):
        self.output_dir = str(output_dir) if output_dir else str(get_output_dir())
        self.scripts_dir = str(scripts_dir) if scripts_dir else str(get_inference_scripts_dir())
        self.services: Dict[str, DeployService] = {}
        self.port_index: Dict[int, str] = {}
        self.uuid_index: Dict[str, str] = {}
        self.state_file = str(Path(self.output_dir) / "_deploy_services.json")
        self._lock = threading.Lock()
        self.base_port = 9000
        self.max_port = 9999
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.scripts_dir).mkdir(parents=True, exist_ok=True)
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

    def _kill_process_tree(self, pid: int) -> bool:
        """Cross-platform process tree termination using psutil."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            parent.terminate()
            gone, alive = psutil.wait_procs(children + [parent], timeout=3)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _allocate_port(self) -> int:
        """使用 socket 让系统自动分配可用端口"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _find_models(self, project_id: str, task_uuid: str) -> Dict[str, str]:
        """
        查找模型路径
        支持 PatchCore 模型 (memory_bank.npz 或 patchcore_model.pt)
        """
        models = {}
        base_path = Path(self.output_dir) / str(project_id) / task_uuid

        if not base_path.exists():
            return models

        labels_file = base_path / "labels.txt"
        is_unified = labels_file.exists()

        if is_unified:
            # 检查新的 anomalib PatchCore 模型格式
            patchcore_model_path = base_path / "patchcore_model.pt"
            has_patchcore_model = patchcore_model_path.exists()
            # 检查旧的 memory_bank 格式
            memory_bank_path = base_path / "memory_bank.npz"
            has_memory_bank = memory_bank_path.exists()
            config_path = base_path / "config.json"
            has_config = config_path.exists()

            if (has_patchcore_model or has_memory_bank) and has_config:
                labels = []
                try:
                    with open(labels_file, "r") as f:
                        labels = [line.strip() for line in f if line.strip()]
                except:
                    pass

                models["multi_position"] = str(base_path)
                models["_labels"] = labels
                models["_model_type"] = "patchcore"
        else:
            train_mode = "by_pos_id"  # 默认值
            for label_dir in base_path.iterdir():
                if not label_dir.is_dir():
                    continue

                # 检查新的 anomalib PatchCore 模型格式
                has_patchcore_model = (label_dir / "patchcore_model.pt").exists()
                # 检查旧的 memory_bank 格式
                has_memory_bank = (label_dir / "memory_bank.npz").exists()
                has_config = (label_dir / "config.json").exists()

                if (has_patchcore_model or has_memory_bank) and has_config:
                    models[label_dir.name] = str(label_dir)
                    models[f"{label_dir.name}_type"] = "patchcore"

                    # 从 config.json 读取 train_mode（读第一个即可）
                    if train_mode == "by_pos_id":
                        try:
                            with open(label_dir / "config.json", "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                                train_mode = cfg.get("train_mode", "by_pos_id")
                        except Exception:
                            pass

            if models:
                models["_train_mode"] = train_mode

        return models
    
    def _create_inference_service(self, service: DeployService) -> str:
        """使用 Jinja2 模板生成推理服务脚本"""
        # 设置 Jinja2 环境
        template_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("inference_service.py.j2")
        
        # 构建 annotations.json 路径
        # 路径格式: {PRODUCT_DIR}/{project_id}/train/{task_uuid}/annotations.json
        annotations_path = path_config.get_project_product_path(service.project_id, service.task_uuid) / "annotations.json"
        annotations_path_str = str(annotations_path) if annotations_path.exists() else ""
        
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
            annotations_path=annotations_path_str,
            train_mode=service.train_mode
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
        logger.info(f"deploy_service called: project_id={project_id}, task_uuid={task_uuid}, port={port}")

        # 步骤1: 在锁内检查已有服务
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

        # 步骤2: I/O 操作在锁外执行
        model_paths = self._find_models(project_id, task_uuid)
        logger.info(f"_find_models returned: {model_paths}")
        if not model_paths:
            return {
                "status": "error",
                "message": f"No models found: project={project_id}, uuid={task_uuid}"
            }

        # 步骤3: 在锁内分配端口和创建服务记录
        with self._lock:
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
            logger.info(f"label_keys: {label_keys}")

            service = DeployService(
                service_id=service_id,
                project_id=project_id,
                task_uuid=task_uuid,
                port=port,
                status=ServiceStatus.STARTING.value,
                labels=label_keys,
                model_paths=model_paths,
                created_at=time.time(),
                inference_url=f"http://0.0.0.0:{port}",
                train_mode=model_paths.get("_train_mode", "by_pos_id")
            )

            # 预先注册服务，防止并发冲突
            self.services[service_id] = service
            self.port_index[port] = service_id
            self.uuid_index[task_uuid] = service_id

        # 步骤4: I/O 操作在锁外执行
        try:
            script_path = self._create_inference_service(service)

            log_dir = Path(self.output_dir) / "logs"
            log_dir.mkdir(exist_ok=True)
            stdout_file = log_dir / f"service_{service_id}.stdout.log"
            stderr_file = log_dir / f"service_{service_id}.stderr.log"

            import sys
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=open(stdout_file, "w"),
                stderr=open(stderr_file, "w"),
                cwd=os.getcwd(),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            logger.info(f"Popen succeeded: pid={proc.pid}")

            time.sleep(2)

            poll_result = proc.poll()
            if poll_result is not None:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()

                # 步骤5: 在锁内更新失败状态
                with self._lock:
                    service.status = ServiceStatus.FAILED.value
                    service.error = f"Process terminated immediately with code {poll_result}"
                    self._save_state()

                return {
                    "status": "error",
                    "message": f"Service process terminated immediately. Check logs: {stderr_file}",
                    "stderr_file": str(stderr_file),
                    "stdout_file": str(stdout_file)
                }

            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

            # 步骤6: 在锁内更新成功状态
            with self._lock:
                service.pid = proc.pid
                service.status = ServiceStatus.RUNNING.value
                logger.info(f"Service started successfully: pid={service.pid}, service_id={service_id}")
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
            # 步骤7: 在锁内更新失败状态
            with self._lock:
                service.status = ServiceStatus.FAILED.value
                service.error = str(e)
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
                    self._kill_process_tree(service.pid)
                    time.sleep(1)
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
            
            # 直接停止进程，避免死锁（stop_service也在同一个锁内）
            if service.pid and self._is_process_alive(service.pid):
                try:
                    self._kill_process_tree(service.pid)
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Error killing process during delete: {e}")
            
            # 更新服务状态
            service.status = ServiceStatus.STOPPED.value
            service.pid = None
            
            script_path = Path(self.scripts_dir) / f"service_{service_id}.py"
            logger.info(f"Attempting to delete script: {script_path}")
            logger.info(f"Script path exists: {script_path.exists()}")
            
            # 尝试多种可能的脚本路径
            possible_paths = [
                script_path,
                Path(self.scripts_dir) / f"service_{service_id}.py",
                Path(f"inference_services/service_{service_id}.py"),
                Path(f"service_{service_id}.py"),
            ]
            
            deleted = False
            for path in possible_paths:
                logger.info(f"Checking path: {path}, exists: {path.exists()}")
                if path.exists():
                    try:
                        path.unlink()
                        logger.info(f"Deleted script: {path}")
                        deleted = True
                    except Exception as e:
                        logger.warning(f"Failed to delete script {path}: {e}")
            
            if not deleted:
                # 使用模糊匹配查找相关脚本文件
                scripts_dir = Path(self.scripts_dir)
                if scripts_dir.exists():
                    # 提取 service_id 中的数字部分用于匹配
                    search_parts = service_id.split("_")
                    
                    # 查找包含 service_id 的文件
                    for f in scripts_dir.glob("*.py"):
                        # 检查多种可能的匹配方式
                        if (service_id in f.name or 
                            f"service_{service_id}" in f.name or
                            any(part in f.name for part in search_parts if len(part) > 5)):
                            try:
                                f.unlink()
                                logger.info(f"Deleted script (fuzzy match): {f}")
                                deleted = True
                                break
                            except Exception as e:
                                logger.warning(f"Failed to delete script {f}: {e}")
                    
                    if not deleted:
                        # 列出所有文件供参考
                        files = list(scripts_dir.glob("*.py"))
                        logger.info(f"Files in {scripts_dir}: {files}")
                        logger.info(f"Looking for service_id: {service_id}")
            
            # 删除日志文件
            logs_dir = Path("logs")
            if logs_dir.exists():
                # 精确匹配: service_svc_xxx_*.log 格式
                exact_pattern = f"service_{service_id}_*.log"
                exact_files = list(logs_dir.glob(exact_pattern))
                
                if exact_files:
                    for log_file in exact_files:
                        try:
                            log_file.unlink()
                            logger.info(f"Deleted log file: {log_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete log file {log_file}: {e}")
                else:
                    # 精确匹配没找到，尝试模糊匹配：文件名包含完整 service_id
                    for log_file in logs_dir.glob("service_*.log"):
                        if f"service_{service_id}" in log_file.name:
                            try:
                                log_file.unlink()
                                logger.info(f"Deleted log file (fuzzy): {log_file}")
                            except Exception as e:
                                logger.warning(f"Failed to delete log file {log_file}: {e}")
            
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

    def get_service_logs(self, service_id: str, lines: int = 100, from_line: int = -1) -> Dict:
        """获取推理服务的日志
        
        增量获取模式（推荐）：
        - from_line=-1 (默认): 获取最新的日志（最后 lines 行）
        - from_line>=0: 从指定行号开始获取增量日志
        
        返回包含 next_line，前端保存用于下次增量获取
        """
        service = self.services.get(service_id)
        if not service:
            return {
                "status": "not_found",
                "service_id": service_id,
                "message": "服务不存在"
            }
        
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return {
                "status": "success",
                "service_id": service_id,
                "logs": "",
                "message": "日志目录不存在",
                "next_line": 0,
                "has_more": False
            }
        
        log_pattern = f"service_{service_id}_*.log"
        log_files = sorted(logs_dir.glob(log_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not log_files:
            return {
                "status": "success",
                "service_id": service_id,
                "logs": "",
                "message": "日志文件不存在",
                "next_line": 0,
                "has_more": False
            }
        
        latest_log = log_files[0]
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                total_lines = len(all_lines)
                
                if from_line < 0:
                    start_line = max(0, total_lines - lines)
                else:
                    start_line = from_line
                
                if start_line >= total_lines:
                    return {
                        "status": "success",
                        "service_id": service_id,
                        "log_file": str(latest_log.name),
                        "logs": "",
                        "total_lines": total_lines,
                        "next_line": total_lines,
                        "has_more": False
                    }
                
                end_line = min(start_line + lines, total_lines)
                log_content = ''.join(all_lines[start_line:end_line])
                
                return {
                    "status": "success",
                    "service_id": service_id,
                    "log_file": str(latest_log.name),
                    "logs": log_content,
                    "total_lines": total_lines,
                    "next_line": end_line,
                    "has_more": end_line < total_lines
                }
        except Exception as e:
            return {
                "status": "error",
                "service_id": service_id,
                "message": f"读取日志失败: {str(e)}"
            }

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
