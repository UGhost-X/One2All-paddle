import os
import threading
import time
import shutil
import json
import re
import random
import ctypes
import logging
import traceback
import warnings
import subprocess
import signal
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ[" paddle_infer_flag_info " ] = "1"

pdx_repos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PaddleX", "paddlex", "repo_manager", "repos", "PaddleSeg")
os.environ["PADDLE_PDX_PADDLESEG_PATH"] = pdx_repos_dir

# 源码PaddleX配置文件路径
pdx_source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PaddleX", "paddlex")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 导入 PaddleSeg 注册模块以确保 STFPM 被注册
import paddlex.repo_apis.PaddleSeg_api.seg.register as _seg_register

import paddlex as pdx
from paddlex.utils.config import AttrDict

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """
    负责监控日志文件并更新训练状态
    """
    def __init__(self, task_id: str, log_file: str, trainer_instance, timeout_seconds: int = 600, init_timeout_seconds: int = 120):
        self.task_id = task_id
        self.log_file = log_file
        self.trainer = trainer_instance
        self.stop_event = threading.Event()
        self.timeout_seconds = timeout_seconds
        self.init_timeout_seconds = init_timeout_seconds
        self.last_log_time = time.time()
        self.last_iter = 0
        self.start_time = time.time()
        self.init_log_received = False

    def start(self):
        self.trainer._add_log(self.task_id, f"Log monitor started for {os.path.basename(self.log_file)}")
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _check_stalled(self):
        """检测训练是否停滞（长时间没有新日志）"""
        current_time = time.time()
        status = self.trainer.training_status.get(self.task_id)
        if not status:
            return False
        
        progress = status.get("progress", 0)
        iter_progress = status.get("metrics", [])
        current_iter = len(iter_progress) if iter_progress else 0
        
        time_since_last_log = current_time - self.last_log_time
        
        # 检测初始化阶段超时：如果超过 init_timeout_seconds 没有收到任何训练日志
        if not self.init_log_received:
            time_since_start = current_time - self.start_time
            if time_since_start > self.init_timeout_seconds:
                self.trainer._add_log(self.task_id, f"ERROR: Training initialization timeout after {self.init_timeout_seconds}s. No training log received.")
                return True
            return False
        
        if progress > 0 and progress < 100:
            if time_since_last_log > self.timeout_seconds:
                if current_iter == self.last_iter:
                    self.trainer._add_log(self.task_id, f"ERROR: Training stalled for {self.timeout_seconds}s without progress. Current iter: {current_iter}")
                    return True
        
        self.last_iter = current_iter
        return False

    def _monitor_loop(self):
        """
        轮询日志文件并解析指标
        """
        # 等待日志文件创建
        start_wait = time.time()
        while not os.path.exists(self.log_file) and time.time() - start_wait < 30:
            if self.stop_event.is_set():
                return
            time.sleep(1)

        if not os.path.exists(self.log_file):
            self.trainer._add_log(self.task_id, f"Warning: Log file {self.log_file} not found after timeout.")
            return

        with open(self.log_file, "r") as f:
            while True:
                if self.stop_event.is_set():
                    remaining_line = f.readline()
                    if remaining_line:
                        self._parse_line(remaining_line)
                    break
                
                line = f.readline()
                if not line:
                    if self._check_stalled():
                        self.trainer._add_log(self.task_id, "Training timeout detected. Forcing termination.")
                        self.trainer.training_status[self.task_id]["status"] = "failed"
                        self.trainer.training_status[self.task_id]["error"] = "Training stalled - timeout"
                        self.trainer.training_status[self.task_id]["_training_timeout"] = True
                        self.stop_event.set()
                        break
                    time.sleep(1)
                    continue
                
                self.last_log_time = time.time()
                self.init_log_received = True
                self._parse_line(line)

        self.trainer._add_log(self.task_id, "Log monitor thread finished.")

    def _read_log_lines(self, last_pos: int) -> int:
        """读取并处理新日志行"""
        try:
            if not os.path.exists(self.log_file):
                return last_pos
                
            with open(self.log_file, "r") as f:
                f.seek(last_pos)
                lines = f.readlines()
                new_pos = f.tell()
                
                for line in lines:
                    self._parse_line(line)
                return new_pos
        except Exception as e:
            logger.error(f"Error reading log {self.log_file}: {e}")
            return last_pos

    def _parse_line(self, line: str):
        """
        解析日志行
        示例: [2026/02/10 16:59:57] INFO: [TRAIN] epoch: 1, iter: 1/2, loss: 3.4739, lr: 0.001000, ...
        """
        # 解析训练指标
        train_match = re.search(r"\[TRAIN\]\s+epoch:\s*(\d+),\s*iter:\s*(\d+)/(\d+),\s*loss:\s*([\d.]+),\s*lr:\s*([\d.]+)", line)
        if train_match:
            epoch = int(train_match.group(1))
            curr_iter = int(train_match.group(2))
            total_iters = int(train_match.group(3))
            loss = float(train_match.group(4))
            lr = float(train_match.group(5))
            
            self.last_log_time = time.time()
            
            current_status = self.trainer.training_status[self.task_id].get("status")
            if current_status != "training":
                self.trainer.training_status[self.task_id]["status"] = "training"
                self.trainer._add_log(self.task_id, "Training started - first iteration detected.")
            
            metrics = {
                "epoch": epoch,
                "iter": curr_iter,
                "total_iters": total_iters,
                "loss": loss,
                "lr": lr,
                "timestamp": time.time()
            }
            
            status = self.trainer.training_status[self.task_id]
            total_epochs = status.get("total_epochs", 1)
            total_iters_overall = status.get("total_iters", 1)
            use_iters_mode = status.get("use_iters_mode", False)
            
            if use_iters_mode:
                current_overall_iter = (epoch - 1) * total_iters + curr_iter
                training_ratio = min(1.0, current_overall_iter / total_iters_overall)
                progress_desc = f"Iter: {current_overall_iter}/{total_iters_overall}"
            else:
                training_ratio = (epoch - 1) / total_epochs + (curr_iter / total_iters) / total_epochs
                progress_desc = f"Epoch: {epoch}/{total_epochs} | Iter: {curr_iter}/{total_iters}"
            
            progress = 10 + (training_ratio * 85)
            new_progress = min(int(progress), 98)
            old_progress = status.get("progress", 0)
            
            if new_progress > old_progress or curr_iter == total_iters:
                log_msg = f"Training Progress: {new_progress}% | {progress_desc} | Loss: {loss:.4f} | LR: {lr:.6f}"
                self.trainer._add_log(self.task_id, log_msg)
            
            self.trainer.training_status[self.task_id]["progress"] = new_progress
            if "metrics" not in self.trainer.training_status[self.task_id]:
                self.trainer.training_status[self.task_id]["metrics"] = []
            self.trainer.training_status[self.task_id]["metrics"].append(metrics)
            
            if len(self.trainer.training_status[self.task_id]["metrics"]) > 1000:
                self.trainer.training_status[self.task_id]["metrics"] = self.trainer.training_status[self.task_id]["metrics"][-1000:]

        # 解析评估指标
        eval_match = re.search(r"\[EVAL\]\s+#Images:\s*(\d+)\s+mIoU:\s*([\d.]+)(?:\s+Acc:\s*([\w.]+))?(?:\s+AUROC:\s*([\d.]+))?", line)
        if eval_match:
            miou = float(eval_match.group(2))
            acc = eval_match.group(3)
            auroc = eval_match.group(4)
            
            eval_info = f"Evaluation: mIoU = {miou:.4f}"
            if auroc:
                eval_info += f", AUROC = {float(auroc):.4f}"
            if acc and acc != "nan":
                eval_info += f", Acc = {acc}"
            
            if miou == 0.0:
                 eval_info += " (Note: mIoU=0 usually means no defects in validation set)"
                
            self.trainer._add_log(self.task_id, eval_info)
            
            if "eval_metrics" not in self.trainer.training_status[self.task_id]:
                self.trainer.training_status[self.task_id]["eval_metrics"] = []
            
            metrics_entry = {"miou": miou, "timestamp": time.time()}
            if auroc: metrics_entry["auroc"] = float(auroc)
            if acc and acc != "nan": metrics_entry["acc"] = float(acc)
            
            self.trainer.training_status[self.task_id]["eval_metrics"].append(metrics_entry)

        # ── 新增：解析 STFPM threshold 行 ──────────────────────────────────────
        # PaddleSeg STFPM 训练结束时会打印类似:
        #   [EVAL] threshold: 0.3456
        # 或写入 best_model/metric.json 中的 threshold 字段
        threshold_match = re.search(r"threshold[:\s=]+([0-9.eE+\-]+)", line, re.IGNORECASE)
        if threshold_match:
            try:
                threshold_val = float(threshold_match.group(1))
                self.trainer.training_status[self.task_id]["threshold"] = threshold_val
                self.trainer._add_log(self.task_id, f"Captured threshold from log: {threshold_val:.6f}")
            except ValueError:
                pass


class AnomalyTrainer:
    """
    负责管理 PaddleX 异常检测训练流程
    """
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.training_status = {}
        self.groups = {}
        self.threads = {}
        self.processes = {}
        self.task_key_index = {}
        self.state_file = str(Path(self.output_dir) / "_one2all_trainer_state.json")
        self._state_lock = threading.Lock()
        
        gpu_count = self._get_gpu_count()
        self._gpu_semaphore = threading.Semaphore(gpu_count) if gpu_count > 0 else threading.Semaphore(1)
        self._gpu_count = gpu_count
        self._gpu_assignment = {}
        self._gpu_assignment_lock = threading.Lock()
        self._last_persist_ts = 0.0
        self._thread_local = threading.local()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.training_status = data.get("training_status", {}) or {}
            self.groups = data.get("groups", {}) or {}
            self.task_key_index = data.get("task_key_index", {}) or {}

            now = time.time()
            for task_id, s in self.training_status.items():
                if s.get("status") in {"starting", "training", "pending", "preparing"}:
                    s["status"] = "interrupted"
                    s["interrupted_at"] = now
                    logs = s.setdefault("logs", [])
                    logs.append(f"[{time.strftime('%H:%M:%S', time.localtime())}] Service restarted; task marked as interrupted.")
                    if len(logs) > 500:
                        s["logs"] = logs[-500:]
        except Exception as e:
            logger.error(f"Failed to load trainer state: {e}")

    def _persist_state(self):
        data = {
            "version": 1,
            "updated_at": time.time(),
            "training_status": self.training_status.copy(),
            "groups": self.groups.copy(),
            "task_key_index": self.task_key_index.copy(),
        }
        tmp_path = f"{self.state_file}.tmp"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, self.state_file)

    def _get_available_gpus(self) -> list:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().split('\n')
            if not lines:
                return []
            available = []
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    gpu_id = int(parts[0].strip())
                    used = float(parts[1].strip())
                    total = float(parts[2].strip())
                    free = total - used
                    if free > 2000:
                        available.append({"id": gpu_id, "free": free, "total": total})
            return available
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return []

    def _get_gpu_count(self) -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                return int(lines[0].strip())
            return 1
        except Exception:
            return 1

    def _check_gpu_for_batch_size(self, batch_size: int) -> tuple:
        gpus = self._get_available_gpus()
        if not gpus:
            return False, [], "No GPU available"
        estimated_memory_per_sample = 500
        required_memory = batch_size * estimated_memory_per_sample
        single_gpu = [g for g in gpus if g["free"] >= required_memory]
        if single_gpu:
            return True, [single_gpu[0]["id"]], f"Using GPU {single_gpu[0]['id']} (free: {single_gpu[0]['free']}MB)"
        if len(gpus) >= 2:
            total_free = sum(g["free"] for g in gpus)
            if total_free >= required_memory:
                gpu_ids = [g["id"] for g in gpus]
                return True, gpu_ids, f"Using {len(gpus)} GPUs (total free: {total_free}MB)"
        return False, [], f"Insufficient GPU memory for batch_size={batch_size} (need ~{required_memory}MB)"

    def _assign_gpu(self, task_id: str) -> int:
        with self._gpu_assignment_lock:
            used_gpus = set(self._gpu_assignment.values())
            for gpu_id in range(self._gpu_count):
                if gpu_id not in used_gpus:
                    self._gpu_assignment[task_id] = gpu_id
                    return gpu_id
            return 0

    def _release_gpu(self, task_id: str):
        with self._gpu_assignment_lock:
            self._gpu_assignment.pop(task_id, None)
        self._gpu_semaphore.release()

    def _check_gpu_available(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False
            lines = result.stdout.strip().split('\n')
            if not lines:
                return False
            total_free = 0
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    used = float(parts[0].strip())
                    total = float(parts[1].strip())
                    free = total - used
                    if free > 2000:
                        total_free += free
            if total_free < 2000:
                logger.warning(f"Low GPU memory: {total_free}MB available")
                return False
            return True
        except Exception as e:
            logger.warning(f"Failed to check GPU status: {e}")
            return False

    def _persist_state_if_due(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_persist_ts < 1.0:
            return
        with self._state_lock:
            now = time.time()
            if not force and now - self._last_persist_ts < 1.0:
                return
            try:
                self._persist_state()
                self._last_persist_ts = now
            except Exception as e:
                logger.error(f"Failed to persist trainer state: {e}")

    def _make_task_key(self, dataset_dir: str, config: dict) -> str:
        return "|".join([
            str(config.get("project_id", "")),
            str(config.get("task_uuid", "")),
            str(config.get("model_name", "")),
            str(config.get("label_name", "")),
            str(dataset_dir),
        ])

    def _find_latest_resume_path(self, save_dir: str):
        latest_file = None
        latest_mtime = -1.0
        logger.info(f"Searching for checkpoints in: {save_dir}")
        for root, _, files in os.walk(save_dir):
            if "model.pdparams" not in files:
                continue
            if "model.pdopt" not in files:
                logger.debug(f"Skipping {root}: model.pdopt not found")
                continue
            pdparams = os.path.join(root, "model.pdparams")
            try:
                mtime = os.path.getmtime(pdparams)
            except Exception:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = pdparams
        if latest_file:
            logger.info(f"Found latest checkpoint: {latest_file} (mtime: {latest_mtime})")
        else:
            logger.info(f"No valid checkpoint (with .pdopt) found in {save_dir}")
        return latest_file

    def run_training_async(self, dataset_dir: str, config: dict, group_id: str = None):
        project_id = config.get("project_id", "unknown")
        
        if not self._check_gpu_available():
            logger.warning(f"[{project_id}] Low GPU memory or GPU unavailable. Training may fail.")
        
        task_key = self._make_task_key(dataset_dir, config)
        existing_task_id = self.task_key_index.get(task_key)
        if existing_task_id and existing_task_id in self.training_status:
            existing = self.training_status[existing_task_id]
            if group_id:
                existing["group_id"] = group_id
                if group_id not in self.groups:
                    self.groups[group_id] = []
                if existing_task_id not in self.groups[group_id]:
                    self.groups[group_id].append(existing_task_id)

            status = existing.get("status")
            thread = self.threads.get(existing_task_id)
            if status in {"starting", "training"} and thread and thread.is_alive():
                self._persist_state_if_due()
                return existing_task_id

            save_dir = existing.get("save_dir")
            resume_path = self._find_latest_resume_path(save_dir) if save_dir else None
            if resume_path:
                existing["status"] = "starting"
                existing["progress"] = min(existing.get("progress", 0), 90)
                existing["resume_path"] = resume_path
                existing.setdefault("logs", []).append(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] Resuming from checkpoint: {resume_path}"
                )
                if len(existing.get("logs", [])) > 500:
                    existing["logs"] = existing["logs"][-500:]
                thread = threading.Thread(
                    target=self._train_process,
                    args=(existing_task_id, existing.get("dataset_dir", dataset_dir), existing.get("config", config), True),
                )
                self.threads[existing_task_id] = thread
                thread.start()
                self._persist_state_if_due(force=True)
                return existing_task_id

        if existing_task_id:
            task_id = existing_task_id
            self._add_log(task_id, f"Reusing task_id: {task_id}")
            resume = config.get("resume", False)
            if not resume:
                logger.info(f"Resetting task status for restart: {task_id}")
                self.training_status[task_id].update({
                    "status": "starting",
                    "progress": 0,
                    "logs": [],
                    "metrics": [],
                    "start_time": time.time(),
                    "resume_path": None,
                    "error": None
                })
                self._add_log(task_id, "Task restarted from scratch. Previous logs cleared.")
        else:
            task_id = f"task_{int(time.time())}_{config.get('label_name', 'unknown')}_{random.randint(1000, 9999)}"
            label_name = config.get("label_name", "unknown")
            task_uuid = config.get("task_uuid", "unknown")
            save_dir = os.path.join(self.output_dir, config.get("project_id", "default"), task_uuid, label_name)
            self.training_status[task_id] = {
                "status": "starting",
                "progress": 0,
                "label": label_name,
                "task_uuid": task_uuid,
                "group_id": group_id,
                "logs": [f"Task {task_id} initialized."],
                "metrics": [],
                "total_epochs": config.get("epochs", 50),
                "start_time": time.time(),
                "dataset_dir": dataset_dir,
                "save_dir": save_dir,
                "config": config,
                "task_key": task_key,
            }
            self.task_key_index[task_key] = task_id
        
        if group_id:
            if group_id not in self.groups:
                self.groups[group_id] = []
            self.groups[group_id].append(task_id)
        self._persist_state_if_due(force=True)
        
        thread = threading.Thread(
            target=self._train_process,
            args=(task_id, dataset_dir, config, False)
        )
        self.threads[task_id] = thread
        thread.start()
        return task_id

    def stop_task(self, task_id: str):
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}
        
        status = self.training_status[task_id].get("status")
        if status in ["completed", "failed", "cancelled"]:
            return {"status": "success", "message": f"Task already in {status} state"}

        if status == "pending":
            self.training_status[task_id]["status"] = "cancelled"
            self._add_log(task_id, "Task cancelled while waiting for GPU.")
            if task_id in self.threads:
                del self.threads[task_id]
            if task_id in self.processes:
                del self.processes[task_id]
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Task cancelled"}

        proc = self.processes.get(task_id)
        if proc:
            try:
                self._add_log(task_id, f"Terminating training subprocess (PID: {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                self._add_log(task_id, "Subprocess terminated.")
            except Exception as e:
                self._add_log(task_id, f"Error terminating subprocess: {e}")

        thread = self.threads.get(task_id)
        if thread and thread.is_alive():
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident),
                ctypes.py_object(SystemExit)
            )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), None)
            
            self._add_log(task_id, "Task cancellation requested by user.")
            self.training_status[task_id]["status"] = "cancelled"
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Cancellation requested"}
        else:
            self.training_status[task_id]["status"] = "cancelled"
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Task marked as cancelled"}

    def stop_group(self, group_id: str):
        if group_id not in self.groups:
            return {"status": "error", "message": "Group not found"}
        task_ids = self.groups[group_id]
        results = []
        for tid in task_ids:
            res = self.stop_task(tid)
            results.append({"task_id": tid, "result": res})
        return {"status": "success", "group_id": group_id, "tasks": results}

    def get_group_status(self, group_id: str):
        if group_id not in self.groups:
            return {"status": "not_found"}
        task_ids = self.groups[group_id]
        task_statuses = [self.training_status.get(tid) for tid in task_ids if tid in self.training_status]
        if not task_statuses:
            return {"status": "starting", "progress": 0}
        total_progress = sum(s.get("progress", 0) for s in task_statuses)
        avg_progress = total_progress / len(task_statuses)
        all_completed = all(s.get("status") == "completed" for s in task_statuses)
        any_failed = any(s.get("status") == "failed" for s in task_statuses)
        status = "training"
        if all_completed:
            status = "completed"
        elif any_failed:
            status = "failed"
        return {
            "group_id": group_id,
            "status": status,
            "progress": int(avg_progress),
            "tasks": [
                {
                    "task_id": tid,
                    "label": s.get("label"),
                    "status": s.get("status"),
                    "progress": s.get("progress")
                }
                for tid, s in zip(task_ids, task_statuses)
            ]
        }

    def _add_log(self, task_id: str, message: str):
        if task_id in self.training_status:
            timestamp_str = time.strftime("%H:%M:%S", time.localtime())
            log_entry = f"[{timestamp_str}] {message}"
            logs = self.training_status[task_id].setdefault("logs", [])
            logs.append(log_entry)
            if len(logs) > 500:
                self.training_status[task_id]["logs"] = logs[-500:]
            logger.info(f"[{task_id}] {message}")
            self._persist_state_if_due()

    def _train_process(self, task_id: str, dataset_dir: str, config: dict, resume: bool = False):
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if self.training_status.get(task_id, {}).get("status") == "cancelled":
                    logger.info(f"[{task_id}] Task cancelled before starting/retrying.")
                    return

                if retry_count > 0:
                    self._add_log(task_id, f"Auto-retrying training ({retry_count}/{max_retries})...")
                    resume = True
                
                self._do_train(task_id, dataset_dir, config, resume)
                break
                
            except (SystemExit, KeyboardInterrupt):
                logger.info(f"[{task_id}] Training thread received stop signal (SystemExit/KeyboardInterrupt).")
                if task_id in self.training_status:
                    self.training_status[task_id]["status"] = "cancelled"
                self._persist_state_if_due(force=True)
                return

            except Exception as e:
                current_status = self.training_status.get(task_id, {}).get("status")
                if current_status == "cancelled":
                    logger.info(f"[{task_id}] Task was cancelled via API, skipping retry.")
                    return

                retry_count += 1
                if retry_count > max_retries:
                    tb = traceback.format_exc()
                    error_msg = f"Training failed after {max_retries} retries: {str(e)}"
                    self._add_log(task_id, error_msg)
                    self.training_status[task_id]["status"] = "failed"
                    self.training_status[task_id]["error"] = str(e)
                    self.training_status[task_id]["traceback"] = tb
                    self._persist_state_if_due(force=True)
                    return
                
                self.training_status[task_id]["status"] = "retrying"
                self._add_log(task_id, f"Training attempt {retry_count} failed: {str(e)}")
                self._persist_state_if_due(force=True)
                time.sleep(5)

    def _do_train(self, task_id: str, dataset_dir: str, config: dict, resume: bool = False):
        self._thread_local.task_id = task_id
        
        try:
            try:
                from paddlex import repo_manager
                pdx_repos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paddlex_repos")
                repo_manager.set_parent_dirs(pdx_repos_dir, None)
                repo_manager.setup(["PaddleSeg"])
                repo_manager.initialize(["PaddleSeg"])
            except:
                pass

            label_name = config.get("label_name", "unknown")
            parallel_train = config.get("parallel_train", False)
            
            self._add_log(task_id, f"Training thread started for label: {label_name} (Parallel: {parallel_train})")
            self.training_status[task_id]["status"] = "preparing"
            self.training_status[task_id]["progress"] = 5
            self._add_log(task_id, "Stage 1/4: Preparing training environment...")

            model_name = config.get("model_name", "STFPM")
            task_uuid = config.get("task_uuid", "unknown")
            save_dir = self.training_status.get(task_id, {}).get("save_dir") or os.path.join(
                self.output_dir, config.get("project_id", "default"), task_uuid, label_name
            )
            os.makedirs(save_dir, exist_ok=True)
            if task_id in self.training_status:
                self.training_status[task_id]["dataset_dir"] = dataset_dir
                self.training_status[task_id]["save_dir"] = save_dir
                self.training_status[task_id]["config"] = config
                self._persist_state_if_due(force=True)

            if not os.path.isdir(dataset_dir):
                raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
            
            train_list = os.path.join(dataset_dir, "train.txt")
            if not os.path.exists(train_list):
                raise FileNotFoundError(f"train.txt not found in dataset_dir: {dataset_dir}")
            
            train_iters_input = config.get("train_iters")
            train_epochs_input = config.get("train_epochs") or config.get("epochs", 50)
            iters_per_epoch = 1
            
            try:
                with open(train_list, "r") as f:
                    num_samples = sum(1 for line in f if line.strip())
                batch_size = config.get("batch_size", 8)
                
                max_batch_size = max(1, num_samples - 1)
                if batch_size > max_batch_size:
                    self._add_log(task_id, f"Adjusting batch_size from {batch_size} to {max_batch_size} (dataset has only {num_samples} samples)")
                    batch_size = max_batch_size
                
                iters_per_epoch = max(1, num_samples // batch_size)
                
                if train_iters_input:
                    total_iters = train_iters_input
                    total_epochs = max(1, total_iters // iters_per_epoch)
                    self._add_log(task_id, f"Using explicit train_iters: {total_iters} (approx {total_epochs} epochs)")
                else:
                    total_epochs = train_epochs_input
                    total_iters = total_epochs * iters_per_epoch
                    self._add_log(task_id, f"Converting epochs to iters: {total_epochs} epochs -> {total_iters} total iters (Dataset size: {num_samples})")
            except Exception as e:
                logger.warning(f"Failed to calculate iters: {e}")
                total_iters = train_iters_input or train_epochs_input or 50
                total_epochs = train_epochs_input or 1
                num_samples = 0
                batch_size = config.get("batch_size", 8)
            
            if task_id in self.training_status:
                self.training_status[task_id]["total_epochs"] = total_epochs
                self.training_status[task_id]["total_iters"] = total_iters
                self.training_status[task_id]["use_iters_mode"] = bool(train_iters_input)
                self.training_status[task_id]["num_samples"] = num_samples
                self.training_status[task_id]["batch_size"] = batch_size
            
            if parallel_train:
                assigned_gpu = self._assign_gpu(task_id)
                device = f"gpu:{assigned_gpu}"
                self._add_log(task_id, f"Parallel training: assigned GPU {assigned_gpu}")
            else:
                assigned_gpu = None
                can_use_gpu, gpu_ids, gpu_msg = self._check_gpu_for_batch_size(batch_size)
                if can_use_gpu:
                    if len(gpu_ids) > 1:
                        device = f"gpu:{','.join(map(str, gpu_ids))}"
                        self._add_log(task_id, f"Multi-GPU training: {gpu_msg}")
                    else:
                        device = f"gpu:{gpu_ids[0]}"
                        self._add_log(task_id, f"Single GPU training: {gpu_msg}")
                else:
                    device = "gpu:0"
                    self._add_log(task_id, f"Warning: {gpu_msg}, falling back to gpu:0")
            
            # 根据模型名称确定配置文件路径
            model_config_path = os.path.join(pdx_source_dir, "repo_apis", "PaddleSeg_api", "configs", f"{model_name}.yaml")
            if not os.path.exists(model_config_path):
                model_config_path = None
            
            pdx_cfg = AttrDict({
                "Global": AttrDict({
                    "model": model_name,
                    "dataset_dir": dataset_dir,
                    "output": save_dir,
                    "device": device
                }),
                "Train": AttrDict({
                    "epochs": total_epochs,
                    "epochs_iters": total_iters,
                    "batch_size": batch_size,
                    "learning_rate": config.get("learning_rate", 0.01),
                    "num_classes": 1,
                    "pretrain_weight_path": None,
                    "resume_path": None,
                    "log_interval": 1,
                    "eval_interval": 5,
                    "save_interval": 10,
                    "basic_config_path": model_config_path
                }),
                "Evaluate": AttrDict({
                    "weight_path": None
                })
            })

            self._add_log(task_id, f"Stage 2/4: Initializing {model_name} trainer (Resume: {resume})...")
            if resume:
                resume_path = self._find_latest_resume_path(save_dir)
                if resume_path:
                    self._add_log(task_id, f"Stage 2/4: Found checkpoint for resume: {resume_path}")
                    m = re.search(r"iter_(\d+)", resume_path)
                    resume_iter = 0
                    if m:
                        resume_iter = int(m.group(1))
                    
                    resume_mode = config.get("resume_mode", "interrupted")
                    
                    if total_iters <= resume_iter:
                        total_iters = resume_iter + 1
                        total_epochs = (total_iters + iters_per_epoch - 1) // iters_per_epoch
                        self._add_log(task_id, f"Stage 2/4: Current progress ({resume_iter}) >= target. Adjusting target to {total_iters} iters to allow continuation.")
                    
                    if resume_mode == "extended":
                        self._add_log(task_id, f"Stage 2/4: Extended resume mode. New total target: {total_iters} iters ({total_epochs} epochs)")
                    else:
                        self._add_log(task_id, f"Stage 2/4: Interrupted resume mode. Resuming to target: {total_iters} iters ({total_epochs} epochs)")

                    pdx_cfg["Train"]["epochs_iters"] = total_iters
                    pdx_cfg["Train"]["epochs"] = total_epochs
                    
                    if task_id in self.training_status:
                        self.training_status[task_id]["total_epochs"] = total_epochs
                        self.training_status[task_id]["total_iters"] = total_iters
                        self.training_status[task_id]["resume_path"] = resume_path
                        self._persist_state_if_due(force=True)

                    pdx_cfg["Train"]["resume_path"] = resume_path
                    self._add_log(task_id, f"Stage 2/4: Resuming from checkpoint: {resume_path}")
                else:
                    self._add_log(task_id, "Stage 2/4: No valid checkpoint found; training from scratch.")
            
            from paddlex.modules.anomaly_detection import UadTrainer
            
            try:
                import logging
                for name in ["paddleseg", "paddle", "paddlex"]:
                    l = logging.getLogger(name)
                    for h in l.handlers[:]:
                        if isinstance(h, logging.StreamHandler):
                            l.removeHandler(h)
                    l.propagate = False
            except:
                pass

            import subprocess
            original_popen = subprocess.Popen
            
            def patched_popen(*args, **kwargs):
                proc = original_popen(*args, **kwargs)
                tid = getattr(self._thread_local, 'task_id', None)
                if tid:
                    self.processes[tid] = proc
                return proc
            
            from paddlex.repo_apis.base.utils import subprocess as pdx_subprocess
            pdx_subprocess.subprocess.Popen = patched_popen

            monitor = None
            
            if not parallel_train:
                self.training_status[task_id]["status"] = "pending"
                self._add_log(task_id, "Waiting for GPU resources (Queued)...")
                self._persist_state_if_due(force=True)
            
            def run_core_train():
                try:
                    paddleseg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "PaddleX", "paddlex", "repo_manager", "repos", "PaddleSeg")
                    os.environ["PYTHONPATH"] = paddleseg_path + ":" + os.environ.get("PYTHONPATH", "")
                    os.environ["PADDLE_PDX_PADDLESEG_PATH"] = paddleseg_path
                    
                    if parallel_train and assigned_gpu is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
                        self._add_log(task_id, f"Setting CUDA_VISIBLE_DEVICES={assigned_gpu} for this process")
                    
                    self.training_status[task_id]["status"] = "starting"
                    self._add_log(task_id, "GPU resource acquired. Initializing trainer...")
                    
                    trainer_obj = UadTrainer(pdx_cfg)
                    
                    self.training_status[task_id]["progress"] = 10
                    self._add_log(task_id, "Stage 3/4: Starting core training process...")
                    
                    log_file = os.path.join(save_dir, "train.log")
                    nonlocal monitor
                    monitor = TrainingMonitor(task_id, log_file, self)
                    monitor.start()
                    
                    trainer_obj.train()
                    
                    self._add_log(task_id, "Stage 4/4: Finalizing and saving model...")
                    time.sleep(1)
                    
                    self.training_status[task_id]["progress"] = 100
                    self._add_log(task_id, "Training Progress: 100% | Status: Completed")
                    self._add_log(task_id, "PaddleX training completed successfully.")
                    self.training_status[task_id]["status"] = "completed"
                    
                    # ── 改造点：训练完成后整理产物 ──────────────────────────────
                    self._finalize_model_artifacts(task_id, save_dir, config)
                    
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    self._add_log(task_id, f"Training failed: {error_msg}")
                    self._add_log(task_id, traceback.format_exc())
                    self.training_status[task_id]["status"] = "failed"
                    self.training_status[task_id]["error"] = error_msg
                finally:
                    pdx_subprocess.subprocess.Popen = original_popen
                    if task_id in self.processes:
                        del self.processes[task_id]
                    
                    if monitor:
                        monitor.stop()
                    
                    status_info = self.training_status.get(task_id, {})
                    is_timeout = status_info.get("_training_timeout", False)
                    if is_timeout:
                        status_info.pop("_training_timeout", None)
                        status_info["status"] = "retrying"
                        status_info.pop("error", None)
                        self._persist_state_if_due(force=True)
                        raise TimeoutError("Training timeout, will retry")
                    
                    self._persist_state_if_due(force=True)

            # GPU 信号量调度（与原逻辑一致）
            if parallel_train:
                acquired = self._gpu_semaphore.acquire(blocking=False)
                if acquired:
                    try:
                        run_core_train()
                    finally:
                        self._gpu_semaphore.release()
                else:
                    max_wait_seconds = 300
                    check_interval = 2
                    waited = 0
                    while waited < max_wait_seconds:
                        current_status = self.training_status.get(task_id, {}).get("status")
                        if current_status == "cancelled":
                            self._add_log(task_id, "Task cancelled while waiting for GPU.")
                            return
                        acquired = self._gpu_semaphore.acquire(blocking=False)
                        if acquired:
                            try:
                                run_core_train()
                            finally:
                                self._gpu_semaphore.release()
                            return
                        else:
                            self._add_log(task_id, f"Waiting for GPU resource... ({waited}s elapsed)")
                            self._persist_state_if_due(force=True)
                            time.sleep(check_interval)
                            waited += check_interval
                    self._add_log(task_id, "Timeout waiting for GPU resource.")
                    self.training_status[task_id]["status"] = "failed"
                    self.training_status[task_id]["error"] = "GPU resource timeout"
                    self._persist_state_if_due(force=True)
            else:
                max_wait_seconds = 1800
                check_interval = 2
                waited = 0
                while waited < max_wait_seconds:
                    current_status = self.training_status.get(task_id, {}).get("status")
                    if current_status == "cancelled":
                        self._add_log(task_id, "Task cancelled while waiting for GPU.")
                        return
                    acquired = self._gpu_semaphore.acquire(blocking=False)
                    if acquired:
                        try:
                            run_core_train()
                        finally:
                            self._gpu_semaphore.release()
                        return
                    else:
                        self._add_log(task_id, f"Waiting for GPU resource... ({waited}s elapsed)")
                        self._persist_state_if_due(force=True)
                        time.sleep(check_interval)
                        waited += check_interval
                self._add_log(task_id, "Timeout waiting for GPU resource.")
                self.training_status[task_id]["status"] = "failed"
                self.training_status[task_id]["error"] = "GPU resource timeout"
                self._persist_state_if_due(force=True)

        except Exception:
            raise

    # ══════════════════════════════════════════════════════════════════════════
    #  新方法：整理训练产物，只保留 model.pdparams + config.json
    # ══════════════════════════════════════════════════════════════════════════

    def _finalize_model_artifacts(self, task_id: str, save_dir: str, config: dict):
        """
        训练完成后将 PaddleX/PaddleSeg 的产物整理为与 train_stfpm_multi.py
        一致的扁平结构：
            save_dir/
                model.pdparams   ← 来自 best_model/model.pdparams
                config.json      ← 生成，与 train_stfpm_multi.py 格式对齐

        所有子目录（best_model、iter_xxx、epoch_xxx 等）以及 .pdopt 等
        中间文件在整理完成后统一删除。
        """
        if not save_dir or not os.path.exists(save_dir):
            self._add_log(task_id, "Warning: save_dir not found, skipping artifact finalization.")
            return

        self._add_log(task_id, "Finalizing model artifacts: extracting best_model to flat structure...")

        try:
            # ── Step 1: 定位 best_model/model.pdparams ────────────────────────
            best_model_dir = os.path.join(save_dir, "best_model")
            src_pdparams = os.path.join(best_model_dir, "model.pdparams")

            # PaddleSeg 有时会在 best_model 下再嵌套一层，做递归查找兜底
            if not os.path.exists(src_pdparams):
                for root, _, files in os.walk(save_dir):
                    if "model.pdparams" in files and "best_model" in root:
                        src_pdparams = os.path.join(root, "model.pdparams")
                        break

            if not os.path.exists(src_pdparams):
                self._add_log(task_id, "Warning: best_model/model.pdparams not found. Skipping artifact finalization.")
                return

            # ── Step 2: 提升 model.pdparams 到 save_dir 根目录 ───────────────
            dst_pdparams = os.path.join(save_dir, "model.pdparams")
            if os.path.abspath(src_pdparams) != os.path.abspath(dst_pdparams):
                shutil.copy2(src_pdparams, dst_pdparams)
                self._add_log(task_id, f"Copied model.pdparams: {src_pdparams} → {dst_pdparams}")
            else:
                self._add_log(task_id, "model.pdparams already in save_dir root, skipping copy.")

            # ── Step 3: 尝试从 best_model/metric.json 读取 threshold ──────────
            threshold = self._read_threshold_from_artifacts(task_id, best_model_dir, save_dir)

            # 如果仍为默认值0.5，尝试用训练集样本计算阈值
            if threshold == 0.5:
                threshold = self._compute_threshold_from_training_data(task_id, save_dir, dst_pdparams, config)
                if threshold > 0:
                    self._add_log(task_id, f"Computed threshold from training data: {threshold:.6f}")

            # ── Step 4: 生成 config.json（与 train_stfpm_multi.py 格式对齐）──
            status_info = self.training_status.get(task_id, {})
            label_name = config.get("label_name", status_info.get("label", "unknown"))
            num_samples = status_info.get("num_samples", 0)
            input_size = config.get("input_size", [224, 224])

            config_data = {
                "category": label_name,
                "threshold": threshold,
                "input_size": input_size,
                "num_samples": num_samples,
                # 附加训练元信息（供调试/溯源，不影响推理）
                "model_name": config.get("model_name", "STFPM"),
                "total_iters": status_info.get("total_iters", 0),
                "batch_size": status_info.get("batch_size", config.get("batch_size", 8)),
                "learning_rate": config.get("learning_rate", 0.01),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            }

            config_path = os.path.join(save_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            self._add_log(task_id, f"Generated config.json: category={label_name}, threshold={threshold:.6f}, num_samples={num_samples}")

            # ── Step 5: 保存训练数据和模板图────────────────────────
            self._save_training_data_and_template(task_id, save_dir, config)

            # ── Step 6: 删除所有子目录及无用中间文件 ─────────────────────────
            removed_dirs = 0
            removed_files = 0
            keep_files = {"model.pdparams", "config.json", "train.log", "template.jpg"}  # 保留关键文件
            keep_dirs = {"training_data"}  # 保留训练数据目录

            for item in os.listdir(save_dir):
                item_path = os.path.join(save_dir, item)

                if os.path.isdir(item_path):
                    if item in keep_dirs:
                        self._add_log(task_id, f"Kept directory: {item}")
                        continue
                    # 删除其他子目录：best_model、iter_xxx、epoch_xxx 等
                    shutil.rmtree(item_path)
                    removed_dirs += 1
                    self._add_log(task_id, f"Removed directory: {item}")

                elif os.path.isfile(item_path):
                    if item not in keep_files:
                        os.remove(item_path)
                        removed_files += 1

            self._add_log(
                task_id,
                f"Cleanup complete: removed {removed_dirs} directories, {removed_files} extra files. "
                f"Final artifacts: model.pdparams, config.json, template.jpg, training_data/"
            )

        except Exception as e:
            self._add_log(task_id, f"Warning: Failed to finalize model artifacts: {e}\n{traceback.format_exc()}")

    def _read_threshold_from_artifacts(self, task_id: str, best_model_dir: str, save_dir: str) -> float:
        """
        按优先级尝试读取 STFPM 训练后的 threshold 值：
          1. training_status 中由日志解析器捕获的值（最实时）
          2. best_model/metric.json
          3. save_dir/metric.json
          4. 兜底：从 eval_metrics 中估算（取最佳 AUROC 对应的值）
          5. 最终兜底：0.5
        """
        status_info = self.training_status.get(task_id, {})

        # 优先级 1：日志解析器已捕获
        if "threshold" in status_info:
            threshold = float(status_info["threshold"])
            self._add_log(task_id, f"Using threshold captured from training log: {threshold:.6f}")
            return threshold

        # 优先级 2 & 3：从 metric.json 文件读取
        for metric_file in [
            os.path.join(best_model_dir, "metric.json"),
            os.path.join(save_dir, "metric.json"),
        ]:
            if os.path.exists(metric_file):
                try:
                    with open(metric_file, "r", encoding="utf-8") as f:
                        metric_data = json.load(f)
                    # PaddleSeg STFPM 可能以 "threshold" 或 "best_threshold" 存储
                    for key in ("threshold", "best_threshold", "optimal_threshold"):
                        if key in metric_data:
                            threshold = float(metric_data[key])
                            self._add_log(task_id, f"Read threshold from {os.path.basename(metric_file)}: {threshold:.6f}")
                            return threshold
                except Exception as e:
                    self._add_log(task_id, f"Warning: Could not parse {metric_file}: {e}")

        # 优先级 4：从 eval_metrics 估算（用最后一次 AUROC 近似）
        eval_metrics = status_info.get("eval_metrics", [])
        if eval_metrics:
            last_metric = eval_metrics[-1]
            if "auroc" in last_metric:
                # AUROC 不直接等于 threshold，但在没有其他信息时用 1 - AUROC 作为粗略参考
                estimated = round(1.0 - float(last_metric["auroc"]), 4)
                self._add_log(task_id, f"Estimating threshold from AUROC ({last_metric['auroc']:.4f}): {estimated:.6f} (approximate)")
                return estimated

        # 最终兜底
        self._add_log(task_id, "Warning: Could not determine threshold, using default 0.5")
        return 0.5

    def _save_training_data_and_template(self, task_id: str, save_dir: str, config: dict):
        """
        保存原始训练数据和标注文件到模型目录，并生成模板图
        
        保存结构:
            save_dir/
                model.pdparams
                config.json
                template.jpg          ← 角度最接近0的图像
                training_data/        ← 原始训练数据
                    annotations.json
                    image_xxx.jpg
                    ...
        """
        try:
            status_info = self.training_status.get(task_id, {})
            dataset_dir = status_info.get("dataset_dir", config.get("dataset_dir", ""))
            
            if not dataset_dir:
                self._add_log(task_id, "Warning: dataset_dir not found, cannot save training data")
                return
            
            # 原始数据在父目录的 raw_images 中
            # dataset_dir 结构: {project_id}/train/{uuid}/{label}
            # 原始数据在: {project_id}/train/{uuid}/
            raw_data_dir = Path(dataset_dir).parent
            raw_images_dir = raw_data_dir / "raw_images"
            annotation_file = raw_data_dir / "annotations.json"
            
            if not annotation_file.exists():
                self._add_log(task_id, f"Warning: annotation file not found: {annotation_file}")
                return
            
            if not raw_images_dir.exists():
                self._add_log(task_id, f"Warning: raw_images directory not found: {raw_images_dir}")
                return
            
            # 创建 training_data 目录
            training_data_dir = os.path.join(save_dir, "training_data")
            os.makedirs(training_data_dir, exist_ok=True)
            
            # 复制标注文件
            dst_annotation = os.path.join(training_data_dir, "annotations.json")
            shutil.copy2(annotation_file, dst_annotation)
            self._add_log(task_id, f"Saved annotations.json to training_data/")
            
            # 读取标注文件
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 找到角度最接近0的图像
            images = annotations.get('images', [])
            if not images:
                self._add_log(task_id, "Warning: no images found in annotations")
                return
            
            # 按角度排序，找到最接近0的
            closest_image = None
            min_angle = float('inf')
            
            for img in images:
                angle = abs(img.get('angle', 0))
                if angle < min_angle:
                    min_angle = angle
                    closest_image = img
            
            if not closest_image:
                self._add_log(task_id, "Warning: could not find image with angle info")
                return
            
            # 复制所有原始图像到 training_data 目录
            copied_count = 0
            for img in images:
                src_path = raw_images_dir / img['file_name']
                dst_path = os.path.join(training_data_dir, img['file_name'])
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
            
            self._add_log(task_id, f"Saved {copied_count} training images to training_data/")
            
            # 复制模板图到模型目录根目录
            if closest_image:
                src_template_path = os.path.join(training_data_dir, closest_image['file_name'])
                dst_template_path = os.path.join(save_dir, "template.jpg")
                
                if os.path.exists(src_template_path):
                    shutil.copy2(src_template_path, dst_template_path)
                    self._add_log(task_id, f"Saved template image: {closest_image['file_name']} (angle={closest_image.get('angle', 0)}°) → template.jpg")
                else:
                    self._add_log(task_id, f"Warning: template source image not found: {src_template_path}")
                
        except Exception as e:
            self._add_log(task_id, f"Warning: Failed to save training data and template: {e}")

    def _compute_threshold_from_training_data(self, task_id: str, save_dir: str, model_path: str, config: dict) -> float:
        """
        使用训练集样本计算 STFPM 阈值
        与 train_stfpm_multi.py 保持一致：threshold = mean(errors) * 2.5
        """
        import paddle
        import paddle.nn as nn
        import numpy as np
        import cv2

        try:
            self._add_log(task_id, "Computing threshold from training data samples...")

            status_info = self.training_status.get(task_id, {})
            label = config.get("label_name", status_info.get("label", ""))
            dataset_dir = status_info.get("dataset_dir", "")
            
            self._add_log(task_id, f"Debug: dataset_dir={dataset_dir}, label={label}")
            
            if not dataset_dir:
                self._add_log(task_id, "Warning: dataset_dir not found in status_info, trying config...")
                dataset_dir = config.get("dataset_dir", "")
            if not label:
                self._add_log(task_id, "Warning: label not found in status_info, trying config...")
                label = config.get("label_name", "")
                
            self._add_log(task_id, f"Debug: after config check - dataset_dir={dataset_dir}, label={label}")
            
            if not dataset_dir or not label:
                self._add_log(task_id, "Warning: dataset_dir or label not found, cannot compute threshold")
                return 0.0

            # 尝试多种可能的训练目录结构
            possible_dirs = [
                os.path.join(dataset_dir, "train", label),
                os.path.join(dataset_dir, label, "train"),
                os.path.join(dataset_dir, label),
                dataset_dir,
            ]
            
            train_dir = None
            for d in possible_dirs:
                if os.path.exists(d):
                    train_dir = d
                    break
            
            self._add_log(task_id, f"Debug: trying train_dir={train_dir}, exists={os.path.exists(train_dir) if train_dir else False}")
            if not train_dir or not os.path.exists(train_dir):
                self._add_log(task_id, f"Warning: training directory not found. Tried: {possible_dirs}")
                return 0.0

            class STFPMModel(nn.Layer):
                def __init__(self):
                    super(STFPMModel, self).__init__()
                    self.teacher = nn.Sequential(
                        nn.Conv2D(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
                        nn.Conv2D(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
                        nn.Conv2D(128, 256, 3, padding=1), nn.ReLU(),
                    )
                    self.student = nn.Sequential(
                        nn.Conv2D(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
                        nn.Conv2D(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
                        nn.Conv2D(128, 256, 3, padding=1), nn.ReLU(),
                    )
                    for param in self.teacher.parameters():
                        param.stop_gradient = True

                def forward(self, x):
                    return self.teacher(x), self.student(x)

                def compute_loss(self, t, s):
                    return paddle.mean(paddle.square(t - s))

            model = STFPMModel()
            model.set_state_dict(paddle.load(model_path))
            model.eval()

            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(Path(train_dir).glob(ext))
            
            self._add_log(task_id, f"Debug: Found {len(image_files)} images in {train_dir}")
            
            if not image_files:
                # 尝试递归搜索
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(Path(train_dir).rglob(ext))
                self._add_log(task_id, f"Debug: Recursive search found {len(image_files)} images")
            
            if not image_files:
                self._add_log(task_id, f"Warning: No images found in {train_dir}")
                return 0.0

            max_samples = min(len(image_files), 50)
            image_files = image_files[:max_samples]

            errors = []
            with paddle.no_grad():
                for img_path in image_files:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
                        img_tensor = paddle.to_tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0)

                        t_feat, s_feat = model(img_tensor)
                        error = model.compute_loss(t_feat, s_feat).item()
                        errors.append(error)
                    except Exception as e:
                        self._add_log(task_id, f"Warning: Failed to process {img_path}: {e}")
                        continue

            if not errors:
                self._add_log(task_id, "Warning: No valid errors computed from training data")
                return 0.0

            threshold = np.mean(errors) * 2.5
            self._add_log(task_id, f"Computed threshold: mean={np.mean(errors):.6f}, threshold={threshold:.6f} (from {len(errors)} samples)")
            return float(threshold)

        except Exception as e:
            self._add_log(task_id, f"Warning: Failed to compute threshold from training data: {e}")
            return 0.0

    def resume_task(self, task_id: str, resume_path: str = None, resume_mode: str = None):
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}
        
        task_info = self.training_status[task_id]
        status = task_info.get("status")
        
        thread = self.threads.get(task_id)
        if status in {"starting", "training"} and thread and thread.is_alive():
            return {"status": "error", "message": "Task is already running"}
        
        if resume_mode:
            if "config" not in task_info:
                task_info["config"] = {}
            task_info["config"]["resume_mode"] = resume_mode

        save_dir = task_info.get("save_dir")
        actual_resume_path = resume_path or self._find_latest_resume_path(save_dir)
        
        if not actual_resume_path or not os.path.exists(actual_resume_path):
            return {"status": "error", "message": "No valid checkpoint found to resume from"}

        task_info["status"] = "starting"
        task_info["progress"] = min(task_info.get("progress", 0), 90)
        task_info["resume_path"] = actual_resume_path
        self._add_log(task_id, f"Manual resume requested. Using checkpoint: {actual_resume_path}")
        
        thread = threading.Thread(
            target=self._train_process,
            args=(task_id, task_info.get("dataset_dir"), task_info.get("config"), True),
        )
        self.threads[task_id] = thread
        thread.start()
        
        self._persist_state_if_due(force=True)
        return {"status": "success", "task_id": task_id, "resume_path": actual_resume_path}

    def get_status(self, task_id: str):
        if task_id in self.training_status:
            return self.training_status[task_id]
        for tid, status in self.training_status.items():
            if status.get("task_uuid") == task_id:
                return status
        return {"status": "not_found"}


# 全局单例
trainer = AnomalyTrainer()