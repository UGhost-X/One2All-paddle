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
# from utils.messaging import messenger  # 已移除 ZeroMQ 依赖
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ[" paddle_infer_flag_info " ] = "1"

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
        # 允许更多的空格灵活性，并处理可能的逗号/空格组合
        train_match = re.search(r"\[TRAIN\]\s+epoch:\s*(\d+),\s*iter:\s*(\d+)/(\d+),\s*loss:\s*([\d.]+),\s*lr:\s*([\d.]+)", line)
        if train_match:
            epoch = int(train_match.group(1))
            curr_iter = int(train_match.group(2))
            total_iters = int(train_match.group(3))
            loss = float(train_match.group(4))
            lr = float(train_match.group(5))
            
            self.last_log_time = time.time()
            
            # 首次检测到训练日志时，更新状态为 training
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
            
            # 更新进度
            status = self.trainer.training_status[self.task_id]
            total_epochs = status.get("total_epochs", 1)
            total_iters_overall = status.get("total_iters", 1)
            use_iters_mode = status.get("use_iters_mode", False)
            
            # 改进进度计算：
            # 0-5%: 初始化, 5-10%: 模型准备, 10-95%: 核心训练阶段, 95-100%: 完成
            
            if use_iters_mode:
                # 基于迭代次数的模式：直接用当前总 iteration 计算
                # PaddleSeg 日志中的 epoch 可能会变，但我们关心的是总进度
                # 计算当前总 iter = (epoch-1)*total_iters_in_this_epoch + curr_iter
                # 但更简单的是：直接信任 PaddleSeg 最终会跑到我们设定的 total_iters_overall
                # 我们假设每个 epoch 的 total_iters (即 total_iters 参数) 是一致的
                current_overall_iter = (epoch - 1) * total_iters + curr_iter
                training_ratio = min(1.0, current_overall_iter / total_iters_overall)
                progress_desc = f"Iter: {current_overall_iter}/{total_iters_overall}"
            else:
                # 基于 Epoch 的模式
                training_ratio = (epoch - 1) / total_epochs + (curr_iter / total_iters) / total_epochs
                progress_desc = f"Epoch: {epoch}/{total_epochs} | Iter: {curr_iter}/{total_iters}"
            
            progress = 10 + (training_ratio * 85)
            new_progress = min(int(progress), 98)
            old_progress = status.get("progress", 0)
            
            # 记录详细日志
            if new_progress > old_progress or curr_iter == total_iters:
                log_msg = f"Training Progress: {new_progress}% | {progress_desc} | Loss: {loss:.4f} | LR: {lr:.6f}"
                self.trainer._add_log(self.task_id, log_msg)
            
            self.trainer.training_status[self.task_id]["progress"] = new_progress
            if "metrics" not in self.trainer.training_status[self.task_id]:
                self.trainer.training_status[self.task_id]["metrics"] = []
            self.trainer.training_status[self.task_id]["metrics"].append(metrics)
            
            # 状态更新已存储在 training_status 内存中，通过 SSE 直接推送
            # messenger.publish(f"task_status_{self.task_id}", {
            #     "task_id": self.task_id,
            #     "type": "status_update",
            #     "progress": new_progress,
            #     "metrics": metrics
            # })
            
            # 限制 metrics 数量
            if len(self.trainer.training_status[self.task_id]["metrics"]) > 1000:
                self.trainer.training_status[self.task_id]["metrics"] = self.trainer.training_status[self.task_id]["metrics"][-1000:]

        # 解析评估指标
        # 匹配示例: [2026/02/11 15:10:37] INFO: [EVAL] #Images: 7 mIoU: 0.0000 Acc: nan Kappa: 0.0000 Dice: 0.0000
        # 增加对更多指标的捕获，如 AUROC (如果存在)
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
            
            # 解释 mIoU = 0 的情况
            if miou == 0.0:
                 eval_info += " (Note: mIoU=0 usually means no defects in validation set)"
                
            self.trainer._add_log(self.task_id, eval_info)
            
            if "eval_metrics" not in self.trainer.training_status[self.task_id]:
                self.trainer.training_status[self.task_id]["eval_metrics"] = []
            
            metrics_entry = {"miou": miou, "timestamp": time.time()}
            if auroc: metrics_entry["auroc"] = float(auroc)
            if acc and acc != "nan": metrics_entry["acc"] = float(acc)
            
            self.trainer.training_status[self.task_id]["eval_metrics"].append(metrics_entry)

class AnomalyTrainer:
    """
    负责管理 PaddleX 异常检测训练流程
    """
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.training_status = {} # 用于跟踪训练状态 task_id -> status
        self.groups = {} # 用于跟踪任务组 group_id -> [task_id1, task_id2, ...]
        self.threads = {} # 用于存储训练线程 task_id -> thread
        self.processes = {} # task_id -> subprocess.Popen (新增加以支持强制停止)
        self.task_key_index = {}
        self.state_file = str(Path(self.output_dir) / "_one2all_trainer_state.json")
        self._state_lock = threading.Lock()
        
        gpu_count = self._get_gpu_count()
        self._gpu_semaphore = threading.Semaphore(gpu_count) if gpu_count > 0 else threading.Semaphore(1)
        self._gpu_count = gpu_count
        self._gpu_assignment = {}  # task_id -> gpu_id
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
            "training_status": self.training_status,
            "groups": self.groups,
            "task_key_index": self.task_key_index,
        }
        tmp_path = f"{self.state_file}.tmp"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, self.state_file)

    def _get_available_gpus(self) -> list:
        """获取可用 GPU 列表"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
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
                        available.append({
                            "id": gpu_id,
                            "free": free,
                            "total": total
                        })
            
            return available
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return []

    def _get_gpu_count(self) -> int:
        """获取 GPU 数量"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                return int(lines[0].strip())
            return 1
        except Exception:
            return 1

    def _check_gpu_for_batch_size(self, batch_size: int) -> tuple:
        """
        检查是否有足够的显存支持指定 batch_size
        返回: (bool, gpu_ids, message)
        """
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
        """
        为任务分配一个 GPU（用于并行训练）
        返回: gpu_id
        """
        with self._gpu_assignment_lock:
            used_gpus = set(self._gpu_assignment.values())
            for gpu_id in range(self._gpu_count):
                if gpu_id not in used_gpus:
                    self._gpu_assignment[task_id] = gpu_id
                    return gpu_id
            return 0

    def _release_gpu(self, task_id: str):
        """
        释放任务占用的 GPU
        """
        with self._gpu_assignment_lock:
            self._gpu_assignment.pop(task_id, None)

    def _check_gpu_available(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
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
        return "|".join(
            [
                str(config.get("project_id", "")),
                str(config.get("task_uuid", "")),
                str(config.get("model_name", "")),
                str(config.get("label_name", "")),
                str(dataset_dir),
            ]
        )

    def _find_latest_resume_path(self, save_dir: str):
        latest_file = None
        latest_mtime = -1.0
        logger.info(f"Searching for checkpoints in: {save_dir}")
        for root, _, files in os.walk(save_dir):
            if "model.pdparams" not in files:
                continue
            # 恢复训练通常需要优化器状态 .pdopt
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
                latest_file = pdparams # 返回文件路径，PaddleSeg 要求指向 model.pdparams
        
        if latest_file:
            logger.info(f"Found latest checkpoint: {latest_file} (mtime: {latest_mtime})")
        else:
            logger.info(f"No valid checkpoint (with .pdopt) found in {save_dir}")
        return latest_file

    def run_training_async(self, dataset_dir: str, config: dict, group_id: str = None):
        """
        启动后台线程执行训练
        """
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
            
            # 只有在非续训模式下才重置状态
            # 如果是续训 (resume=True)，则保留 logs 和 metrics 以便查看历史
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
                "metrics": [], # 存储 Epoch 级别的指标
                "total_epochs": config.get("epochs", 50),
                "start_time": time.time(),
                "dataset_dir": dataset_dir,
                "save_dir": save_dir,
                "config": config,
                "task_key": task_key,
            }
            self.task_key_index[task_key] = task_id
        
        # 记录到组
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
        """
        停止单个训练任务
        """
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}
        
        status = self.training_status[task_id].get("status")
        if status in ["completed", "failed", "cancelled"]:
            return {"status": "success", "message": f"Task already in {status} state"}

        # 如果任务在 pending 状态等待 GPU 信号量，需要特殊处理
        if status == "pending":
            self.training_status[task_id]["status"] = "cancelled"
            self._add_log(task_id, "Task cancelled while waiting for GPU.")
            if task_id in self.threads:
                del self.threads[task_id]
            if task_id in self.processes:
                del self.processes[task_id]
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Task cancelled"}

        # 1. 尝试终止子进程 (PaddleSeg 训练进程)
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

        # 2. 终止 Python 管理线程
        thread = self.threads.get(task_id)
        if thread and thread.is_alive():
            # 使用 ctypes 强制抛出异常
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), 
                ctypes.py_object(SystemExit)
            )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), None)
            
            self._add_log(task_id, "Task cancellation requested by user.")
            self.training_status[task_id]["status"] = "cancelled"
            
            # 发布停止消息 - 状态已存储在内存中，通过 SSE 推送
            # messenger.publish(f"task_status_{task_id}", {
            #     "task_id": task_id,
            #     "type": "cancelled",
            #     "message": "Task stopped by user"
            # })
            
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Cancellation requested"}
        else:
            self.training_status[task_id]["status"] = "cancelled"
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Task marked as cancelled"}

    def stop_group(self, group_id: str):
        """
        停止整个任务组
        """
        if group_id not in self.groups:
            return {"status": "error", "message": "Group not found"}
        
        task_ids = self.groups[group_id]
        results = []
        for tid in task_ids:
            res = self.stop_task(tid)
            results.append({"task_id": tid, "result": res})
            
        return {"status": "success", "group_id": group_id, "tasks": results}

    def get_group_status(self, group_id: str):
        """
        获取一组任务的聚合状态
        """
        if group_id not in self.groups:
            return {"status": "not_found"}
        
        task_ids = self.groups[group_id]
        task_statuses = [self.training_status.get(tid) for tid in task_ids if tid in self.training_status]
        
        if not task_statuses:
            return {"status": "starting", "progress": 0}
            
        total_progress = sum(s.get("progress", 0) for s in task_statuses)
        avg_progress = total_progress / len(task_statuses)
        
        # 确定整体状态
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
        """添加日志到任务状态"""
        if task_id in self.training_status:
            timestamp_str = time.strftime("%H:%M:%S", time.localtime())
            log_entry = f"[{timestamp_str}] {message}"
            logs = self.training_status[task_id].setdefault("logs", [])
            logs.append(log_entry)
            if len(logs) > 500:
                self.training_status[task_id]["logs"] = logs[-500:]
            
            # 通过消息队列发布日志 - 日志已存储在内存中，通过 SSE 推送
            # messenger.publish(f"task_log_{task_id}", {
            #     "task_id": task_id,
            #     "type": "log",
            #     "message": log_entry,
            #     "raw_message": message
            # })
            
            logger.info(f"[{task_id}] {message}")
            self._persist_state_if_due()

    def _train_process(self, task_id: str, dataset_dir: str, config: dict, resume: bool = False):
        """
        实际的训练进程 (运行在后台)，包含自动重试逻辑
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # 检查任务是否已被手动取消
                if self.training_status.get(task_id, {}).get("status") == "cancelled":
                    logger.info(f"[{task_id}] Task cancelled before starting/retrying.")
                    return

                if retry_count > 0:
                    self._add_log(task_id, f"Auto-retrying training ({retry_count}/{max_retries})...")
                    # 重试时强制开启续训模式
                    resume = True
                
                self._do_train(task_id, dataset_dir, config, resume)
                # 如果执行到这里没有抛出异常，说明训练成功，退出循环
                break
                
            except (SystemExit, KeyboardInterrupt):
                # 显式捕获手动停止引发的异常 (ctypes 注入的 SystemExit 或终端 Ctrl+C)
                logger.info(f"[{task_id}] Training thread received stop signal (SystemExit/KeyboardInterrupt).")
                # 再次确认状态，确保状态一致性
                if task_id in self.training_status:
                    self.training_status[task_id]["status"] = "cancelled"
                self._persist_state_if_due(force=True)
                return

            except Exception as e:
                # 关键：检查任务是否是通过 API 手动中止的
                # 在 stop_task API 调用时，状态会被设置为 'cancelled'
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
                    # 发布失败消息 - 状态已存储在内存中，通过 SSE 推送
                    # messenger.publish(f"task_status_{task_id}", {
                    #     "task_id": task_id,
                    #     "type": "failed",
                    #     "error": str(e)
                    # })
                    self._persist_state_if_due(force=True)
                    return
                
                self.training_status[task_id]["status"] = "retrying"
                self._add_log(task_id, f"Training attempt {retry_count} failed: {str(e)}")
                self._persist_state_if_due(force=True)
                time.sleep(5)

    def _do_train(self, task_id: str, dataset_dir: str, config: dict, resume: bool = False):
        """
        单次训练执行逻辑
        """
        # 设置线程局部的 task_id，供 patched_popen 使用
        self._thread_local.task_id = task_id
        
        try:
            # 在训练线程中再次确保仓库初始化，防止多线程环境下的模型注册失效
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

            # 1. 构造 PaddleX 训练配置
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
            
            # 计算训练参数：优先使用 train_iters，否则使用 train_epochs 换算
            train_iters_input = config.get("train_iters")
            train_epochs_input = config.get("train_epochs") or config.get("epochs", 50)
            iters_per_epoch = 1
            
            try:
                with open(train_list, "r") as f:
                    num_samples = sum(1 for line in f if line.strip())
                batch_size = config.get("batch_size", 8)
                
                # 自适应 batch_size：确保 batch_size 不超过数据集大小
                # 否则会导致训练无法启动（尤其是一个 label 数据少的情况）
                max_batch_size = max(1, num_samples-1)
                if batch_size > max_batch_size:
                    self._add_log(task_id, f"Adjusting batch_size from {batch_size} to {max_batch_size} (dataset has only {num_samples} samples)")
                    batch_size = max_batch_size
                
                iters_per_epoch = max(1, num_samples // batch_size)
                
                if train_iters_input:
                    total_iters = train_iters_input
                    # 如果提供了 iters，我们也换算一个名义上的 epochs 用于日志显示
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
            
            # 更新状态中的总轮数/总迭代数，供监控器使用
            if task_id in self.training_status:
                self.training_status[task_id]["total_epochs"] = total_epochs
                self.training_status[task_id]["total_iters"] = total_iters
                self.training_status[task_id]["use_iters_mode"] = bool(train_iters_input)
            
            if parallel_train:
                assigned_gpu = self._assign_gpu(task_id)
                device = f"gpu:{assigned_gpu}"
                self._add_log(task_id, f"Parallel training: assigned GPU {assigned_gpu}")
            else:
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
                    "save_interval": 10
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
                        # 无论哪种模式，如果目标轮数不大于当前进度，都至少增加 1 个 iter 以允许程序运行
                        total_iters = resume_iter + 1
                        total_epochs = (total_iters + iters_per_epoch - 1) // iters_per_epoch
                        self._add_log(task_id, f"Stage 2/4: Current progress ({resume_iter}) >= target. Adjusting target to {total_iters} iters to allow continuation.")
                    
                    if resume_mode == "extended":
                        # 完结续训：前端已经传回了累加后的总目标轮次
                        self._add_log(task_id, f"Stage 2/4: Extended resume mode. New total target: {total_iters} iters ({total_epochs} epochs)")
                    else:
                        # 中断续训：恢复到原本设定的总轮数
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
            
            # 尝试静默掉 PaddleSeg 的控制台输出，只保留文件日志
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

            # 注入 monkey-patch 以获取底层的 Popen 对象 (Thread-safe)
            import subprocess
            original_popen = subprocess.Popen
            
            def patched_popen(*args, **kwargs):
                proc = original_popen(*args, **kwargs)
                # 从线程局部变量中获取当前任务 ID
                tid = getattr(self._thread_local, 'task_id', None)
                if tid:
                    self.processes[tid] = proc
                return proc
            
            from paddlex.repo_apis.base.utils import subprocess as pdx_subprocess
            pdx_subprocess.subprocess.Popen = patched_popen

            monitor = None
            
            # 标记为等待队列状态
            if not parallel_train:
                self.training_status[task_id]["status"] = "pending"
                self._add_log(task_id, "Waiting for GPU resources (Queued)...")
                # 立即持久化状态，确保前端能看到 "pending"
                self._persist_state_if_due(force=True)
            
            # 定义实际执行训练的逻辑闭包
            def run_core_train():
                try:
                    # 并行训练时，设置 CUDA_VISIBLE_DEVICES 只使用分配的单个 GPU
                    # 防止 PaddleX 错误地使用多个 GPU 进行分布式训练
                    if parallel_train and assigned_gpu is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
                        self._add_log(task_id, f"Setting CUDA_VISIBLE_DEVICES={assigned_gpu} for this process")
                    
                    # 获取到 GPU，更新状态为 starting
                    self.training_status[task_id]["status"] = "starting"
                    self._add_log(task_id, "GPU resource acquired. Initializing trainer...")
                    
                    trainer_obj = UadTrainer(pdx_cfg)
                    
                    self.training_status[task_id]["progress"] = 10
                    self._add_log(task_id, "Stage 3/4: Starting core training process...")
                    
                    # 2. 启动日志监控
                    log_file = os.path.join(save_dir, "train.log")
                    monitor = TrainingMonitor(task_id, log_file, self)
                    monitor.start()
                    
                    # 3. 执行训练
                    trainer_obj.train()
                    
                    # 4. 训练成功后的处理
                    self._add_log(task_id, "Stage 4/4: Finalizing and saving model...")
                    time.sleep(1) # 给日志监控一点时间同步最后几行
                    
                    # 显式输出 100% 进度日志
                    self.training_status[task_id]["progress"] = 100
                    self._add_log(task_id, "Training Progress: 100% | Status: Completed")
                    self._add_log(task_id, "PaddleX training completed successfully.")
                    self.training_status[task_id]["status"] = "completed"
                    
                    # 训练成功后清理中间检查点
                    self._cleanup_checkpoints(task_id, save_dir)
                    
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    self._add_log(task_id, f"Training failed: {error_msg}")
                    self._add_log(task_id, traceback.format_exc())
                    self.training_status[task_id]["status"] = "failed"
                    self.training_status[task_id]["error"] = error_msg
                finally:
                    # 恢复原始的 Popen
                    pdx_subprocess.subprocess.Popen = original_popen
                    if task_id in self.processes:
                        del self.processes[task_id]
                    
                    if monitor:
                        monitor.stop()
                    
                    # 释放分配的 GPU
                    self._release_gpu(task_id)
                    
                    # 检查是否发生了超时错误，需要让外层重试逻辑捕获
                    status_info = self.training_status.get(task_id, {})
                    is_timeout = status_info.get("_training_timeout", False)
                    if is_timeout:
                        status_info.pop("_training_timeout", None)
                        # 清除 failed 状态，让外层知道要重试
                        status_info["status"] = "retrying"
                        status_info.pop("error", None)
                        self._persist_state_if_due(force=True)
                        raise TimeoutError("Training timeout, will retry")
                    
                    self._persist_state_if_due(force=True)

            # 根据 parallel_train 决定是否使用 GPU 资源
            # 重要：先获取信号量，再执行训练，确保资源管理正确
            if parallel_train:
                # 并行训练：先获取信号量，再执行训练（GPU 已在上面分配）
                acquired = self._gpu_semaphore.acquire(blocking=False)
                if acquired:
                    try:
                        run_core_train()
                    finally:
                        self._gpu_semaphore.release()
                else:
                    # 如果获取不到，就用非阻塞轮询方式
                    max_wait_seconds = 300  # 最多等待 5 分钟
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
                    
                    # 超时
                    self._add_log(task_id, "Timeout waiting for GPU resource.")
                    self.training_status[task_id]["status"] = "failed"
                    self.training_status[task_id]["error"] = "GPU resource timeout"
                    self._persist_state_if_due(force=True)
            else:
                # 使用非阻塞轮询方式获取 GPU 信号量，避免线程永久阻塞
                max_wait_seconds = 1800  # 最多等待 30 分钟
                check_interval = 2  # 每 2 秒检查一次
                waited = 0
                while waited < max_wait_seconds:
                    # 检查任务是否已被取消
                    current_status = self.training_status.get(task_id, {}).get("status")
                    if current_status == "cancelled":
                        self._add_log(task_id, "Task cancelled while waiting for GPU.")
                        return
                    
                    # 尝试获取信号量（非阻塞）
                    acquired = self._gpu_semaphore.acquire(blocking=False)
                    if acquired:
                        try:
                            run_core_train()
                        finally:
                            # 确保信号量一定会被释放
                            self._gpu_semaphore.release()
                        return
                    else:
                        # 信号量被占用，等待一会儿再试
                        self._add_log(task_id, f"Waiting for GPU resource... ({waited}s elapsed)")
                        self._persist_state_if_due(force=True)
                        time.sleep(check_interval)
                        waited += check_interval
                
                # 超时
                self._add_log(task_id, "Timeout waiting for GPU resource.")
                self.training_status[task_id]["status"] = "failed"
                self.training_status[task_id]["error"] = "GPU resource timeout"
                self._persist_state_if_due(force=True)

        except Exception:
            raise

    def resume_task(self, task_id: str, resume_path: str = None, resume_mode: str = None):
        """
        恢复一个已停止或中断的任务
        """
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}
        
        task_info = self.training_status[task_id]
        status = task_info.get("status")
        
        # 只有不在运行中的任务可以恢复
        thread = self.threads.get(task_id)
        if status in {"starting", "training"} and thread and thread.is_alive():
            return {"status": "error", "message": "Task is already running"}
        
        # 如果传入了新的 resume_mode，则更新配置
        if resume_mode:
            if "config" not in task_info:
                task_info["config"] = {}
            task_info["config"]["resume_mode"] = resume_mode

        # 确定恢复路径
        save_dir = task_info.get("save_dir")
        actual_resume_path = resume_path or self._find_latest_resume_path(save_dir)
        
        if not actual_resume_path or not os.path.exists(actual_resume_path):
             return {"status": "error", "message": "No valid checkpoint found to resume from"}

        # 更新状态并启动
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

    def _cleanup_checkpoints(self, task_id: str, save_dir: str):
        """
        训练完成后清理中间检查点，只保留 best_model
        """
        if not save_dir or not os.path.exists(save_dir):
            return
            
        self._add_log(task_id, "Cleaning up intermediate checkpoints, keeping best_model...")
        try:
            cleaned_count = 0
            # 遍历 save_dir 下的一级子目录
            for item in os.listdir(save_dir):
                item_path = os.path.join(save_dir, item)
                if not os.path.isdir(item_path):
                    continue
                
                # 策略：删除所有以 epoch_ 开头的目录，保留 best_model 和其他可能的重要目录
                if item.startswith("epoch_"):
                    shutil.rmtree(item_path)
                    cleaned_count += 1
            
            self._add_log(task_id, f"Cleanup finished. Removed {cleaned_count} intermediate checkpoint directories.")
        except Exception as e:
            self._add_log(task_id, f"Warning: Failed to cleanup checkpoints: {e}")

    def get_status(self, task_id: str):
        """
        查询训练状态
        支持通过 task_id 或 task_uuid 查找
        """
        if task_id in self.training_status:
            return self.training_status[task_id]
        
        for tid, status in self.training_status.items():
            if status.get("task_uuid") == task_id:
                return status
        
        return {"status": "not_found"}

# 全局单例
trainer = AnomalyTrainer()
