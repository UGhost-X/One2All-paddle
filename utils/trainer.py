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
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ[" paddle_infer_flag_info "] = "1"

pdx_repos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PaddleX", "paddlex", "repo_manager", "repos", "PaddleSeg")
os.environ["PADDLE_PDX_PADDLESEG_PATH"] = pdx_repos_dir
pdx_source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PaddleX", "paddlex")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import paddlex.repo_apis.PaddleSeg_api.seg.register as _seg_register
import paddlex as pdx
from paddlex.utils.config import AttrDict

logger = logging.getLogger(__name__)


class TrainingMonitor:
    def __init__(self, task_id, log_file, trainer_instance, timeout_seconds=600, init_timeout_seconds=120):
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
        current_time = time.time()
        status = self.trainer.training_status.get(self.task_id)
        if not status:
            return False
        progress = status.get("progress", 0)
        current_iter = len(status.get("metrics", []))
        time_since_last_log = current_time - self.last_log_time
        if not self.init_log_received:
            if current_time - self.start_time > self.init_timeout_seconds:
                self.trainer._add_log(self.task_id, f"ERROR: Initialization timeout after {self.init_timeout_seconds}s.")
                return True
            return False
        if 0 < progress < 100 and time_since_last_log > self.timeout_seconds:
            if current_iter == self.last_iter:
                self.trainer._add_log(self.task_id, f"ERROR: Training stalled for {self.timeout_seconds}s.")
                return True
        self.last_iter = current_iter
        return False

    def _monitor_loop(self):
        start_wait = time.time()
        while not os.path.exists(self.log_file) and time.time() - start_wait < 30:
            if self.stop_event.is_set():
                return
            time.sleep(1)
        if not os.path.exists(self.log_file):
            self.trainer._add_log(self.task_id, f"Warning: Log file not found: {self.log_file}")
            return
        with open(self.log_file, "r") as f:
            while True:
                if self.stop_event.is_set():
                    line = f.readline()
                    if line:
                        self._parse_line(line)
                    break
                line = f.readline()
                if not line:
                    if self._check_stalled():
                        self.trainer._add_log(self.task_id, "Training timeout. Forcing termination.")
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

    def _parse_line(self, line):
        train_match = re.search(r"\[TRAIN\]\s+epoch:\s*(\d+),\s*iter:\s*(\d+)/(\d+),\s*loss:\s*([\d.]+),\s*lr:\s*([\d.]+)", line)
        if train_match:
            epoch = int(train_match.group(1))
            curr_iter = int(train_match.group(2))
            total_iters = int(train_match.group(3))
            loss = float(train_match.group(4))
            lr = float(train_match.group(5))
            self.last_log_time = time.time()
            if self.trainer.training_status[self.task_id].get("status") != "training":
                self.trainer.training_status[self.task_id]["status"] = "training"
                self.trainer._add_log(self.task_id, "Training started - first iteration detected.")
            s = self.trainer.training_status[self.task_id]
            total_epochs = s.get("total_epochs", 1)
            total_iters_overall = s.get("total_iters", 1)
            use_iters_mode = s.get("use_iters_mode", False)
            if use_iters_mode:
                overall = (epoch - 1) * total_iters + curr_iter
                ratio = min(1.0, overall / total_iters_overall)
                desc = f"Iter: {overall}/{total_iters_overall}"
            else:
                ratio = (epoch - 1) / total_epochs + (curr_iter / total_iters) / total_epochs
                desc = f"Epoch: {epoch}/{total_epochs} | Iter: {curr_iter}/{total_iters}"
            new_progress = min(int(10 + ratio * 80), 89)
            if new_progress > s.get("progress", 0) or curr_iter == total_iters:
                self.trainer._add_log(self.task_id, f"Progress: {new_progress}% | {desc} | Loss: {loss:.4f} | LR: {lr:.6f}")
            s["progress"] = new_progress
            s.setdefault("metrics", []).append({
                "epoch": epoch, "iter": curr_iter, "total_iters": total_iters,
                "loss": loss, "lr": lr, "timestamp": time.time()
            })
            if len(s["metrics"]) > 1000:
                s["metrics"] = s["metrics"][-1000:]

        eval_match = re.search(r"\[EVAL\]\s+#Images:\s*(\d+)\s+mIoU:\s*([\d.]+)(?:\s+Acc:\s*([\w.]+))?(?:\s+AUROC:\s*([\d.]+))?", line)
        if eval_match:
            miou = float(eval_match.group(2))
            auroc = eval_match.group(4)
            acc = eval_match.group(3)
            info = f"Evaluation: mIoU={miou:.4f}"
            if auroc:
                info += f", AUROC={float(auroc):.4f}"
            if acc and acc != "nan":
                info += f", Acc={acc}"
            if miou == 0.0:
                info += " (mIoU=0: no defects in val set, normal for anomaly detection)"
            self.trainer._add_log(self.task_id, info)
            entry = {"miou": miou, "timestamp": time.time()}
            if auroc:
                entry["auroc"] = float(auroc)
            self.trainer.training_status[self.task_id].setdefault("eval_metrics", []).append(entry)

        t_match = re.search(r"threshold[:\s=]+([0-9.eE+\-]+)", line, re.IGNORECASE)
        if t_match:
            try:
                val = float(t_match.group(1))
                if val > 0.0:
                    self.trainer.training_status[self.task_id]["threshold_from_log"] = val
                    self.trainer._add_log(self.task_id, f"Captured threshold from log: {val:.6f}")
            except ValueError:
                pass


class AnomalyTrainer:
    def __init__(self, output_dir="output"):
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
                    logs.append(f"[{time.strftime('%H:%M:%S')}] Service restarted; task marked as interrupted.")
                    if len(logs) > 500:
                        s["logs"] = logs[-500:]
        except Exception as e:
            logger.error(f"Failed to load trainer state: {e}")

    def _persist_state(self):
        data = {
            "version": 1, "updated_at": time.time(),
            "training_status": self.training_status.copy(),
            "groups": self.groups.copy(),
            "task_key_index": self.task_key_index.copy(),
        }
        tmp = f"{self.state_file}.tmp"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.state_file)

    def _persist_state_if_due(self, force=False):
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
                logger.error(f"Failed to persist state: {e}")

    def _get_available_gpus(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return []
            available = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 3:
                    gpu_id = int(parts[0].strip())
                    used = float(parts[1].strip())
                    total = float(parts[2].strip())
                    if total - used > 2000:
                        available.append({"id": gpu_id, "free": total - used, "total": total})
            return available
        except Exception:
            return []

    def _get_gpu_count(self):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0].strip())
        except Exception:
            pass
        return 1

    def _check_gpu_for_batch_size(self, batch_size):
        gpus = self._get_available_gpus()
        if not gpus:
            return False, [], "No GPU available"
        req = batch_size * 500
        single = [g for g in gpus if g["free"] >= req]
        if single:
            return True, [single[0]["id"]], f"Using GPU {single[0]['id']} (free: {single[0]['free']}MB)"
        if len(gpus) >= 2 and sum(g["free"] for g in gpus) >= req:
            return True, [g["id"] for g in gpus], f"Using {len(gpus)} GPUs"
        return False, [], f"Insufficient GPU memory (need ~{req}MB)"

    def _assign_gpu(self, task_id):
        with self._gpu_assignment_lock:
            used = set(self._gpu_assignment.values())
            for i in range(self._gpu_count):
                if i not in used:
                    self._gpu_assignment[task_id] = i
                    return i
            return 0

    def _check_gpu_available(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False
            total_free = 0
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 2:
                    total_free += float(parts[1].strip()) - float(parts[0].strip())
            return total_free >= 2000
        except Exception:
            return False

    def _make_task_key(self, dataset_dir, config):
        return "|".join([
            str(config.get("project_id", "")), str(config.get("task_uuid", "")),
            str(config.get("model_name", "")), str(config.get("label_name", "")), str(dataset_dir),
        ])

    def _find_latest_resume_path(self, save_dir):
        latest_file, latest_mtime = None, -1.0
        for root, _, files in os.walk(save_dir):
            if "model.pdparams" in files and "model.pdopt" in files:
                pdparams = os.path.join(root, "model.pdparams")
                try:
                    mtime = os.path.getmtime(pdparams)
                    if mtime > latest_mtime:
                        latest_mtime, latest_file = mtime, pdparams
                except Exception:
                    pass
        return latest_file

    def _add_log(self, task_id, message):
        if task_id in self.training_status:
            entry = f"[{time.strftime('%H:%M:%S')}] {message}"
            logs = self.training_status[task_id].setdefault("logs", [])
            logs.append(entry)
            if len(logs) > 500:
                self.training_status[task_id]["logs"] = logs[-500:]
            logger.info(f"[{task_id}] {message}")
            self._persist_state_if_due()

    def run_training_async(self, dataset_dir, config, group_id=None):
        if not self._check_gpu_available():
            logger.warning("Low GPU memory or GPU unavailable.")
        task_key = self._make_task_key(dataset_dir, config)
        existing_task_id = self.task_key_index.get(task_key)
        if existing_task_id and existing_task_id in self.training_status:
            existing = self.training_status[existing_task_id]
            if group_id:
                existing["group_id"] = group_id
                self.groups.setdefault(group_id, [])
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
                existing.update({"status": "starting", "progress": min(existing.get("progress", 0), 90), "resume_path": resume_path})
                existing.setdefault("logs", []).append(f"[{time.strftime('%H:%M:%S')}] Resuming from: {resume_path}")
                if len(existing.get("logs", [])) > 500:
                    existing["logs"] = existing["logs"][-500:]
                t = threading.Thread(target=self._train_process,
                                     args=(existing_task_id, existing.get("dataset_dir", dataset_dir), existing.get("config", config), True))
                self.threads[existing_task_id] = t
                t.start()
                self._persist_state_if_due(force=True)
                return existing_task_id

        if existing_task_id:
            task_id = existing_task_id
            if not config.get("resume", False):
                self.training_status[task_id].update({
                    "status": "starting", "progress": 0, "logs": [], "metrics": [],
                    "start_time": time.time(), "resume_path": None, "error": None
                })
                self._add_log(task_id, "Task restarted from scratch.")
        else:
            task_id = f"task_{int(time.time())}_{config.get('label_name', 'unknown')}_{random.randint(1000, 9999)}"
            label_name = config.get("label_name", "unknown")
            task_uuid = config.get("task_uuid", "unknown")
            save_dir = os.path.join(self.output_dir, config.get("project_id", "default"), task_uuid, label_name)
            self.training_status[task_id] = {
                "status": "starting", "progress": 0, "label": label_name, "task_uuid": task_uuid,
                "group_id": group_id, "logs": [f"Task {task_id} initialized."], "metrics": [],
                "total_epochs": config.get("epochs", 50), "start_time": time.time(),
                "dataset_dir": dataset_dir, "save_dir": save_dir, "config": config, "task_key": task_key,
            }
            self.task_key_index[task_key] = task_id

        if group_id:
            self.groups.setdefault(group_id, []).append(task_id)
        self._persist_state_if_due(force=True)
        t = threading.Thread(target=self._train_process, args=(task_id, dataset_dir, config, False))
        self.threads[task_id] = t
        t.start()
        return task_id

    def stop_task(self, task_id):
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}
        status = self.training_status[task_id].get("status")
        if status in ["completed", "failed", "cancelled"]:
            return {"status": "success", "message": f"Task already in {status} state"}
        if status == "pending":
            self.training_status[task_id]["status"] = "cancelled"
            self._add_log(task_id, "Task cancelled while waiting for GPU.")
            self.threads.pop(task_id, None)
            self.processes.pop(task_id, None)
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Task cancelled"}
        proc = self.processes.get(task_id)
        if proc:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                self._add_log(task_id, f"Error terminating subprocess: {e}")
        thread = self.threads.get(task_id)
        if thread and thread.is_alive():
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), None)
            self.training_status[task_id]["status"] = "cancelled"
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Cancellation requested"}
        self.training_status[task_id]["status"] = "cancelled"
        self._persist_state_if_due(force=True)
        return {"status": "success", "message": "Task marked as cancelled"}

    def stop_group(self, group_id):
        if group_id not in self.groups:
            return {"status": "error", "message": "Group not found"}
        return {"status": "success", "group_id": group_id,
                "tasks": [{"task_id": tid, "result": self.stop_task(tid)} for tid in self.groups[group_id]]}

    def get_group_status(self, group_id):
        if group_id not in self.groups:
            return {"status": "not_found"}
        task_ids = self.groups[group_id]
        ss = [self.training_status.get(tid) for tid in task_ids if tid in self.training_status]
        if not ss:
            return {"status": "starting", "progress": 0}
        avg = sum(s.get("progress", 0) for s in ss) / len(ss)
        stat = "completed" if all(s.get("status") == "completed" for s in ss) else \
               ("failed" if any(s.get("status") == "failed" for s in ss) else "training")
        return {"group_id": group_id, "status": stat, "progress": int(avg),
                "tasks": [{"task_id": t, "label": s.get("label"), "status": s.get("status"), "progress": s.get("progress")}
                          for t, s in zip(task_ids, ss)]}

    def _train_process(self, task_id, dataset_dir, config, resume=False):
        for retry in range(4):
            try:
                if self.training_status.get(task_id, {}).get("status") == "cancelled":
                    return
                if retry > 0:
                    self._add_log(task_id, f"Auto-retrying ({retry}/3)...")
                    resume = True
                self._do_train(task_id, dataset_dir, config, resume)
                return
            except (SystemExit, KeyboardInterrupt):
                if task_id in self.training_status:
                    self.training_status[task_id]["status"] = "cancelled"
                self._persist_state_if_due(force=True)
                return
            except Exception as e:
                if self.training_status.get(task_id, {}).get("status") == "cancelled":
                    return
                if retry >= 3:
                    self._add_log(task_id, f"Training failed after 3 retries: {e}")
                    self.training_status[task_id].update({
                        "status": "failed", "error": str(e), "traceback": traceback.format_exc()
                    })
                    self._persist_state_if_due(force=True)
                    return
                self.training_status[task_id]["status"] = "retrying"
                self._add_log(task_id, f"Attempt {retry + 1} failed: {e}")
                self._persist_state_if_due(force=True)
                time.sleep(5)

    def _do_train(self, task_id, dataset_dir, config, resume=False):
        self._thread_local.task_id = task_id
        try:
            try:
                from paddlex import repo_manager
                rdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paddlex_repos")
                repo_manager.set_parent_dirs(rdir, None)
                repo_manager.setup(["PaddleSeg"])
                repo_manager.initialize(["PaddleSeg"])
            except Exception:
                pass

            label_name = config.get("label_name", "unknown")
            parallel_train = config.get("parallel_train", False)
            self._add_log(task_id, f"Training started: label={label_name}, parallel={parallel_train}")
            self.training_status[task_id]["status"] = "preparing"
            self.training_status[task_id]["progress"] = 5

            model_name = config.get("model_name", "STFPM")
            task_uuid = config.get("task_uuid", "unknown")
            save_dir = self.training_status.get(task_id, {}).get("save_dir") or \
                       os.path.join(self.output_dir, config.get("project_id", "default"), task_uuid, label_name)
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Training save_dir: {save_dir}")
            logger.info(f"Training dataset_dir: {dataset_dir}")
            if task_id in self.training_status:
                self.training_status[task_id].update({"dataset_dir": dataset_dir, "save_dir": save_dir, "config": config})
                self._persist_state_if_due(force=True)

            if not os.path.isdir(dataset_dir):
                raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
            train_list = os.path.join(dataset_dir, "train.txt")
            if not os.path.exists(train_list):
                raise FileNotFoundError(f"train.txt not found: {train_list}")

            train_iters_input = config.get("train_iters")
            train_epochs_input = config.get("train_epochs") or config.get("epochs", 50)
            iters_per_epoch = 1
            try:
                with open(train_list, "r") as f:
                    num_samples = sum(1 for line in f if line.strip())
                batch_size = config.get("batch_size", 8)
                batch_size = min(batch_size, max(1, num_samples))  # FIX: 原代码 num_samples-1 有误
                iters_per_epoch = max(1, num_samples // batch_size)
                if train_iters_input:
                    total_iters = train_iters_input
                    total_epochs = max(1, total_iters // iters_per_epoch)
                    self._add_log(task_id, f"train_iters={total_iters} (~{total_epochs} epochs)")
                else:
                    total_epochs = train_epochs_input
                    total_iters = total_epochs * iters_per_epoch
                    self._add_log(task_id, f"{total_epochs} epochs x {iters_per_epoch} iters/epoch = {total_iters} iters (n={num_samples})")
            except Exception as e:
                logger.warning(f"iter calc failed: {e}")
                total_iters = train_iters_input or train_epochs_input or 50
                total_epochs = train_epochs_input or 1
                num_samples, batch_size = 0, config.get("batch_size", 8)

            self.training_status[task_id].update({
                "total_epochs": total_epochs, "total_iters": total_iters,
                "use_iters_mode": bool(train_iters_input), "num_samples": num_samples, "batch_size": batch_size,
            })

            if parallel_train:
                assigned_gpu = self._assign_gpu(task_id)
                device, gpu_ids = f"gpu:{assigned_gpu}", [assigned_gpu]
            else:
                assigned_gpu = None
                can, gpu_ids, msg = self._check_gpu_for_batch_size(batch_size)
                if can:
                    device = f"gpu:{','.join(map(str, gpu_ids))}" if len(gpu_ids) > 1 else f"gpu:{gpu_ids[0]}"
                    self._add_log(task_id, f"GPU: {msg}")
                else:
                    device, gpu_ids = "gpu:0", [0]
                    self._add_log(task_id, f"Warning: {msg}, fallback to gpu:0")

            # FIX: 非并行模式也设置 CUDA_VISIBLE_DEVICES
            if not parallel_train:
                cuda_str = ",".join(map(str, gpu_ids))
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str
                self._add_log(task_id, f"CUDA_VISIBLE_DEVICES={cuda_str}")

            model_config_path = os.path.join(pdx_source_dir, "repo_apis", "PaddleSeg_api", "configs", f"{model_name}.yaml")
            if not os.path.exists(model_config_path):
                model_config_path = None

            pdx_cfg = AttrDict({
                "Global": AttrDict({"model": model_name, "dataset_dir": dataset_dir, "output": save_dir, "device": device}),
                "Train": AttrDict({
                    "epochs": total_epochs, "epochs_iters": total_iters, "batch_size": batch_size,
                    "learning_rate": config.get("learning_rate", 0.01), "num_classes": 1,
                    "pretrain_weight_path": None, "resume_path": None,
                    "log_interval": 1, "eval_interval": 5, "save_interval": 10,
                    "basic_config_path": model_config_path
                }),
                "Evaluate": AttrDict({"weight_path": None})
            })

            if "backbone" not in config:
                config["backbone"] = "resnet18"

            if resume:
                resume_path = self._find_latest_resume_path(save_dir)
                if resume_path:
                    m = re.search(r"iter_(\d+)", resume_path)
                    resume_iter = int(m.group(1)) if m else 0
                    if total_iters <= resume_iter:
                        total_iters = resume_iter + 1
                        total_epochs = (total_iters + iters_per_epoch - 1) // iters_per_epoch
                    pdx_cfg["Train"]["epochs_iters"] = total_iters
                    pdx_cfg["Train"]["epochs"] = total_epochs
                    pdx_cfg["Train"]["resume_path"] = resume_path
                    self.training_status[task_id].update({
                        "total_epochs": total_epochs, "total_iters": total_iters, "resume_path": resume_path
                    })
                    self._persist_state_if_due(force=True)
                    self._add_log(task_id, f"Resume from: {resume_path}")
                else:
                    self._add_log(task_id, "No checkpoint found, training from scratch.")

            from paddlex.modules.anomaly_detection import UadTrainer
            try:
                for name in ["paddleseg", "paddle", "paddlex"]:
                    l = logging.getLogger(name)
                    for h in l.handlers[:]:
                        if isinstance(h, logging.StreamHandler):
                            l.removeHandler(h)
                    l.propagate = False
            except Exception:
                pass

            import subprocess as _sp
            original_popen = _sp.Popen

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
                self._add_log(task_id, "Waiting for GPU (Queued)...")
                self._persist_state_if_due(force=True)

            def run_core_train():
                nonlocal monitor
                try:
                    paddleseg_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "PaddleX", "paddlex", "repo_manager", "repos", "PaddleSeg"
                    )
                    os.environ["PYTHONPATH"] = paddleseg_path + ":" + os.environ.get("PYTHONPATH", "")
                    os.environ["PADDLE_PDX_PADDLESEG_PATH"] = paddleseg_path
                    if parallel_train and assigned_gpu is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

                    self.training_status[task_id]["status"] = "starting"
                    self._add_log(task_id, "GPU acquired. Starting UadTrainer...")
                    trainer_obj = UadTrainer(pdx_cfg)
                    self.training_status[task_id]["progress"] = 10
                    self._add_log(task_id, "Stage 3/4: Core training...")

                    log_file = os.path.join(save_dir, "train.log")
                    monitor = TrainingMonitor(task_id, log_file, self)
                    monitor.start()

                    # ── 核心训练 ────────────────────────────────────────────────────
                    trainer_obj.train()

                    # ── 关键：训练完立即用 trainer_obj 校准 threshold ──────────────
                    # 此时模型已在 GPU 内存中加载，可直接推理
                    # 不依赖文件导出格式，是最可靠的方式
                    self._add_log(task_id, "Stage 4/4: Calibrating threshold with trainer_obj...")
                    self.training_status[task_id]["progress"] = 90
                    self._calibrate_threshold_with_trainer(
                        task_id=task_id,
                        trainer_obj=trainer_obj,
                        dataset_dir=dataset_dir,
                        save_dir=save_dir,
                        config=config,
                    )

                    self.training_status[task_id]["status"] = "completed"
                    self._finalize_model_artifacts(task_id, save_dir, config)
                    self.training_status[task_id]["progress"] = 100
                    self._add_log(task_id, "All stages complete.")

                except Exception as e:
                    self._add_log(task_id, f"Training failed: {e}")
                    self._add_log(task_id, traceback.format_exc())
                    self.training_status[task_id]["status"] = "failed"
                    self.training_status[task_id]["error"] = str(e)
                finally:
                    pdx_subprocess.subprocess.Popen = original_popen
                    self.processes.pop(task_id, None)
                    if monitor:
                        monitor.stop()
                    s = self.training_status.get(task_id, {})
                    if s.get("_training_timeout"):
                        s.pop("_training_timeout")
                        s.update({"status": "retrying", "error": None})
                        self._persist_state_if_due(force=True)
                        raise TimeoutError("Training timeout, will retry")
                    self._persist_state_if_due(force=True)

            if parallel_train:
                acquired = self._gpu_semaphore.acquire(blocking=False)
                if acquired:
                    try:
                        run_core_train()
                    finally:
                        self._gpu_semaphore.release()
                else:
                    self._wait_for_gpu_and_train(task_id, run_core_train, 300)
            else:
                self._wait_for_gpu_and_train(task_id, run_core_train, 1800)
        except Exception:
            raise

    def _wait_for_gpu_and_train(self, task_id, run_fn, max_wait):
        waited = 0
        while waited < max_wait:
            if self.training_status.get(task_id, {}).get("status") == "cancelled":
                self._add_log(task_id, "Cancelled while waiting for GPU.")
                return
            if self._gpu_semaphore.acquire(blocking=False):
                try:
                    run_fn()
                finally:
                    self._gpu_semaphore.release()
                return
            self._add_log(task_id, f"Waiting for GPU... ({waited}s)")
            self._persist_state_if_due(force=True)
            time.sleep(2)
            waited += 2
        self._add_log(task_id, "GPU wait timeout.")
        self.training_status[task_id].update({"status": "failed", "error": "GPU resource timeout"})
        self._persist_state_if_due(force=True)

    # ═══════════════════════════════════════════════════════════════════════
    #  核心修复：train() 完成后立即调用，用 trainer_obj 内置模型推理校准
    # ═══════════════════════════════════════════════════════════════════════
    def _calibrate_threshold_with_trainer(self, task_id, trainer_obj, dataset_dir, save_dir, config):
        self._add_log(task_id, "=== Threshold Calibration Start ===")
        status_info = self.training_status.get(task_id, {})

        # 优先级 1: 日志捕获
        val_from_log = status_info.get("threshold_from_log")
        if val_from_log and float(val_from_log) > 0.0:
            self._add_log(task_id, f"[P1] Threshold from training log: {float(val_from_log):.6f}")
            status_info.update({"threshold": float(val_from_log), "threshold_source": "training_log"})
            return

        # 优先级 2: metric.json
        best_model_dir = os.path.join(save_dir, "best_model")
        for mf in self._find_metric_jsons(save_dir, best_model_dir):
            val = self._read_threshold_from_metric_json(task_id, mf)
            if val and val > 0.0:
                status_info.update({"threshold": val, "threshold_source": f"metric_json:{os.path.relpath(mf, save_dir)}"})
                return

        # 优先级 3: 推理校准
        image_paths = self._collect_train_images(task_id, dataset_dir)
        if not image_paths:
            self._add_log(task_id, "WARNING: No train images found. threshold=-1.0")
            status_info.update({"threshold": -1.0, "threshold_source": "not_found"})
            return

        # 均匀采样最多 200 张
        max_samples = min(len(image_paths), 200)
        if len(image_paths) > max_samples:
            step = len(image_paths) / max_samples
            image_paths = [image_paths[int(i * step)] for i in range(max_samples)]
        self._add_log(task_id, f"[P3] Inferring {len(image_paths)} training images for calibration...")

        scores = self._infer_scores_via_trainer(task_id, trainer_obj, image_paths, config)

        if not scores:
            self._add_log(task_id, "WARNING: No scores from inference. threshold=-1.0")
            status_info.update({"threshold": -1.0, "threshold_source": "not_found"})
            return

        import numpy as np
        arr = np.array(scores, dtype=np.float32)
        percentile = config.get("threshold_percentile", 99)
        threshold = float(np.percentile(arr, percentile))
        stats = {
            "n": len(scores),
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "percentile_used": percentile,
            "threshold": threshold,
        }
        status_info.update({
            "threshold": threshold,
            "threshold_source": f"calibration_p{percentile}",
            "calibration_stats": stats,
        })
        self._add_log(
            task_id,
            f"Calibration OK: n={stats['n']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"p90={stats['p90']:.6f}, p95={stats['p95']:.6f}, p99={stats['p99']:.6f} "
            f"-> threshold={threshold:.6f} (p{percentile})"
        )
        self._add_log(task_id, "=== Threshold Calibration End ===")

    def _infer_scores_via_trainer(self, task_id, trainer_obj, image_paths, config):
        """
        策略 A: trainer_obj.predict() —— 标准接口
        策略 B: trainer_obj.model 直接前向 —— predict 不可用时的兜底
        策略 C: 重新创建 trainer 加载权重后推理 —— 训练子进程已退出的兜底
        """
        scores = []

        # 策略 A: trainer_obj.predict
        if hasattr(trainer_obj, 'predict'):
            self._add_log(task_id, "Trying trainer_obj.predict() for calibration...")
            try:
                # 先尝试批量
                try:
                    results = trainer_obj.predict(image_paths)
                    if results is not None:
                        for r in (results if hasattr(results, '__iter__') else [results]):
                            s = self._extract_score(r)
                            if s is not None:
                                scores.append(s)
                    if scores:
                        self._add_log(task_id, f"Batch predict: {len(scores)} scores.")
                        return scores
                except Exception as e_batch:
                    self._add_log(task_id, f"Batch predict failed: {e_batch}, trying single-image...")

                # 逐张
                for img_path in image_paths:
                    try:
                        result = trainer_obj.predict(img_path)
                        s = self._extract_score(result)
                        if s is not None:
                            scores.append(s)
                    except Exception as e:
                        self._add_log(task_id, f"predict({os.path.basename(img_path)}) err: {e}")
                if scores:
                    self._add_log(task_id, f"Single-image predict: {len(scores)} scores.")
                    return scores
            except Exception as e:
                self._add_log(task_id, f"trainer_obj.predict() unavailable: {e}")

        # 策略 B: 直接前向推理
        self._add_log(task_id, "Trying direct model forward pass for calibration...")
        try:
            scores = self._infer_via_model_forward(task_id, trainer_obj, image_paths, config)
            if scores:
                return scores
        except Exception as e:
            self._add_log(task_id, f"Direct forward failed: {e}")

        # 策略 C: 重新创建 trainer 并加载权重后推理
        self._add_log(task_id, "Trying recreate trainer for calibration...")
        try:
            scores = self._infer_via_new_trainer(task_id, image_paths, config)
            if scores:
                return scores
        except Exception as e:
            self._add_log(task_id, f"Recreate trainer failed: {e}\n{traceback.format_exc()}")

        return scores

    def _infer_via_model_forward(self, task_id, trainer_obj, image_paths, config):
            """
            利用 trainer_obj.pdx_model 中已实例化的网络结构，
            加载子进程训练好并保存在硬盘上的权重，进行纯底层的前向推理。
            """
            import numpy as np
            import cv2
            import paddle
            import os

            scores =[]
            input_size = config.get("input_size", [224, 224])
            h, w = input_size[1], input_size[0]
            # ImageNet 归一化参数 (PaddleX STFPM 标准)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            # 1. 获取网络结构 (遍历 pdx_model 寻找底层的 paddle.nn.Layer)
            net = None
            pdx_model = getattr(trainer_obj, "pdx_model", None)
            if pdx_model is not None:
                # 暴力遍历寻找 paddle.nn.Layer 对象
                for attr_name in dir(pdx_model):
                    try:
                        attr_val = getattr(pdx_model, attr_name)
                        # 只要继承了 paddle.nn.Layer，就是真正的神经网络
                        if isinstance(attr_val, paddle.nn.Layer):
                            net = attr_val
                            self._add_log(task_id, f"Found neural network layer at pdx_model.{attr_name}")
                            break
                    except Exception:
                        continue

            if net is None:
                self._add_log(task_id, "Cannot find a paddle.nn.Layer inside trainer_obj.pdx_model.")
                return scores

            # 2. 定位硬盘上最新训练出的权重文件
            save_dir = self.training_status.get(task_id, {}).get("save_dir", "")
            best_model_dir = os.path.join(save_dir, "best_model")
            pdparams_path = os.path.join(best_model_dir, "model.pdparams")
            if not os.path.exists(pdparams_path):
                pdparams_path = os.path.join(save_dir, "model.pdparams")
                
            if not os.path.exists(pdparams_path):
                self._add_log(task_id, f"Cannot find model.pdparams at {pdparams_path}")
                return scores

            # 3. 将硬盘上的训练权重加载到我们找到的网络中
            try:
                self._add_log(task_id, f"Loading trained weights from {pdparams_path}...")
                state_dict = paddle.load(pdparams_path)
                net.set_state_dict(state_dict)
                self._add_log(task_id, "Weights loaded successfully.")
            except Exception as e:
                self._add_log(task_id, f"Failed to load weights: {e}")
                return scores

            # 4. 执行直接前向推理
            net.eval()
            failed = 0
            with paddle.no_grad():
                for img_path in image_paths:
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            failed += 1
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
                        img = (img - mean) / std
                        tensor = paddle.to_tensor(np.transpose(img, (2, 0, 1))[np.newaxis])  #[1,C,H,W]
                        
                        output = net(tensor)

                        score = None
                        # 解析STFPM或类似模型的输出
                        if isinstance(output, (tuple, list)) and len(output) >= 2:
                            t_feat, s_feat = output[0], output[1]
                            # 计算特征图差异的均方误差作为 score
                            score = float(paddle.mean(paddle.square(t_feat - s_feat)).item())
                        elif isinstance(output, paddle.Tensor):
                            score = float(paddle.mean(paddle.abs(output)).item())
                        elif isinstance(output, dict):
                            for k in ("anomaly_map", "score", "pred", "output"):
                                if k in output and output[k] is not None:
                                    score = float(paddle.mean(paddle.abs(output[k])).item())
                                    break

                        if score is not None and score == score:  # NaN check
                            scores.append(score)
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        if failed <= 5:
                            self._add_log(task_id, f"Forward fail ({os.path.basename(img_path)}): {e}")

            self._add_log(task_id, f"Direct forward: {len(scores)} scored, {failed} failed out of {len(image_paths)}")
            return scores

    def _infer_via_new_trainer(self, task_id, image_paths, config):
        """
        策略 C: 直接从 PaddleSeg 创建 STFPM 模型并加载权重进行推理
        当训练子进程已退出，无法直接使用 trainer_obj 时使用
        """
        import numpy as np
        import cv2
        import paddle
        import paddle.nn as nn
        import paddle.nn.functional as F

        scores = []
        status_info = self.training_status.get(task_id, {})
        save_dir = status_info.get("save_dir", "")

        pdparams_path = os.path.join(save_dir, "best_model", "model.pdparams")
        if not os.path.exists(pdparams_path):
            pdparams_path = os.path.join(save_dir, "model.pdparams")
        if not os.path.exists(pdparams_path):
            self._add_log(task_id, f"Cannot find model.pdparams at {save_dir}")
            return scores

        try:
            import importlib.util
            
            paddleseg_models_path = os.path.join(pdx_repos_dir, "paddleseg")
            resnet_ms3_file = os.path.join(paddleseg_models_path, "models", "backbones", "resnet_ms3.py")
            
            self._add_log(task_id, f"Loading ResNet_MS3 from: {resnet_ms3_file}")
            
            spec = importlib.util.spec_from_file_location("resnet_ms3_module", resnet_ms3_file)
            resnet_ms3_module = importlib.util.module_from_spec(spec)
            
            import sys
            if paddleseg_models_path not in sys.path:
                sys.path.insert(0, paddleseg_models_path)
            
            sys.modules['paddleseg.models.backbones.resnet_ms3'] = resnet_ms3_module
            spec.loader.exec_module(resnet_ms3_module)
            
            ResNet_MS3 = resnet_ms3_module.ResNet_MS3
            self._add_log(task_id, "ResNet_MS3 loaded successfully.")
            
            from paddle.vision.models.resnet import resnet18, resnet34, resnet50, resnet101
            
            backbone_name_map = {
                resnet18: "resnet18",
                resnet34: "resnet34",
                resnet50: "resnet50",
                resnet101: "resnet101",
            }
            
            class STFPM(nn.Layer):
                def __init__(self, num_classes, backbone):
                    super(STFPM, self).__init__()
                    arch = backbone_name_map.get(type(backbone), "resnet18")
                    self.student = ResNet_MS3(pretrained=False, arch=arch)
                    self.teacher = ResNet_MS3(pretrained=True, arch=arch)
                    self.teacher.eval()

                def forward(self, x):
                    stu = self.student(x)
                    if self.teacher.training:
                        self.teacher.eval()
                    with paddle.no_grad():
                        tea = self.teacher(x)
                    if self.student.training:
                        return [[stu, tea]]
                    else:
                        score_map = 1.
                        t_feat = tea
                        s_feat = stu
                        for j in range(len(t_feat)):
                            t_feat[j] = F.normalize(t_feat[j], axis=1)
                            s_feat[j] = F.normalize(s_feat[j], axis=1)
                            sm = paddle.sum((t_feat[j] - s_feat[j])**2, 1, keepdim=True)
                            sm = F.interpolate(sm, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                            score_map = score_map * sm
                        return [score_map]
            
            self._add_log(task_id, "STFPM class defined successfully.")
            
            config_json_path = os.path.join(save_dir, "config.json")
            model_cfg = {}
            if os.path.exists(config_json_path):
                import json
                with open(config_json_path, 'r') as f:
                    model_cfg = json.safe_load(f)
            
            backbone_name = model_cfg.get("backbone", "resnet18")
            self._add_log(task_id, f"Creating STFPM model with backbone: {backbone_name}")
            
            backbone_dict = {
                "resnet18": resnet18,
                "resnet34": resnet34,
                "resnet50": resnet50,
                "resnet101": resnet101,
            }
            backbone_fn = backbone_dict.get(backbone_name, resnet18)
            backbone = backbone_fn(pretrained=False)
            
            net = STFPM(num_classes=1, backbone=backbone)
            self._add_log(task_id, "STFPM model created.")

            state_dict = paddle.load(pdparams_path)
            net.set_state_dict(state_dict)
            self._add_log(task_id, "Weights loaded successfully.")

            input_size = config.get("input_size", [224, 224])
            h, w = input_size[1], input_size[0]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            net.eval()
            failed = 0
            self._add_log(task_id, f"Running inference on {len(image_paths)} images...")
            with paddle.no_grad():
                for img_path in image_paths:
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            failed += 1
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
                        img = (img - mean) / std
                        tensor = paddle.to_tensor(np.transpose(img, (2, 0, 1))[np.newaxis])

                        output = net(tensor)

                        score = None
                        if isinstance(output, (tuple, list)):
                            score_map = output[0]
                            if isinstance(score_map, list):
                                score_map = score_map[0]
                            score = float(paddle.max(score_map).item())
                        elif isinstance(output, paddle.Tensor):
                            score = float(paddle.max(output).item())

                        if score is not None and score == score:
                            scores.append(score)
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        if failed <= 5:
                            self._add_log(task_id, f"Forward fail ({os.path.basename(img_path)}): {e}")

            self._add_log(task_id, f"STFPM inference: {len(scores)} scores, {failed} failed out of {len(image_paths)}")
            return scores

        except Exception as e:
            self._add_log(task_id, f"STFPM inference failed: {e}\n{traceback.format_exc()}")

        return scores

    def _extract_score(self, result):
        """从推理结果中提取 anomaly score 标量，兼容多种返回格式。"""
        if result is None:
            return None
        import numpy as np
        if hasattr(result, '__iter__') and not isinstance(result, dict):
            try:
                result = next(iter(result))
            except StopIteration:
                return None
        if isinstance(result, dict):
            for key in ("score", "anomaly_score", "pred_score", "scores", "label_score"):
                if key in result and result[key] is not None:
                    val = result[key]
                    return float(np.max(val)) if hasattr(val, '__len__') else float(val)
        for attr in ("score", "anomaly_score"):
            if hasattr(result, attr):
                return float(getattr(result, attr))
        return None

    def _collect_train_images(self, task_id, dataset_dir):
        """从 train.txt 收集训练图路径。"""
        train_list = os.path.join(dataset_dir, "train.txt")
        if not os.path.exists(train_list):
            self._add_log(task_id, f"train.txt not found: {train_list}")
            return []
        paths = []
        with open(train_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path = line.split()[0]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(dataset_dir, img_path)
                if os.path.exists(img_path):
                    paths.append(img_path)
        self._add_log(task_id, f"Collected {len(paths)} images from train.txt")
        return paths

    def _find_metric_jsons(self, save_dir, best_model_dir):
        candidates = []
        for p in [os.path.join(best_model_dir, "metric.json"), os.path.join(save_dir, "metric.json")]:
            if os.path.exists(p):
                candidates.append(p)
        for root, _, files in os.walk(save_dir):
            if "metric.json" in files:
                c = os.path.join(root, "metric.json")
                if c not in candidates:
                    candidates.append(c)
        return candidates

    def _read_threshold_from_metric_json(self, task_id, metric_file):
        try:
            with open(metric_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._add_log(task_id, f"metric.json keys ({os.path.basename(os.path.dirname(metric_file))}): {list(data.keys())}")
            for key in ("threshold", "best_threshold", "optimal_threshold", "score_threshold"):
                if key in data:
                    val = float(data[key])
                    if val > 0.0:
                        self._add_log(task_id, f"[P2] Threshold from metric.json['{key}']: {val:.6f}")
                        return val
        except Exception as e:
            self._add_log(task_id, f"Cannot parse {metric_file}: {e}")
        return None

    def _finalize_model_artifacts(self, task_id, save_dir, config):
        if not save_dir or not os.path.exists(save_dir):
            self._add_log(task_id, "Warning: save_dir not found.")
            return
        try:
            # 1. 复制 model.pdparams
            best_model_dir = os.path.join(save_dir, "best_model")
            src = os.path.join(best_model_dir, "model.pdparams")
            if not os.path.exists(src):
                for root, _, files in os.walk(save_dir):
                    if "model.pdparams" in files and "best_model" in root:
                        src = os.path.join(root, "model.pdparams")
                        break
            if not os.path.exists(src):
                self._add_log(task_id, "Warning: best_model/model.pdparams not found.")
                return
            dst = os.path.join(save_dir, "model.pdparams")
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                self._add_log(task_id, "Copied model.pdparams to root")

            # 2. 读取校准好的 threshold（由 _calibrate_threshold_with_trainer 写入）
            status_info = self.training_status.get(task_id, {})
            threshold = status_info.get("threshold", -1.0)
            threshold_source = status_info.get("threshold_source", "not_found")
            calibration_stats = status_info.get("calibration_stats")

            # 3. 写 config.json
            label_name = config.get("label_name", status_info.get("label", "unknown"))
            config_data = {
                "category": label_name,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "input_size": config.get("input_size", [224, 224]),
                "num_samples": status_info.get("num_samples", 0),
                "model_name": config.get("model_name", "STFPM"),
                "backbone": config.get("backbone", "resnet18"),
                "total_iters": status_info.get("total_iters", 0),
                "batch_size": status_info.get("batch_size", config.get("batch_size", 8)),
                "learning_rate": config.get("learning_rate", 0.01),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "calibration_stats": calibration_stats,
            }
            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            self._add_log(task_id, f"config.json: threshold={threshold:.6f}, source={threshold_source}")

            # 4. 保存训练数据和模板
            self._save_training_data_and_template(task_id, save_dir, config)

            # 5. 清理
            keep_files = {"model.pdparams", "config.json", "train.log", "template.jpg"}
            keep_dirs = {"training_data"}
            rd = rf = 0
            for item in os.listdir(save_dir):
                p = os.path.join(save_dir, item)
                if os.path.isdir(p):
                    if item not in keep_dirs:
                        shutil.rmtree(p)
                        rd += 1
                elif os.path.isfile(p) and item not in keep_files:
                    os.remove(p)
                    rf += 1
            self._add_log(task_id, f"Cleanup: removed {rd} dirs, {rf} files.")
        except Exception as e:
            self._add_log(task_id, f"Warning: finalize failed: {e}\n{traceback.format_exc()}")

    def _save_training_data_and_template(self, task_id, save_dir, config):
        try:
            status_info = self.training_status.get(task_id, {})
            dataset_dir = status_info.get("dataset_dir", config.get("dataset_dir", ""))
            if not dataset_dir:
                return
            raw_data_dir = Path(dataset_dir).parent
            raw_images_dir = raw_data_dir / "raw_images"
            annotation_file = raw_data_dir / "annotations.json"
            if not annotation_file.exists() or not raw_images_dir.exists():
                self._add_log(task_id, f"Warning: annotations.json or raw_images/ not found under {raw_data_dir}")
                return
            td = os.path.join(save_dir, "training_data")
            os.makedirs(td, exist_ok=True)
            shutil.copy2(annotation_file, os.path.join(td, "annotations.json"))
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            images = annotations.get('images', [])
            if not images:
                return
            closest = min(images, key=lambda img: abs(img.get('angle', 0)))
            cnt = 0
            for img in images:
                src = raw_images_dir / img['file_name']
                if src.exists():
                    shutil.copy2(src, os.path.join(td, img['file_name']))
                    cnt += 1
            self._add_log(task_id, f"Saved {cnt} training images.")
            tmpl = os.path.join(td, closest['file_name'])
            if os.path.exists(tmpl):
                shutil.copy2(tmpl, os.path.join(save_dir, "template.jpg"))
                self._add_log(task_id, f"Template: {closest['file_name']} (angle={closest.get('angle', 0)})")
        except Exception as e:
            self._add_log(task_id, f"Warning: save training data failed: {e}")

    def resume_task(self, task_id, resume_path=None, resume_mode=None):
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}
        task_info = self.training_status[task_id]
        thread = self.threads.get(task_id)
        if task_info.get("status") in {"starting", "training"} and thread and thread.is_alive():
            return {"status": "error", "message": "Task is already running"}
        if resume_mode:
            task_info.setdefault("config", {})["resume_mode"] = resume_mode
        actual = resume_path or self._find_latest_resume_path(task_info.get("save_dir"))
        if not actual or not os.path.exists(actual):
            return {"status": "error", "message": "No valid checkpoint found"}
        task_info.update({"status": "starting", "progress": min(task_info.get("progress", 0), 90), "resume_path": actual})
        self._add_log(task_id, f"Manual resume: {actual}")
        t = threading.Thread(target=self._train_process, args=(task_id, task_info.get("dataset_dir"), task_info.get("config"), True))
        self.threads[task_id] = t
        t.start()
        self._persist_state_if_due(force=True)
        return {"status": "success", "task_id": task_id, "resume_path": actual}

    def get_status(self, task_id):
        if task_id in self.training_status:
            return self.training_status[task_id]
        for tid, status in self.training_status.items():
            if status.get("task_uuid") == task_id:
                return status
        return {"status": "not_found"}


# 全局单例
trainer = AnomalyTrainer()
