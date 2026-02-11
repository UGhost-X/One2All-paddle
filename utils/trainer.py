import os
import threading
import time
import shutil
import json
import re
import random
import ctypes
import logging
from pathlib import Path
import paddlex as pdx
from paddlex.utils.config import AttrDict

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """
    负责监控日志文件并更新训练状态
    """
    def __init__(self, task_id: str, log_file: str, trainer_instance):
        self.task_id = task_id
        self.log_file = log_file
        self.trainer = trainer_instance
        self.stop_event = threading.Event()

    def start(self):
        self.trainer._add_log(self.task_id, f"Log monitor started for {os.path.basename(self.log_file)}")
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stop_event.set()

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
                line = f.readline()
                if not line:
                    if self.stop_event.is_set():
                        # 在退出前再尝试读一次，确保不漏掉最后几行
                        remaining_line = f.readline()
                        if remaining_line:
                            self._parse_line(remaining_line)
                        break
                    time.sleep(0.5)
                    continue
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
            
            metrics = {
                "epoch": epoch,
                "iter": curr_iter,
                "total_iters": total_iters,
                "loss": loss,
                "lr": lr,
                "timestamp": time.time()
            }
            
            # 更新进度 (iter 级别)
            total_epochs = self.trainer.training_status[self.task_id].get("total_epochs", 50)
            
            # 改进进度计算：
            # 0-5%: 初始化
            # 5-10%: 模型准备
            # 10-95%: 核心训练阶段
            # 95-100%: 完成
            
            training_ratio = (epoch - 1) / total_epochs + (curr_iter / total_iters) / total_epochs
            progress = 10 + (training_ratio * 85)
            
            new_progress = min(int(progress), 98)
            old_progress = self.trainer.training_status[self.task_id].get("progress", 0)
            
            # 记录详细日志
            if new_progress > old_progress or curr_iter == total_iters:
                log_msg = f"Training Progress: {new_progress}% | Epoch: {epoch}/{total_epochs} | Iter: {curr_iter}/{total_iters} | Loss: {loss:.4f} | LR: {lr:.6f}"
                self.trainer._add_log(self.task_id, log_msg)
            
            self.trainer.training_status[self.task_id]["progress"] = new_progress
            if "metrics" not in self.trainer.training_status[self.task_id]:
                self.trainer.training_status[self.task_id]["metrics"] = []
            self.trainer.training_status[self.task_id]["metrics"].append(metrics)
            
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

    def run_training_async(self, dataset_dir: str, config: dict, group_id: str = None):
        """
        启动后台线程执行训练
        """
        task_id = f"task_{int(time.time())}_{config.get('label_name', 'unknown')}_{random.randint(1000, 9999)}"
        self.training_status[task_id] = {
            "status": "starting", 
            "progress": 0,
            "label": config.get("label_name", "unknown"),
            "group_id": group_id,
            "logs": [f"Task {task_id} initialized."],
            "metrics": [], # 存储 Epoch 级别的指标
            "total_epochs": config.get("epochs", 50),
            "start_time": time.time()
        }
        
        # 记录到组
        if group_id:
            if group_id not in self.groups:
                self.groups[group_id] = []
            self.groups[group_id].append(task_id)
        
        thread = threading.Thread(
            target=self._train_process,
            args=(task_id, dataset_dir, config)
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

        # 尝试停止线程
        thread = self.threads.get(task_id)
        if thread and thread.is_alive():
            # 使用 ctypes 强制停止线程
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), 
                ctypes.py_object(SystemExit)
            )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), None)
            
            self._add_log(task_id, "Task cancellation requested by user.")
            self.training_status[task_id]["status"] = "cancelled"
            return {"status": "success", "message": "Cancellation requested"}
        else:
            self.training_status[task_id]["status"] = "cancelled"
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
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            log_entry = f"[{timestamp}] {message}"
            self.training_status[task_id].setdefault("logs", []).append(log_entry)
            logger.info(f"[{task_id}] {message}")

    def _train_process(self, task_id: str, dataset_dir: str, config: dict):
        """
        实际的训练进程 (运行在后台)
        """
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
            self._add_log(task_id, f"Training thread started for label: {label_name}")
            self.training_status[task_id]["status"] = "training"
            self.training_status[task_id]["progress"] = 5
            self._add_log(task_id, "Stage 1/4: Preparing training environment...")

            # 1. 构造 PaddleX 训练配置
            model_name = config.get("model_name", "STFPM")
            save_dir = os.path.join(self.output_dir, config.get("project_id", "default"), label_name, task_id)
            
            pdx_cfg = AttrDict({
                "Global": AttrDict({
                    "model": model_name,
                    "dataset_dir": dataset_dir,
                    "output": save_dir,
                    "device": "gpu:0"
                }),
                "Train": AttrDict({
                    "epochs": config.get("epochs", 50),
                    "epochs_iters": config.get("epochs", 50),
                    "batch_size": config.get("batch_size", 8),
                    "learning_rate": config.get("learning_rate", 0.01),
                    "num_classes": 1,
                    "pretrain_weight_path": None,
                    "resume_path": None,
                    "log_interval": 1,
                    "eval_interval": 1,
                    "save_interval": 1
                }),
                "Evaluate": AttrDict({
                    "weight_path": None
                })
            })
            
            self._add_log(task_id, f"Stage 2/4: Initializing {model_name} trainer...")
            from paddlex.modules.anomaly_detection import UadTrainer
            trainer_obj = UadTrainer(pdx_cfg)
            
            self.training_status[task_id]["progress"] = 10
            self._add_log(task_id, "Stage 3/4: Starting core training process...")
            
            # 2. 启动日志监控
            log_file = os.path.join(save_dir, "train.log")
            monitor = TrainingMonitor(task_id, log_file, self)
            monitor.start()
            
            # 3. 执行训练
            try:
                trainer_obj.train()
                self._add_log(task_id, "Stage 4/4: Finalizing and saving model...")
                time.sleep(1) # 给日志监控一点时间同步最后几行
                
                # 显式输出 100% 进度日志
                self.training_status[task_id]["progress"] = 100
                self._add_log(task_id, "Training Progress: 100% | Status: Completed")
                self._add_log(task_id, "PaddleX training completed successfully.")
                self.training_status[task_id]["status"] = "completed"
            finally:
                monitor.stop()
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self._add_log(task_id, error_msg)
            self.training_status[task_id]["status"] = "failed"
            self.training_status[task_id]["error"] = str(e)

    def get_status(self, task_id: str):
        """
        查询训练状态
        """
        return self.training_status.get(task_id, {"status": "not_found"})

# 全局单例
trainer = AnomalyTrainer()
