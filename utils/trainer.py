import os
import threading
import time
import shutil
import json
import re
from pathlib import Path
import paddlex as pdx
from paddlex.utils.config import AttrDict

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
            return

        last_pos = 0
        while not self.stop_event.is_set():
            last_pos = self._read_log_lines(last_pos)
            time.sleep(1) # 增加频率到 1秒
            
        # 训练结束后，最后再读一次以防遗漏
        self._read_log_lines(last_pos)

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
            print(f"Error reading log {self.log_file}: {e}")
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
            # 改进进度计算：考虑 iter 和 epoch
            # 如果 epoch 始终为 1，则主要看 iter/total_iters
            if total_epochs == 1 or epoch == 1:
                # 如果只有 1 个 epoch，或者当前是第 1 个 epoch
                progress_in_epoch = curr_iter / total_iters
                if total_epochs > 1:
                    progress = 30 + (progress_in_epoch / total_epochs) * 60
                else:
                    progress = 30 + progress_in_epoch * 60
            else:
                progress = 30 + ((epoch - 1) / total_epochs + (curr_iter / total_iters) / total_epochs) * 60
            
            self.trainer.training_status[self.task_id]["progress"] = min(int(progress), 99)
            if "metrics" not in self.trainer.training_status[self.task_id]:
                self.trainer.training_status[self.task_id]["metrics"] = []
            self.trainer.training_status[self.task_id]["metrics"].append(metrics)
            
            # 限制 metrics 数量
            if len(self.trainer.training_status[self.task_id]["metrics"]) > 1000:
                self.trainer.training_status[self.task_id]["metrics"] = self.trainer.training_status[self.task_id]["metrics"][-1000:]

        # 解析评估指标
        eval_match = re.search(r"\[EVAL\]\s+#Images:\s*(\d+)\s+mIoU:\s*([\d.]+)", line)
        if eval_match:
            miou = float(eval_match.group(2))
            if "eval_metrics" not in self.trainer.training_status[self.task_id]:
                self.trainer.training_status[self.task_id]["eval_metrics"] = []
            self.trainer.training_status[self.task_id]["eval_metrics"].append({"miou": miou, "timestamp": time.time()})

class AnomalyTrainer:
    """
    负责管理 PaddleX 异常检测训练流程
    """
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.training_status = {} # 用于跟踪训练状态

    def run_training_async(self, dataset_dir: str, config: dict):
        """
        启动后台线程执行训练
        """
        task_id = f"task_{int(time.time())}_{config.get('label_name', 'unknown')}"
        self.training_status[task_id] = {
            "status": "starting", 
            "progress": 0,
            "label": config.get("label_name", "unknown"),
            "logs": [f"Task {task_id} initialized."],
            "metrics": [], # 存储 Epoch 级别的指标
            "total_epochs": config.get("epochs", 50),
            "start_time": time.time()
        }
        
        thread = threading.Thread(
            target=self._train_process,
            args=(task_id, dataset_dir, config)
        )
        thread.start()
        return task_id

    def _add_log(self, task_id: str, message: str):
        """添加日志到任务状态"""
        if task_id in self.training_status:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            log_entry = f"[{timestamp}] {message}"
            self.training_status[task_id].setdefault("logs", []).append(log_entry)
            print(f"[{task_id}] {message}")

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
            self.training_status[task_id]["progress"] = 10

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

            self._add_log(task_id, f"Initializing {model_name} trainer...")
            from paddlex.modules.anomaly_detection import UadTrainer
            trainer_obj = UadTrainer(pdx_cfg)
            
            self.training_status[task_id]["progress"] = 30
            self._add_log(task_id, "Starting PaddleX training process...")
            
            # 2. 启动日志监控
            log_file = os.path.join(save_dir, "train.log")
            monitor = TrainingMonitor(task_id, log_file, self)
            monitor.start()
            
            # 3. 执行训练
            try:
                trainer_obj.train()
                self._add_log(task_id, "PaddleX training completed successfully.")
                self.training_status[task_id]["status"] = "completed"
                self.training_status[task_id]["progress"] = 100
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
