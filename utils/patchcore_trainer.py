#!/usr/bin/env python3
"""
PatchCore 异常检测模型训练脚本
遵循与 trainer.py 相同的输出格式和训练流程
"""
import os
import sys
import time
import json
import random
import shutil
import logging
import traceback
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import cv2

# 延迟导入paddle，避免初始化冲突
_paddle_module = None
_nn_module = None
_F_module = None

def _import_paddle():
    global _paddle_module, _nn_module, _F_module
    if _paddle_module is None:
        import paddle
        import paddle.nn as nn
        import paddle.nn.functional as F
        _paddle_module = paddle
        _nn_module = nn
        _F_module = F
    return _paddle_module, _nn_module, _F_module

def _get_resnet(backbone_name):
    """动态获取ResNet模型"""
    paddle, _, _ = _import_paddle()

    # 手动定义ResNet18/50，避免通过paddle.vision.models导入
    from paddle.vision.models.resnet import BasicBlock, BottleneckBlock

    class SimpleResNet(paddle.nn.Layer):
        def __init__(self, block, depth, num_classes=1000, with_pool=True):
            super(SimpleResNet, self).__init__()
            layer_cfg = {
                18: [2, 2, 2, 2],
                34: [3, 4, 6, 3],
                50: [3, 4, 6, 3],
                101: [3, 4, 23, 3],
                152: [3, 8, 36, 3],
            }
            layers = layer_cfg[depth]
            self.with_pool = with_pool
            self.num_classes = num_classes

            self.conv1 = paddle.nn.Conv2D(3, 64, 7, stride=2, padding=3, bias_attr=False)
            self.bn1 = paddle.nn.BatchNorm2D(64)
            self.relu = paddle.nn.ReLU()
            self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

            self.inplanes = 64
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = paddle.nn.Sequential(
                    paddle.nn.Conv2D(self.inplanes, planes * block.expansion, 1, stride=stride, bias_attr=False),
                    paddle.nn.BatchNorm2D(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return paddle.nn.Sequential(*layers)

    if backbone_name == "resnet18":
        model = SimpleResNet(BasicBlock, 18)
        # 加载预训练权重
        try:
            state_dict = paddle.load('/root/.paddle/weights/resnet18_pretrained.pdparams')
            model.set_state_dict(state_dict)
        except:
            pass  # 如果没有预训练权重，使用随机初始化
    elif backbone_name == "resnet50":
        model = SimpleResNet(BottleneckBlock, 50)
        try:
            state_dict = paddle.load('/root/.paddle/weights/resnet50_pretrained.pdparams')
            model.set_state_dict(state_dict)
        except:
            pass
    else:
        model = SimpleResNet(BasicBlock, 18)

    return model

logger = logging.getLogger(__name__)


class PatchCoreTrainer:
    """
    PatchCore 模型训练器
    使用预训练ResNet提取特征，构建内存库，通过核心集采样减少内存占用
    """

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.training_status = {}
        self.threads = {}
        self._state_lock = threading.Lock()
        self._last_persist_ts = 0.0
        self.state_file = str(Path(output_dir) / "_patchcore_trainer_state.json")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        """加载训练状态"""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.training_status = data.get("training_status", {}) or {}
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
        """持久化训练状态"""
        data = {
            "version": 1,
            "updated_at": time.time(),
            "training_status": self.training_status.copy(),
        }
        tmp = f"{self.state_file}.tmp"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.state_file)

    def _persist_state_if_due(self, force=False):
        """按需持久化状态"""
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

    def _add_log(self, task_id, message):
        """添加日志"""
        if task_id in self.training_status:
            entry = f"[{time.strftime('%H:%M:%S')}] {message}"
            logs = self.training_status[task_id].setdefault("logs", [])
            logs.append(entry)
            if len(logs) > 500:
                self.training_status[task_id]["logs"] = logs[-500:]
            logger.info(f"[{task_id}] {message}")
            self._persist_state_if_due()

    def _make_task_key(self, dataset_dir, config):
        """生成任务唯一标识"""
        return "|".join([
            str(config.get("project_id", "")),
            str(config.get("task_uuid", "")),
            str(config.get("model_name", "")),
            str(config.get("label_name", "")),
            str(dataset_dir),
        ])

    def run_training_async(self, dataset_dir, config):
        """
        异步启动训练
        """
        task_key = self._make_task_key(dataset_dir, config)
        existing_task_id = None

        # 查找现有任务
        for tid, s in self.training_status.items():
            if s.get("task_key") == task_key:
                existing_task_id = tid
                break

        if existing_task_id:
            task_id = existing_task_id
            existing = self.training_status[task_id]
            status = existing.get("status")
            thread = self.threads.get(task_id)

            if status in {"starting", "training"} and thread and thread.is_alive():
                self._persist_state_if_due()
                return task_id

            # 重启任务
            existing.update({
                "status": "starting",
                "progress": 0,
                "logs": [],
                "metrics": [],
                "start_time": time.time(),
                "error": None
            })
            self._add_log(task_id, "Task restarted from scratch.")
        else:
            task_id = f"patchcore_{int(time.time())}_{config.get('label_name', 'unknown')}_{random.randint(1000, 9999)}"
            label_name = config.get("label_name", "unknown")
            task_uuid = config.get("task_uuid", "unknown")
            save_dir = os.path.join(
                self.output_dir,
                config.get("project_id", "default"),
                task_uuid,
                label_name
            )
            self.training_status[task_id] = {
                "status": "starting",
                "progress": 0,
                "label": label_name,
                "task_uuid": task_uuid,
                "logs": [f"Task {task_id} initialized."],
                "metrics": [],
                "total_epochs": 1,  # PatchCore不需要多轮训练
                "start_time": time.time(),
                "dataset_dir": dataset_dir,
                "save_dir": save_dir,
                "config": config,
                "task_key": task_key,
            }
            self._persist_state_if_due(force=True)

        t = threading.Thread(target=self._train_process, args=(task_id, dataset_dir, config))
        self.threads[task_id] = t
        t.start()
        return task_id

    def _train_process(self, task_id, dataset_dir, config):
        """训练进程"""
        try:
            self._do_train(task_id, dataset_dir, config)
        except (SystemExit, KeyboardInterrupt):
            if task_id in self.training_status:
                self.training_status[task_id]["status"] = "cancelled"
            self._persist_state_if_due(force=True)
        except Exception as e:
            self._add_log(task_id, f"Training failed: {e}")
            self.training_status[task_id].update({
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            self._persist_state_if_due(force=True)

    def _do_train(self, task_id, dataset_dir, config):
        """执行训练"""
        self._add_log(task_id, f"PatchCore training started: label={config.get('label_name')}")
        self.training_status[task_id]["status"] = "preparing"
        self.training_status[task_id]["progress"] = 5

        label_name = config.get("label_name", "unknown")
        task_uuid = config.get("task_uuid", "unknown")
        save_dir = self.training_status[task_id].get("save_dir") or \
                   os.path.join(self.output_dir, config.get("project_id", "default"), task_uuid, label_name)
        os.makedirs(save_dir, exist_ok=True)

        self.training_status[task_id].update({
            "dataset_dir": dataset_dir,
            "save_dir": save_dir,
            "config": config
        })
        self._persist_state_if_due(force=True)

        # 加载训练图像
        train_images = self._load_train_images(task_id, dataset_dir)
        if not train_images:
            raise ValueError("No training images found")

        num_samples = len(train_images)
        self.training_status[task_id]["num_samples"] = num_samples
        self._add_log(task_id, f"Loaded {num_samples} training images")

        # 构建PatchCore模型
        backbone_name = config.get("backbone", "resnet18")
        input_size = config.get("input_size", [224, 224])
        coreset_ratio = config.get("coreset_ratio", 0.1)  # 核心集采样比例

        self._add_log(task_id, f"Building PatchCore model (backbone={backbone_name}, coreset_ratio={coreset_ratio})")
        self.training_status[task_id]["status"] = "training"
        self.training_status[task_id]["progress"] = 10

        # 创建特征提取器
        feature_extractor = self._build_feature_extractor(backbone_name)
        feature_extractor.eval()

        # 提取训练集特征
        self._add_log(task_id, "Stage 1/3: Extracting features from training images...")
        features = self._extract_features(task_id, feature_extractor, train_images, input_size)

        if len(features) == 0:
            raise ValueError("No features extracted")

        self._add_log(task_id, f"Extracted {len(features)} feature patches")
        self.training_status[task_id]["progress"] = 50

        # 核心集采样
        self._add_log(task_id, "Stage 2/3: Applying coreset sampling...")
        memory_bank = self._coreset_sampling(features, coreset_ratio)
        self._add_log(task_id, f"Memory bank size: {memory_bank.shape}")
        self.training_status[task_id]["progress"] = 70

        # 保存内存库
        memory_bank_path = os.path.join(save_dir, "memory_bank.npz")
        np.savez_compressed(memory_bank_path, memory_bank=memory_bank)
        self._add_log(task_id, f"Memory bank saved to {memory_bank_path}")

        # 阈值校准
        self._add_log(task_id, "Stage 3/3: Calibrating threshold...")
        self.training_status[task_id]["progress"] = 80
        threshold = self._calibrate_threshold(task_id, feature_extractor, memory_bank, train_images, input_size, config)

        # 保存配置
        config_data = {
            "category": label_name,
            "threshold": threshold,
            "threshold_source": "calibration_p95",
            "input_size": input_size,
            "num_samples": num_samples,
            "model_name": "PatchCore",
            "backbone": backbone_name,
            "coreset_ratio": coreset_ratio,
            "memory_bank_shape": list(memory_bank.shape),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        self._add_log(task_id, f"config.json: threshold={threshold:.6f}")

        # 保存训练数据和模板
        self._save_training_data_and_template(task_id, save_dir, config)

        # 完成任务
        self.training_status[task_id]["status"] = "completed"
        self.training_status[task_id]["progress"] = 100
        self._add_log(task_id, "All stages complete.")
        self._persist_state_if_due(force=True)

    def _build_feature_extractor(self, backbone_name: str):
        """构建特征提取器"""
        backbone = _get_resnet(backbone_name)

        # 冻结参数
        for param in backbone.parameters():
            param.stop_gradient = True

        backbone.eval()
        return backbone

    def _extract_features(self, task_id: str, feature_extractor,
                          image_paths: List[str], input_size: List[int]) -> np.ndarray:
        """从图像中提取patch特征"""
        paddle, _, _ = _import_paddle()

        h, w = input_size[1], input_size[0]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        all_features = []
        failed = 0

        with paddle.no_grad():
            for idx, img_path in enumerate(image_paths):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        failed += 1
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
                    img = (img - mean) / std
                    tensor = paddle.to_tensor(np.transpose(img, (2, 0, 1))[np.newaxis])

                    # 提取多尺度特征
                    features = self._extract_patch_features(feature_extractor, tensor)
                    all_features.append(features)

                    if (idx + 1) % 10 == 0:
                        self._add_log(task_id, f"Processed {idx + 1}/{len(image_paths)} images")
                        self.training_status[task_id]["progress"] = 10 + int((idx + 1) / len(image_paths) * 35)

                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        self._add_log(task_id, f"Feature extraction failed for {os.path.basename(img_path)}: {e}")

        if failed > 0:
            self._add_log(task_id, f"Feature extraction: {len(all_features)} succeeded, {failed} failed")

        if not all_features:
            return np.array([])

        return np.concatenate(all_features, axis=0)

    def _extract_patch_features(self, backbone, x) -> np.ndarray:
        """提取patch级别的特征"""
        paddle, _, F = _import_paddle()

        B, C, H, W = x.shape

        # 前向传播到layer3
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x1 = backbone.layer1(x)
        x2 = backbone.layer2(x1)
        x3 = backbone.layer3(x2)

        # 上采样到相同尺寸
        target_size = (H // 8, W // 8)
        x2_up = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3_up = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)

        # 拼接特征
        features = paddle.concat([x2_up, x3_up], axis=1)

        # 转换为patch特征 [B, H*W, C]
        B, C, H_p, W_p = features.shape
        patches = features.transpose([0, 2, 3, 1]).reshape([B, H_p * W_p, C])

        # 归一化
        patches = F.normalize(patches, axis=-1)

        return patches.numpy()[0]  # [H*W, C]

    def _coreset_sampling(self, features: np.ndarray, ratio: float) -> np.ndarray:
        """
        使用贪心算法进行核心集采样
        选择最具代表性的特征子集
        """
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * ratio))

        if n_coreset >= n_samples:
            return features

        # 贪心核心集选择
        selected_indices = [0]  # 从第一个样本开始
        selected_features = features[0:1]

        for _ in range(1, n_coreset):
            # 计算每个未选样本到已选样本的最小距离
            distances = np.min(
                np.linalg.norm(features[:, np.newaxis] - selected_features, axis=2),
                axis=1
            )
            # 选择距离最远的样本
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
            selected_features = np.vstack([selected_features, features[next_idx:next_idx+1]])

        return selected_features

    def _calibrate_threshold(self, task_id: str, feature_extractor,
                             memory_bank: np.ndarray, image_paths: List[str],
                             input_size: List[int], config: Dict) -> float:
        """校准异常检测阈值"""
        paddle, _, F = _import_paddle()

        h, w = input_size[1], input_size[0]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 均匀采样最多200张图像进行校准
        max_samples = min(len(image_paths), 200)
        if len(image_paths) > max_samples:
            step = len(image_paths) / max_samples
            calib_images = [image_paths[int(i * step)] for i in range(max_samples)]
        else:
            calib_images = image_paths

        scores = []
        memory_tensor = paddle.to_tensor(memory_bank)
        memory_tensor = F.normalize(memory_tensor, axis=-1)

        with paddle.no_grad():
            for img_path in calib_images:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
                    img = (img - mean) / std
                    tensor = paddle.to_tensor(np.transpose(img, (2, 0, 1))[np.newaxis])

                    # 提取特征
                    features = self._extract_patch_features(feature_extractor, tensor)
                    features_tensor = paddle.to_tensor(features)
                    features_tensor = F.normalize(features_tensor, axis=-1)

                    # 计算异常分数
                    similarity = paddle.matmul(features_tensor, memory_tensor, transpose_y=True)
                    max_similarity = paddle.max(similarity, axis=1)
                    distances = 1.0 - max_similarity
                    anomaly_score = float(paddle.max(distances).numpy())

                    scores.append(anomaly_score)
                except Exception as e:
                    pass

        if not scores:
            self._add_log(task_id, "WARNING: No scores from calibration. threshold=0.5")
            return 0.5

        # 使用95分位数作为阈值
        arr = np.array(scores, dtype=np.float32)
        percentile = config.get("threshold_percentile", 95)
        threshold = float(np.percentile(arr, percentile))

        stats = {
            "n": len(scores),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "threshold": threshold,
        }

        self.training_status[task_id]["calibration_stats"] = stats
        self._add_log(
            task_id,
            f"Calibration OK: n={stats['n']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"p90={stats['p90']:.6f}, p95={stats['p95']:.6f}, p99={stats['p99']:.6f} "
            f"-> threshold={threshold:.6f} (p{percentile})"
        )

        return threshold

    def _load_train_images(self, task_id: str, dataset_dir: str) -> List[str]:
        """从train.txt加载训练图像路径"""
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

    def _save_training_data_and_template(self, task_id: str, save_dir: str, config: Dict):
        """保存训练数据和模板图像"""
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

            # 选择角度最接近0的图像作为模板
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

    def stop_task(self, task_id):
        """停止任务"""
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}

        status = self.training_status[task_id].get("status")
        if status in ["completed", "failed", "cancelled"]:
            return {"status": "success", "message": f"Task already in {status} state"}

        thread = self.threads.get(task_id)
        if thread and thread.is_alive():
            # 设置取消标志
            self.training_status[task_id]["status"] = "cancelled"
            self._persist_state_if_due(force=True)
            return {"status": "success", "message": "Cancellation requested"}

        self.training_status[task_id]["status"] = "cancelled"
        self._persist_state_if_due(force=True)
        return {"status": "success", "message": "Task marked as cancelled"}

    def get_status(self, task_id):
        """获取任务状态"""
        if task_id in self.training_status:
            return self.training_status[task_id]
        for tid, status in self.training_status.items():
            if status.get("task_uuid") == task_id:
                return status
        return {"status": "not_found"}


# 全局单例
trainer = PatchCoreTrainer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 示例配置
    test_config = {
        "project_id": "test_project",
        "task_uuid": "test_uuid",
        "label_name": "test_label",
        "model_name": "PatchCore",
        "backbone": "resnet18",
        "input_size": [224, 224],
        "coreset_ratio": 0.1,
        "threshold_percentile": 95,
    }

    # 示例数据集路径（需要替换为实际路径）
    test_dataset_dir = "/path/to/dataset"

    if os.path.exists(test_dataset_dir):
        task_id = trainer.run_training_async(test_dataset_dir, test_config)
        print(f"Training started with task_id: {task_id}")

        # 等待训练完成
        while True:
            status = trainer.get_status(task_id)
            print(f"Status: {status.get('status')}, Progress: {status.get('progress')}%")
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)
    else:
        print(f"Test dataset not found: {test_dataset_dir}")
