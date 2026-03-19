#!/usr/bin/env python3
"""
PatchCore 异常检测模型训练脚本 - ROI级别训练
每个标注框（ROI）训练一个独立的模型
使用 PyTorch 进行特征提取
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
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class PatchCoreTrainer:
    """
    ROI级别PatchCore训练器
    每个标注框（ROI）训练一个独立的模型
    """

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.training_status = {}
        self.threads = {}
        self._state_lock = threading.Lock()
        self._last_persist_ts = 0.0
        self.state_file = str(Path(output_dir) / "_roi_patchcore_trainer_state.json")
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
        with self._state_lock:
            data = {
                "version": 1,
                "updated_at": time.time(),
                "training_status": dict(self.training_status),
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

    def _make_task_key(self, dataset_dir, config, roi_id=None):
        """生成任务唯一标识 - 基于ROI ID"""
        parts = [
            str(config.get("project_id", "")),
            str(config.get("task_uuid", "")),
            str(config.get("model_name", "")),
        ]
        if roi_id is not None:
            parts.append(f"roi_{roi_id}")
        parts.append(str(dataset_dir))
        return "|".join(parts)

    def run_training_async(self, dataset_dir, config):
        """
        异步启动训练 - 为每个类别创建独立的训练任务
        返回所有创建的任务ID列表
        
        Args:
            dataset_dir: 数据目录路径，应该包含 annotations.json 和 raw_images/
            config: 配置字典，可以包含 group_by 字段来指定分组方式
                   - 'label': 按 label 字段分组（默认）
                   - 'category_id': 按 category_id 字段分组
                   - 'roi_id': 按 id 字段分组（每个标注框独立）
        """
        # 加载标注数据 - dataset_dir 直接是数据目录
        raw_data_dir = Path(dataset_dir)
        annotation_file = raw_data_dir / "annotations.json"

        if not annotation_file.exists():
            raise ValueError(f"annotations.json not found: {annotation_file}")

        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)

        annotations = annotations_data.get('annotations', [])
        if not annotations:
            raise ValueError("No annotations found in annotations.json")

        # 根据配置决定分组方式
        group_by = config.get('group_by', 'label')  # 默认按 label 分组
        
        groups = {}
        for ann in annotations:
            if group_by == 'roi_id':
                group_id = ann.get('id')
            elif group_by == 'category_id':
                group_id = ann.get('category_id', 'unknown')
            else:  # 默认按 label
                group_id = ann.get('label', 'unknown')
            
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(ann)

        self._add_log("main", f"Found {len(groups)} unique groups to train (group_by={group_by})")

        # 为每个组创建训练任务
        task_ids = []
        for group_id in sorted(groups.keys()):
            task_id = self._create_group_training_task(dataset_dir, config, group_id, groups[group_id])
            task_ids.append(task_id)

        return task_ids

    def _create_group_training_task(self, dataset_dir, config, group_id, group_annotations):
        """为单个组创建训练任务"""
        # 清理 group_id 用于文件名
        safe_group_id = str(group_id).replace('/', '_').replace('\\', '_')
        task_key = self._make_task_key(dataset_dir, config, safe_group_id)

        # 查找现有任务
        existing_task_id = None
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
            self._add_log(task_id, f"Task restarted for group {group_id}")
        else:
            task_id = f"patchcore_{int(time.time())}_{safe_group_id}_{random.randint(1000, 9999)}"
            task_uuid = config.get("task_uuid", "unknown")

            # 按 group_id 保存模型
            save_dir = os.path.join(
                self.output_dir,
                config.get("project_id", "default"),
                task_uuid,
                str(safe_group_id)
            )

            self.training_status[task_id] = {
                "status": "starting",
                "progress": 0,
                "group_id": group_id,
                "task_uuid": task_uuid,
                "logs": [f"Task {task_id} initialized for group {group_id}."],
                "metrics": [],
                "total_epochs": 1,
                "start_time": time.time(),
                "dataset_dir": dataset_dir,
                "save_dir": save_dir,
                "config": config,
                "task_key": task_key,
                "group_annotations": group_annotations,
            }
            self._persist_state_if_due(force=True)

        t = threading.Thread(target=self._train_group_process, args=(task_id, dataset_dir, config, group_id, group_annotations))
        self.threads[task_id] = t
        t.start()
        return task_id

    def _train_group_process(self, task_id, dataset_dir, config, group_id, group_annotations):
        """组训练进程"""
        try:
            self._do_train_group(task_id, dataset_dir, config, group_id, group_annotations)
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

    def _do_train_group(self, task_id, dataset_dir, config, group_id, group_annotations):
        """为单个组执行训练"""
        self._add_log(task_id, f"PatchCore training started for group '{group_id}'")
        self.training_status[task_id]["status"] = "preparing"
        self.training_status[task_id]["progress"] = 5

        save_dir = self.training_status[task_id].get("save_dir")
        os.makedirs(save_dir, exist_ok=True)

        # 提取ROI图像
        roi_images = self._extract_roi_images(task_id, dataset_dir, group_annotations)
        if not roi_images:
            raise ValueError(f"No ROI images extracted for group {group_id}")

        num_samples = len(roi_images)
        self.training_status[task_id]["num_samples"] = num_samples
        self._add_log(task_id, f"Extracted {num_samples} ROI images for training")

        # 构建PatchCore模型
        backbone_name = config.get("backbone", "resnet18")
        input_size = config.get("input_size", [224, 224])
        coreset_ratio = config.get("coreset_ratio", 0.1)

        self._add_log(task_id, f"Building PatchCore model (backbone={backbone_name}, coreset_ratio={coreset_ratio})")
        self.training_status[task_id]["status"] = "training"
        self.training_status[task_id]["progress"] = 10

        # 创建特征提取器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = self._build_feature_extractor(backbone_name, device)
        feature_extractor.eval()

        # 提取训练集特征
        self._add_log(task_id, "Stage 1/3: Extracting features from ROI images...")
        features = self._extract_features(task_id, feature_extractor, roi_images, input_size, device)

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
        threshold = self._calibrate_threshold(task_id, feature_extractor, memory_bank, roi_images, input_size, config, device)

        # 获取第一个标注的信息用于配置
        first_ann = group_annotations[0] if group_annotations else {}
        bbox = first_ann.get('bbox', [0, 0, 0, 0])
        label = first_ann.get('label', 'unknown')
        category_id = first_ann.get('category_id', 0)

        # 保存配置
        config_data = {
            "group_id": group_id,
            "category": label,
            "category_id": category_id,
            "bbox": bbox,
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

        self._add_log(task_id, f"config.json: threshold={threshold:.6f}, category={label}")

        # 保存训练数据和模板
        self._save_group_training_data(task_id, save_dir, config, group_annotations)

        # 完成任务
        self.training_status[task_id]["status"] = "completed"
        self.training_status[task_id]["progress"] = 100
        self._add_log(task_id, f"Group '{group_id}' training complete.")
        self._persist_state_if_due(force=True)

    def _extract_roi_images(self, task_id: str, dataset_dir: str, group_annotations: List[Dict]) -> List[np.ndarray]:
        """从原始图像中提取ROI区域"""
        raw_data_dir = Path(dataset_dir)
        raw_images_dir = raw_data_dir / "raw_images"

        if not raw_images_dir.exists():
            self._add_log(task_id, f"raw_images directory not found: {raw_images_dir}")
            return []

        # 加载图像映射
        annotation_file = raw_data_dir / "annotations.json"
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = {img['id']: img for img in data.get('images', [])}

        roi_images = []
        failed = 0

        for ann in group_annotations:
            image_id = ann.get('image_id')
            bbox = ann.get('bbox', [0, 0, 0, 0])
            angle = ann.get('angle', 0)
            h_flip = ann.get('horizontal_flip', False)
            v_flip = ann.get('vertical_flip', False)

            img_info = images.get(image_id)
            if not img_info:
                failed += 1
                continue

            img_path = raw_images_dir / img_info['file_name']
            if not img_path.exists():
                failed += 1
                continue

            try:
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    failed += 1
                    continue

                # 提取ROI
                x, y, w, h = [int(v) for v in bbox]
                x = max(0, x)
                y = max(0, y)
                w = max(1, w)
                h = max(1, h)

                # 边界检查
                img_h, img_w = img.shape[:2]
                x = min(x, img_w - 1)
                y = min(y, img_h - 1)
                w = min(w, img_w - x)
                h = min(h, img_h - y)

                roi = img[y:y+h, x:x+w]
                if roi.size == 0:
                    failed += 1
                    continue

                # 应用数据增强（角度、翻转）
                roi = self._apply_augmentation(roi, angle, h_flip, v_flip)

                roi_images.append(roi)

            except Exception as e:
                failed += 1
                self._add_log(task_id, f"Failed to extract ROI from {img_path}: {e}")

        self._add_log(task_id, f"Extracted {len(roi_images)} ROI images, {failed} failed")
        return roi_images

    def _apply_augmentation(self, img: np.ndarray, angle: float, h_flip: bool, v_flip: bool) -> np.ndarray:
        """应用数据增强"""
        # 水平翻转
        if h_flip:
            img = cv2.flip(img, 1)

        # 垂直翻转
        if v_flip:
            img = cv2.flip(img, 0)

        # 旋转
        if angle != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return img

    def _build_feature_extractor(self, backbone_name: str, device: torch.device):
        """构建特征提取器"""
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
        else:
            backbone = models.resnet18(pretrained=True)

        # 移除最后的全连接层
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone = backbone.to(device)
        backbone.eval()

        # 冻结参数
        for param in backbone.parameters():
            param.requires_grad = False

        return backbone

    def _extract_features(self, task_id: str, feature_extractor, roi_images: List[np.ndarray], input_size: List[int], device: torch.device) -> np.ndarray:
        """从ROI图像中提取patch特征"""
        h, w = input_size[1], input_size[0]

        # 图像预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        all_features = []
        failed = 0

        with torch.no_grad():
            for idx, roi in enumerate(roi_images):
                try:
                    # 转换 BGR -> RGB
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    # 预处理
                    tensor = transform(roi_rgb).unsqueeze(0).to(device)

                    # 提取特征
                    features = self._extract_patch_features(feature_extractor, tensor)
                    all_features.append(features)

                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        self._add_log(task_id, f"Feature extraction failed for ROI {idx}: {e}")

        if failed > 5:
            self._add_log(task_id, f"... and {failed - 5} more failures")

        if not all_features:
            return np.array([])

        return np.vstack(all_features)

    def _extract_patch_features(self, feature_extractor, tensor: torch.Tensor) -> np.ndarray:
        """提取patch级别的特征"""
        # 提取特征 [B, C, H, W]
        features = feature_extractor(tensor)

        # 转换为patch特征 [B, H*W, C]
        B, C, H_p, W_p = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(B, H_p * W_p, C)

        # 归一化
        patches = F.normalize(patches, dim=-1)

        return patches.cpu().numpy()[0]

    def _coreset_sampling(self, features: np.ndarray, ratio: float) -> np.ndarray:
        """贪心核心集采样 - 使用 PyTorch 加速"""
        import torch
        
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * ratio))

        if n_coreset >= n_samples:
            return features

        # 转换为 PyTorch 张量并移至 GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features_tensor = torch.from_numpy(features).to(device)
        
        selected_indices = [0]
        selected_features = features_tensor[0:1]

        for i in range(1, n_coreset):
            # 计算每个未选样本到已选样本的最小距离
            # features: [N, C], selected: [M, C]
            # distances: [N, M] -> min -> [N]
            distances = torch.cdist(features_tensor, selected_features).min(dim=1)[0]
            next_idx = torch.argmax(distances).item()
            selected_indices.append(next_idx)
            selected_features = torch.cat([selected_features, features_tensor[next_idx:next_idx+1]], dim=0)
            
            if i % 500 == 0:
                self._add_log(f"coreset", f"  Coreset sampling progress: {i}/{n_coreset}")

        return features_tensor[selected_indices].cpu().numpy()

    def _calibrate_threshold(self, task_id: str, feature_extractor, memory_bank: np.ndarray,
                             roi_images: List[np.ndarray], input_size: List[int], config: Dict, device: torch.device) -> float:
        """校准异常检测阈值"""
        h, w = input_size[1], input_size[0]

        # 图像预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        max_samples = min(len(roi_images), 200)
        if len(roi_images) > max_samples:
            step = len(roi_images) / max_samples
            calib_images = [roi_images[int(i * step)] for i in range(max_samples)]
        else:
            calib_images = roi_images

        scores = []
        memory_tensor = torch.from_numpy(memory_bank).to(device)
        memory_tensor = F.normalize(memory_tensor, dim=-1)

        with torch.no_grad():
            for roi in calib_images:
                try:
                    # 转换 BGR -> RGB
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    # 预处理
                    tensor = transform(roi_rgb).unsqueeze(0).to(device)

                    # 提取特征
                    features = self._extract_patch_features(feature_extractor, tensor)
                    features_tensor = torch.from_numpy(features).to(device)
                    features_tensor = F.normalize(features_tensor, dim=-1)

                    # 计算异常分数
                    similarity = torch.matmul(features_tensor, memory_tensor.T)
                    max_similarity = torch.max(similarity, dim=1)[0]
                    distances = 1.0 - max_similarity
                    anomaly_score = float(torch.max(distances).cpu().numpy())

                    scores.append(anomaly_score)
                except:
                    pass

        if not scores:
            self._add_log(task_id, "WARNING: No scores from calibration. threshold=0.5")
            return 0.5

        arr = np.array(scores, dtype=np.float32)
        percentile = config.get("threshold_percentile", 95)
        threshold = float(np.percentile(arr, percentile))

        stats = {
            "n": len(scores), "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "threshold": threshold,
        }

        if task_id in self.training_status:
            self.training_status[task_id]["calibration_stats"] = stats
        self._add_log(task_id, f"Calibration OK: n={stats['n']}, mean={stats['mean']:.6f}, p95={stats['p95']:.6f} -> threshold={threshold:.6f}")

        return threshold

    def _save_group_training_data(self, task_id: str, save_dir: str, config: Dict, group_annotations: List[Dict]):
        """保存组训练数据"""
        try:
            td = os.path.join(save_dir, "training_data")
            os.makedirs(td, exist_ok=True)

            # 保存组标注信息
            group_data = {
                "group_id": group_annotations[0].get('label') if group_annotations else None,
                "annotations": group_annotations,
                "num_samples": len(group_annotations),
            }

            with open(os.path.join(td, "group_annotations.json"), 'w', encoding='utf-8') as f:
                json.dump(group_data, f, ensure_ascii=False, indent=2)

            self._add_log(task_id, f"Saved group annotations with {len(group_annotations)} samples")

        except Exception as e:
            self._add_log(task_id, f"Warning: save training data failed: {e}")

    def get_training_status(self, task_id: str = None):
        """获取训练状态"""
        if task_id:
            return self.training_status.get(task_id)
        return self.training_status

    def stop_task(self, task_id):
        """停止任务"""
        if task_id not in self.training_status:
            return {"status": "error", "message": "Task not found"}

        status = self.training_status[task_id].get("status")
        if status in ["completed", "failed", "cancelled"]:
            return {"status": "success", "message": f"Task already in {status} state"}

        thread = self.threads.get(task_id)
        if thread and thread.is_alive():
            self.training_status[task_id]["cancel_requested"] = True
            return {"status": "success", "message": "Cancel requested"}

        return {"status": "success", "message": "Task stopped"}


if __name__ == "__main__":
    # 测试代码
    trainer = PatchCoreTrainer(output_dir="test_output")
    print("ROI PatchCore Trainer initialized")
