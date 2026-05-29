#!/usr/bin/env python3
"""
YOLO 检测模型训练器
"""

import os
import sys
import json
import time
import random
import logging
import traceback
import math
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import yaml
import shutil
import tempfile

from utils.config import get_output_dir
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '/home/software/One2All-paddle')

PROJECT_ROOT = Path(__file__).parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"

from utils.synthetic_defect_generator import SyntheticDefectGenerator, load_synthetic_config
import cv2

# Albumentations 导入
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("albumentations not installed, augmentation will be disabled")

logger = logging.getLogger(__name__)

# 默认数据增强配置文件路径
DEFAULT_AUGMENTATION_CONFIG = PROJECT_ROOT / "configs" / "augmentations.yaml"


def normalize_brightness(image: Image.Image, target_mean: float = 168.0, threshold: int = 10) -> Image.Image:
    """对图片进行亮度归一化"""
    arr = np.array(image).astype(np.float32)
    brightness = arr.mean(axis=2)
    hole_mask = brightness > threshold
    if np.sum(hole_mask) == 0:
        return image
    current_mean = arr[hole_mask].mean()
    if current_mean > 0:
        scale = target_mean / current_mean
        arr[hole_mask] = np.clip(arr[hole_mask] * scale, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def normalize_contrast(image: Image.Image, target_std: float = 65.0, threshold: int = 10) -> Image.Image:
    """对图片进行对比度归一化"""
    arr = np.array(image).astype(np.float32)
    brightness = arr.mean(axis=2)
    hole_mask = brightness > threshold
    if np.sum(hole_mask) == 0:
        return image
    current_mean = arr[hole_mask].mean()
    current_std = arr[hole_mask].std()
    if current_std > 0:
        arr[hole_mask] = (arr[hole_mask] - current_mean) / current_std * target_std + current_mean
        arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def letterbox_resize(
    img: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Image.Image, np.ndarray]:
    """
    保持宽高比地将图片 padding 到 target_size，并返回有效区域的 binary mask。

    填充色默认为图片自身的平均颜色（自适应），避免固定黑色填充让模型学到
    "暗色=padding=可忽略"，导致暗背景上的暗色异物漏检。

    Returns:
        img_padded: PIL Image，尺寸为 target_size (H×W)
        mask:       np.ndarray bool (H, W)，True 表示原始像素，False 表示填充像素
    """
    th, tw = target_size
    ow, oh = img.size

    scale = min(tw / ow, th / oh)
    new_w = int(ow * scale)
    new_h = int(oh * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    if fill_color is None:
        arr = np.array(img_resized)
        mean_color = tuple(int(c) for c in arr.reshape(-1, 3).mean(axis=0))
        fill_color = mean_color

    pad_left = (tw - new_w) // 2
    pad_top = (th - new_h) // 2

    img_padded = Image.new("RGB", (tw, th), fill_color)
    img_padded.paste(img_resized, (pad_left, pad_top))

    mask = np.zeros((th, tw), dtype=bool)
    mask[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = True

    return img_padded, mask


def extract_polygon_region(image_path: Path, segmentation: List, target_size: Tuple[int, int] = (224, 224)) -> Tuple[Image.Image, np.ndarray]:
    """
    使用多边形segmentation从图片中提取区域，并使用 letterbox_resize 保持宽高比。
    
    Returns:
        roi_image: PIL Image，尺寸为 target_size (H×W)
        mask:      np.ndarray bool (H, W)，True 表示原始像素，False 表示填充像素
    """
    coords = segmentation[0]
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    with Image.open(image_path) as img:
        width, height = img.size
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        if bbox_width <= 0 or bbox_height <= 0:
            # 返回空白图像和全 False mask
            return Image.new('RGB', target_size, (0, 0, 0)), np.zeros(target_size, dtype=bool)

        cropped = img.crop((x_min, y_min, x_max, y_max))
        
        # 使用 letterbox_resize 保持宽高比
        roi_image, mask = letterbox_resize(cropped.convert("RGB"), target_size)

        return roi_image, mask


def load_augmentation_transform(config_path: Optional[str] = None):
    """加载 albumentations 数据增强配置
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        
    Returns:
        albumentations.Compose 或 None
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
        
    if config_path is None:
        config_path = DEFAULT_AUGMENTATION_CONFIG
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Augmentation config not found: {config_path}")
        return None
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        transform = A.from_dict(config)
        logger.info(f"Loaded augmentation config from {config_path}")
        return transform
    except Exception as e:
        logger.warning(f"Failed to load augmentation config: {e}")
        return None


def calculate_adaptive_brightness_range(image: Image.Image) -> Tuple[float, float]:
    """根据图片当前亮度计算自适应亮度调整范围
    
    Args:
        image: 输入图片 (PIL.Image)
        
    Returns:
        (min_brightness, max_brightness): 亮度调整范围
    """
    arr = np.array(image).astype(np.float32)
    mean_brightness = arr.mean()
    
    # 暗图（平均亮度 < 80）：只允许增亮
    if mean_brightness < 80:
        return (0.0, 0.4)  # 可以增亮 0-40%
    # 亮图（平均亮度 > 180）：只允许调暗
    elif mean_brightness > 180:
        return (-0.4, 0.0)  # 可以调暗 0-40%
    # 正常亮度：允许微调
    else:
        return (-0.2, 0.2)  # 可以微调 ±20%


class AdaptiveBrightnessContrast(A.core.transforms_interface.ImageOnlyTransform):
    """自适应亮度对比度调整
    
    根据图片当前亮度动态调整亮度范围：
    - 暗图只允许增亮
    - 亮图只允许调暗
    - 正常图允许微调
    """
    
    def __init__(
        self,
        contrast_limit: Tuple[float, float] = (-0.5, 0.5),
        always_apply: bool = False,
        p: float = 0.7,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.contrast_limit = contrast_limit
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # 计算当前亮度
        mean_brightness = img.mean()
        
        # 根据亮度确定调整范围
        if mean_brightness < 80:
            brightness_limit = (0.0, 0.4)  # 暗图：只增亮
        elif mean_brightness > 180:
            brightness_limit = (-0.4, 0.0)  # 亮图：只调暗
        else:
            brightness_limit = (-0.2, 0.2)  # 正常：微调
        
        # 随机选择调整值
        brightness = random.uniform(brightness_limit[0], brightness_limit[1])
        contrast = random.uniform(self.contrast_limit[0], self.contrast_limit[1])
        
        # 应用调整
        # 亮度调整
        if brightness != 0:
            img = img * (1 + brightness)
        
        # 对比度调整
        if contrast != 0:
            mean = img.mean()
            img = (img - mean) * (1 + contrast) + mean
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def get_transform_init_args_names(self):
        return ("contrast_limit",)


def apply_augmentation(image: Image.Image, transform) -> Image.Image:
    """应用数据增强
    
    Args:
        image: 输入图片 (PIL.Image)
        transform: albumentations transform
        
    Returns:
        增强后的图片 (PIL.Image)
    """
        
    # albumentations 需要 numpy array 格式
    image_np = np.array(image)
    augmented = transform(image=image_np)
    return Image.fromarray(augmented['image'])


def extract_roi_images(
    dataset_dir: str,
    output_dir: str,
    mask_output_dir: str,
    group_annotations: List[Dict],
    normalize_brightness: bool = False,
    normalize_contrast: bool = False,
    augment: bool = False,
    num_augmentations: int = 1,
    train_mode: str = "by_pos_id",
    augmentation_config: Optional[str] = None,
    target_size: Tuple[int, int] = (320, 320),
) -> List[Path]:
    """
    从annotations中提取ROI图片并保存到output_dir，同时保存mask到mask_output_dir
    支持数据增强，根据训练模式决定增强策略
    返回提取的ROI图片路径列表
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    mask_output_dir = Path(mask_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    raw_images_dir = dataset_dir / "raw_images"

    # 加载annotations.json获取image_id到file_name的映射
    json_path = dataset_dir / "annotations.json"
    image_map = {}
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image_map = {img['id']: img['file_name'] for img in data.get('images', [])}

    # 加载数据增强配置
    augment_transform = None
    if augment and ALBUMENTATIONS_AVAILABLE:
        augment_transform = load_augmentation_transform(augmentation_config)

    logger.info(f"Extracting {len(group_annotations)} ROI images to {output_dir}")
    if augment:
        logger.info(f"Augmentation enabled: mode={train_mode}, target={num_augmentations}")

    extracted_paths = []
    
    # 首先提取所有原始ROI
    original_rois = []
    for ann in group_annotations:
        image_id = ann['image_id']
        ann_id = ann['id']
        file_name = image_map.get(image_id)

        if not file_name:
            logger.warning(f"找不到image_id={image_id}对应的图片")
            continue

        image_path = raw_images_dir / file_name
        if not image_path.exists():
            logger.warning(f"图片不存在: {image_path}")
            continue

        segmentation = ann.get('segmentation', [])
        if not segmentation:
            logger.warning(f"annotation {ann_id} 没有segmentation数据")
            continue

        try:
            roi_image, roi_mask = extract_polygon_region(image_path, segmentation, target_size=target_size)

            if normalize_brightness:
                roi_image = normalize_brightness(roi_image)
            if normalize_contrast:
                roi_image = normalize_contrast(roi_image)

            original_rois.append((roi_image, roi_mask, Path(file_name).stem, ann_id))

        except Exception as e:
            logger.error(f"处理annotation {ann_id}时出错: {e}")

    num_original = len(original_rois)
    if num_original == 0:
        logger.warning("No ROI images extracted")
        return extracted_paths

    # 保存原始图片和mask
    for roi_image, roi_mask, file_stem, ann_id in original_rois:
        output_filename = f"{file_stem}_ann{ann_id}.png"
        output_path = output_dir / output_filename
        roi_image.save(output_path)
        extracted_paths.append(output_path)
        
        # 保存mask
        mask_filename = f"{file_stem}_ann{ann_id}.npy"
        mask_path = mask_output_dir / mask_filename
        np.save(mask_path, roi_mask)

    # 数据增强策略
    if augment and num_augmentations > num_original:
        if train_mode == "by_category":
            # by_category 模式：增强到 num_augmentations 总数
            num_needed = num_augmentations - num_original
            logger.info(f"by_category augmentation: {num_original} original, {num_needed} augmented needed")
            
            for i in range(num_needed):
                # 随机选择一张原始图片进行增强
                roi_image, roi_mask, file_stem, ann_id = random.choice(original_rois)
                if augment_transform:
                    aug_image = apply_augmentation(roi_image, augment_transform)
                else:
                    aug_image = roi_image
                    
                aug_filename = f"{file_stem}_ann{ann_id}_aug{i:04d}.png"
                aug_path = output_dir / aug_filename
                aug_image.save(aug_path)
                extracted_paths.append(aug_path)
                
                # 保存增强后的mask（与原mask相同）
                mask_filename = f"{file_stem}_ann{ann_id}_aug{i:04d}.npy"
                mask_path = mask_output_dir / mask_filename
                np.save(mask_path, roi_mask)
        else:
            # by_pos_id 模式：每个 ROI 增强 num_augmentations 次
            logger.info(f"by_pos_id augmentation: {num_original} original, {num_augmentations} per image")
            
            for roi_image, roi_mask, file_stem, ann_id in original_rois:
                for i in range(num_augmentations - 1):  # -1 because original is already saved
                    if augment_transform:
                        aug_image = apply_augmentation(roi_image, augment_transform)
                    else:
                        aug_image = roi_image
                        
                    aug_filename = f"{file_stem}_ann{ann_id}_aug{i:04d}.png"
                    aug_path = output_dir / aug_filename
                    aug_image.save(aug_path)
                    extracted_paths.append(aug_path)
                    
                    # 保存增强后的mask（与原mask相同）
                    mask_filename = f"{file_stem}_ann{ann_id}_aug{i:04d}.npy"
                    mask_path = mask_output_dir / mask_filename
                    np.save(mask_path, roi_mask)

    logger.info(f"ROI extraction completed. Saved {len(extracted_paths)} images ({num_original} original + {len(extracted_paths) - num_original} augmented) to {output_dir}")
    return extracted_paths


class ModelTrainer:
    """
    异常检测模型训练器
    支持 PatchCore 和 Dinomaly 算法
    """

    def __init__(
        self,
        output_dir: str = None,
        max_concurrent: int = 2,
    ):
        if output_dir is None:
            output_dir = str(get_output_dir())
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_tasks = 0
        self._active_lock = threading.Lock()

        self.threads: Dict[str, threading.Thread] = {}
        self.training_status: Dict[str, Dict] = {}
        self._stop_events: Dict[str, threading.Event] = {}

        self._persist_lock = threading.Lock()
        self._last_persist = 0
        self._persist_interval = 5

        # 状态字典的锁保护
        self._status_lock = threading.RLock()

        self._load_state()

    def _state_path(self) -> Path:
        return Path(self.output_dir) / "training_state.json"

    def _load_state(self):
        state_path = self._state_path()
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                with self._status_lock:
                    for tid, status in loaded.items():
                        if status.get("status") in {"starting", "training", "preparing"}:
                            status["status"] = "failed"
                            status["error"] = "Interrupted during reload"
                        self.training_status[tid] = status
                logger.info(f"Loaded {len(self.training_status)} tasks from state file")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _persist_state_if_due(self, force: bool = False):
        with self._persist_lock:
            now = time.time()
            if not force and now - self._last_persist < self._persist_interval:
                return
            self._last_persist = now
            try:
                state_path = self._state_path()
                state_path.parent.mkdir(parents=True, exist_ok=True)
                with self._status_lock:
                    state_copy = {}
                    for tid, s in self.training_status.items():
                        s_copy = {k: v for k, v in s.items() if k != "group_annotations"}
                        s_copy["_annotation_count"] = len(s.get("group_annotations") or [])
                        state_copy[tid] = s_copy
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state_copy, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to persist state: {e}")

    def _add_log(self, task_id: str, message: str, persist: bool = False):
        """添加日志"""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {message}"
        logger.info(f"[Task {task_id}] {message}")
        with self._status_lock:
            if task_id in self.training_status:
                self.training_status[task_id].setdefault("logs", []).append(entry)
        if persist:
            self._persist_state_if_due(force=True)
        else:
            self._persist_state_if_due()

    def _is_task_cancelled(self, task_id: str) -> bool:
        evt = self._stop_events.get(task_id)
        return evt is not None and evt.is_set()

    def _get_active_task_count(self) -> int:
        """获取当前活跃任务数（线程安全）"""
        with self._status_lock:
            return sum(
                1 for s in self.training_status.values()
                if s.get("status") in {"starting", "training", "preparing"}
            )

    def run_batch_training_async(
        self,
        dataset_dir: str,
        config: dict,
        groups: Dict[Any, List[Dict]],
        group_id: str = None,
    ) -> tuple[List[str], List[str]]:
        """批量启动训练任务"""
        if group_id:
            config = dict(config)
            config["external_group_id"] = group_id

        new_max_concurrent = config.get("max_concurrent", self.max_concurrent)
        if new_max_concurrent != self.max_concurrent:
            active_count = self._get_active_task_count()
            if active_count == 0:
                self.max_concurrent = new_max_concurrent
                self._semaphore = threading.Semaphore(new_max_concurrent)
                logger.info(f"Updated max_concurrent to {new_max_concurrent}")
            else:
                logger.warning(
                    f"Cannot update max_concurrent: {active_count} tasks running. "
                    f"Current={self.max_concurrent}, Requested={new_max_concurrent}"
                )

        # 排除 label 为 "工件主体" 的 group
        excluded_label = "工件主体"
        filtered_groups = {}
        for grp_id, annotations in groups.items():
            if annotations and annotations[0].get("label") == excluded_label:
                logger.info(f"Skipping training for excluded label '{excluded_label}' (group: {grp_id})")
                continue
            filtered_groups[grp_id] = annotations

        task_ids: List[str] = []
        filtered_keys: List[str] = []
        for grp_id in sorted(filtered_groups.keys(), key=str):
            task_id = self._create_group_training_task(
                dataset_dir, config, grp_id, filtered_groups[grp_id]
            )
            task_ids.append(task_id)
            filtered_keys.append(str(grp_id))
        return task_ids, filtered_keys

    def _create_group_training_task(
        self,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ) -> str:
        """创建单个组的训练任务"""
        safe_group_id = str(group_id).replace("/", "_").replace("\\", "_")
        model_name = config.get("model_name", "YOLO")

        # 为当前 group 设置 category（使用 category_id 而不是 label 避免中文路径问题）
        config = dict(config)
        train_mode = config.get("train_mode", "by_pos_id")
        if group_annotations:
            first_ann = group_annotations[0]
            category_id = first_ann.get("category_id", 0)
            config["category"] = str(category_id)
            config["category_label"] = first_ann.get("label", "unknown")  # 保存原始标签用于日志
            # 根据训练模式确定用于路径的ID（避免中文）
            if train_mode == "by_category":
                path_id = str(category_id)
            else:  # by_pos_id
                path_id = str(first_ann.get("pos_id", safe_group_id))
        else:
            path_id = safe_group_id
        
        # 将 path_id 保存到 config，确保 _do_train_yolo 使用相同的值
        config["path_id"] = path_id

        task_key = self._make_task_key(dataset_dir, config, safe_group_id)

        with self._status_lock:
            existing_task_id = None
            for tid, s in self.training_status.items():
                if s.get("task_key") == task_key:
                    existing_task_id = tid
                    break

            if existing_task_id:
                task_id = existing_task_id
                existing = self.training_status[task_id]
                thread = self.threads.get(task_id)
                if existing.get("status") in {"starting", "training", "preparing"} and thread and thread.is_alive():
                    self._persist_state_if_due()
                    return task_id
                existing.update({
                    "status": "starting",
                    "progress": 0,
                    "logs": [],
                    "metrics": [],
                    "start_time": time.time(),
                    "error": None,
                })
                self._add_log(task_id, f"Task restarted for group {group_id}")
            else:
                task_id = f"{model_name.lower()}_{int(time.time())}_{safe_group_id}_{random.randint(1000, 9999)}"
                task_uuid = config.get("task_uuid", "unknown")
                # 使用 path_id（数字ID）而不是 safe_group_id（可能是中文）
                save_dir = os.path.join(
                    self.output_dir,
                    config.get("project_id", "default"),
                    task_uuid,
                    str(path_id),
                )
                external_group_id = config.get("external_group_id")
                # 判断任务类型：有 base_model_dir 说明是重训，否则是首次训练
                is_retrain = bool(config.get("base_model_dir"))
                task_type = "yolo_retrain" if is_retrain else "yolo_initial"
                self.training_status[task_id] = {
                    "status": "starting",
                    "progress": 0,
                    "task_type": task_type,
                    "group_id": external_group_id or group_id,
                    "internal_group_id": group_id,
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

        self._stop_events[task_id] = threading.Event()
        t = threading.Thread(
            target=self._train_group_process,
            args=(task_id, dataset_dir, config, group_id, group_annotations),
            daemon=True,
        )
        self.threads[task_id] = t
        t.start()
        return task_id

    def _create_yolo_only_training_task(
        self,
        dataset_dir: str,
        config: dict,
        path_id: str,
        base_model_dir: str,
    ) -> str:
        """创建纯 YOLO 训练任务（无 FP，仅 FN 图像）。复制基础 YOLO 模型 + 训练 YOLO 检测器。"""
        model_name = config.get("model_name", "YOLO")
        safe_path_id = str(path_id).replace("/", "_").replace("\\", "_")
        task_id = f"{model_name.lower()}_yolo_{int(time.time())}_{safe_path_id}_{random.randint(1000, 9999)}"

        task_uuid = config.get("task_uuid", "unknown")
        save_dir = os.path.join(
            self.output_dir,
            config.get("project_id", "default"),
            task_uuid,
            str(path_id),
        )

        external_group_id = config.get("external_group_id")
        task_key = self._make_task_key(dataset_dir, config, roi_id=safe_path_id)

        self.training_status[task_id] = {
            "status": "starting",
            "progress": 0,
            "task_type": "yolo_only_retrain",
            "group_id": external_group_id or f"yolo_only_{path_id}",
            "internal_group_id": path_id,
            "task_uuid": task_uuid,
            "logs": [f"YOLO-only task {task_id} initialized for path_id={path_id}"],
            "metrics": [],
            "total_epochs": 1,
            "start_time": time.time(),
            "dataset_dir": dataset_dir,
            "save_dir": save_dir,
            "config": config,
            "task_key": task_key,
            "group_annotations": [],
        }
        self._persist_state_if_due(force=True)

        self._stop_events[task_id] = threading.Event()
        t = threading.Thread(
            target=self._train_yolo_group_process,
            args=(task_id, dataset_dir, config, path_id, base_model_dir),
            daemon=True,
        )
        self.threads[task_id] = t
        t.start()
        return task_id

    def _train_yolo_group_process(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        path_id: str,
        base_model_dir: str,
    ):
        """纯 YOLO 训练进程入口（detection 模式）。"""
        self._semaphore.acquire()
        with self._active_lock:
            self._active_tasks += 1
        try:
            self._do_train_yolo_only(task_id, dataset_dir, config, path_id, base_model_dir)
        except SystemExit:
            self._add_log(task_id, "Training cancelled")
            self._update_task_status(task_id, status="cancelled")
        except Exception as e:
            logger.exception(f"Training failed for {task_id}")
            error_msg = str(e)
            if "out of memory" in str(e).lower():
                error_msg = f"[GPU 显存不足] {error_msg}"
                self._add_log(task_id, f"训练失败：GPU 显存不足 (OOM)。建议释放 GPU 资源后重试。")
            self._update_task_status(task_id, status="failed", error=error_msg)
            self._add_log(task_id, f"Error: {e}")
        finally:
            with self._active_lock:
                self._active_tasks -= 1
            self._semaphore.release()
            self._persist_state_if_due(force=True)

    def _make_task_key(self, dataset_dir: str, config: dict, roi_id=None) -> str:
        """生成任务唯一标识"""
        parts = [
            str(config.get("project_id", "")),
            str(config.get("task_uuid", "")),
            str(config.get("model_name", "")),
            str(config.get("train_mode", "by_pos_id")),
        ]
        if roi_id is not None:
            parts.append(f"roi_{roi_id}")
        parts.append(str(dataset_dir))
        return "|".join(parts)

    def _update_task_status(self, task_id: str, **kwargs):
        """线程安全地更新任务状态"""
        with self._status_lock:
            if task_id in self.training_status:
                self.training_status[task_id].update(kwargs)

    def _train_group_process(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ):
        """训练进程入口（YOLO）"""
        self._semaphore.acquire()
        with self._active_lock:
            self._active_tasks += 1
        try:
            self._do_train_yolo(task_id, dataset_dir, config, group_id, group_annotations)
        except SystemExit:
            self._add_log(task_id, "Training cancelled")
            self._update_task_status(task_id, status="cancelled")
        except Exception as e:
            logger.exception(f"Training failed for {task_id}")
            error_msg = str(e)
            if "out of memory" in str(e).lower():
                error_msg = f"[GPU 显存不足] {error_msg}"
                self._add_log(task_id, f"训练失败：GPU 显存不足 (OOM)。建议减小 batch_size 或释放 GPU 资源。")
            self._update_task_status(task_id, status="failed", error=error_msg)
            self._add_log(task_id, f"Error: {e}")
        finally:
            with self._active_lock:
                self._active_tasks -= 1
            self._semaphore.release()
            self._persist_state_if_due(force=True)

    # ==================== YOLO 训练 ====================

    def _do_train_yolo(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ):
        """执行 YOLO 训练"""
        self._add_log(task_id, f"YOLO training started for group '{group_id}'")
        self._update_task_status(task_id, status="preparing", progress=5)

        with self._status_lock:
            save_dir = self.training_status[task_id]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        train_mode = config.get("train_mode", "by_pos_id")
        category_label = config.get("category_label", "unknown")
        use_synthetic = config.get("synthetic_defects", True)
        synthetic_per_normal = config.get("synthetic_per_normal", 3)
        is_retrain = bool(config.get("base_model_dir"))

        self._update_task_status(task_id, progress=10)

        path_id = config.get("path_id")
        if path_id is None:
            if group_annotations:
                first_ann = group_annotations[0]
                path_id = str(first_ann.get("category_id", 0)) if train_mode == "by_category" \
                          else str(first_ann.get("pos_id", group_id))
            else:
                path_id = str(group_id)

        # Stage 1: Extract ROI images
        self._update_task_status(task_id, status="preparing", progress=15, stage="1/5")
        self._add_log(task_id, f"Stage 1/5: Extracting ROI images ({len(group_annotations)} annotations, label={category_label})...")

        roi_save_dir = Path(dataset_dir) / "roi" / str(path_id)
        mask_save_dir = Path(dataset_dir) / "masks" / str(path_id)
        roi_save_dir.mkdir(parents=True, exist_ok=True)
        mask_save_dir.mkdir(parents=True, exist_ok=True)

        if is_retrain:
            base_model_dir = Path(config.get("base_model_dir"))
            base_roi_dir = Path(dataset_dir) / "roi" / str(path_id)
            for img_path in sorted(list(base_roi_dir.glob("*.png")) + list(base_roi_dir.glob("*.jpg"))):
                if img_path.name.startswith("fp_"):
                    continue
                dst = roi_save_dir / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
            self._add_log(task_id, f"Copied base ROI images from {base_roi_dir}")

        yolo_imgsz = config.get("yolo_imgsz", 320)
        image_paths = extract_roi_images(
            dataset_dir=dataset_dir,
            output_dir=str(roi_save_dir),
            mask_output_dir=str(mask_save_dir),
            group_annotations=group_annotations,
            normalize_brightness=config.get("normalize_brightness", False),
            normalize_contrast=config.get("normalize_contrast", False),
            augment=False,
            num_augmentations=1,
            train_mode=train_mode,
            augmentation_config=config.get("augmentation_config"),
            target_size=(yolo_imgsz, yolo_imgsz),
        )

        normal_roi_paths = sorted(list(roi_save_dir.glob("*.png")) + list(roi_save_dir.glob("*.jpg")))
        self._add_log(task_id, f"ROI extraction done: {len(normal_roi_paths)} normal ROI images")

        # Stage 2: 收集所有真实 FN（当前 + 历史），一次收集、直接传列表
        self._update_task_status(task_id, progress=25, stage="2/5")
        fn_images_dir = Path(save_dir) / "fn_images"

        all_fn_files: List[Path] = []
        if fn_images_dir.exists():
            all_fn_files.extend(fn_images_dir.rglob("*fn_*.jpg"))
        if is_retrain:
            project_dir = Path(save_dir).parent.parent
            current_path_id_val = Path(save_dir).name
            current_task_uuid = Path(save_dir).parent.name
            for task_dir in project_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name == current_task_uuid:
                    continue
                hd = task_dir / current_path_id_val / "fn_images"
                if hd.exists():
                    all_fn_files.extend(hd.rglob("*fn_*.jpg"))

        n_real_fn = len(all_fn_files)
        self._add_log(task_id, f"Stage 2/5: Found {n_real_fn} real FN images (current + historical)")

        # 计算缺陷目标数量
        max_aug_ratio = config.get("max_augmentation_ratio", 40)
        max_total = config.get("max_total_samples", 500)
        min_total = config.get("min_total_samples", 50)

        # 合成缺陷始终生成，保底 min_total 张，保证"纹理破坏=异常"的泛化概念不退化
        synthetic_defect_paths: List[Path] = []
        if use_synthetic:
            syn_target = max(min_total, min_total)  # 始终至少 50 张合成
            syn_per_normal = max(1, (syn_target + len(normal_roi_paths) - 1) // len(normal_roi_paths))
            self._add_log(task_id,
                f"Stage 2/5: Generating ~{syn_target} synthetic defects for diversity "
                f"(real FN: {n_real_fn})"
            )
            synthetic_defect_paths = self._generate_synthetic_defects(
                task_id, normal_roi_paths, save_dir, syn_per_normal,
            )
            synthetic_defect_paths = synthetic_defect_paths[:syn_target]
            self._add_log(task_id, f"Stage 2/5: Generated {len(synthetic_defect_paths)} synthetic")

        # defect_target 基于真实FN+合成的总数，与正常类平衡
        total_defect_originals = n_real_fn + len(synthetic_defect_paths)
        defect_target = min(total_defect_originals * max_aug_ratio, max_total)
        defect_target = max(defect_target, min_total, total_defect_originals)

        # Stage 3: Build YOLO dataset（FN 直接传列表，不再重复收集）
        self._update_task_status(task_id, progress=30, stage="3/5")
        self._add_log(task_id, "Stage 3/5: Building YOLO detection dataset...")

        yolo_dataset_dir = Path(save_dir) / "_yolo_dataset"
        historical_yolo_fp_dirs: List[str] = []

        if is_retrain:
            project_dir = Path(save_dir).parent.parent
            current_path_id_val = Path(save_dir).name
            current_task_uuid = Path(save_dir).parent.name
            for task_dir in sorted(project_dir.iterdir()):
                if not task_dir.is_dir() or task_dir.name == current_task_uuid:
                    continue
                hist_dir = task_dir / current_path_id_val
                hist_yolo_fp_dir = hist_dir / "yolo_fp_images"
                if hist_yolo_fp_dir.exists() and (
                    list(hist_yolo_fp_dir.rglob("yolo_fp_*.jpg")) or list(hist_yolo_fp_dir.rglob("yolo_fp_*.png"))
                ):
                    historical_yolo_fp_dirs.append(str(hist_yolo_fp_dir))
            if historical_yolo_fp_dirs:
                self._add_log(task_id, f"[YOLO] Found {len(historical_yolo_fp_dirs)} historical YOLO FP dirs")

        count = self._prepare_yolo_dataset(
            task_id=task_id,
            normal_roi_dir=str(roi_save_dir),
            output_dir=yolo_dataset_dir,
            config=config,
            yolo_fp_dir=str(Path(save_dir) / "yolo_fp_images"),
            historical_yolo_fp_dirs=historical_yolo_fp_dirs or None,
            all_fn_files=all_fn_files,
            synthetic_defect_paths=synthetic_defect_paths if synthetic_defect_paths else None,
            defect_target=defect_target,
        )

        if count <= 0:
            raise ValueError(f"No training samples for YOLO (normal={len(normal_roi_paths)}, synthetic={len(synthetic_defect_paths)})")

        # Stage 4: Train YOLO
        self._update_task_status(task_id, status="training", progress=40, stage="4/5")
        self._add_log(task_id, "Stage 4/5: Training YOLO detector...")
        self._train_yolo_detector(task_id, yolo_dataset_dir, save_dir, config, progress_range=(40, 80))

        # Stage 5: Compute threshold and save
        self._update_task_status(task_id, progress=85, stage="5/5")
        self._add_log(task_id, "Stage 5/5: Computing threshold and saving model...")

        model_path = Path(save_dir) / "yolo_model.pt"
        threshold, metrics = self._compute_yolo_threshold_and_metrics(
            task_id, model_path, normal_roi_paths, config, save_dir,
        )

        self._save_yolo_model(task_id, save_dir, dataset_dir, config, threshold, metrics, group_annotations)

        self._update_task_status(
            task_id, status="completed", progress=100, stage="completed", end_time=time.time(),
        )
        self._persist_state_if_due(force=True)
        self._add_log(task_id, f"YOLO training completed for group '{group_id}'")

    def _compute_threshold_from_normal_scores(
        self,
        task_id: str,
        scores: np.ndarray,
        method: str = "percentile",
        percentile: float = 99,
        n_sigma: float = 3.0,
        fallback: float = 0.5,
    ) -> float:
        """
        仅有正常样本时，从分数分布计算阈值。

        method:
            percentile  → P{percentile} 分位数，推荐首选
            sigma       → mean + n_sigma * std，分布接近正态时效果好
            max         → 正常样本最大值，最保守（误报率最低，漏报率最高）
        """
        if scores is None or len(scores) == 0:
            self._add_log(task_id, f"No scores available, using fallback threshold={fallback:.4f}")
            return fallback

        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        s_mean = float(np.mean(scores))
        s_std = float(np.std(scores))

        self._add_log(
            task_id,
            f"Normal score stats: min={s_min:.4f}, max={s_max:.4f}, "
            f"mean={s_mean:.4f}, std={s_std:.4f}, n={len(scores)}"
        )

        # 样本极少时，百分位统计意义有限，切换到 max 方法
        min_samples_for_percentile = 10
        if len(scores) < min_samples_for_percentile and method == "percentile":
            method = "max"
            self._add_log(task_id, f"Too few samples ({len(scores)}), switching to 'max' method")

        if method == "percentile":
            threshold = float(np.percentile(scores, percentile))
            self._add_log(task_id, f"Threshold (P{percentile}): {threshold:.4f}")

        elif method == "sigma":
            threshold = s_mean + n_sigma * s_std
            self._add_log(task_id, f"Threshold (mean + {n_sigma}σ): {threshold:.4f}")

        elif method == "max":
            # 安全余量 = 3σ 或 s_max 的 30%，取大者（与正常得分分布成比例）
            safety_margin = max(3.0 * s_std, s_max * 0.3)
            threshold = s_max + safety_margin
            self._add_log(task_id, f"Threshold (max={s_max:.4f} + margin={safety_margin:.4f}): {threshold:.4f}")

        else:
            self._add_log(task_id, f"Unknown method '{method}', using fallback={fallback:.4f}")
            threshold = fallback

        # 不低于 mean（否则一半正常样本误报）
        if threshold < s_mean:
            threshold = s_mean + 0.5 * s_std
            self._add_log(task_id, f"WARNING: threshold below mean, clamping to {threshold:.4f}")

        # 钳位到 [0.5, 0.95]，确保阈值在 YOLO 置信度有效范围内
        if threshold < 0.5:
            self._add_log(task_id, f"[YOLO] Threshold {threshold:.4f} below minimum 0.5, clamping")
            threshold = 0.5
        if threshold > 0.95:
            self._add_log(task_id, f"[YOLO] Threshold {threshold:.4f} above maximum 0.95, clamping")
            threshold = 0.95

        return threshold

    def _augment_fn_images(
        self,
        task_id: str,
        fn_files: List[Path],
        output_dir: Path,
        target_total: int = 200,
        label: str = "defect",
        input_size: Tuple[int, int] = (320, 320),
    ) -> int:
        """预处理 (letterbox_resize) + albumentations 增强填满 target_total。YOLO 在线增强叠加生效。"""
        n_orig = len(fn_files)
        self._add_log(task_id, f"[YOLO] Found {n_orig} original {label} images, target={target_total}")

        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        preprocessed = []
        for p in fn_files:
            try:
                img = Image.open(p).convert("RGB")
                img_padded, _ = letterbox_resize(img, input_size)
                rel_name = f"{p.parent.name}_{p.name}"
                dst = output_dir / rel_name
                img_padded.save(str(dst), quality=95)
                preprocessed.append((dst, img_padded))
            except Exception as e:
                self._add_log(task_id, f"[YOLO] Failed to preprocess {label} image {p}: {e}")

        n_valid = len(preprocessed)
        if n_valid == 0:
            self._add_log(task_id, "[YOLO] No valid images after preprocessing")
            return 0

        if n_valid >= target_total:
            self._add_log(task_id, f"[YOLO] {label} images already >= target ({n_valid} >= {target_total}), skip augmentation")
            return n_valid

        # albumentations 几何/模糊增强 + 自适应亮度对比度（暗图不更暗，亮图不更亮）
        aug_per_image = max(1, math.ceil((target_total - n_valid) / n_valid))
        self._add_log(task_id, f"[YOLO] Augmenting {aug_per_image}x per {label} image (need {target_total - n_valid}, have {n_valid})")

        fn_aug_config = PROJECT_ROOT / "configs" / "augmentations_fn.yaml"
        transform = load_augmentation_transform(str(fn_aug_config))
        adaptive_bc = AdaptiveBrightnessContrast(p=0.7)

        total = n_valid
        for img_path, img_padded in preprocessed:
            for j in range(aug_per_image):
                if total >= target_total:
                    break
                try:
                    aug_img = img_padded
                    if transform:
                        aug_img = apply_augmentation(aug_img, transform)
                    aug_np = np.array(aug_img)
                    aug_np = adaptive_bc(image=aug_np)['image']
                    aug_img = Image.fromarray(aug_np)
                except Exception:
                    pass
                aug_path = output_dir / f"{img_path.stem}_aug{j:04d}.jpg"
                aug_img.save(str(aug_path), quality=95)
                total += 1

        self._add_log(task_id, f"[YOLO] {label} done: {n_valid} originals + {total - n_valid} augmented = {total}")
        return total

    def _augment_yolo_fp_images(
        self,
        task_id: str,
        yolo_fp_files: List[Path],
        output_dir: Path,
        target_total: int = 30,
        input_size: Tuple[int, int] = (320, 320),
    ) -> List[Path]:
        """预处理 YOLO FP 原图 (letterbox_resize) + albumentations 增强。返回处理后的图片路径列表。"""
        if not yolo_fp_files:
            return []

        n_orig = len(yolo_fp_files)
        self._add_log(task_id, f"[YOLO] Found {n_orig} original YOLO FP images, target={target_total}")

        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_paths = []
        preprocessed = []
        for p in yolo_fp_files:
            try:
                img = Image.open(p).convert("RGB")
                img_padded, _ = letterbox_resize(img, input_size)
                dst = output_dir / p.name
                img_padded.save(str(dst), quality=95)
                processed_paths.append(dst)
                preprocessed.append((dst, img_padded))
            except Exception as e:
                self._add_log(task_id, f"[YOLO] Failed to preprocess YOLO FP image {p}: {e}")

        n_valid = len(preprocessed)
        if n_valid == 0 or n_valid >= target_total:
            self._add_log(task_id, f"[YOLO] YOLO FP: {n_valid} valid images, skip augmentation")
            return processed_paths

        aug_per_image = max(1, math.ceil((target_total - n_valid) / n_valid))
        self._add_log(task_id, f"[YOLO] Augmenting YOLO FP {aug_per_image}x per image (need {target_total - n_valid}, have {n_valid})")

        yolo_fp_aug_config = PROJECT_ROOT / "configs" / "augmentations_fn.yaml"
        transform = load_augmentation_transform(str(yolo_fp_aug_config))
        adaptive_bc = AdaptiveBrightnessContrast(p=0.7)

        total = n_valid
        for img_path, img_padded in preprocessed:
            for j in range(aug_per_image):
                if total >= target_total:
                    break
                try:
                    aug_img = img_padded
                    if transform:
                        aug_img = apply_augmentation(aug_img, transform)
                    aug_np = np.array(aug_img)
                    aug_np = adaptive_bc(image=aug_np)['image']
                    aug_img = Image.fromarray(aug_np)
                except Exception:
                    aug_img = img_padded
                aug_path = output_dir / f"{img_path.stem}_aug{j:04d}.jpg"
                aug_img.save(str(aug_path), quality=95)
                processed_paths.append(aug_path)
                total += 1

        self._add_log(task_id, f"[YOLO] YOLO FP done: {n_valid} originals + {total - n_valid} augmented = {total}")
        return processed_paths

    def _prepare_yolo_dataset(
        self,
        task_id: str,
        normal_roi_dir: str,
        output_dir: Path,
        config: dict,
        yolo_fp_dir: str = None,
        historical_yolo_fp_dirs: List[str] = None,
        all_fn_files: List[Path] = None,
        synthetic_defect_paths: List[Path] = None,
        defect_target: int = 200,
    ) -> int:
        """
        构建 YOLO 检测二分类数据集（detection 格式）。
        - normal 图 → 空 label（无目标）+ YOLO FP 图片（如果提供）
        - defect 图 → 整图 bbox（class 0 = defect）
        all_fn_files: 调用方预先收集的所有真实 FN 路径列表（不再重复收集）
        synthetic_defect_paths: 合成缺陷图片路径列表
        defect_target: 缺陷类目标总数
        返回 defect 类样本数；返回 0 表示跳过训练。
        """
        # 合并真实 FN + 合成缺陷（FN 由调用方预先收集，不再重复扫目录）
        all_fn_files = list(all_fn_files or [])
        all_fn_files.extend(synthetic_defect_paths or [])

        min_samples = config.get("yolo_min_samples", 1)
        hist_info = ""
        if synthetic_defect_paths:
            hist_info = f" + {len(synthetic_defect_paths)} synthetic"
        self._add_log(task_id, f"[YOLO] Preparing detection dataset: normal={normal_roi_dir}, fn={len(all_fn_files)} files{hist_info}, min_samples={min_samples}")

        if len(all_fn_files) < min_samples:
            self._add_log(task_id, f"[YOLO] Insufficient FN images ({len(all_fn_files)} < {min_samples}), skipping")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)

        # YOLO detection 目录结构: train/images/, train/labels/, val/images/, val/labels/
        train_img_dir = output_dir / "train" / "images"
        train_lbl_dir = output_dir / "train" / "labels"
        val_img_dir = output_dir / "val" / "images"
        val_lbl_dir = output_dir / "val" / "labels"
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        yolo_imgsz = config.get("yolo_imgsz", 320)
        input_size = (yolo_imgsz, yolo_imgsz)

        # 1. defect 类：FN 原图 + 增强，整图作为 bbox
        # 增强倍率限制：每张原图最多增强 max_aug_ratio 倍，且与 normal 保持平衡
        max_aug_ratio = config.get("max_augmentation_ratio", 40)
        defect_max = min(defect_target, len(all_fn_files) * max_aug_ratio)

        # 先找到 normal ROI 目录并计算 normal 的最大可增强数量
        normal_roi_path = Path(normal_roi_dir)
        if not normal_roi_path.exists():
            train_base = normal_roi_path.parent.parent.parent
            path_id_str = normal_roi_path.name
            found_roi_dirs = sorted(train_base.glob(f"*/roi/{path_id_str}"), key=lambda p: p.stat().st_mtime, reverse=True)
            if found_roi_dirs:
                normal_roi_path = found_roi_dirs[0]
                self._add_log(task_id, f"[YOLO] Fallback to ancestor normal ROI: {normal_roi_path}")
            else:
                self._add_log(task_id, f"[YOLO] WARNING: normal_roi_dir not found: {normal_roi_dir}")
                return 0

        normal_raw = list(normal_roi_path.glob("*.png")) + list(normal_roi_path.glob("*.jpg"))
        self._add_log(task_id, f"[YOLO] Found {len(normal_raw)} normal ROI, {len(all_fn_files)} defect originals")

        # ---- 验证集：从增强后数据池 80/20 拆分 ----
        # 原图太少时无法独立预留验证集，只能从增强池拆分保证 val 样本量充足

        # 平衡目标：双方都能达到的数量，且至少包含所有原始图
        normal_max = len(normal_raw) * max_aug_ratio
        balanced_target = min(defect_max, normal_max)
        balanced_target = max(balanced_target, len(all_fn_files), len(normal_raw))
        self._add_log(task_id, f"[YOLO] Balanced target: {balanced_target} (defect originals={len(all_fn_files)}, normal originals={len(normal_raw)}, defect_max={defect_max}, normal_max={normal_max})")

        defect_aug_dir = output_dir / "_defect_aug"
        defect_count = self._augment_fn_images(task_id, all_fn_files, defect_aug_dir, target_total=balanced_target, label="defect", input_size=input_size)

        # 2. normal 类：统一使用 max_aug_ratio，与 defect 相同倍率
        normal_aug_dir = output_dir / "_normal_aug"
        normal_target = min(defect_count, normal_max)
        normal_target = max(normal_target, len(normal_raw))
        if len(normal_raw) < normal_target:
            self._add_log(task_id, f"[YOLO] Augmenting normal images: {len(normal_raw)} → target {normal_target}")
            self._augment_fn_images(task_id, normal_raw, normal_aug_dir, target_total=normal_target, label="normal", input_size=input_size)
            normal_paths = sorted(list(normal_aug_dir.glob("*.jpg")))
        else:
            normal_paths = [Path(p) for p in normal_raw]
            normal_aug_dir = None

        # 2b. YOLO FP 图片 → 加入 normal 类
        yolo_fp_raw: List[Path] = []
        if yolo_fp_dir:
            yolo_fp_path = Path(yolo_fp_dir)
            if yolo_fp_path.exists():
                yolo_fp_raw.extend(yolo_fp_path.glob("yolo_fp_*.jpg"))
                yolo_fp_raw.extend(yolo_fp_path.glob("yolo_fp_*.png"))
        for hist_dir_str in (historical_yolo_fp_dirs or []):
            hist_dir = Path(hist_dir_str)
            if hist_dir.exists():
                yolo_fp_raw.extend(hist_dir.glob("yolo_fp_*.jpg"))
                yolo_fp_raw.extend(hist_dir.glob("yolo_fp_*.png"))

        if yolo_fp_raw:
            yolo_fp_aug_dir = output_dir / "_yolo_fp_aug"
            yolo_fp_aug_target = min(50, max(10, len(normal_paths) // 5))
            yolo_fp_processed = self._augment_yolo_fp_images(
                task_id, yolo_fp_raw, yolo_fp_aug_dir, target_total=yolo_fp_aug_target, input_size=input_size,
            )
            if yolo_fp_processed:
                self._add_log(task_id, f"[YOLO] Adding {len(yolo_fp_processed)} preprocessed YOLO FP images to normal class "
                               f"(from {len(yolo_fp_raw)} raw, current + {len(historical_yolo_fp_dirs or [])} historical)")
                normal_paths = normal_paths + yolo_fp_processed

        # 3. 从增强池 shuffle 后 80/20 拆分 train/val
        all_normal = [p for p in normal_paths]
        all_defect = [p for p in defect_aug_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')]

        random.shuffle(all_normal)
        random.shuffle(all_defect)

        # val 至少 3 张/类，保证 mAP 稳定
        n_val_normal = max(3, int(len(all_normal) * 0.2))
        n_val_defect = max(3, int(len(all_defect) * 0.2))
        n_val_normal = min(n_val_normal, len(all_normal) - 3)  # train 至少留 3 张
        n_val_defect = min(n_val_defect, len(all_defect) - 3)

        train_normals = all_normal[n_val_normal:]
        val_normals = all_normal[:n_val_normal]
        train_defects = all_defect[n_val_defect:]
        val_defects = all_defect[:n_val_defect]

        # 4. 复制图片并生成 label
        def _place_images(paths, img_dir, lbl_dir, is_defect):
            for p in paths:
                dst_name = p.name
                shutil.copy2(str(p), str(img_dir / dst_name))
                label_path = lbl_dir / (Path(dst_name).stem + ".txt")
                if is_defect:
                    label_path.write_text("0 0.5 0.5 1.0 1.0\n")
                else:
                    label_path.write_text("")

        _place_images(train_normals, train_img_dir, train_lbl_dir, is_defect=False)
        _place_images(train_defects, train_img_dir, train_lbl_dir, is_defect=True)
        _place_images(val_normals, val_img_dir, val_lbl_dir, is_defect=False)
        _place_images(val_defects, val_img_dir, val_lbl_dir, is_defect=True)

        # 5. 生成 YAML 配置文件
        yaml_path = output_dir / "dataset.yaml"
        yaml_content = f"""# YOLO detection dataset (auto-generated)
path: {output_dir}
train: train/images
val: val/images

names:
  0: defect
"""
        yaml_path.write_text(yaml_content)

        self._add_log(
            task_id,
            f"[YOLO] Dataset built: train normal={len(train_normals)}, train defect={len(train_defects)}, "
            f"val normal={len(val_normals)}, val defect={len(val_defects)}, "
            f"yaml={yaml_path}"
        )

        # 清理临时目录
        shutil.rmtree(defect_aug_dir, ignore_errors=True)
        if normal_aug_dir:
            shutil.rmtree(normal_aug_dir, ignore_errors=True)
        if yolo_fp_raw:
            shutil.rmtree(yolo_fp_aug_dir, ignore_errors=True)

        return defect_count

    def _train_yolo_detector(
        self,
        task_id: str,
        dataset_dir: Path,
        save_dir: str,
        config: dict,
        progress_range: tuple = (92, 98),
    ):
        """训练 YOLOv8 检测器（epochs=100, patience=15 早停）。"""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._add_log(task_id, "[YOLO] ERROR: ultralytics not installed, skip YOLO training")
            return

        # 始终从 yolo26n 开始训练（数据累积保证知识不丢失，避免微调遗忘）
        pretrained_path = PROJECT_ROOT / "models" / "pretrained" / "yolo26n.pt"
        if not pretrained_path.exists():
            self._add_log(task_id, f"[YOLO] Local yolo26n not found at {pretrained_path}, will try auto-download")
            pretrained_path = "yolo26n.pt"

        yolo_epochs = config.get("yolo_epochs", 40)
        yolo_imgsz = config.get("yolo_imgsz", 320)
        yaml_path = dataset_dir / "dataset.yaml"
        self._add_log(task_id, f"[YOLO] Starting detection training: epochs={yolo_epochs}, patience=15, imgsz={yolo_imgsz}, lr0=0.005, cos_lr=True, warmup=3, data={yaml_path}")
        train_start = time.time()

        trainer_ref = self
        p_min, p_max = progress_range

        def on_epoch_end(trainer):
            current = trainer.epoch + 1
            progress = int(p_min + (current / yolo_epochs) * (p_max - p_min))
            trainer_ref._update_task_status(task_id, progress=min(progress, 99))
            loss = trainer.loss
            metrics = getattr(trainer, 'metrics', {}) or {}
            mAP50 = metrics.get('metrics/mAP50(B)', None)
            mAP50_95 = metrics.get('metrics/mAP50-95(B)', None)
            parts = [f"loss={loss:.4f}"] if loss is not None else []
            if mAP50 is not None:
                parts.append(f"mAP50={mAP50:.4f}")
            if mAP50_95 is not None:
                parts.append(f"mAP50-95={mAP50_95:.4f}")
            metrics_str = " ".join(parts) if parts else ""
            trainer_ref._add_log(
                task_id,
                f"[YOLO] Epoch {current}/{yolo_epochs} {metrics_str}"
            )

        model = YOLO(str(pretrained_path))
        model.add_callback("on_train_epoch_end", on_epoch_end)

        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            results = model.train(
                data=str(yaml_path),
                epochs=yolo_epochs,
                patience=15,
                imgsz=yolo_imgsz,
                batch=config.get("yolo_batch", 16),
                workers=8,
                lr0=0.005,
                lrf=0.01,
                warmup_epochs=3,
                amp=True,
                cos_lr=True,
                seed=42,
                verbose=False,
                exist_ok=True,
                project=str(Path(save_dir) / "yolo_runs"),
                name="train",
                # 在线增强：模拟推理端 warpAffine 引入的角度/平移/错切扰动
                augment=True,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.3,
                fliplr=0.5,
                flipud=0.5,
                degrees=15,
                translate=0.15,
                scale=0.15,
                shear=5,
                perspective=0.0005,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
            )
        finally:
            sys.stdout = old_stdout

        # 记录训练结果指标
        elapsed = time.time() - train_start
        results_dict = getattr(results, 'results_dict', {}) or {}
        mAP50 = results_dict.get('metrics/mAP50(B)', None)
        mAP50_95 = results_dict.get('metrics/mAP50-95(B)', None)
        precision = results_dict.get('metrics/precision(B)', None)
        recall = results_dict.get('metrics/recall(B)', None)

        # 从 savedir 获取实际完成的 epoch 数
        actual_epochs = "?"
        if hasattr(results, "save_dir") and results.save_dir:
            results_csv = Path(results.save_dir) / "results.csv"
            if results_csv.exists():
                try:
                    lines = results_csv.read_text().strip().split("\n")
                    actual_epochs = str(len(lines) - 1)
                except Exception:
                    pass

        self._add_log(task_id,
            f"[YOLO] Training finished in {elapsed:.1f}s: "
            f"epochs={actual_epochs}/{yolo_epochs}, "
            f"mAP50={mAP50 if mAP50 is not None else 'N/A'}, "
            f"mAP50-95={mAP50_95 if mAP50_95 is not None else 'N/A'}, "
            f"P={precision if precision is not None else 'N/A'}, "
            f"R={recall if recall is not None else 'N/A'}"
        )

        # 查找 best.pt 并复制到模型目录
        if hasattr(results, "save_dir") and results.save_dir:
            best_pt = Path(results.save_dir) / "weights" / "best.pt"
        else:
            # 回退：project/name 组合路径
            best_pt = Path(save_dir) / "yolo_runs" / "train" / "weights" / "best.pt"

        if best_pt and best_pt.exists():
            dst = Path(save_dir) / "yolo_model.pt"
            shutil.copy2(str(best_pt), str(dst))
            self._add_log(task_id, f"[YOLO] Model saved: {dst} ({elapsed:.1f}s)")
        else:
            self._add_log(task_id, "[YOLO] WARNING: best.pt not found after training")

    def get_training_status(self, task_id: str) -> dict:
        """获取任务状态"""
        with self._status_lock:
            status = self.training_status.get(task_id, {}).copy()
        status.pop("group_annotations", None)
        return status

    get_task_status = get_training_status

    def get_task_group_status(self, group_id: str) -> dict:
        """获取任务组状态"""
        with self._status_lock:
            group_tasks = [
                {"task_id": tid, **s}
                for tid, s in self.training_status.items()
                if s.get("group_id") == group_id or s.get("task_uuid") == group_id
            ]

        if not group_tasks:
            return {"status": "not_found", "message": "Group not found"}

        statuses = [t.get("status") for t in group_tasks]
        if all(s == "completed" for s in statuses):
            overall = "completed"
        elif any(s == "failed" for s in statuses):
            overall = "failed"
        elif any(s in {"training", "preparing"} for s in statuses):
            overall = "training"
        else:
            overall = statuses[0] if statuses else "unknown"

        progresses = [t.get("progress", 0) for t in group_tasks]
        avg_progress = sum(progresses) / len(progresses) if progresses else 0

        for task in group_tasks:
            task.pop("group_annotations", None)

        return {
            "status": overall,
            "progress": avg_progress,
            "group_id": group_id,
            "tasks": group_tasks,
            "total_tasks": len(group_tasks),
            "completed_tasks": sum(1 for s in statuses if s == "completed"),
        }


    def _generate_synthetic_defects(
        self,
        task_id: str,
        normal_roi_paths: List[Path],
        save_dir: str,
        per_image: int = 3,
    ) -> List[Path]:
        """从正常 ROI 生成合成缺陷，返回合成缺陷图片路径列表"""
        synthetic_dir = Path(save_dir) / "_synthetic_defects"
        if synthetic_dir.exists():
            shutil.rmtree(str(synthetic_dir))
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        synthetic_config = load_synthetic_config(None)
        strategies = synthetic_config.get("strategies", {})
        generator = SyntheticDefectGenerator(
            strategies=list(strategies.keys()) if strategies else ["texture_swap", "solid_color"],
            weights=list(strategies.values()) if strategies else [0.6, 0.4],
            min_defect_size_ratio=synthetic_config.get("min_defect_size_ratio", 0.10),
            max_defect_size_ratio=synthetic_config.get("max_defect_size_ratio", 0.60),
        )

        # 预加载所有正常 ROI 作为纹理替换池
        all_roi_imgs: List[np.ndarray] = []
        for p in normal_roi_paths:
            img = cv2.imread(str(p))
            if img is not None:
                all_roi_imgs.append(img)

        synthetic_paths: List[Path] = []
        for idx, img_path in enumerate(normal_roi_paths):
            try:
                img = all_roi_imgs[idx] if idx < len(all_roi_imgs) else cv2.imread(str(img_path))
                if img is None:
                    continue
                # 纹理替换池 = 其他正常 ROI（排除自己）
                pool = [all_roi_imgs[i] for i in range(len(all_roi_imgs)) if i != idx]
                generator.set_patch_pool(pool)
                defects = generator.generate_multiple(img, per_image)
                for j, (defect_img, bbox) in enumerate(defects):
                    out_path = synthetic_dir / f"syn_{img_path.stem}_{j:02d}.jpg"
                    cv2.imwrite(str(out_path), defect_img)
                    synthetic_paths.append(out_path)
            except Exception as e:
                self._add_log(task_id, f"[Synthetic] Failed for {img_path.name}: {e}")

        self._add_log(task_id, f"[Synthetic] Generated {len(synthetic_paths)} synthetic defects from {len(normal_roi_paths)} normal images")
        return synthetic_paths

    def _compute_yolo_threshold_and_metrics(
        self,
        task_id: str,
        model_path: Path,
        normal_roi_paths: List[Path],
        config: dict,
        save_dir: str,
    ) -> Tuple[float, Dict]:
        """阈值固定为 0.5，不做动态计算。"""
        self._add_log(task_id, "[YOLO] Threshold fixed at 0.5 (dynamic computation disabled)")
        return 0.5, {"n_samples": len(normal_roi_paths)}

    def _save_yolo_model(
        self,
        task_id: str,
        save_dir: str,
        dataset_dir: str,
        config: dict,
        threshold: float,
        metrics: Dict,
        group_annotations: List[Dict],
    ):
        """保存 YOLO 模型配置和阈值"""
        save_path = Path(save_dir)
        config_path = save_path / "config.json"
        threshold_path = save_path / "threshold.json"

        # Compute ROI size stats
        roi_sizes = []
        padding = 10
        for ann in group_annotations:
            segmentation = ann.get("segmentation", [])
            if segmentation and len(segmentation) > 0:
                coords = segmentation[0]
                if len(coords) >= 8:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    w = int(x_max - x_min + 2 * padding)
                    h = int(y_max - y_min + 2 * padding)
                    roi_sizes.append({"width": w, "height": h})

        if roi_sizes:
            avg_width = sum(s["width"] for s in roi_sizes) / len(roi_sizes)
            avg_height = sum(s["height"] for s in roi_sizes) / len(roi_sizes)
            target_size = (int(avg_width), int(avg_height))
        else:
            target_size = (224, 224)

        config_data = {
            "model_type": "yolo",
            "train_mode": config.get("train_mode", "by_pos_id"),
            "category": config.get("category", ""),
            "category_label": config.get("category_label", ""),
            "threshold": threshold,
            "target_size": target_size,
            "normalize_brightness": config.get("normalize_brightness", False),
            "normalize_contrast": config.get("normalize_contrast", False),
            "train_images": len(group_annotations),
            "yolo_imgsz": config.get("yolo_imgsz", 320),
            "yolo_epochs": config.get("yolo_epochs", 100),
            "yolo_batch": config.get("yolo_batch", 16),
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        threshold_data = {
            "threshold": threshold,
            "method": config.get("threshold_method", "percentile"),
            "percentile": config.get("threshold_percentile", 99),
            "score_stats": {
                "min": metrics.get("score_min"),
                "max": metrics.get("score_max"),
                "mean": metrics.get("score_mean"),
                "std": metrics.get("score_std"),
                "n": metrics.get("n_samples"),
            },
        }
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_data, f, ensure_ascii=False, indent=2)

        self._add_log(task_id, f"Model config saved to: {config_path}")
    def stop_task(self, task_id: str) -> bool:
        """停止单个任务"""
        with self._status_lock:
            status = self.training_status.get(task_id)
            if not status:
                return False
            if status.get("status") not in {"starting", "training", "preparing"}:
                return False
            evt = self._stop_events.get(task_id)
            if evt:
                evt.set()
            status["status"] = "cancelled"
        self._persist_state_if_due(force=True)
        return True

    def stop_group(self, group_id: str) -> int:
        """停止整个组的任务"""
        stopped = 0
        with self._status_lock:
            task_ids_to_stop = [
                tid for tid, s in self.training_status.items()
                if (s.get("group_id") == group_id or s.get("task_uuid") == group_id)
                and s.get("status") in {"starting", "training", "preparing"}
            ]
        for tid in task_ids_to_stop:
            if self.stop_task(tid):
                stopped += 1
        return stopped

    def resume_task(self, task_id: str) -> str:
        """恢复任务"""
        with self._status_lock:
            status = self.training_status.get(task_id)
            if not status:
                raise ValueError(f"Task {task_id} not found")
            if status.get("status") not in {"failed", "cancelled"}:
                raise ValueError(f"Cannot resume task with status: {status.get('status')}")

            dataset_dir = status.get("dataset_dir")
            config = status.get("config", {})
            group_id = status.get("internal_group_id")
            group_annotations = status.get("group_annotations", [])

        if not group_annotations:
            raise ValueError("No group_annotations found for resume")

        new_task_id = self._create_group_training_task(
            dataset_dir, config, group_id, group_annotations
        )
        return new_task_id

    def list_tasks(self, status_filter: str = None) -> List[Dict]:
        """列出所有任务"""
        with self._status_lock:
            result = []
            for tid, s in self.training_status.items():
                if status_filter and s.get("status") != status_filter:
                    continue
                s_copy = {k: v for k, v in s.items() if k != "group_annotations"}
                s_copy["task_id"] = tid
                result.append(s_copy)
        return sorted(result, key=lambda x: x.get("start_time", 0), reverse=True)

    def cleanup_task(self, task_id: str) -> bool:
        """清理任务"""
        with self._status_lock:
            if task_id in self.training_status:
                del self.training_status[task_id]
                self._stop_events.pop(task_id, None)
        self._persist_state_if_due(force=True)
        return True


# 保持向后兼容
PatchCoreTrainer = ModelTrainer
