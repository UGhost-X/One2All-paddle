#!/usr/bin/env python3
"""
异常检测模型训练器
仅支持 Dinomaly 算法
"""

import os
import sys
import json
import time
import random
import logging
import traceback
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
import shutil
import tempfile

from utils.config import get_output_dir
from tqdm import tqdm
from collections import defaultdict

# 设置 timm 模型缓存目录为本地路径（必须在导入 timm/anomalib 之前设置）
PROJECT_ROOT = Path(__file__).parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
HUB_DIR = PRETRAINED_DIR / "hub"
os.environ["TIMM_HOME"] = str(PRETRAINED_DIR)
os.environ["HF_HOME"] = str(PRETRAINED_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(PRETRAINED_DIR / "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUB_DIR)
# 使用 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 允许联网下载
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

sys.path.insert(0, '/home/software/One2All-paddle')

# Anomalib 导入
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine

# PyTorch Lightning Callback
from pytorch_lightning.callbacks import Callback

try:
    from anomalib.models import Dinomaly
    DINORMALY_AVAILABLE = True
except ImportError:
    DINORMALY_AVAILABLE = False
    logging.warning("Dinomaly model not available")

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
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, np.ndarray]:
    """
    保持宽高比地将图片 padding 到 target_size，并返回有效区域的 binary mask。

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


def apply_augmentation(image: Image.Image, transform) -> Image.Image:
    """应用数据增强
    
    Args:
        image: 输入图片 (PIL.Image)
        transform: albumentations transform
        
    Returns:
        增强后的图片 (PIL.Image)
    """
    if transform is None:
        return image
        
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
            roi_image, roi_mask = extract_polygon_region(image_path, segmentation)

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
        model_name = config.get("model_name", "PatchCore")

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
        
        # 将 path_id 保存到 config，确保 _do_train_dinomaly 使用相同的值
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
                self.training_status[task_id] = {
                    "status": "starting",
                    "progress": 0,
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
        """训练进程入口（仅支持 Dinomaly）"""
        self._semaphore.acquire()
        with self._active_lock:
            self._active_tasks += 1
        try:
            self._do_train_dinomaly(task_id, dataset_dir, config, group_id, group_annotations)
        except SystemExit:
            self._add_log(task_id, "Training cancelled")
            self._update_task_status(task_id, status="cancelled")
        except Exception as e:
            logger.exception(f"Training failed for {task_id}")
            self._update_task_status(task_id, status="failed", error=str(e))
            self._add_log(task_id, f"Error: {e}")
        finally:
            with self._active_lock:
                self._active_tasks -= 1
            self._semaphore.release()
            self._persist_state_if_due(force=True)

    # ==================== Dinomaly 训练 ====================

    def _do_train_dinomaly(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ):
        """执行 Dinomaly 训练"""
        self._add_log(task_id, f"Dinomaly training started for group '{group_id}'")
        self._update_task_status(task_id, status="preparing", progress=5)

        with self._status_lock:
            save_dir = self.training_status[task_id]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        # 配置参数
        encoder_name = config.get("encoder_name", "dinov2_vit_base_14")
        decoder_depth = config.get("decoder_depth", 8)
        bottleneck_dropout = config.get("bottleneck_dropout", 0.2)
        # epochs = config.get("epochs", 20)
        epochs = 20
        batch_size = config.get("batch_size", 1)
        num_workers = config.get("num_workers", 4)
        normalize_brightness = config.get("normalize_brightness", False)
        normalize_contrast = config.get("normalize_contrast", False)
        freeze_encoder = config.get("freeze_encoder", True)
        category = config.get("category", "unknown")
        category_label = config.get("category_label", "unknown")
        # 数据增强参数
        augment = config.get("augment", False)
        num_augmentations = config.get("num_augmentations", 1)
        augmentation_config = config.get("augmentation_config", None)
        num_samples = len(group_annotations)

        self._update_task_status(task_id, num_samples=num_samples)
        self._add_log(task_id, f"Training with {num_samples} annotations for group '{group_id}' (label: {category_label})")
        self._add_log(task_id, f"Encoder: {encoder_name}, Decoder depth: {decoder_depth}, Epochs: {epochs}")

        if not DINORMALY_AVAILABLE:
            raise RuntimeError("Dinomaly model is not available")

        # Stage 1: 提取ROI图片
        self._update_task_status(task_id, status="preparing", progress=10, stage="1/3")
        self._add_log(task_id, "Stage 1/3: Preparing ROI images...")

        # 根据训练模式确定ROI保存路径
        train_mode = config.get("train_mode", "by_pos_id")
        task_uuid = config.get("task_uuid", "unknown")
        project_id = config.get("project_id", "default")

        # 从 config 中获取 path_id（由 _create_group_training_task 设置），确保一致性
        path_id = config.get("path_id")
        if path_id is None:
            # 兼容旧逻辑（如果 path_id 未设置）
            if group_annotations:
                first_ann = group_annotations[0]
                if train_mode == "by_category":
                    path_id = str(first_ann.get("category_id", 0))
                else:  # by_pos_id
                    path_id = str(first_ann.get("pos_id", group_id))
            else:
                path_id = str(group_id)
            self._add_log(task_id, f"Warning: path_id not in config, fallback to {path_id}")

        # ROI 保存路径: dataset_dir 已经是 product/{project_id}/train/{task_uuid}/
        # 直接在 dataset_dir 下创建 roi 子目录
        roi_save_dir = Path(dataset_dir) / "roi" / str(path_id)
        mask_save_dir = Path(dataset_dir) / "masks" / str(path_id)
        roi_save_dir.mkdir(parents=True, exist_ok=True)
        mask_save_dir.mkdir(parents=True, exist_ok=True)

        # 提取ROI图片到保存目录（包含数据增强），同时保存 mask
        image_paths = extract_roi_images(
            dataset_dir=dataset_dir,
            output_dir=str(roi_save_dir),
            mask_output_dir=str(mask_save_dir),
            group_annotations=group_annotations,
            normalize_brightness=normalize_brightness,
            normalize_contrast=normalize_contrast,
            augment=augment,
            num_augmentations=num_augmentations,
            train_mode=train_mode,
            augmentation_config=augmentation_config,
        )

        if not image_paths:
            raise ValueError(f"No ROI images extracted for group {group_id}")

        self._add_log(task_id, f"Saved {len(image_paths)} ROI images to {roi_save_dir}")

        # 直接使用 roi_save_dir 作为训练数据目录（禁止复制）
        self._add_log(task_id, f"Using ROI images directly from {roi_save_dir}")

        # Stage 2: 创建 Anomalib datamodule 和模型
        self._update_task_status(task_id, status="training", progress=30, stage="2/3")
        self._add_log(task_id, "Stage 2/3: Creating model and datamodule...")

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        try:
            model = Dinomaly(
                encoder_name=encoder_name,
                decoder_depth=decoder_depth,
                bottleneck_dropout=bottleneck_dropout,
            )
        except (OSError, IOError, Exception) as e:
            if 'dinov2' in str(e).lower() or 'download' in str(e).lower():
                raise RuntimeError(f"DINOv2 预训练权重下载失败: {e}")
            raise

        # 冻结编码器
        if freeze_encoder:
            encoder = None
            if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
                encoder = model.model.encoder
            elif hasattr(model, 'encoder'):
                encoder = model.encoder

            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False
                self._add_log(task_id, "Encoder frozen for faster training")

        # ✅ 自定义 Callback，每个 epoch 结束更新进度和日志
        class EpochProgressCallback(Callback):
            def __init__(self, trainer_ref, task_id, total_epochs):
                self.trainer_ref = trainer_ref
                self.task_id = task_id
                self.total_epochs = total_epochs

            def on_train_epoch_end(self, trainer, pl_module):
                current = trainer.current_epoch + 1
                # 进度从 50 到 80 之间分配给训练阶段
                progress = 50 + int((current / self.total_epochs) * 30)
                self.trainer_ref._update_task_status(self.task_id, progress=progress)
                self.trainer_ref._add_log(
                    self.task_id,
                    f"[task:{self.task_id[-8:]}] Epoch {current}/{self.total_epochs} done"
                )

        epoch_cb = EpochProgressCallback(self, task_id, epochs)

        # 使用临时目录作为 anomalib 日志目录，训练结束后自动清理
        import tempfile
        anomalib_temp_dir = tempfile.mkdtemp(prefix="anomalib_")
        
        engine = Engine(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            enable_progress_bar=False,   # ✅ 关掉 tqdm，避免多线程输出混乱
            enable_model_summary=False,
            check_val_every_n_epoch=epochs,
            callbacks=[epoch_cb],        # ✅ 加入 callback
            default_root_dir=anomalib_temp_dir,  # ✅ 使用临时目录，不保留日志
        )

        # 创建 Folder datamodule - 直接使用 roi_save_dir
        # 创建一个临时根目录，roi_save_dir 作为 normal 子目录
        temp_root = Path(save_dir) / "temp_anomalib"
        temp_root.mkdir(parents=True, exist_ok=True)

        # ✅ 修复：线程内强制 num_workers=0，避免 DataLoader fork 死锁
        safe_num_workers = 0
        self._add_log(task_id, f"DataLoader num_workers forced to 0 (threading mode)")

        # 创建符号链接或直接使用 - 这里使用 . 作为 root，roi_save_dir 作为 normal_dir 的绝对路径
        datamodule = Folder(
            name="train",
            root=temp_root,
            normal_dir=str(roi_save_dir),  # 直接使用 ROI 保存目录
            normal_test_dir=str(roi_save_dir),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=safe_num_workers,  # ✅ 强制使用 0
            test_split_mode=TestSplitMode.FROM_DIR,
            val_split_mode=ValSplitMode.SAME_AS_TEST,
            val_split_ratio=0.0,
        )
        datamodule.setup()

        # Stage 3: 训练模型
        self._update_task_status(task_id, progress=50, stage="3/3")
        self._add_log(task_id, f"{'='*20} Task {task_id[-8:]} Stage3 fit() START {'='*20}")
        self._add_log(task_id, "Stage 3/3: Training model...")

        train_start = time.time()

        # ✅ 加超时检测，防止永久卡死
        fit_exception = [None]

        def run_fit():
            try:
                engine.fit(model=model, datamodule=datamodule)
            except Exception as e:
                fit_exception[0] = e

        fit_thread = threading.Thread(target=run_fit, daemon=True)
        fit_thread.start()

        # 每 epoch 最多 10 分钟，可按需调整
        timeout_seconds = epochs * 600
        fit_thread.join(timeout=timeout_seconds)

        if fit_thread.is_alive():
            raise RuntimeError(f"engine.fit() timed out after {timeout_seconds}s (stuck in stage 3)")
        if fit_exception[0]:
            raise fit_exception[0]

        train_time = time.time() - train_start
        self._add_log(task_id, f"{'='*20} Task {task_id[-8:]} Stage3 fit() END {'='*20}")
        self._add_log(task_id, f"Training completed in {train_time:.2f}s")

        # 计算阈值（在训练集上推理）
        self._update_task_status(task_id, progress=80)
        self._add_log(task_id, "Computing threshold...")

        test_dataloader = datamodule.test_dataloader()
        predictions = engine.predict(model=model, dataloaders=test_dataloader)
        scores = self._extract_dinomaly_scores(predictions)

        # 获取阈值
        threshold = self._get_dinomaly_threshold(model)

        if len(scores) > 0:
            self._add_log(task_id, f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}], Threshold: {threshold:.4f}")

        # 保存模型
        self._update_task_status(task_id, progress=90)
        self._save_dinomaly_model(task_id, save_dir, model, engine, config, threshold, group_annotations)

        # 清理临时目录
        if temp_root.exists():
            shutil.rmtree(temp_root)
        
        # 清理 anomalib 临时日志目录
        if os.path.exists(anomalib_temp_dir):
            shutil.rmtree(anomalib_temp_dir)

        # 清理模型和显存
        self._cleanup_training_resources(model, engine, datamodule)

        self._update_task_status(
            task_id,
            status="completed",
            progress=100,
            stage="completed",
            end_time=time.time()
        )
        self._persist_state_if_due(force=True)
        self._add_log(task_id, f"Training completed successfully for group '{group_id}'")

    def _extract_dinomaly_scores(self, predictions) -> np.ndarray:
        """从 Dinomaly 预测结果中提取分数"""
        scores = []
        score_keys = ['pred_score', 'anomaly_score', 'pred_scores', 'anomaly_scores']

        for batch in predictions:
            found = False
            for key in score_keys:
                val = getattr(batch, key, None)
                if val is None and isinstance(batch, dict):
                    val = batch.get(key)
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        scores.extend(val.detach().cpu().numpy().flatten().tolist())
                    else:
                        scores.append(float(val))
                    found = True
                    break

        return np.array(scores) if scores else np.array([0.0])

    def _get_dinomaly_threshold(self, model) -> float:
        """获取 Dinomaly 阈值"""
        for attr in ('image_threshold', 'threshold'):
            obj = getattr(model, attr, None)
            if obj is not None:
                val = getattr(obj, 'value', None)
                if val is not None:
                    return float(val)
        return 0.55

    def _save_dinomaly_model(
        self,
        task_id: str,
        save_dir: str,
        model,
        engine: Engine,
        config: dict,
        threshold: float,
        group_annotations: List[Dict]
    ):
        """保存 Dinomaly 模型"""
        model_path = Path(save_dir) / "model.ckpt"
        config_path = Path(save_dir) / "config.json"
        threshold_path = Path(save_dir) / "threshold.json"

        # 保存 checkpoint
        if hasattr(engine, 'trainer') and engine.trainer is not None:
            engine.trainer.save_checkpoint(str(model_path))
            self._add_log(task_id, f"Checkpoint saved to: {model_path}")
        else:
            # 直接保存模型状态
            torch.save({
                'state_dict': model.state_dict(),
                'model_type': 'dinomaly',
            }, model_path)
            self._add_log(task_id, f"Model state saved to: {model_path}")

        # 计算ROI尺寸统计
        roi_sizes = []
        padding = 10
        for ann in group_annotations:
            segmentation = ann.get('segmentation', [])
            if segmentation and len(segmentation) > 0:
                coords = segmentation[0]
                if len(coords) >= 8:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    w = int(x_max - x_min + 2 * padding)
                    h = int(y_max - y_min + 2 * padding)
                    roi_sizes.append({'width': w, 'height': h})

        if roi_sizes:
            avg_width = sum(s['width'] for s in roi_sizes) / len(roi_sizes)
            avg_height = sum(s['height'] for s in roi_sizes) / len(roi_sizes)
            target_size = (int(avg_width), int(avg_height))
        else:
            target_size = (224, 224)
            avg_width = avg_height = 224

        config_data = {
            'model_type': 'dinomaly',
            'encoder_name': config.get("encoder_name", "dinov2_vit_base_14"),
            'decoder_depth': config.get("decoder_depth", 8),
            'bottleneck_dropout': config.get("bottleneck_dropout", 0.2),
            'train_images': len(group_annotations),
            'threshold': threshold,
            'normalize_brightness': config.get("normalize_brightness", False),
            'normalize_contrast': config.get("normalize_contrast", False),
            'target_size': target_size,
            'train_mode': config.get("train_mode", "by_pos_id"),
            'category': config.get("category", ""),
            'category_label': config.get("category_label", ""),
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        threshold_data = {
            'threshold': threshold,
            'method': 'anomalib_auto',
        }
        with open(threshold_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_data, f, ensure_ascii=False, indent=2)

        self._add_log(task_id, f"Config saved to: {config_path}")

    def _cleanup_training_resources(self, model, engine: Engine = None, datamodule=None):
        """清理训练资源 - 彻底释放显存"""
        try:
            # 删除 datamodule
            if datamodule is not None:
                del datamodule

            # 删除 engine 和 trainer
            if engine is not None:
                if hasattr(engine, 'trainer') and engine.trainer is not None:
                    # 清理 trainer 中的模型引用
                    if hasattr(engine.trainer, 'model'):
                        engine.trainer.model = None
                    if hasattr(engine.trainer, 'lightning_module'):
                        engine.trainer.lightning_module = None
                del engine

            # 删除模型及其组件
            if model is not None:
                # 清理 Dinomaly 模型的各个组件
                if hasattr(model, 'model'):
                    inner_model = model.model
                    if hasattr(inner_model, 'encoder'):
                        del inner_model.encoder
                    if hasattr(inner_model, 'decoder'):
                        del inner_model.decoder
                    if hasattr(inner_model, 'bottleneck'):
                        del inner_model.bottleneck
                    del inner_model
                if hasattr(model, 'memory_bank'):
                    del model.memory_bank
                if hasattr(model, 'feature_extractor'):
                    del model.feature_extractor
                if hasattr(model, 'feature_pooler'):
                    del model.feature_pooler
                del model

            # 强制垃圾回收
            import gc
            gc.collect()

            # 清理 CUDA 显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # 重置峰值显存统计
                torch.cuda.reset_peak_memory_stats()
                logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    # ==================== 任务管理接口 ====================

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
