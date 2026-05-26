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
# 设置 anomalib 预训练模型缓存目录为本地路径
os.environ["ANOMALIB_CACHE_DIR"] = str(PRETRAINED_DIR)
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


try:
    from anomalib.models.components.dinov2.dinov2_loader import DinoV2Loader as _DinoV2Loader
except ImportError:
    from anomalib.models.image.dinomaly.components.dinov2_loader import DinoV2Loader as _DinoV2Loader
_original_dinov2loader_init = _DinoV2Loader.__init__
def _patched_dinov2loader_init(self, cache_dir=None, vit_factory=None):
    if cache_dir is None:
        cache_dir = str(PRETRAINED_DIR / "dinov2")
    _original_dinov2loader_init(self, cache_dir, vit_factory)
_DinoV2Loader.__init__ = _patched_dinov2loader_init

# PyTorch Lightning Callback
from pytorch_lightning.callbacks import Callback, EarlyStopping

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

    def _create_yolo_only_training_task(
        self,
        dataset_dir: str,
        config: dict,
        path_id: str,
        base_model_dir: str,
    ) -> str:
        """创建纯 YOLO 训练任务（无 FP，仅 FN 图像）。复制基础 Dinomaly 模型 + 训练 YOLO 检测器。"""
        model_name = config.get("model_name", "Dinomaly")
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
            self._add_log(task_id, "YOLO training cancelled")
            self._update_task_status(task_id, status="cancelled")
        except Exception as e:
            logger.exception(f"YOLO training failed for {task_id}")
            self._update_task_status(task_id, status="failed", error=str(e))
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

    def _prepare_fp_combined_dataset(
        self,
        task_id: str,
        base_roi_dir: Path,
        fp_group_annotations: List[Dict],
        dataset_dir: str,
        output_dir: Path,
        mask_output_dir: Path,
        num_fp_augmentations: int = 30,
        normalize_brightness_flag: bool = False,
        normalize_contrast_flag: bool = False,
        augmentation_config: Optional[str] = None,
        fp_image_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        合并基础模型 ROI 图 + 增强后的 FP 图。

        策略：
          - 原始 ROI 图：直接复制，不做额外增强
          - FP 图：提取 ROI 后做 num_fp_augmentations 张增强（含原图本身）
          - 如果 fp_group_annotations 为空，可从 fp_image_dir 读取预存的 FP 图片

        Returns:
            合并后所有图片的路径列表
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        combined_paths: List[Path] = []

        base_images = sorted(
            list(base_roi_dir.glob("base_*.png")) + list(base_roi_dir.glob("base_*.jpg"))
        )
        if not base_images:
            # 回退：如果没有 base_ 前缀文件（初次训练的原始数据），读取所有文件
            base_images = sorted(
                list(base_roi_dir.glob("*.png")) + list(base_roi_dir.glob("*.jpg"))
            )
        if not base_images:
            self._add_log(task_id, f"WARNING: base_roi_dir is empty: {base_roi_dir}")

        for img_path in base_images:
            dst = output_dir / f"base_{img_path.name}"
            if not dst.exists():
                shutil.copy2(img_path, dst)
            combined_paths.append(dst)

            base_mask_dir = base_roi_dir.parent.parent / "masks" / base_roi_dir.name
            mask_src = base_mask_dir / (img_path.stem + ".npy")
            if mask_src.exists():
                mask_dst = mask_output_dir / f"base_{img_path.stem}.npy"
                if not mask_dst.exists():
                    shutil.copy2(mask_src, mask_dst)

        self._add_log(task_id, f"Copied {len(base_images)} base ROI images from {base_roi_dir}")

        fp_image_paths: List[Path] = []
        if not fp_group_annotations and fp_image_dir and fp_image_dir.exists():
            fp_image_paths = sorted(
                list(fp_image_dir.glob("fp_*.jpg")) + list(fp_image_dir.glob("fp_*.png"))
            )
            if fp_image_paths:
                self._add_log(task_id, f"Found {len(fp_image_paths)} pre-existing FP images in {fp_image_dir}")

        if not fp_group_annotations and not fp_image_paths:
            self._add_log(task_id, "No FP annotations or pre-existing FP images found, skipping FP augmentation")
            return combined_paths

        augment_transform = load_augmentation_transform(augmentation_config)
        if augment_transform is None:
            self._add_log(task_id, "WARNING: No augmentation transform loaded, FP images will be duplicated without augmentation")

        fp_original_count = 0
        fp_aug_count = 0

        dataset_path = Path(dataset_dir)
        raw_images_dir = dataset_path / "raw_images"
        json_path = dataset_path / "annotations.json"

        image_map: Dict[int, str] = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            image_map = {img["id"]: img["file_name"] for img in data.get("images", [])}

        if fp_group_annotations:
            for ann in fp_group_annotations:
                image_id = ann["image_id"]
                ann_id = ann["id"]
                file_name = image_map.get(image_id)

                if not file_name:
                    self._add_log(task_id, f"WARNING: image_id={image_id} not found, skipping")
                    continue

                image_path = raw_images_dir / file_name
                if not image_path.exists():
                    self._add_log(task_id, f"WARNING: FP image not found: {image_path}")
                    continue

                segmentation = ann.get("segmentation", [])
                if not segmentation:
                    self._add_log(task_id, f"WARNING: annotation {ann_id} has no segmentation, skipping")
                    continue

                try:
                    roi_image, roi_mask = extract_polygon_region(image_path, segmentation)

                    if normalize_brightness_flag:
                        roi_image = normalize_brightness(roi_image)
                    if normalize_contrast_flag:
                        roi_image = normalize_contrast(roi_image)

                    file_stem = Path(file_name).stem

                    orig_name = f"fp_{file_stem}_ann{ann_id}.png"
                    orig_path = output_dir / orig_name
                    roi_image.save(orig_path)
                    combined_paths.append(orig_path)
                    np.save(mask_output_dir / f"fp_{file_stem}_ann{ann_id}.npy", roi_mask)
                    fp_original_count += 1

                    for i in range(num_fp_augmentations - 1):
                        if augment_transform:
                            aug_image = apply_augmentation(roi_image, augment_transform)
                        else:
                            aug_image = roi_image

                        aug_name = f"fp_{file_stem}_ann{ann_id}_aug{i:04d}.png"
                        aug_path = output_dir / aug_name
                        aug_image.save(aug_path)
                        combined_paths.append(aug_path)
                        np.save(mask_output_dir / f"fp_{file_stem}_ann{ann_id}_aug{i:04d}.npy", roi_mask)
                        fp_aug_count += 1

                except Exception as e:
                    self._add_log(task_id, f"ERROR: Failed to process FP annotation {ann_id}: {e}")
        else:
            for img_path in fp_image_paths:
                try:
                    roi_image = Image.open(img_path).convert("RGB")
                    roi_image, roi_mask = letterbox_resize(roi_image, (224, 224))

                    if normalize_brightness_flag:
                        roi_image = normalize_brightness(roi_image)
                    if normalize_contrast_flag:
                        roi_image = normalize_contrast(roi_image)

                    file_stem = img_path.stem

                    orig_name = f"{file_stem}.png"
                    orig_path = output_dir / orig_name
                    roi_image.save(orig_path)
                    combined_paths.append(orig_path)
                    np.save(mask_output_dir / f"{file_stem}.npy", roi_mask)
                    fp_original_count += 1

                    for i in range(num_fp_augmentations - 1):
                        if augment_transform:
                            aug_image = apply_augmentation(roi_image, augment_transform)
                        else:
                            aug_image = roi_image

                        aug_name = f"{file_stem}_aug{i:04d}.png"
                        aug_path = output_dir / aug_name
                        aug_image.save(aug_path)
                        combined_paths.append(aug_path)
                        np.save(mask_output_dir / f"{file_stem}_aug{i:04d}.npy", roi_mask)
                        fp_aug_count += 1

                except Exception as e:
                    self._add_log(task_id, f"ERROR: Failed to process FP image {img_path.name}: {e}")

        self._add_log(
            task_id,
            f"FP augmentation done: {fp_original_count} FP × {num_fp_augmentations} = "
            f"{fp_original_count + fp_aug_count} FP images"
        )
        self._add_log(
            task_id,
            f"Combined dataset total: {len(base_images)} base + "
            f"{fp_original_count + fp_aug_count} FP augmented = {len(combined_paths)} images"
        )
        return combined_paths

    def _load_annotations_for_path_id(
        self,
        task_id: str,
        dataset_dir: str,
        path_id: str,
        train_mode: str = "by_pos_id",
    ) -> List[Dict]:
        """
        从 dataset_dir/annotations.json 中筛选匹配指定 path_id 的标注。

        用于重训时 base_roi_dir 缺失的回退场景：
        根据 train_mode 选择匹配字段（by_category → category_id, by_pos_id → pos_id）。
        """
        annotations_path = Path(dataset_dir) / "annotations.json"
        if not annotations_path.exists():
            self._add_log(task_id, f"annotations.json not found in {dataset_dir}")
            return []

        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._add_log(task_id, f"Failed to read annotations.json: {e}")
            return []

        all_annotations = data.get("annotations", [])
        filtered = []
        for ann in all_annotations:
            if train_mode == "by_category":
                if str(ann.get("category_id", 0)) == str(path_id):
                    filtered.append(ann)
            else:  # by_pos_id or default
                if str(ann.get("pos_id", "")) == str(path_id):
                    filtered.append(ann)

        self._add_log(
            task_id,
            f"Loaded {len(filtered)} annotations for path_id={path_id} "
            f"(train_mode={train_mode}, total={len(all_annotations)})"
        )
        return filtered

    # ==================== Dinomaly 训练 ====================

    def _do_train_dinomaly(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ):
        """执行 Dinomaly 训练（支持从检查点恢复进行微调）"""
        self._add_log(task_id, f"Dinomaly training started for group '{group_id}'")
        self._update_task_status(task_id, status="preparing", progress=5)

        with self._status_lock:
            save_dir = self.training_status[task_id]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        # ✅ 固定随机种子，保证可复现性
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  # benchmark=True 会引入不确定性
        self._add_log(task_id, f"Random seed fixed: {seed}")

        # 配置参数
        encoder_name = config.get("encoder_name", "dinov2_vit_base_14")
        decoder_depth = config.get("decoder_depth", 8)
        bottleneck_dropout = config.get("bottleneck_dropout", 0.2)
        epochs = config.get("epochs", 12)
        batch_size = config.get("batch_size", 2)
        num_workers = config.get("num_workers", 4)
        normalize_brightness = config.get("normalize_brightness", False)
        normalize_contrast = config.get("normalize_contrast", False)
        freeze_encoder = config.get("freeze_encoder", True)
        category = config.get("category", "unknown")
        category_label = config.get("category_label", "unknown")
        augment = config.get("augment", False)
        num_augmentations = config.get("num_augmentations", 1)
        augmentation_config = config.get("augmentation_config", None)
        num_samples = len(group_annotations)

        self._update_task_status(task_id, num_samples=num_samples)
        self._add_log(task_id, f"Training with {num_samples} annotations for group '{group_id}' (label: {category_label})")
        self._add_log(task_id, f"Encoder: {encoder_name}, Decoder depth: {decoder_depth}, Epochs: {epochs}")

        if not DINORMALY_AVAILABLE:
            raise RuntimeError("Dinomaly model is not available")

        # 初始化变量，用于 finally 中清理
        model = None
        engine = None
        datamodule = None
        temp_root = None
        anomalib_temp_dir = None
        fit_thread = None
        predictions = None
        scores = None
        eval_dataloader = None
        val_split_ratio_actual = 0.0  # 将在后面根据样本数动态设置

        try:
            # Stage 1: 准备训练图片
            self._update_task_status(task_id, status="preparing", progress=10, stage="1/3")
            self._add_log(task_id, "Stage 1/3: Preparing ROI images...")

            train_mode = config.get("train_mode", "by_pos_id")
            task_uuid = config.get("task_uuid", "unknown")
            project_id = config.get("project_id", "default")

            path_id = config.get("path_id")
            if path_id is None:
                if group_annotations:
                    first_ann = group_annotations[0]
                    path_id = str(first_ann.get("category_id", 0)) if train_mode == "by_category" \
                              else str(first_ann.get("pos_id", group_id))
                else:
                    path_id = str(group_id)
                self._add_log(task_id, f"Warning: path_id not in config, fallback to {path_id}")

            roi_save_dir = Path(dataset_dir) / "roi" / str(path_id)
            mask_save_dir = Path(dataset_dir) / "masks" / str(path_id)
            roi_save_dir.mkdir(parents=True, exist_ok=True)
            mask_save_dir.mkdir(parents=True, exist_ok=True)

            # 判断是否为 FP 重训模式
            # 触发条件：提供了 base_roi_dir
            base_roi_dir_str = config.get("base_roi_dir")
            is_fp_retrain = bool(base_roi_dir_str)

            if is_fp_retrain:
                base_roi_dir = Path(base_roi_dir_str)
                num_fp_augmentations = config.get("num_fp_augmentations", 30)

                self._add_log(task_id, f"[FP Retrain] mode: base_roi_dir={base_roi_dir}, fp_aug={num_fp_augmentations}x")

                # 检查 base_roi_dir 是否存在（可能因上一轮是纯 FN 重训而没有 roi 目录）
                if not base_roi_dir.exists():
                    self._add_log(task_id, f"[FP Retrain] base_roi_dir not found, falling back to annotation-based extraction")
                    # 回退：从 dataset_dir 的 annotations.json 中提取匹配当前 path_id 的 ROI
                    filtered_anns = self._load_annotations_for_path_id(
                        task_id=task_id,
                        dataset_dir=str(dataset_dir),
                        path_id=str(path_id),
                        train_mode=train_mode,
                    )
                    if filtered_anns:
                        image_paths = extract_roi_images(
                            dataset_dir=str(dataset_dir),
                            output_dir=str(roi_save_dir),
                            mask_output_dir=str(mask_save_dir),
                            group_annotations=filtered_anns,
                            normalize_brightness=normalize_brightness,
                            normalize_contrast=normalize_contrast,
                            augment=False,
                            num_augmentations=1,
                            train_mode=train_mode,
                            augmentation_config=augmentation_config,
                        )
                        n_base = len(filtered_anns)
                        self._add_log(task_id, f"[FP Retrain] Extracted {len(image_paths)} base ROI images from annotations (n_base={n_base})")
                    else:
                        n_base = 0
                        image_paths = []
                        self._add_log(task_id, f"[FP Retrain] WARNING: No annotations found for path_id={path_id}")

                    # 对 FP 图片进行增强（由 retrain 端点预存到 roi_save_dir 的 fp_*.jpg）
                    fp_images_in_dir = list(roi_save_dir.glob("fp_*.jpg")) + list(roi_save_dir.glob("fp_*.png"))
                    n_fp_raw = len(fp_images_in_dir)
                    if n_fp_raw > 0 and n_base > 0:
                        # 增强 FP 图片：使用 _prepare_fp_combined_dataset 但 base_roi_dir 指向
                        # 已提取好的 roi_save_dir，且只处理 fp_ 前缀文件
                        max_fp_aug_total = int(n_base * 0.5)
                        if n_fp_raw * num_fp_augmentations > max_fp_aug_total:
                            safe_aug_per_fp = max(1, max_fp_aug_total // n_fp_raw)
                            self._add_log(task_id, f"[FP Retrain] FP aug capped: {num_fp_augmentations} → {safe_aug_per_fp} per image (base={n_base}, fp_raw={n_fp_raw}, limit={max_fp_aug_total})")
                            num_fp_augmentations = safe_aug_per_fp

                        augment_transform = load_augmentation_transform(augmentation_config)
                        for fp_img_path in fp_images_in_dir:
                            try:
                                roi_image = Image.open(fp_img_path).convert("RGB")
                                roi_image, roi_mask = letterbox_resize(roi_image, (224, 224))
                                file_stem = fp_img_path.stem
                                # 保存原图
                                orig_path = roi_save_dir / f"{file_stem}.png"
                                if not orig_path.exists():
                                    roi_image.save(orig_path)
                                image_paths.append(orig_path)
                                np.save(mask_save_dir / f"{file_stem}.npy", roi_mask)
                                # 生成增强变体
                                for i in range(num_fp_augmentations - 1):
                                    if augment_transform:
                                        aug_image = apply_augmentation(roi_image, augment_transform)
                                    else:
                                        aug_image = roi_image
                                    aug_path = roi_save_dir / f"{file_stem}_aug{i:04d}.png"
                                    aug_image.save(aug_path)
                                    image_paths.append(aug_path)
                                    np.save(mask_save_dir / f"{file_stem}_aug{i:04d}.npy", roi_mask)
                            except Exception as e:
                                self._add_log(task_id, f"ERROR: Failed to augment FP image {fp_img_path.name}: {e}")

                        n_fp_total = n_fp_raw * num_fp_augmentations
                        self._add_log(task_id, f"[FP Retrain] FP augmentation: {n_fp_raw} × {num_fp_augmentations} = {n_fp_total} images")
                    elif n_fp_raw == 0:
                        self._add_log(task_id, f"[FP Retrain] No FP images found in {roi_save_dir}")

                    n_total = len(image_paths)
                    actual_epochs = epochs
                    self._add_log(task_id, f"[FP Retrain] Fallback training with {n_total} images, epochs={actual_epochs}")

                else:
                    # 正常 FP 重训路径：base_roi_dir 存在
                    # 只统计 base_* 前缀文件（排除上次重训混入的 fp_* 文件）
                    n_base = len(list(base_roi_dir.glob("base_*.png")) + list(base_roi_dir.glob("base_*.jpg")))
                    if n_base == 0:
                        # 回退：初次训练数据没有 base_ 前缀
                        n_base = len(list(base_roi_dir.glob("*.png")) + list(base_roi_dir.glob("*.jpg")))
                    n_fp_raw = len(group_annotations)

                    if n_fp_raw == 0:
                        fp_images_in_dir = list(roi_save_dir.glob("fp_*.jpg")) + list(roi_save_dir.glob("fp_*.png"))
                        n_fp_raw = len(fp_images_in_dir)
                        if n_fp_raw > 0:
                            self._add_log(task_id, f"[FP Retrain] Found {n_fp_raw} pre-existing FP images in {roi_save_dir}")

                    if n_fp_raw > 0:
                        max_fp_aug_total = int(n_base * 0.5)
                        if n_fp_raw * num_fp_augmentations > max_fp_aug_total:
                            safe_aug_per_fp = max(1, max_fp_aug_total // n_fp_raw)
                            self._add_log(task_id, f"[FP Retrain] FP aug capped: {num_fp_augmentations} → {safe_aug_per_fp} per image (base={n_base}, fp_raw={n_fp_raw}, limit={max_fp_aug_total})")
                            num_fp_augmentations = safe_aug_per_fp

                    # 1. 先复制基础模型 ROI 到当前 roi 目录
                    # 2. 然后对 FP 图片进行增强，也保存到 roi 目录
                    image_paths = self._prepare_fp_combined_dataset(
                        task_id=task_id,
                        base_roi_dir=base_roi_dir,
                        fp_group_annotations=group_annotations,
                        dataset_dir=dataset_dir,
                        output_dir=roi_save_dir,
                        mask_output_dir=mask_save_dir,
                        num_fp_augmentations=num_fp_augmentations,
                        normalize_brightness_flag=normalize_brightness,
                        normalize_contrast_flag=normalize_contrast,
                        augmentation_config=augmentation_config,
                        fp_image_dir=roi_save_dir,
                    )

                    n_total = len(image_paths)
                    if n_total > n_base * 1.5:
                        actual_epochs = max(8, epochs - 2)
                        self._add_log(task_id, f"[FP Retrain] Large dataset ({n_total} imgs), reducing epochs: {epochs} → {actual_epochs}")
                    else:
                        actual_epochs = epochs

                    self._add_log(task_id, f"[FP Retrain] Training with {len(image_paths)} images, epochs={actual_epochs}")

            else:
                # 初次训练：从 raw_images 提取 ROI
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
                actual_epochs = epochs

            if not image_paths:
                raise ValueError(f"No ROI images for group {group_id}")

            self._add_log(task_id, f"Training directory: {roi_save_dir} ({len(image_paths)} images)")

            # Stage 2: 创建 Anomalib datamodule 和模型
            self._update_task_status(task_id, status="training", progress=30, stage="2/3")
            self._add_log(task_id, "Stage 2/3: Creating model and datamodule...")

            accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
            model = None

            # 从不加载 checkpoint，始终从头训练新模型
            self._add_log(task_id, f"Training new model from scratch: epochs={actual_epochs}")
            if model is None:
                try:
                    model = Dinomaly(
                        encoder_name=encoder_name,
                        decoder_depth=decoder_depth,
                        bottleneck_dropout=bottleneck_dropout,
                    )
                    # 注意：即使是新模型，在重训练场景下也应该使用 finetune_epochs
                    # 因为用户期望的是微调而不是从头训练
                    self._add_log(task_id, f"Training new model (retrain mode): using {actual_epochs} epochs")
                except (OSError, IOError, Exception) as e:
                    if 'dinov2' in str(e).lower() or 'download' in str(e).lower():
                        raise RuntimeError(f"DINOv2 预训练权重下载失败: {e}")
                    raise

            # 冻结编码器并启用 inference_mode 以释放激活值显存
            if freeze_encoder:
                encoder = None
                if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
                    encoder = model.model.encoder
                elif hasattr(model, 'encoder'):
                    encoder = model.encoder

                if encoder is not None:
                    for param in encoder.parameters():
                        param.requires_grad = False
                    # 使用 inference_mode 包裹 encoder forward，释放中间激活值显存
                    _orig_fwd = encoder.forward
                    def _no_grad_fwd(*a, **kw):
                        with torch.inference_mode():
                            return _orig_fwd(*a, **kw)
                    encoder.forward = _no_grad_fwd
                    self._add_log(task_id, "Encoder frozen with inference_mode for faster training and lower memory")

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

            epoch_cb = EpochProgressCallback(self, task_id, actual_epochs)

            # 使用临时目录作为 anomalib 日志目录，训练结束后自动清理
            import tempfile
            anomalib_temp_dir = tempfile.mkdtemp(prefix="anomalib_")

            # 创建 Folder datamodule - 直接使用 roi_save_dir
            # 创建一个临时根目录，roi_save_dir 作为 normal 子目录
            temp_root = Path(save_dir) / "temp_anomalib"
            temp_root.mkdir(parents=True, exist_ok=True)

            # ✅ 修复：线程内强制 num_workers=0，避免 DataLoader fork 死锁
            safe_num_workers = 0
            self._add_log(task_id, f"DataLoader num_workers forced to 0 (threading mode)")

            # 根据样本数量动态决定是否分割验证集
            num_images = len(image_paths)
            min_val_samples = 3  # 验证集最少需要的样本数
            val_split_ratio_actual = 0.2
            check_val_every_n_epoch = actual_epochs  # 默认只在最后验证

            # 计算验证集样本数
            val_samples = int(num_images * val_split_ratio_actual)

            if val_samples < min_val_samples:
                # 样本太少，不分割验证集
                val_split_ratio_actual = 0.0
                val_split_mode = ValSplitMode.SAME_AS_TEST
                check_val_every_n_epoch = epochs
                callbacks = [epoch_cb]
                self._add_log(task_id, f"WARNING: Only {num_images} samples, skipping val split (need >= {int(min_val_samples / 0.2)} images)")
            else:
                # 样本足够，分割 20% 作为验证集
                val_split_mode = ValSplitMode.FROM_TEST
                check_val_every_n_epoch = 1  # 每个 epoch 都验证
                # 添加 Early Stopping（样本数 > 50 时启用）
                # 注意：使用 train_loss 而不是 val_loss，因为验证集可能只有正常样本
                if num_images > 50:
                    early_stop_cb = EarlyStopping(
                        monitor="train_loss",
                        patience=3,  # 从 5 改为 3，encoder 冻结时收敛更快
                        mode="min",
                        verbose=False,
                    )
                    callbacks = [epoch_cb, early_stop_cb]
                    self._add_log(task_id, f"Using val_split_ratio=0.2 ({val_samples} val samples) with EarlyStopping(patience=3, monitor=train_loss)")
                else:
                    callbacks = [epoch_cb]
                    self._add_log(task_id, f"Using val_split_ratio=0.2 ({val_samples} val samples), EarlyStopping disabled (samples <= 50)")

            engine = Engine(
                max_epochs=actual_epochs,
                accelerator=accelerator,
                devices=1,
                enable_progress_bar=False,   # ✅ 关掉 tqdm，避免多线程输出混乱
                enable_model_summary=False,
                check_val_every_n_epoch=check_val_every_n_epoch,
                callbacks=callbacks,        # ✅ 加入 callbacks
                default_root_dir=anomalib_temp_dir,  # ✅ 使用临时目录，不保留日志
                deterministic=False,        # ✅ AMP (BF16) 必须关闭确定性
                precision="bf16-mixed",     # ✅ BF16 混合精度，显存 -35%，速度 +20~30%
                logger=False,               # ✅ 关闭 logger 减少磁盘 I/O
            )

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
                val_split_mode=val_split_mode,
                val_split_ratio=val_split_ratio_actual,
            )
            datamodule.setup()

            # ✅ 验证数据集分割结果
            try:
                train_size = len(datamodule.train_dataloader().dataset)
                val_size = len(datamodule.val_dataloader().dataset)
                test_size = len(datamodule.test_dataloader().dataset)
                self._add_log(task_id, f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
            except Exception as e:
                self._add_log(task_id, f"Warning: Could not log dataset split: {e}")

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

            # 计算阈值（在验证集上推理，如果验证集存在；否则使用测试集）
            self._update_task_status(task_id, progress=80)

            # 优先使用验证集计算阈值（held-out 数据更具代表性）
            if val_split_ratio_actual > 0 and hasattr(datamodule, 'val_dataloader'):
                self._add_log(task_id, "Computing threshold on validation samples (held-out)...")
                eval_dataloader = datamodule.val_dataloader()
                eval_source = "validation"
            else:
                self._add_log(task_id, "Computing threshold on test samples (no validation split)...")
                eval_dataloader = datamodule.test_dataloader()
                eval_source = "test"

            predictions = engine.predict(model=model, dataloaders=eval_dataloader)
            scores = self._extract_dinomaly_scores(predictions)

            # 仅有正常样本时，使用统计方法计算阈值（anomalib 的 F1AdaptiveThreshold 需要正负样本）
            threshold = self._compute_threshold_from_normal_scores(
                task_id=task_id,
                scores=scores,
                method=config.get("threshold_method", "percentile"),
                percentile=config.get("threshold_percentile", 99),
                n_sigma=config.get("threshold_n_sigma", 3.0),
                fallback=self._get_dinomaly_threshold(model),
            )

            if len(scores) > 0:
                self._add_log(task_id, f"Score range ({eval_source}): [{np.min(scores):.4f}, {np.max(scores):.4f}], Threshold: {threshold:.4f}")

            # 保存模型（在清理变量之前，因为需要用到 scores）
            self._update_task_status(task_id, progress=90)
            self._save_dinomaly_model(task_id, save_dir, dataset_dir, model, engine, config, threshold, scores, group_annotations)

            # Stage 4/4: 如果存在 FN 图像，训练 YOLO 检测器
            fn_images_dir = Path(save_dir) / "fn_images"
            if fn_images_dir.exists() and list(fn_images_dir.rglob("fn_*.jpg")):
                # 累积历史 FN 图像（从基础模型目录合并，保留历史批次的学习成果）
                checkpoint_path = config.get("checkpoint_path")
                if checkpoint_path:
                    base_model_dir = Path(checkpoint_path).parent
                    base_fn_dir = base_model_dir / "fn_images"
                    if base_fn_dir.exists():
                        merged = 0
                        for fn_file in base_fn_dir.rglob("fn_*.jpg"):
                            rel_path = fn_file.relative_to(base_fn_dir)
                            dst = fn_images_dir / rel_path
                            if not dst.exists():
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(fn_file), str(dst))
                                merged += 1
                        if merged > 0:
                            self._add_log(task_id, f"[YOLO] Merged {merged} historical FN images from {base_fn_dir}")

                self._update_task_status(task_id, progress=92, stage="YOLO")
                self._add_log(task_id, "Stage 4/4: Preparing YOLO detection dataset...")

                yolo_dataset_dir = Path(save_dir) / "_yolo_dataset"
                normal_roi_dir = Path(dataset_dir) / "roi" / str(path_id)
                count = self._prepare_yolo_dataset(
                    task_id=task_id,
                    normal_roi_dir=str(normal_roi_dir),
                    fn_images_dir=str(fn_images_dir),
                    output_dir=yolo_dataset_dir,
                    config=config,
                )
                if count > 0:
                    self._add_log(task_id, "Stage 4/4: Training YOLO detector on FN images...")
                    self._train_yolo_detector(task_id, yolo_dataset_dir, save_dir, config, progress_range=(93, 98))
                    try:
                        keep = True
                        if keep:
                            self._add_log(task_id, f"[YOLO] Dataset kept: {yolo_dataset_dir}")
                        else:
                            shutil.rmtree(yolo_dataset_dir, ignore_errors=True)
                    except Exception:
                        pass
                    self._add_log(task_id, f"YOLO detector training done ({count} defect samples)")
                else:
                    self._add_log(task_id, "YOLO training skipped (insufficient FN images or missing normal ROI)")
            else:
                self._add_log(task_id, "No FN images found, skipping YOLO training")

            self._update_task_status(
                task_id,
                status="completed",
                progress=100,
                stage="completed",
                end_time=time.time()
            )
            self._persist_state_if_due(force=True)
            self._add_log(task_id, f"Training completed successfully for group '{group_id}'")

        finally:
            # 无论成功或失败，都执行清理
            self._add_log(task_id, "Cleaning up training resources...")

            # 清理推理相关变量
            if predictions is not None:
                del predictions
            if scores is not None:
                del scores
            if eval_dataloader is not None:
                del eval_dataloader

            # 清理线程对象
            if fit_thread is not None:
                del fit_thread

            # 清理临时目录
            if temp_root is not None and temp_root.exists():
                try:
                    shutil.rmtree(temp_root)
                except Exception as e:
                    logger.warning(f"Failed to remove temp_root: {e}")
            
            # 清理 anomalib 临时日志目录
            if anomalib_temp_dir is not None and os.path.exists(anomalib_temp_dir):
                try:
                    shutil.rmtree(anomalib_temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to remove anomalib_temp_dir: {e}")

            # 清理模型和显存
            self._cleanup_training_resources(model, engine, datamodule)
            self._add_log(task_id, "Cleanup completed")

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
        """获取 Dinomaly 阈值（anomalib 自动计算的兜底值）"""
        for attr in ('image_threshold', 'threshold'):
            obj = getattr(model, attr, None)
            if obj is not None:
                val = getattr(obj, 'value', None)
                if val is not None:
                    return float(val)
        return 0.50

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
            threshold = s_max
            self._add_log(task_id, f"Threshold (max normal score): {threshold:.4f}")

        else:
            self._add_log(task_id, f"Unknown method '{method}', using fallback={fallback:.4f}")
            threshold = fallback

        # 合理性检查：阈值不能低于均值（否则一半正常样本都会报警）
        if threshold < s_mean:
            self._add_log(
                task_id,
                f"WARNING: threshold {threshold:.4f} < mean {s_mean:.4f}, "
                f"clamping to mean + 0.5σ"
            )
            threshold = s_mean + 0.5 * s_std

        return threshold

    def _save_dinomaly_model(
        self,
        task_id: str,
        save_dir: str,
        dataset_dir: str,
        model,
        engine: Engine,
        config: dict,
        threshold: float,
        scores: np.ndarray,
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
            'method': config.get("threshold_method", "percentile"),
            'percentile': config.get("threshold_percentile", 99),
            'score_stats': {
                'min': float(np.min(scores)) if len(scores) else None,
                'max': float(np.max(scores)) if len(scores) else None,
                'mean': float(np.mean(scores)) if len(scores) else None,
                'std': float(np.std(scores)) if len(scores) else None,
                'n': len(scores),
            },
        }
        with open(threshold_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_data, f, ensure_ascii=False, indent=2)

        self._add_log(task_id, f"Config saved to: {config_path}")

        # 预计算并保存24角度模板图（用于LightGlue推理加速）
        self._precompute_and_save_template_variants(
            task_id, save_dir, dataset_dir, group_annotations
        )

    def _precompute_and_save_template_variants(
        self,
        task_id: str,
        save_dir: str,
        dataset_dir: str,
        group_annotations: List[Dict]
    ):
        """
        预计算24角度模板图及其SuperPoint特征并保存，用于LightGlue推理加速。
        选择最佳模板图（无翻转、角度最接近0），提取工件主体区域，
        生成24个旋转角度（每15度一个）的模板图变体，并预提取SuperPoint特征。
        """
        try:
            import cv2
            import torch
        except ImportError:
            logger.warning("cv2 or torch not available, skipping template variant precomputation")
            return

        # 尝试导入LightGlue
        try:
            from lightglue import SuperPoint
            from lightglue.utils import rbd
            LIGHTGLUE_AVAILABLE = True
        except ImportError:
            logger.warning("lightglue not available, skipping template variant precomputation")
            return

        # 查找最佳模板图（无翻转、角度最接近0的标注）
        best_ann = None

        for ann in group_annotations:
            # 跳过有翻转的标注
            if ann.get('horizontal_flip', False) or ann.get('vertical_flip', False):
                continue
            # 选择角度最接近0的
            angle = ann.get('angle', 0)
            if best_ann is None or abs(angle) < abs(best_ann.get('angle', 0)):
                best_ann = ann

        if best_ann is None:
            logger.warning("No suitable template annotation found for variant precomputation")
            return

        # 获取图片路径
        image_id = best_ann.get('image_id')
        dataset_path = Path(dataset_dir)
        annotations_path = dataset_path / "annotations.json"

        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            image_map = {img['id']: img['file_name'] for img in data.get('images', [])}
            file_name = image_map.get(image_id)
            if not file_name:
                logger.warning(f"Image file not found for image_id={image_id}")
                return

            raw_images_dir = dataset_path / "raw_images"
            image_path = raw_images_dir / file_name
            if not image_path.exists():
                logger.warning(f"Template image not found: {image_path}")
                return

            # 读取模板图
            template_image = cv2.imread(str(image_path))
            if template_image is None:
                logger.warning(f"Failed to load template image: {image_path}")
                return

            # 查找工件主体标注
            subject_ann = None
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for ann in data.get('annotations', []):
                if ann.get('image_id') == image_id:
                    label = ann.get('label', '')
                    if label == "工件主体":
                        subject_ann = ann
                        break

            # 提取工件主体区域
            if subject_ann:
                segmentation = subject_ann.get('segmentation', [])
                if segmentation and len(segmentation) > 0:
                    coords = segmentation[0]
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))
                    subject_img = template_image[y_min:y_max, x_min:x_max].copy()
                    logger.info(f"Extracted subject region: ({x_min},{y_min},{x_max-x_min},{y_max-y_min})")
                else:
                    bbox = subject_ann.get('bbox', [])
                    if bbox and len(bbox) >= 4:
                        x, y, w, h = [int(v) for v in bbox]
                        subject_img = template_image[y:y+h, x:x+w].copy()
                        logger.info(f"Extracted subject region from bbox: ({x},{y},{w},{h})")
                    else:
                        subject_img = template_image.copy()
                        logger.info("Using full template image as subject")
            else:
                subject_img = template_image.copy()
                logger.info("No subject annotation found, using full template image")

            # 生成24个旋转角度（每15度一个）
            rotations = list(range(0, 360, 30))
            variants_dir = Path(save_dir) / "template_variants"
            variants_dir.mkdir(parents=True, exist_ok=True)

            h, w = subject_img.shape[:2]
            saved_count = 0

            # 加载SuperPoint模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extractor = SuperPoint(max_num_keypoints=8192).eval().to(device)
            logger.info(f"[LightGlue] Loaded SuperPoint for feature extraction (device={device})")

            # 预提取特征
            t_start = time.time()

            for angle in rotations:
                cx, cy = w / 2, h / 2
                M_rot = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
                cos_a = abs(M_rot[0, 0])
                sin_a = abs(M_rot[0, 1])
                canvas_w = int(h * sin_a + w * cos_a)
                canvas_h = int(h * cos_a + w * sin_a)
                M_rot[0, 2] += canvas_w / 2 - cx
                M_rot[1, 2] += canvas_h / 2 - cy

                rotated = cv2.warpAffine(
                    subject_img, M_rot, (canvas_w, canvas_h),
                    flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
                )

                # 提取并保存SuperPoint特征（不再保存模板图，只保存特征）
                rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    feats = extractor.extract(tensor)
                    # 将特征移到CPU并转换为numpy以便保存
                    feats_dict = {
                        'keypoints': feats['keypoints'].cpu().numpy(),
                        'descriptors': feats['descriptors'].cpu().numpy(),
                    }
                    # SuperPoint输出可能包含scores或keypoint_scores
                    if 'scores' in feats:
                        feats_dict['scores'] = feats['scores'].cpu().numpy()
                    elif 'keypoint_scores' in feats:
                        feats_dict['scores'] = feats['keypoint_scores'].cpu().numpy()
                    if 'scales' in feats:
                        feats_dict['scales'] = feats['scales'].cpu().numpy()
                    if 'oris' in feats:
                        feats_dict['oris'] = feats['oris'].cpu().numpy()

                # 保存特征
                feature_path = variants_dir / f"features_{angle:03d}.npz"
                np.savez_compressed(str(feature_path), **feats_dict)

                saved_count += 1

            # 清理SuperPoint模型
            del extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 保存元数据
            metadata = {
                'source_image': str(image_path),
                'source_annotation_id': best_ann.get('id'),
                'rotations': rotations,
                'num_variants': saved_count,
                'original_size': {'width': w, 'height': h},
                'features_extracted': True,
                'feature_type': 'superpoint',
            }
            metadata_path = variants_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - t_start
            self._add_log(task_id, f"Precomputed and saved {saved_count} template variants with features to {variants_dir} (took {elapsed:.2f}s)")

        except Exception as e:
            logger.warning(f"Failed to precompute template variants: {e}")
            import traceback
            traceback.print_exc()

    # ==================== YOLO 检测训练 ====================

    def _sample_normal_images(self, roi_dir: Path, num_samples: int) -> List[Path]:
        """从正常 ROI 目录随机采样指定数量的图像（不增强）。"""
        all_images = list(roi_dir.glob("*.png")) + list(roi_dir.glob("*.jpg"))
        if len(all_images) <= num_samples:
            return all_images
        return random.sample(all_images, num_samples)

    def _augment_fn_images(
        self,
        task_id: str,
        fn_dir: Path,
        output_dir: Path,
        target_total: int = 200,
    ) -> int:
        """对 FN 原图预处理 (letterbox_resize) 后应用轻度增强，使总数接近 target_total。返回最终总数。"""
        fn_originals = sorted(list(fn_dir.rglob("fn_*.jpg")))
        n_orig = len(fn_originals)
        self._add_log(task_id, f"[YOLO] Found {n_orig} original FN images, target={target_total}")

        # 清理旧数据，避免上次运行的残留文件污染数据集
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 先对 FN 原图做预处理（与 Dinomaly 训练时 extract_roi_images 一致），再保存
        input_size = (224, 224)
        preprocessed_originals = []
        for p in fn_originals:
            try:
                img = Image.open(p).convert("RGB")
                # letterbox_resize 保持宽高比，自适应填充色，与训练预处理一致
                img_padded, _ = letterbox_resize(img, input_size)
                # 用相对路径作为文件名，避免不同 pos_id 下同名文件互相覆盖
                rel_name = str(p.relative_to(fn_dir)).replace("/", "_").replace("\\", "_")
                dst = output_dir / rel_name
                img_padded.save(str(dst), quality=95)
                preprocessed_originals.append((dst, img_padded))
            except Exception as e:
                self._add_log(task_id, f"[YOLO] Failed to preprocess FN image {p}: {e}")

        n_valid = len(preprocessed_originals)
        if n_valid == 0:
            self._add_log(task_id, "[YOLO] No valid FN images after preprocessing")
            return 0

        if n_valid >= target_total:
            self._add_log(task_id, f"[YOLO] FN images already >= target ({n_valid} >= {target_total}), skip augmentation")
            return n_valid

        # 计算每张原图需要的增强数量
        aug_per_image = max(1, (target_total - n_valid) // n_valid)
        self._add_log(task_id, f"[YOLO] Augmenting {aug_per_image}x per FN image ({(target_total - n_valid)} needed)")

        fn_aug_config = PROJECT_ROOT / "configs" / "augmentations_fn.yaml"
        transform = load_augmentation_transform(str(fn_aug_config))

        total = n_valid
        for img_path, img_padded in preprocessed_originals:
            for j in range(aug_per_image):
                if total >= target_total:
                    break
                try:
                    aug_img = apply_augmentation(img_padded, transform) if transform else img_padded
                except Exception:
                    aug_img = img_padded
                aug_path = output_dir / f"{img_path.stem}_aug{j:04d}.jpg"
                aug_img.save(str(aug_path), quality=95)
                total += 1

        self._add_log(task_id, f"[YOLO] FN augmentation done: {n_valid} preprocessed + {total - n_valid} augmented = {total}")
        return total

    def _prepare_yolo_dataset(
        self,
        task_id: str,
        normal_roi_dir: str,
        fn_images_dir: str,
        output_dir: Path,
        config: dict,
    ) -> int:
        """
        构建 YOLO 检测二分类数据集（detection 格式）。
        - normal 图 → 空 label（无目标）
        - defect 图 → 整图 bbox（class 0 = defect）
        返回 defect 类样本数；返回 0 表示跳过训练。
        """
        fn_dir = Path(fn_images_dir)
        fn_originals = list(fn_dir.rglob("fn_*.jpg"))
        min_samples = config.get("yolo_min_samples", 1)

        self._add_log(task_id, f"[YOLO] Preparing detection dataset: normal={normal_roi_dir}, fn={fn_images_dir}, min_samples={min_samples}")

        if len(fn_originals) < min_samples:
            self._add_log(task_id, f"[YOLO] Insufficient FN images ({len(fn_originals)} < {min_samples}), skipping")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)

        # YOLO detection 目录结构: train/images/, train/labels/, val/images/, val/labels/
        train_img_dir = output_dir / "train" / "images"
        train_lbl_dir = output_dir / "train" / "labels"
        val_img_dir = output_dir / "val" / "images"
        val_lbl_dir = output_dir / "val" / "labels"
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. defect 类：FN 原图 + 增强（目标 200），整图作为 bbox
        defect_aug_dir = output_dir / "_defect_aug"
        defect_count = self._augment_fn_images(task_id, fn_dir, defect_aug_dir, target_total=200)

        # 2. normal 类：从 Dinomaly 训练数据随机采样（不增强），与 defect 数量匹配
        normal_roi_path = Path(normal_roi_dir)

        # 如果当前 task_uuid 的 roi 目录不存在，向上追溯到父级 train 目录
        if not normal_roi_path.exists():
            train_base = normal_roi_path.parent.parent.parent  # product/{project_id}/train/
            path_id_str = normal_roi_path.name  # path_id
            found_roi_dirs = sorted(train_base.glob(f"*/roi/{path_id_str}"), key=lambda p: p.stat().st_mtime, reverse=True)
            if found_roi_dirs:
                normal_roi_path = found_roi_dirs[0]
                self._add_log(task_id, f"[YOLO] Fallback to ancestor normal ROI: {normal_roi_path}")
            else:
                self._add_log(task_id, f"[YOLO] WARNING: normal_roi_dir not found in current or any ancestor train dir: {normal_roi_dir}")
                return 0

        normal_paths = list(normal_roi_path.glob("*.png")) + list(normal_roi_path.glob("*.jpg"))
        self._add_log(task_id, f"[YOLO] Using all {len(normal_paths)} normal images from {normal_roi_path}")

        # 3. 收集所有图片列表并拆分 train/val (80/20)
        defect_paths = sorted(defect_aug_dir.iterdir())
        all_normal = [p for p in normal_paths]
        all_defect = [p for p in defect_paths if p.suffix.lower() in ('.jpg', '.png')]

        random.shuffle(all_normal)
        random.shuffle(all_defect)

        n_val_normal = max(1, int(len(all_normal) * 0.2))
        n_val_defect = max(1, int(len(all_defect) * 0.2))

        train_normals = all_normal[n_val_normal:]
        val_normals = all_normal[:n_val_normal]
        train_defects = all_defect[n_val_defect:]
        val_defects = all_defect[:n_val_defect]

        # 4. 复制图片并生成 label 文件（normal=空label, defect=整图bbox）
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

        pretrained_path = PROJECT_ROOT / "models" / "pretrained" / "yolov8n.pt"
        if not pretrained_path.exists():
            self._add_log(task_id, f"[YOLO] Local detection model not found at {pretrained_path}, will try auto-download")
            pretrained_path = "yolov8n.pt"

        yolo_epochs =40
        yolo_imgsz = 320
        yaml_path = dataset_dir / "dataset.yaml"
        self._add_log(task_id, f"[YOLO] Starting detection training: epochs={yolo_epochs}, patience=15, imgsz={yolo_imgsz}, lr0=0.01, cos_lr=True, data={yaml_path}")
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
                workers=0,
                lr0=0.01,
                amp=False,
                cos_lr=True,
                seed=42,
                verbose=False,
                exist_ok=True,
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
            runs_dir = Path.cwd() / "runs" / "detect"
            train_dirs = sorted(runs_dir.glob("train*"), key=lambda p: p.stat().st_mtime, reverse=True)
            best_pt = train_dirs[0] / "weights" / "best.pt" if train_dirs else None

        if best_pt and best_pt.exists():
            dst = Path(save_dir) / "yolo_model.pt"
            shutil.copy2(str(best_pt), str(dst))
            self._add_log(task_id, f"[YOLO] Model saved: {dst} ({elapsed:.1f}s)")
        else:
            self._add_log(task_id, "[YOLO] WARNING: best.pt not found after training")

    def _do_train_yolo_only(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        path_id: str,
        base_model_dir: str,
    ):
        """纯 FN 重训：复制基础 Dinomaly 模型 + 训练 YOLO 检测器。"""
        self._add_log(task_id, f"YOLO-only training started for path_id={path_id}")
        self._update_task_status(task_id, status="preparing", progress=10)

        with self._status_lock:
            save_dir = self.training_status[task_id]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        # 1. 复制基础 Dinomaly 模型文件
        self._add_log(task_id, "Stage 1/2: Copying base Dinomaly model...")
        base_dir = Path(base_model_dir)
        copied = []
        for fname in ["model.ckpt", "dinomaly_model.pt", "config.json", "threshold.json"]:
            src = base_dir / fname
            if src.exists():
                dst = Path(save_dir) / fname
                shutil.copy2(str(src), str(dst))
                copied.append(fname)

        # 复制 yolo_model.pt（如果基础模型也有）
        src_yolo = base_dir / "yolo_model.pt"
        if src_yolo.exists():
            shutil.copy2(str(src_yolo), str(Path(save_dir) / "yolo_model.pt"))
            copied.append("yolo_model.pt")

        # 复制 template_variants
        src_variants = base_dir / "template_variants"
        if src_variants.exists() and src_variants.is_dir():
            dst_variants = Path(save_dir) / "template_variants"
            if dst_variants.exists():
                shutil.rmtree(str(dst_variants))
            shutil.copytree(str(src_variants), str(dst_variants))
            self._add_log(task_id, "Copied template_variants")

        self._add_log(task_id, f"Copied base model files: {copied}")

        # 2. 训练 YOLO 检测器
        self._update_task_status(task_id, status="training", progress=30, stage="YOLO")
        self._add_log(task_id, "Stage 2/2: Preparing YOLO detection dataset...")

        fn_images_dir = Path(save_dir) / "fn_images"
        normal_roi_dir = Path(dataset_dir) / "roi" / str(path_id)

        # 累积历史 FN 图像（从基础模型目录合并，保留历史批次的学习成果）
        base_fn_dir = base_dir / "fn_images"
        if base_fn_dir.exists():
            merged = 0
            for fn_file in base_fn_dir.rglob("fn_*.jpg"):
                rel_path = fn_file.relative_to(base_fn_dir)
                dst = fn_images_dir / rel_path
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(fn_file), str(dst))
                    merged += 1
            if merged > 0:
                self._add_log(task_id, f"[YOLO] Merged {merged} historical FN images from {base_fn_dir}")

        if fn_images_dir.exists() and list(fn_images_dir.rglob("fn_*.jpg")):
            yolo_dataset_dir = Path(save_dir) / "_yolo_dataset"
            count = self._prepare_yolo_dataset(
                task_id=task_id,
                normal_roi_dir=str(normal_roi_dir),
                fn_images_dir=str(fn_images_dir),
                output_dir=yolo_dataset_dir,
                config=config,
            )
            if count > 0:
                self._add_log(task_id, "Stage 2/2: Training YOLO detector on FN images...")
                self._train_yolo_detector(task_id, yolo_dataset_dir, save_dir, config, progress_range=(40, 98))
                self._add_log(task_id, f"[YOLO] Dataset kept: {yolo_dataset_dir}")
                self._add_log(task_id, f"YOLO detector training done ({count} defect samples)")
            else:
                self._add_log(task_id, "YOLO training skipped (insufficient FN images or missing normal ROI)")
        else:
            self._add_log(task_id, "No FN images found, skipping YOLO training")

        self._update_task_status(task_id, status="completed", progress=100, stage="completed", end_time=time.time())
        self._persist_state_if_due(force=True)
        self._add_log(task_id, f"YOLO-only training completed for path_id={path_id}")

    def _cleanup_training_resources(self, model, engine: Engine = None, datamodule=None):
        """清理训练资源 - 彻底释放显存"""
        try:
            # 删除 datamodule
            if datamodule is not None:
                del datamodule

            # 删除 engine 和 trainer
            if engine is not None:
                if hasattr(engine, 'trainer') and engine.trainer is not None:
                    # 清理 trainer 中的模型引用（使用try-except避免只读属性错误）
                    try:
                        if hasattr(engine.trainer, 'model'):
                            engine.trainer.model = None
                    except (AttributeError, TypeError):
                        pass
                    try:
                        if hasattr(engine.trainer, 'lightning_module'):
                            engine.trainer.lightning_module = None
                    except (AttributeError, TypeError):
                        pass
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
