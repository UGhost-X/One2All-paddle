#!/usr/bin/env python3
"""
PatchCore模型训练器
为每个pos_id训练一个独立的PatchCore模型
参考: /home/software/One2All-paddle/test/train_patchcore.py
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

from utils.config import get_output_dir
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict

# Albumentations 导入
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("albumentations not installed, will use torchvision transforms for augmentation")

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
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

logger = logging.getLogger(__name__)


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


class PatchCoreDataset(Dataset):
    """PatchCore数据集 - 从目录读取图片

    数据增强策略：
    - by_pos_id 模式：每个 ROI 增强 num_augmentations 次（原逻辑）
    - by_category 模式：如果 ROI 数量 < num_augmentations，则均匀增强到 num_augmentations 总数
    - 使用 albumentations 进行数据增强（通过配置文件）
    """

    # 默认配置文件路径
    DEFAULT_AUGMENTATION_CONFIG = Path(__file__).parent.parent / "configs" / "augmentations.yaml"

    def __init__(
        self,
        image_dir: str,
        transform=None,
        augment: bool = False,
        num_augmentations: int = 1,
        save_images: bool = False,
        save_dir: str = None,
        train_mode: str = "by_pos_id",  # "by_pos_id" | "by_category"
        normalize_brightness: bool = False,
        normalize_contrast: bool = False,
        augmentation_config: Optional[str] = None,  # albumentations 配置文件路径
    ):
        self.image_dir = Path(image_dir)
        # 读取所有png和jpg图片
        self.image_paths = sorted(
            list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        )
        self.transform = transform or self._default_transform()
        self.save_images = save_images
        self.save_dir = Path(save_dir) if save_dir else None
        self.train_mode = train_mode
        self.normalize_brightness = normalize_brightness
        self.normalize_contrast = normalize_contrast
        # 根据训练模式计算数据增强策略
        num_original = len(self.image_paths)

        if train_mode == "by_category" and augment and num_original > 0:
            # by_category 模式：增强到 num_augmentations 总数
            if num_original >= num_augmentations:
                # 原始数量已足够，不增强
                self.augment = False
                self.augmentations_per_image = 0
                self.extra_augmentations = 0
                self.total_samples = num_original
            else:
                # 需要增强到目标数量
                self.augment = True
                num_needed = num_augmentations - num_original
                self.augmentations_per_image = num_needed // num_original
                self.extra_augmentations = num_needed % num_original
                self.total_samples = num_augmentations
        else:
            # by_pos_id 模式或其他：每个 ROI 增强 num_augmentations 次
            self.augment = augment
            self.num_augmentations = num_augmentations if augment else 1
            self.total_samples = num_original * self.num_augmentations

        # 加载 albumentations 数据增强配置
        self._augment_transform = self._load_augmentation_transform(augmentation_config)

        # 如果启用保存图片且有内容需要保存，创建目录
        has_normalization = self.normalize_brightness or self.normalize_contrast
        has_augmentation = self.augment
        if self.save_images and self.save_dir and (has_normalization or has_augmentation):
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if has_normalization:
                (self.save_dir / "original").mkdir(exist_ok=True)
            if has_augmentation:
                (self.save_dir / "augmented").mkdir(exist_ok=True)
            logger.info(f"图片将保存到: {self.save_dir}")
            if train_mode == "by_category":
                logger.info(f"数据增强策略(by_category): 原始={num_original}, 增强={self.augment}, "
                           f"目标总数={self.total_samples}")
            else:
                logger.info(f"数据增强策略(by_pos_id): 原始={num_original}, 增强={self.augment}, "
                           f"每图增强={getattr(self, 'num_augmentations', 1)}, 总数={self.total_samples}")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        num_original = len(self.image_paths)

        if self.train_mode == "by_category":
            # by_category 模式：随机挑选图片进行增强
            if idx < num_original:
                # 原始图片
                img_idx = idx
                is_augmented = False
                aug_idx = 0
            else:
                # 增强图片 - 随机挑选一张原始图片
                is_augmented = True
                aug_idx = idx - num_original  # 增强图片的索引 (0-based)
                img_idx = random.randint(0, num_original - 1)
        else:
            # by_pos_id 模式：原逻辑，每个 ROI 增强 num_augmentations 次
            num_augmentations = getattr(self, 'num_augmentations', 1)
            img_idx = idx // num_augmentations
            aug_idx = idx % num_augmentations
            is_augmented = self.augment and aug_idx > 0

        img_path = self.image_paths[img_idx]
        image = Image.open(img_path).convert('RGB')

        # 仅在需要保存图片且有归一化操作时才复制原始图片
        has_normalization = self.normalize_brightness or self.normalize_contrast
        if self.save_images and self.save_dir and not is_augmented and has_normalization:
            original_image = image.copy()
            save_path = self.save_dir / "original" / f"{img_path.stem}_normalized.png"
            original_image.save(save_path)

        # 数据增强 - 使用预构建的 transform pipeline
        if is_augmented and self.augment:
            image = self._apply_augmentation(image)
            # 保存增强后的图片
            if self.save_images and self.save_dir:
                save_path = self.save_dir / "augmented" / f"{img_path.stem}_aug{aug_idx}.png"
                image.save(save_path)

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)

    def _load_augmentation_transform(self, config_path: Optional[str] = None):
        """加载 albumentations 数据增强配置

        Args:
            config_path: 配置文件路径，如果为 None 则使用默认配置

        Returns:
            albumentations.Compose 或 torchvision.transforms.Compose
        """
        if not self.augment:
            return None

        # 确定配置文件路径
        if config_path is None:
            config_path = self.DEFAULT_AUGMENTATION_CONFIG
        else:
            config_path = Path(config_path)

        # 如果 albumentations 可用且配置文件存在，则使用 albumentations
        if ALBUMENTATIONS_AVAILABLE and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # 从配置构建 albumentations transform
                transform = A.from_dict(config)
                logger.info(f"Loaded albumentations config from {config_path}")
                return transform
            except Exception as e:
                logger.warning(f"Failed to load albumentations config: {e}, falling back to torchvision")

        # 回退到 torchvision transforms
        logger.info("Using torchvision transforms for augmentation")
        return transforms.Compose([
            transforms.RandomRotation(degrees=20, fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """应用数据增强

        Args:
            image: 输入图片 (PIL.Image)

        Returns:
            增强后的图片 (PIL.Image)
        """
        if self._augment_transform is None:
            return image

        # 判断是 albumentations 还是 torchvision
        if ALBUMENTATIONS_AVAILABLE and hasattr(self._augment_transform, 'transforms'):
            # albumentations 需要 numpy array 格式
            image_np = np.array(image)
            augmented = self._augment_transform(image=image_np)
            image = Image.fromarray(augmented['image'])
        else:
            # torchvision transforms 直接处理 PIL Image
            image = self._augment_transform(image)

        return image


def extract_polygon_region(image_path: Path, segmentation: List, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """使用多边形segmentation从图片中提取区域，并缩放到固定大小

    Args:
        image_path: 原始图片路径
        segmentation: 多边形坐标列表
        target_size: 目标尺寸 (width, height)，默认 (224, 224)

    Returns:
        裁剪并缩放后的 ROI 图片
    """
    coords = segmentation[0]  # 取第一个segmentation

    # 计算边界框
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size

        # 确保边界框不超出图片边界
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        # 裁剪出边界框区域
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        if bbox_width <= 0 or bbox_height <= 0:
            # 如果边界框无效，返回空白图片
            return Image.new('RGB', target_size, (0, 0, 0))

        # 裁剪出 tight bounding box
        cropped = img.crop((x_min, y_min, x_max, y_max))

        # 缩放到目标大小
        cropped_resized = cropped.resize(target_size, Image.Resampling.LANCZOS)

        return cropped_resized


def extract_roi_images(
    dataset_dir: str,
    output_dir: str,
    group_annotations: List[Dict],
    normalize_brightness: bool = False,
    normalize_contrast: bool = False,
) -> str:
    """
    从annotations中提取ROI图片并保存到output_dir
    
    Returns:
        提取的ROI图片目录路径
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_images_dir = dataset_dir / "raw_images"

    # 加载annotations.json获取image_id到file_name的映射
    json_path = dataset_dir / "annotations.json"
    image_map = {}
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image_map = {img['id']: img['file_name'] for img in data.get('images', [])}

    logger.info(f"Extracting {len(group_annotations)} ROI images to {output_dir}")

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

        # 获取segmentation
        segmentation = ann.get('segmentation', [])
        if not segmentation:
            logger.warning(f"annotation {ann_id} 没有segmentation数据")
            continue

        try:
            # 提取ROI区域
            roi_image = extract_polygon_region(image_path, segmentation)

            # 亮度归一化
            if normalize_brightness:
                roi_image = normalize_brightness(roi_image)

            # 对比度归一化
            if normalize_contrast:
                roi_image = normalize_contrast(roi_image)

            # 保存ROI图片
            output_filename = f"{Path(file_name).stem}_ann{ann_id}.png"
            output_path = output_dir / output_filename
            roi_image.save(output_path)

        except Exception as e:
            logger.error(f"处理annotation {ann_id}时出错: {e}")

    logger.info(f"ROI extraction completed. Saved to {output_dir}")
    return str(output_dir)


class PatchCoreTrainer:
    """PatchCore训练器 - 参考train_patchcore.py"""

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
                # 在锁保护下复制状态数据
                with self._status_lock:
                    # 移除group_annotations后再保存
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
        """添加日志
        
        Args:
            task_id: 任务ID
            message: 日志消息
            persist: 是否立即持久化状态，默认为False减少I/O开销
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {message}"
        logger.info(f"[Task {task_id}] {message}")
        with self._status_lock:
            if task_id in self.training_status:
                self.training_status[task_id].setdefault("logs", []).append(entry)
        # 仅在需要时触发持久化，减少I/O开销
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
        """批量启动训练任务
        Returns:
            tuple: (task_ids, filtered_group_keys) - 返回任务ID列表和实际训练的分组合并后的keys
        """
        if group_id:
            config = dict(config)
            config["external_group_id"] = group_id

        new_max_concurrent = config.get("max_concurrent", self.max_concurrent)
        if new_max_concurrent != self.max_concurrent:
            # 使用锁保护的活跃任务计数
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
            # 从第一个 annotation 的 label 判断是否需要排除
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

        # 为当前 group 设置 category（从第一个 annotation 的 label 获取）
        config = dict(config)  # 复制 config，避免修改原始配置
        if group_annotations:
            first_label = group_annotations[0].get("label", "unknown")
            config["category"] = first_label

        task_key = self._make_task_key(dataset_dir, config, safe_group_id)

        # 检查是否已存在相同任务（在锁保护下）
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
                # 重启任务
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
                # 创建新任务
                task_id = f"patchcore_{int(time.time())}_{safe_group_id}_{random.randint(1000, 9999)}"
                task_uuid = config.get("task_uuid", "unknown")
                save_dir = os.path.join(
                    self.output_dir,
                    config.get("project_id", "default"),
                    task_uuid,
                    str(safe_group_id),
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

        # 启动训练线程
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
        """训练进程入口"""
        self._semaphore.acquire()
        with self._active_lock:
            self._active_tasks += 1
        try:
            self._do_train_group(task_id, dataset_dir, config, group_id, group_annotations)
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

    def _do_train_group(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ):
        """执行训练 - 参考train_patchcore.py"""
        self._add_log(task_id, f"PatchCore training started for group '{group_id}'")
        self._update_task_status(task_id, status="preparing", progress=5)

        # 在锁保护下读取 save_dir
        with self._status_lock:
            save_dir = self.training_status[task_id]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        # 配置参数
        backbone_name = config.get("backbone", "wide_resnet50_2")
        layers = config.get("layers", ["layer2", "layer3"])
        num_neighbors = int(config.get("num_neighbors", 9))
        normalize_brightness = config.get("normalize_brightness", False)
        normalize_contrast = config.get("normalize_contrast", False)
        augment = config.get("augment", True)
        num_augmentations = config.get("num_augmentations", 1)
        save_images = config.get("save_images", True)
        train_mode = config.get("train_mode", "by_pos_id")

        # 如果 category 是 "工件主体"，跳过数据增强
        category = config.get("category", "unknown")
        if category == "工件主体":
            if augment:
                augment = False
                num_augmentations = 1

        num_samples = len(group_annotations)
        self._update_task_status(task_id, num_samples=num_samples)
        self._add_log(task_id, f"Training with {num_samples} annotations for group '{group_id}'")
        self._add_log(
            task_id,
            f"Backbone: {backbone_name}, Layers: {layers}, Num neighbors: {num_neighbors}"
        )

        self._add_log(
            task_id,
            f"Brightness norm: {normalize_brightness}, Contrast norm: {normalize_contrast}, "
            f"Augment: {augment}, Num augmentations: {num_augmentations}, Train mode: {train_mode}"
        )

        # 检测设备：优先尝试 CUDA，如果失败则回退到 CPU
        device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                # 尝试初始化 CUDA，验证驱动兼容性
                torch.cuda.init()
                device = torch.device("cuda")
                self._add_log(task_id, f"CUDA initialized successfully")
            except RuntimeError as e:
                self._add_log(task_id, f"CUDA initialization failed: {e}")
                self._add_log(task_id, f"Falling back to CPU")
                device = torch.device("cpu")
        self._add_log(task_id, f"Using device: {device}")

        # Stage 1: 提取ROI图片到训练目录
        self._update_task_status(task_id, status="preparing", progress=10, stage="1/4")
        self._add_log(task_id, "Stage 1/4: Extracting ROI images to training directory...")

        # 创建ROI图片保存目录（在训练目录下）
        # 路径: dataset_dir/{category}/{group_id}/
        category = config.get("category", "unknown")
        roi_images_dir = Path(dataset_dir) / str(category) / str(group_id)
        roi_images_dir.mkdir(parents=True, exist_ok=True)

        # 提取ROI图片
        extract_roi_images(
            dataset_dir=dataset_dir,
            output_dir=str(roi_images_dir),
            group_annotations=group_annotations,
            normalize_brightness=normalize_brightness,
            normalize_contrast=normalize_contrast,
        )

        # Stage 2: 创建数据集
        self._update_task_status(task_id, status="training", progress=20, stage="2/4")
        self._add_log(task_id, "Stage 2/4: Creating dataset...")

        # 创建图片保存目录（用于调试，也在训练目录下）
        # 路径: dataset_dir/{category}/{group_id}/training_images/
        training_images_dir = roi_images_dir / "training_images" if save_images else None

        dataset = PatchCoreDataset(
            image_dir=str(roi_images_dir),
            augment=augment,
            num_augmentations=num_augmentations,
            save_images=save_images,
            save_dir=str(training_images_dir) if training_images_dir else None,
            train_mode=train_mode,
            normalize_brightness=normalize_brightness,
            normalize_contrast=normalize_contrast,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )

        self._add_log(task_id, f"Dataset created with {len(dataset)} images")

        # Stage 3: 创建模型并提取特征
        self._update_task_status(task_id, progress=30, stage="3/4")
        self._add_log(task_id, "Stage 3/4: Creating model and extracting features...")

        model = PatchcoreModel(
            layers=layers,
            backbone=backbone_name,
            pre_trained=True,
            num_neighbors=num_neighbors,
        ).to(device)
        model.eval()

        # 收集特征 - 使用分批处理减少内存峰值
        embeddings_list = []
        total = len(dataset)
        batch_accumulate_size = 32  # 每32个embedding合并一次，减少内存碎片

        with torch.no_grad():
            batch_embeddings = []
            for i, (images, _) in enumerate(dataloader):
                if self._is_task_cancelled(task_id):
                    self._add_log(task_id, "Training cancelled by user")
                    raise SystemExit("Task cancelled")

                images = images.to(device)

                # 提取特征
                features = model.feature_extractor(images)
                features = {layer: model.feature_pooler(feature) for layer, feature in features.items()}
                embedding = model.generate_embedding(features)

                # reshape并添加到批次
                embedding = model.reshape_embedding(embedding)
                batch_embeddings.append(embedding)

                # 批量合并，减少内存碎片
                if len(batch_embeddings) >= batch_accumulate_size:
                    embeddings_list.append(torch.cat(batch_embeddings, dim=0))
                    batch_embeddings = []
                    # 清理缓存
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                # 更新进度
                progress = 30 + int((i + 1) / total * 30)
                self._update_task_status(task_id, progress=progress)

            # 处理剩余的embedding
            if batch_embeddings:
                embeddings_list.append(torch.cat(batch_embeddings, dim=0))

        # 最终合并所有embedding
        model.memory_bank = torch.cat(embeddings_list, dim=0)

        self._add_log(task_id, f"Memory bank created with shape: {model.memory_bank.shape}")

        # Stage 4: 计算阈值
        self._update_task_status(task_id, progress=70, stage="4/4")
        self._add_log(task_id, "Stage 4/4: Computing threshold...")

        threshold = self._compute_threshold(model, dataloader, device, num_neighbors, task_id)
        self._add_log(task_id, f"Threshold computed: {threshold:.4f}")

        # 保存模型
        self._update_task_status(task_id, progress=90)
        self._add_log(task_id, "Saving model...")

        model_path = Path(save_dir) / "patchcore_model.pt"
        config_path = Path(save_dir) / "config.json"
        threshold_path = Path(save_dir) / "threshold.json"

        torch.save({
            'model_state_dict': model.state_dict(),
            'memory_bank': model.memory_bank,
            'backbone': backbone_name,
            'layers': layers,
            'num_neighbors': num_neighbors,
        }, model_path)

        # 计算ROI尺寸统计信息
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
                    roi_sizes.append({'width': w, 'height': h, 'max': max(w, h), 'min': min(w, h)})

        if roi_sizes:
            avg_width = sum(s['width'] for s in roi_sizes) / len(roi_sizes)
            avg_height = sum(s['height'] for s in roi_sizes) / len(roi_sizes)
            target_size = (int(avg_width), int(avg_height))
        else:
            target_size = (224, 224)
            avg_width = avg_height = 224

        # 保存配置
        config_data = {
            'backbone': backbone_name,
            'layers': layers,
            'num_neighbors': num_neighbors,
            'train_images': num_samples,
            'augmented_images': len(dataset),
            'embedding_shape': list(model.memory_bank.shape),
            'threshold': threshold,
            'normalize_brightness': normalize_brightness,
            'normalize_contrast': normalize_contrast,
            'augment': augment,
            'num_augmentations': num_augmentations,
            'target_size': target_size,
            'train_mode': config.get("train_mode", "by_pos_id"),
            'roi_size_stats': {
                'avg_width': avg_width,
                'avg_height': avg_height,
            },
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        # 保存阈值
        threshold_data = {
            'threshold': threshold,
            'method': 'mean_plus_3std',
            'description': 'Mean + 3 * standard deviation of anomaly scores on training data',
        }
        with open(threshold_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_data, f, ensure_ascii=False, indent=2)

        self._add_log(task_id, f"Model saved to: {model_path}")
        self._add_log(task_id, f"Config saved to: {config_path}")

        # 更新状态
        self._update_task_status(
            task_id,
            status="completed",
            progress=100,
            stage="completed",
            end_time=time.time()
        )
        self._persist_state_if_due(force=True)

        self._add_log(task_id, f"Training completed successfully for group '{group_id}'")

    def _compute_threshold(
        self,
        model: PatchcoreModel,
        dataloader: DataLoader,
        device: torch.device,
        num_neighbors: int,
        task_id: str
    ) -> float:
        """计算异常阈值（使用统计方法：均值 + 3倍标准差）"""
        model.eval()
        all_scores = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Computing threshold"):
                if self._is_task_cancelled(task_id):
                    raise SystemExit("Task cancelled")

                images = images.to(device)

                # 提取特征
                features = model.feature_extractor(images)
                features = {layer: model.feature_pooler(feature) for layer, feature in features.items()}
                embedding = model.generate_embedding(features)

                # reshape embedding
                batch_size, channels, height, width = embedding.shape
                embedding_reshaped = embedding.permute(0, 2, 3, 1).reshape(-1, channels)

                # 计算与memory bank的距离
                distances = torch.cdist(embedding_reshaped, model.memory_bank)

                # 取k+1个最近邻（包含自身），然后排除第一个（自身距离≈0）
                top_k_plus_1_distances, _ = torch.topk(
                    distances, k=num_neighbors + 1, largest=False, dim=1
                )
                top_k_distances = top_k_plus_1_distances[:, 1:]
                patch_scores = top_k_distances.mean(dim=1)

                # reshape回图像形状
                patch_scores_map = patch_scores.reshape(batch_size, height, width)

                # 上采样到原图大小
                anomaly_map = torch.nn.functional.interpolate(
                    patch_scores_map.unsqueeze(1),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

                # 计算每张图像的异常分数（取异常图的最大值）
                for i in range(batch_size):
                    score = anomaly_map[i].max().item()
                    all_scores.append(score)

        # 使用统计方法计算阈值：均值 + 3倍标准差
        import numpy as np
        scores_array = np.array(all_scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        threshold = mean_score + 3 * std_score

        self._add_log(task_id, f"Threshold stats: mean={mean_score:.4f}, std={std_score:.4f}, threshold={threshold:.4f}")
        self._add_log(task_id, f"Score range: min={np.min(scores_array):.4f}, max={np.max(scores_array):.4f}")

        return float(threshold)

    # ------------------------------------------------------------------ #
    #  任务管理（与原版一致）
    # ------------------------------------------------------------------ #

    def run_training_async(self, dataset_dir: str, config: dict, group_id: str = None):
        """启动训练任务"""
        if group_id:
            config = dict(config)
            config["external_group_id"] = group_id

        raw_data_dir = Path(dataset_dir)
        annotation_file = raw_data_dir / "annotations.json"

        if not annotation_file.exists():
            raise ValueError(f"annotations.json not found: {annotation_file}")

        with open(annotation_file, "r", encoding="utf-8") as f:
            annotations_data = json.load(f)

        annotations = annotations_data.get("annotations", [])
        if not annotations:
            raise ValueError("No annotations found in annotations.json")

        # 排除 label 为 "工件主体" 的 annotations
        excluded_label = "工件主体"
        original_count = len(annotations)
        annotations = [a for a in annotations if a.get("label") != excluded_label]
        filtered_count = len(annotations)
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} annotations with label '{excluded_label}'")

        target_pos_id = config.get("target_pos_id")
        use_pos_id = config.get("use_pos_id", False)

        if target_pos_id is not None and use_pos_id:
            annotations = [a for a in annotations if a.get("pos_id") == target_pos_id]
            if not annotations:
                raise ValueError(f"No annotations found for target_pos_id: {target_pos_id}")
        elif target_pos_id is not None:
            annotations = [a for a in annotations if a.get("label") == target_pos_id]
            if not annotations:
                raise ValueError(f"No annotations found for label: {target_pos_id}")

        group_by = config.get("group_by", "label")
        groups: Dict[Any, List[Dict]] = {}

        for ann in annotations:
            if use_pos_id and ann.get("pos_id") is not None:
                key = ann["pos_id"]
            elif group_by == "roi_id":
                key = ann.get("id")
            elif group_by == "category_id":
                key = ann.get("category_id", "unknown")
            else:
                key = ann.get("label", "unknown")
            groups.setdefault(key, []).append(ann)

        # 获取category（从第一个annotation的label）
        if annotations:
            first_label = annotations[0].get("label", "unknown")
            config["category"] = first_label

        task_ids = self.run_batch_training_async(dataset_dir, config, groups, group_id=None)
        return task_ids[0] if len(task_ids) == 1 else task_ids

    def get_training_status(self, task_id: str) -> dict:
        """获取任务状态（线程安全）"""
        with self._status_lock:
            status = self.training_status.get(task_id, {}).copy()
        status.pop("group_annotations", None)
        return status

    # 别名，保持兼容性
    get_task_status = get_training_status

    def get_task_group_status(self, group_id: str) -> dict:
        """获取任务组状态（线程安全）"""
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
        """停止单个任务（线程安全）"""
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
        """停止整个组的任务（线程安全）"""
        stopped = 0
        with self._status_lock:
            task_ids_to_stop = [
                tid for tid, s in self.training_status.items()
                if (s.get("group_id") == group_id or s.get("task_uuid") == group_id)
                and s.get("status") in {"starting", "training", "preparing"}
            ]
        # 在锁外调用 stop_task，避免死锁
        for tid in task_ids_to_stop:
            if self.stop_task(tid):
                stopped += 1
        return stopped

    def resume_task(self, task_id: str) -> str:
        """恢复任务（线程安全）"""
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
        """列出所有任务（线程安全）"""
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
        """清理任务（线程安全）"""
        with self._status_lock:
            if task_id in self.training_status:
                del self.training_status[task_id]
                self._stop_events.pop(task_id, None)
        self._persist_state_if_due(force=True)
        return True
