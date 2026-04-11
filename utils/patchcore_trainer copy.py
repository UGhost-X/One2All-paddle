#!/usr/bin/env python3
"""
PatchCore 异常检测模型训练脚本 - ROI级别训练
每个标注框（ROI）训练一个独立的模型
使用 PyTorch 进行特征提取

修复说明（v2）：
  1. [致命Bug修复] 移除 ChannelAttention 模块 —— 其 Linear 层随机初始化后立即被冻结，
     等价于对特征乘随机矩阵，严重污染 Memory Bank 特征空间，是误报率上升的主因。
  2. [逻辑Bug修复] 修复 _compute_anomaly_score 中空间一致性判断：
     原代码对 topk 返回的"按值排序的 indices"做 diff，不代表空间距离，逻辑完全错误。
     现改为正确的基于 top-k 统计量的孤立点检测。
  3. [阈值修复] threshold_k 从 1.5 提高到 2.5，percentile 策略整体上移一档，减少误报。
  4. [冗余修复] 灰度分支从 32 通道缩减为 4 通道（32 通道归一化后完全冗余）。
  5. 其余稳定逻辑（固定 224x224、纯黑填充、Chunk 采样）保持不变。
"""

import os
import time
import json
import random
import logging
import traceback
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 多尺度特征提取器（小ROI优化版）
# 优化：针对小ROI(<=64x64)场景，使用浅层特征提取器，避免深层特征丢失空间信息
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleFeatureExtractor(nn.Module):
    """
    从 ResNet 提取多尺度特征并融合。
    
    针对小ROI优化策略：
    - 小ROI (<=64x64): 只使用 layer1 + layer2，避免深层特征图过小
    - 大ROI (>64x64): 使用 layer1 + layer2 + layer3 完整特征
    
    注意：不使用 ChannelAttention —— 未经训练的随机线性层冻结后等价于
    对特征乘随机矩阵，会破坏特征空间的一致性，显著增加误报。
    """

    # 灰度分支通道数：4 通道足以携带亮度信息
    GRAY_CHANNELS = 4
    # 小ROI阈值：小于等于此尺寸使用浅层特征
    SMALL_ROI_THRESHOLD = 64

    def __init__(self, backbone_name: str = "resnet18"):
        super().__init__()

        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.layer1_channels = 256
            self.layer2_channels = 512
            self.layer3_channels = 1024
        else:
            backbone = models.resnet18(pretrained=True)
            self.layer1_channels = 64
            self.layer2_channels = 128
            self.layer3_channels = 256

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # 小ROI模式输出通道数 (layer1 + layer2 + gray)
        self.small_roi_channels = self.layer1_channels + self.layer2_channels + self.GRAY_CHANNELS
        # 完整模式输出通道数 (layer1 + layer2 + layer3 + gray)
        self.full_channels = self.layer1_channels + self.layer2_channels + self.layer3_channels + self.GRAY_CHANNELS
        
        # 默认输出通道数（保持兼容性）
        self.out_channels = self.full_channels

        # 局部平均池化：缓解平移敏感性
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 冻结所有参数（仅用于特征提取，不参与训练）
        for param in self.parameters():
            param.requires_grad = False

    def _get_input_size(self, x: torch.Tensor) -> int:
        """获取输入图像尺寸"""
        return max(x.shape[2], x.shape[3])

    def _is_small_roi(self, x: torch.Tensor) -> bool:
        """判断是否为小ROI场景"""
        return self._get_input_size(x) <= self.SMALL_ROI_THRESHOLD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回拼接后的 patch 特征张量 [B, H*W, C]，已 L2 归一化。
        
        小ROI优化：
        - 输入 <=64x64: 只使用 layer1 + layer2，输出特征图较大(16x16)，保留更多空间信息
        - 输入 >64x64: 使用完整三层特征
        """
        is_small = self._is_small_roi(x)
        
        # 灰度强度分支
        if x.shape[1] == 3:
            gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            # 小ROI使用更小的下采样率，保持更多细节
            if is_small:
                gray_feat = F.interpolate(
                    gray,
                    size=(x.shape[2] // 4, x.shape[3] // 4),  # 对应 layer2 尺寸
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                gray_feat = F.interpolate(
                    gray,
                    size=(x.shape[2] // 8, x.shape[3] // 8),  # 对应 layer2 尺寸
                    mode="bilinear",
                    align_corners=False,
                )
            gray_feat = gray_feat.repeat(1, self.GRAY_CHANNELS, 1, 1)
            gray_feat = F.normalize(gray_feat, dim=1)
        else:
            gray_feat = None

        x_stem = self.stem(x)
        f1 = self.layer1(x_stem)   # [B, C1, H/4,  W/4]
        f2 = self.layer2(f1)       # [B, C2, H/8,  W/8] 或 [B, C2, H/4, W/4] (小ROI)

        if is_small:
            # 小ROI模式：只使用 layer1 + layer2，对齐到 layer2 尺寸
            # 将 f1 上采样到 f2 尺寸
            f1_up = F.interpolate(f1, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            
            # 逐通道 L2 归一化
            f1_up   = F.normalize(f1_up,   dim=1)
            f2_norm = F.normalize(f2,       dim=1)
            
            # 拼接：layer1 + layer2 + 灰度分支
            if gray_feat is not None:
                if gray_feat.shape[-2:] != f2.shape[-2:]:
                    gray_feat = F.interpolate(
                        gray_feat, size=f2.shape[-2:], mode="bilinear", align_corners=False
                    )
                combined = torch.cat([f1_up, f2_norm, gray_feat], dim=1)
            else:
                combined = torch.cat([f1_up, f2_norm], dim=1)
            
            # 更新输出通道数
            self.out_channels = self.small_roi_channels
        else:
            # 完整模式：使用 layer1 + layer2 + layer3
            f3 = self.layer3(f2)       # [B, C3, H/16, W/16]
            
            # 对齐到 f2 的尺寸 (H/8, W/8)
            f1_up = F.interpolate(f1, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)

            # 逐通道 L2 归一化
            f1_up   = F.normalize(f1_up,   dim=1)
            f2_norm = F.normalize(f2,       dim=1)
            f3_up   = F.normalize(f3_up,   dim=1)

            # 拼接：backbone 三层 + 灰度分支
            if gray_feat is not None:
                if gray_feat.shape[-2:] != f2.shape[-2:]:
                    gray_feat = F.interpolate(
                        gray_feat, size=f2.shape[-2:], mode="bilinear", align_corners=False
                    )
                combined = torch.cat([f1_up, f2_norm, f3_up, gray_feat], dim=1)
            else:
                combined = torch.cat([f1_up, f2_norm, f3_up], dim=1)
            
            # 更新输出通道数
            self.out_channels = self.full_channels

        # 局部平均池化（缓解平移敏感）
        combined = self.avg_pool(combined)

        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patches = F.normalize(patches, dim=-1)   # patch 级别 L2 归一化
        return patches


# ─────────────────────────────────────────────────────────────────────────────
# 主训练器
# ─────────────────────────────────────────────────────────────────────────────

class PatchCoreTrainer:
    """
    ROI 级别 PatchCore 训练器。
    每个标注框（ROI）训练一个独立的模型。
    """

    def __init__(self, output_dir: str = "output", max_concurrent: int = None):
        self.output_dir = output_dir
        self.training_status: Dict[str, dict] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self._state_lock = threading.RLock()
        self._last_persist_ts = 0.0
        self.state_file = str(Path(output_dir) / "_roi_patchcore_trainer_state.json")

        # 检测系统环境
        self._detect_system_environment()

        # 计算最优并发量
        if max_concurrent is not None:
            # 用户指定了并发量，使用用户值但给出警告
            self.max_concurrent = max_concurrent
            self._auto_configured = False
        else:
            # 自动计算最优并发量
            self.max_concurrent = self._calculate_optimal_concurrency()
            self._auto_configured = True

        self._semaphore = threading.Semaphore(self.max_concurrent)
        self._active_tasks = 0
        self._active_lock = threading.Lock()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._load_state()

        # 构建详细的配置日志
        config_info = self._build_config_log()
        self._add_log("system", config_info)

    def _detect_system_environment(self):
        """
        检测系统环境（CPU核心数、GPU可用性、显存/内存大小等）
        用于智能计算最优并发量
        """
        # CPU 信息
        try:
            self.cpu_count = os.cpu_count() or 4
        except Exception:
            self.cpu_count = 4

        # 内存信息 (GB)
        try:
            import psutil
            self.total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            self.total_memory_gb = 8  # 默认假设 8GB

        # GPU 信息 (CUDA)
        try:
            self.has_gpu = torch.cuda.is_available()
            self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
            self.gpu_memory_gb = []
            if self.has_gpu:
                for i in range(self.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.gpu_memory_gb.append(props.total_memory / (1024 ** 3))
        except Exception:
            self.has_gpu = False
            self.gpu_count = 0
            self.gpu_memory_gb = []

        # MPS 支持 (Apple Silicon)
        try:
            self.has_mps = torch.backends.mps.is_available()
        except Exception:
            self.has_mps = False

    def _calculate_optimal_concurrency(self) -> int:
        """
        根据硬件环境自动计算最优并发量

        策略：
        1. GPU 环境：基于显存大小计算，每 ~2GB 显存可支持 1 个并发
           - ResNet18 内存库约占用 500MB-1GB
           - ResNet50 内存库约占用 1-2GB
           - 预留 20% 显存缓冲
        2. MPS 环境 (Apple Silicon)：基于统一内存，保守设置
        3. CPU 环境：基于 CPU 核心数和内存，每个任务约需 2GB 内存

        Returns:
            最优并发任务数
        """
        if self.has_gpu and self.gpu_count > 0:
            # GPU 环境：基于最小显存 GPU 计算
            min_gpu_memory = min(self.gpu_memory_gb) if self.gpu_memory_gb else 4
            # 每 2.5GB 显存支持 1 个并发，预留 20% 缓冲
            concurrent_per_gpu = max(1, int(min_gpu_memory / 2.5 * 0.8))
            # 多 GPU 时适当增加，但不超过显存最小的 GPU 的限制
            max_concurrent = min(concurrent_per_gpu * self.gpu_count, concurrent_per_gpu + 4)
            # 上限保护：不超过 16
            return min(max_concurrent, 16)

        elif self.has_mps:
            # Apple Silicon MPS：统一内存，保守估计
            # 假设至少 8GB 统一内存可用
            available_memory = max(self.total_memory_gb - 4, 4)  # 预留 4GB 系统
            # 每任务约 2GB
            return max(2, min(int(available_memory / 2), 6))

        else:
            # CPU 环境：基于核心数和内存
            # 每个任务需要 1-2 个核心 + 2GB 内存
            cpu_based = max(1, self.cpu_count // 2)
            memory_based = max(1, int(self.total_memory_gb / 2.5))
            # 取较小值，但至少 2 个，最多 8 个
            return max(2, min(min(cpu_based, memory_based), 8))

    def _build_config_log(self) -> str:
        """
        构建详细的配置日志信息

        Returns:
            格式化的配置信息字符串
        """
        lines = []
        lines.append("=" * 50)
        lines.append("PatchCore Trainer 配置信息")
        lines.append("=" * 50)

        # 硬件信息
        lines.append(f"CPU: {self.cpu_count} 核心")
        lines.append(f"内存: {self.total_memory_gb:.1f} GB")

        if self.has_gpu:
            lines.append(f"GPU: {self.gpu_count} 个 CUDA 设备")
            for i, mem in enumerate(self.gpu_memory_gb):
                lines.append(f"  - GPU {i}: {mem:.1f} GB 显存")
        elif self.has_mps:
            lines.append("GPU: Apple Silicon MPS (统一内存)")
        else:
            lines.append("GPU: 无 (CPU 模式)")

        # 并发配置
        lines.append("-" * 50)
        if self._auto_configured:
            lines.append(f"并发量: {self.max_concurrent} (自动配置)")
        else:
            lines.append(f"并发量: {self.max_concurrent} (用户指定)")

        # 配置建议
        if self.has_gpu:
            min_mem = min(self.gpu_memory_gb) if self.gpu_memory_gb else 0
            if min_mem < 4:
                lines.append("提示: 显存较小，建议减少并发或使用 ResNet18")
            elif min_mem > 16:
                lines.append("提示: 显存充足，可适当增加并发量")
        elif not self.has_mps:
            if self.total_memory_gb < 8:
                lines.append("提示: 内存较小，建议减少并发量")

        lines.append("=" * 50)
        return "\n".join(lines)

    def _load_state(self):
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
                    logs.append(
                        f"[{time.strftime('%H:%M:%S')}] Service restarted; task marked as interrupted."
                    )
                    if len(logs) > 500:
                        s["logs"] = logs[-500:]
        except Exception as e:
            logger.error(f"Failed to load trainer state: {e}")

    def _persist_state(self):
        data = {
            "version": 1,
            "updated_at": time.time(),
            "training_status": {},
        }
        for tid, status in self.training_status.items():
            status_copy = {k: v for k, v in status.items() if k != "group_annotations"}
            status_copy["_annotation_count"] = len(status.get("group_annotations") or [])
            data["training_status"][tid] = status_copy

        tmp = f"{self.state_file}.tmp"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.error(f"[PersistState] Failed: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)

    def _persist_state_if_due(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_persist_ts < 1.0:
            return
        acquired = self._state_lock.acquire(timeout=5)
        if not acquired:
            return
        try:
            now = time.time()
            if not force and now - self._last_persist_ts < 1.0:
                return
            self._persist_state()
            self._last_persist_ts = time.time()
        except Exception as e:
            logger.error(f"[Persist] Failed to persist state: {e}")
        finally:
            self._state_lock.release()

    def _add_log(self, task_id: str, message: str):
        pos_id = (
            self.training_status[task_id].get("internal_group_id", "")
            if task_id in self.training_status
            else ""
        )
        log_prefix = f"[pos_id:{pos_id}]" if pos_id else ""
        logger.info(f"[{task_id}] {log_prefix} {message}")
        if task_id in self.training_status:
            entry = f"[{time.strftime('%H:%M:%S')}] {log_prefix} {message}"
            logs = self.training_status[task_id].setdefault("logs", [])
            logs.append(entry)
            if len(logs) > 500:
                self.training_status[task_id]["logs"] = logs[-500:]
            self._persist_state_if_due()

    def _make_task_key(self, dataset_dir: str, config: dict, roi_id=None) -> str:
        parts = [
            str(config.get("project_id", "")),
            str(config.get("task_uuid", "")),
            str(config.get("model_name", "")),
        ]
        if roi_id is not None:
            parts.append(f"roi_{roi_id}")
        parts.append(str(dataset_dir))
        return "|".join(parts)

    def run_batch_training_async(
        self,
        dataset_dir: str,
        config: dict,
        groups: Dict[Any, List[Dict]],
        group_id: str = None,
    ) -> List[str]:
        if group_id:
            config = dict(config)
            config["external_group_id"] = group_id

        task_ids: List[str] = []
        for grp_id in sorted(groups.keys(), key=str):
            task_id = self._create_group_training_task(
                dataset_dir, config, grp_id, groups[grp_id]
            )
            task_ids.append(task_id)
        return task_ids

    def run_training_async(
        self, dataset_dir: str, config: dict, group_id: str = None
    ):
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

        task_ids = self.run_batch_training_async(dataset_dir, config, groups, group_id=None)
        return task_ids[0] if len(task_ids) == 1 else task_ids

    def _create_group_training_task(
        self,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ) -> str:
        safe_group_id = str(group_id).replace("/", "_").replace("\\", "_")
        task_key = self._make_task_key(dataset_dir, config, safe_group_id)

        existing_task_id = None
        for tid, s in self.training_status.items():
            if s.get("task_key") == task_key:
                existing_task_id = tid
                break

        if existing_task_id:
            task_id = existing_task_id
            existing = self.training_status[task_id]
            thread = self.threads.get(task_id)
            if existing.get("status") in {"starting", "training"} and thread and thread.is_alive():
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

        t = threading.Thread(
            target=self._train_group_process,
            args=(task_id, dataset_dir, config, group_id, group_annotations),
            daemon=True,
        )
        self.threads[task_id] = t
        t.start()
        return task_id

    def _train_group_process(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ):
        self._semaphore.acquire()
        with self._active_lock:
            self._active_tasks += 1

        try:
            self._do_train_group(task_id, dataset_dir, config, group_id, group_annotations)
        except (SystemExit, KeyboardInterrupt):
            if task_id in self.training_status:
                self.training_status[task_id]["status"] = "cancelled"
            self._persist_state_if_due(force=True)
        except Exception as e:
            self._add_log(task_id, f"Training failed: {e}")
            if task_id in self.training_status:
                self.training_status[task_id].update({
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            self._persist_state_if_due(force=True)
        finally:
            with self._active_lock:
                self._active_tasks -= 1
            self._semaphore.release()

    def _do_train_group(
        self,
        task_id: str,
        dataset_dir: str,
        config: dict,
        group_id: Any,
        group_annotations: List[Dict],
    ):
        self._add_log(task_id, f"PatchCore training started for group '{group_id}'")
        self.training_status[task_id]["status"] = "preparing"
        self.training_status[task_id]["progress"] = 5

        save_dir = self.training_status[task_id]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        roi_images = self._extract_roi_images(task_id, dataset_dir, group_annotations)
        if not roi_images:
            raise ValueError(f"No ROI images extracted for group {group_id}")

        num_samples = len(roi_images)
        self.training_status[task_id]["num_samples"] = num_samples
        self._add_log(task_id, f"Extracted {num_samples} ROI images for training")

        backbone_name = config.get("backbone", "resnet18")
        
        # 使用固定输入尺寸，避免GPU显存碎片化
        # 通过配置控制是否使用小ROI模式，而不是动态计算
        input_size = config.get("input_size", [224, 224])
        # 通过配置或ROI平均尺寸判断是否使用小ROI模式
        avg_roi_size = self._compute_avg_roi_size(roi_images)
        is_small_roi = config.get("force_small_roi_mode", False) or avg_roi_size <= 64
        
        if is_small_roi:
            self._add_log(task_id, f"Small ROI mode: avg_roi_size={avg_roi_size:.1f}, input_size={input_size}")
        else:
            self._add_log(task_id, f"Standard ROI mode: avg_roi_size={avg_roi_size:.1f}, input_size={input_size}")

        coreset_ratio = self._compute_coreset_ratio(config, num_samples, input_size)
        self._add_log(task_id, f"Adaptive coreset_ratio={coreset_ratio:.3f} for {num_samples} samples")

        self.training_status[task_id]["status"] = "training"
        self.training_status[task_id]["progress"] = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_extractor = MultiScaleFeatureExtractor(backbone_name).to(device)
        feature_extractor.eval()

        self.training_status[task_id]["stage"] = "1/3"
        self._add_log(task_id, "Stage 1/3: Extracting multi-scale features from ROI images...")
        features = self._extract_features(task_id, feature_extractor, roi_images, input_size, device)

        if len(features) == 0:
            raise ValueError("No features extracted")

        self.training_status[task_id]["progress"] = 50
        self.training_status[task_id]["stage"] = "2/3"

        self._add_log(task_id, "Stage 2/3: Applying coreset sampling...")
        memory_bank = self._coreset_sampling(task_id, features, coreset_ratio)
        self.training_status[task_id]["progress"] = 70

        memory_bank_path = os.path.join(save_dir, "memory_bank.npz")
        np.savez_compressed(memory_bank_path, memory_bank=memory_bank)
        self.training_status[task_id]["stage"] = "3/3"

        self._add_log(task_id, "Stage 3/3: Calibrating threshold...")
        self.training_status[task_id]["progress"] = 80
        threshold = self._calibrate_threshold(
            task_id, feature_extractor, memory_bank, roi_images, input_size, config, device, is_small_roi=is_small_roi
        )
        self.training_status[task_id]["threshold"] = float(threshold)

        first_ann = group_annotations[0] if group_annotations else {}
        label = first_ann.get("label", "unknown")
        category_id = first_ann.get("category_id", 0)
        bbox = first_ann.get("bbox", [0, 0, 0, 0])

        config_data = {
            "group_id": group_id,
            "category": label,
            "category_id": category_id,
            "bbox": bbox,
            "threshold": threshold,
            "threshold_source": "adaptive_calibration",
            "input_size": input_size,
            "num_samples": num_samples,
            "model_name": "PatchCore",
            "backbone": backbone_name,
            "coreset_ratio": coreset_ratio,
            "memory_bank_shape": list(memory_bank.shape),
            "feature_channels": feature_extractor.out_channels,
            "is_small_roi": is_small_roi,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        self._save_group_training_data(task_id, save_dir, config, group_annotations)

        self.training_status[task_id]["status"] = "completed"
        self.training_status[task_id]["progress"] = 100
        self._add_log(task_id, f"Group '{group_id}' training complete.")
        self._persist_state_if_due(force=True)

    def _compute_avg_roi_size(
        self, roi_images: List[np.ndarray]
    ) -> float:
        """
        计算ROI的平均尺寸。
        
        用于判断是否使用小ROI模式，而不改变输入尺寸。
        
        返回: 平均最大边长
        """
        if not roi_images:
            return 224.0
        
        # 计算所有ROI的平均尺寸
        max_sizes = [max(roi.shape[:2]) for roi in roi_images]
        avg_max_size = sum(max_sizes) / len(max_sizes)
        
        return avg_max_size

    def _compute_coreset_ratio(
        self, config: dict, num_samples: int, input_size: List[int]
    ) -> float:
        if "coreset_ratio" in config:
            return float(config["coreset_ratio"])

        # 小ROI优化：根据实际输入尺寸计算patch数量
        is_small_roi = max(input_size) <= 64
        if is_small_roi:
            # 小ROI使用 layer2 尺寸 (H/4, W/4) 因为不使用 layer3
            h_patch = input_size[1] // 4
            w_patch = input_size[0] // 4
        else:
            h_patch = input_size[1] // 8
            w_patch = input_size[0] // 8
        
        estimated_patches = num_samples * h_patch * w_patch

        if estimated_patches < 5_000:
            return 1.0
        elif estimated_patches < 20_000:
            return 0.8
        elif estimated_patches < 100_000:
            return 0.5
        elif estimated_patches < 500_000:
            return 0.3
        else:
            return 0.1

    def _extract_roi_images(
        self, task_id: str, dataset_dir: str, group_annotations: List[Dict]
    ) -> List[np.ndarray]:
        raw_data_dir = Path(dataset_dir)
        raw_images_dir = raw_data_dir / "raw_images"

        if not raw_images_dir.exists():
            return []

        annotation_file = raw_data_dir / "annotations.json"
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = {img["id"]: img for img in data.get("images", [])}

        roi_images: List[np.ndarray] = []

        for ann in group_annotations:
            image_id = ann.get("image_id")
            bbox = ann.get("bbox", [0, 0, 0, 0])
            angle = ann.get("angle", 0)
            h_flip = ann.get("horizontal_flip", False)
            v_flip = ann.get("vertical_flip", False)

            if angle != 0 or h_flip or v_flip:
                self._add_log(
                    task_id,
                    "WARNING: Augmentation (flip/rotate) used! "
                    "This may destroy structural context for PatchCore.",
                )

            img_info = images.get(image_id)
            if not img_info:
                continue

            img_path = raw_images_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                x, y, w, h = [int(v) for v in bbox]
                img_h, img_w = img.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = max(1, min(w, img_w - x))
                h = max(1, min(h, img_h - y))

                roi = img[y: y + h, x: x + w]
                if roi.size == 0:
                    continue

                roi = self._apply_augmentation(roi, angle, h_flip, v_flip)
                roi_images.append(roi)

            except Exception:
                pass

        return roi_images

    def _apply_augmentation(
        self, img: np.ndarray, angle: float, h_flip: bool, v_flip: bool
    ) -> np.ndarray:
        if h_flip:
            img = cv2.flip(img, 1)
        if v_flip:
            img = cv2.flip(img, 0)
        if angle != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
        return img

    @staticmethod
    def _pad_to_square(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == w:
            return img
        side = max(h, w)
        pad_h = (side - h) // 2
        pad_w = (side - w) // 2
        # 纯黑填充，避免边缘拉伸复制产生条纹噪点
        return cv2.copyMakeBorder(
            img,
            pad_h, side - h - pad_h,
            pad_w, side - w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def _make_transform(self, input_size: List[int]) -> transforms.Compose:
        h, w = input_size[1], input_size[0]
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess_roi(self, roi: np.ndarray, transform) -> torch.Tensor:
        roi = self._pad_to_square(roi)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return transform(roi_rgb)

    def _extract_features(
        self,
        task_id: str,
        feature_extractor: MultiScaleFeatureExtractor,
        roi_images: List[np.ndarray],
        input_size: List[int],
        device: torch.device,
        batch_size: int = None,
    ) -> np.ndarray:
        if batch_size is None:
            batch_size = 32 if device.type == "cuda" else 16

        transform = self._make_transform(input_size)
        all_features: List[np.ndarray] = []
        total = len(roi_images)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_images = roi_images[batch_start:batch_end]

            try:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(8, len(batch_images))) as executor:
                    batch_tensors = list(
                        executor.map(lambda roi: self._preprocess_roi(roi, transform), batch_images)
                    )

                batch_tensor = torch.stack(batch_tensors).to(device)

                with torch.no_grad():
                    patches = feature_extractor(batch_tensor)  # [B, H*W, C]

                all_features.extend(patches.cpu().numpy())

            except Exception:
                for idx in range(batch_start, batch_end):
                    try:
                        tensor = self._preprocess_roi(roi_images[idx], transform)
                        tensor = tensor.unsqueeze(0).to(device)
                        with torch.no_grad():
                            patch = feature_extractor(tensor)[0]
                        all_features.append(patch.cpu().numpy())
                    except Exception:
                        pass

            if batch_start % max(1, total // 10) == 0 or batch_end == total:
                progress = 10 + int(40 * batch_end / total)
                self.training_status[task_id]["progress"] = progress

        if not all_features:
            return np.array([])

        return np.vstack(all_features)

    # ─────────────────────────────────────────────────────────────────────────
    # 核心集采样
    # ─────────────────────────────────────────────────────────────────────────

    _CORESET_K_MAX = int(os.environ.get("CORESET_K_MAX", "3000"))
    _CORESET_PRESAMPLE_N = int(os.environ.get("CORESET_PRESAMPLE_N", "8000"))

    def _coreset_sampling(
        self, task_id: str, features: np.ndarray, ratio: float
    ) -> np.ndarray:
        n_samples = features.shape[0]
        n_coreset_raw = max(1, int(n_samples * ratio))
        n_coreset = min(n_coreset_raw, self._CORESET_K_MAX)

        if n_coreset >= n_samples:
            return features

        presample_n = self._CORESET_PRESAMPLE_N
        if n_samples > presample_n * 2:
            presample_n = min(presample_n, n_samples // 10)

        if n_samples > presample_n:
            rng = np.random.default_rng(seed=42)
            pre_idx = rng.choice(n_samples, size=presample_n, replace=False)
            pre_idx.sort()
            pool = features[pre_idx]
        else:
            pool = features

        n_coreset = min(n_coreset, len(pool))

        device = torch.device("cpu")
        chunk_mb   = int(os.environ.get("CORESET_CHUNK_MB", "512"))
        C          = pool.shape[1]
        chunk_size = max(8192, (chunk_mb * 1024 * 1024) // (C * 4))

        selected_local: List[int] = [0]
        anchor = torch.from_numpy(pool[0:1]).float()
        min_distances = self._chunked_dist_to_anchor(pool, anchor, chunk_size, device)

        for i in range(1, n_coreset):
            next_local = int(np.argmax(min_distances))
            selected_local.append(next_local)

            new_anchor = torch.from_numpy(pool[next_local: next_local + 1]).float()
            new_dists  = self._chunked_dist_to_anchor(pool, new_anchor, chunk_size, device)
            np.minimum(min_distances, new_dists, out=min_distances)

            if i % max(1, n_coreset // 10) == 0 or i == n_coreset - 1:
                self.training_status[task_id]["progress"] = 50 + int(20 * i / n_coreset)

        return pool[selected_local]

    @staticmethod
    def _chunked_dist_to_anchor(
        features_np: np.ndarray,
        anchor: torch.Tensor,
        chunk_size: int,
        device: torch.device,
    ) -> np.ndarray:
        N = features_np.shape[0]
        result = np.empty(N, dtype=np.float32)
        for start in range(0, N, chunk_size):
            end   = min(start + chunk_size, N)
            chunk = torch.from_numpy(features_np[start:end]).float().to(device)
            dists = torch.cdist(chunk, anchor).squeeze(1)
            result[start:end] = dists.cpu().numpy()
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 阈值校准
    # 修复：threshold_k 提高到 2.5（原 1.5 过于激进）
    # 修复：percentile 策略整体上移一档，减少误报
    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate_threshold(
        self,
        task_id: str,
        feature_extractor: MultiScaleFeatureExtractor,
        memory_bank: np.ndarray,
        roi_images: List[np.ndarray],
        input_size: List[int],
        config: dict,
        device: torch.device,
        batch_size: int = None,
        is_small_roi: bool = False,
    ) -> float:
        if batch_size is None:
            batch_size = 32 if device.type == "cuda" else 16

        transform = self._make_transform(input_size)

        max_samples = min(len(roi_images), 200)
        if len(roi_images) > max_samples:
            step = len(roi_images) / max_samples
            calib_images = [roi_images[int(i * step)] for i in range(max_samples)]
        else:
            calib_images = roi_images

        scores: List[float] = []
        memory_tensor = torch.from_numpy(memory_bank).float().to(device)
        memory_tensor = F.normalize(memory_tensor, dim=-1)
        total_calib = len(calib_images)

        for batch_start in range(0, total_calib, batch_size):
            batch_end = min(batch_start + batch_size, total_calib)
            batch_images = calib_images[batch_start:batch_end]

            try:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(8, len(batch_images))) as executor:
                    batch_tensors = list(
                        executor.map(lambda roi: self._preprocess_roi(roi, transform), batch_images)
                    )

                batch_tensor = torch.stack(batch_tensors).to(device)

                with torch.no_grad():
                    patches = feature_extractor(batch_tensor)  # [B, H*W, C]
                    # 小ROI优化：传递 is_small_roi 参数
                    batch_scores = self._compute_anomaly_scores_batch(patches, memory_tensor, is_small_roi)
                    scores.extend(batch_scores.cpu().numpy().tolist())

            except Exception:
                for idx in range(batch_start, batch_end):
                    try:
                        tensor = self._preprocess_roi(calib_images[idx], transform)
                        tensor = tensor.unsqueeze(0).to(device)
                        with torch.no_grad():
                            patch = feature_extractor(tensor)[0]
                        # 小ROI优化：传递 is_small_roi 参数
                        score = self._compute_anomaly_score(patch, memory_tensor, is_small_roi)
                        scores.append(score)
                    except Exception:
                        pass

        if not scores:
            return 0.5

        arr = np.array(scores, dtype=np.float32)

        mean_score   = float(arr.mean())
        std_score    = float(arr.std())
        median_score = float(np.median(arr))

        # 小ROI优化：调整阈值计算参数
        if is_small_roi:
            # 小ROI使用更保守的阈值参数
            threshold_k   = float(config.get("threshold_k", 2.5))
            min_margin    = float(config.get("threshold_min_margin", 0.12))
        else:
            # 标准ROI参数
            threshold_k   = float(config.get("threshold_k", 2.5))
            min_margin    = float(config.get("threshold_min_margin", 0.15))

        margin             = max(threshold_k * std_score, min_margin)
        threshold_adaptive = median_score + margin

        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        p97 = float(np.percentile(arr, 97))
        p99 = float(np.percentile(arr, 99))

        # 小ROI优化：调整percentile策略
        if is_small_roi:
            # 小ROI使用更严格的阈值策略
            if std_score < 0.03:
                # 分布非常集中：使用 p99
                threshold = max(threshold_adaptive, p99)
            elif std_score > 0.15:
                # 分布分散：使用 p95
                threshold = max(threshold_adaptive, p95)
            else:
                # 正常分布：使用 p97
                threshold = max(threshold_adaptive, p97)
        else:
            # 标准ROI策略
            if std_score < 0.05:
                threshold = max(threshold_adaptive, p99)
            elif std_score > 0.2:
                threshold = max(threshold_adaptive, p95)
            else:
                threshold = max(threshold_adaptive, p97)

        if config.get("force_percentile"):
            pct       = int(config.get("threshold_percentile", 95))
            threshold = float(np.percentile(arr, pct))

        self._add_log(
            task_id,
            f"Calibration: n={len(scores)}, mean={mean_score:.4f}, std={std_score:.4f}, "
            f"median={median_score:.4f}, adaptive={threshold_adaptive:.4f}, "
            f"p95={p95:.4f}, p97={p97:.4f}, p99={p99:.4f} → threshold={threshold:.6f}",
        )
        return threshold

    # ─────────────────────────────────────────────────────────────────────────
    # 异常得分计算
    # 修复：空间一致性判断逻辑错误
    #   原代码对 topk 返回的"按值大小排序的 indices"做 diff，这些 indices 不代表
    #   空间位置，diff 结果毫无意义，导致 spatial_consistency 判断完全随机。
    #   修复后：直接使用 top-k 统计量做孤立点判断，逻辑清晰可靠。
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_anomaly_score(
        patch_features: torch.Tensor, memory_tensor: torch.Tensor, is_small_roi: bool = False
    ) -> float:
        """
        异常得分计算：基于 top-k 统计量的孤立点感知评分。
        
        小ROI优化：
        - 小ROI特征图较大(16x16=256 patches)，有更多空间信息可利用
        - 调整噪声过滤阈值，适应小ROI的特征分布
        - 优化孤立点检测策略

        核心思路：
        - 真实缺陷区域：多个相邻 patch 距离都偏高，top1 与 top3 均值相近
        - 噪声/死像素：仅 top1 极高，top2 之后迅速下降（孤立点）
        - 通过 isolation = top1 - top2 判断孤立程度，对噪声降权
        """
        similarity = torch.matmul(patch_features, memory_tensor.T)
        distances  = 1.0 - torch.max(similarity, dim=-1)[0]  # [H*W]

        n = distances.shape[0]
        if n < 4:
            return float(distances.max().item())

        # 取 top-k 最高距离（按值排序，非空间排序）
        top_k = min(9, n)
        sorted_dists, _ = torch.sort(distances, descending=True)
        top1      = sorted_dists[0].item()
        top3_mean = sorted_dists[:min(3, n)].mean().item()
        top5_mean = sorted_dists[:min(5, n)].mean().item()
        mean_dist = distances.mean().item()

        # 孤立点指标：top1 与 top2 的差距
        isolation = top1 - sorted_dists[1].item() if n > 1 else 0.0

        # 小ROI优化：调整噪声过滤阈值
        if is_small_roi:
            # 小ROI使用更宽松的噪声过滤阈值
            if top1 < 0.05:
                return float(top1 * 0.2)
            
            # 小ROI优化：使用更多patch的统计信息，更稳健
            if isolation > 0.06:
                # 强孤立点（典型噪声）：大幅降低 top1 权重，增加top5_mean权重
                score = top3_mean * 0.4 + top5_mean * 0.3 + mean_dist * 0.3
            elif isolation > 0.03:
                # 轻度孤立：适度降低 top1 权重
                score = top1 * 0.25 + top3_mean * 0.35 + top5_mean * 0.25 + mean_dist * 0.15
            else:
                # 非孤立（多点高距离，倾向真实缺陷）：保留较高的 top1 权重
                score = top1 * 0.45 + top3_mean * 0.30 + top5_mean * 0.15 + mean_dist * 0.10
        else:
            # 标准ROI模式（原逻辑）
            # 噪声过滤：距离很小时直接返回小值，避免放大微弱信号
            if top1 < 0.08:
                return float(top1 * 0.3)

            # 基于孤立程度的加权融合
            if isolation > 0.08:
                # 强孤立点（典型噪声）：大幅降低 top1 权重
                score = top3_mean * 0.5 + mean_dist * 0.5
            elif isolation > 0.04:
                # 轻度孤立：适度降低 top1 权重
                score = top1 * 0.3 + top3_mean * 0.4 + mean_dist * 0.3
            else:
                # 非孤立（多点高距离，倾向真实缺陷）：保留较高的 top1 权重
                score = top1 * 0.5 + top3_mean * 0.35 + mean_dist * 0.15

        return float(score)

    @staticmethod
    def _compute_anomaly_scores_batch(
        patch_features_batch: torch.Tensor, memory_tensor: torch.Tensor, is_small_roi: bool = False
    ) -> torch.Tensor:
        """批量计算异常得分"""
        B = patch_features_batch.shape[0]
        scores = torch.zeros(B, device=patch_features_batch.device)
        for i in range(B):
            scores[i] = PatchCoreTrainer._compute_anomaly_score(
                patch_features_batch[i], memory_tensor, is_small_roi
            )
        return scores

    def _save_group_training_data(
        self, task_id: str, save_dir: str, config: dict, group_annotations: List[Dict]
    ):
        try:
            td = os.path.join(save_dir, "training_data")
            os.makedirs(td, exist_ok=True)
            group_data = {
                "group_id": group_annotations[0].get("label") if group_annotations else None,
                "annotations": group_annotations,
                "num_samples": len(group_annotations),
            }
            with open(os.path.join(td, "group_annotations.json"), "w", encoding="utf-8") as f:
                json.dump(group_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_training_status(self, task_id: str = None):
        if task_id:
            return self.training_status.get(task_id)
        return self.training_status

    def get_status(self, task_id: str) -> dict:
        status = self.training_status.get(task_id)
        if not status:
            return {"status": "not_found", "message": "Task not found"}
        return status

    def get_group_status(self, group_id: str) -> dict:
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

        progresses  = [t.get("progress", 0) for t in group_tasks]
        avg_progress = sum(progresses) / len(progresses) if progresses else 0

        return {
            "status": overall,
            "progress": avg_progress,
            "group_id": group_id,
            "tasks": group_tasks,
            "total_tasks": len(group_tasks),
            "completed_tasks": sum(1 for s in statuses if s == "completed"),
        }


if __name__ == "__main__":
    trainer = PatchCoreTrainer(output_dir="test_output")
    print("ROI PatchCore Trainer initialized")
    print(f"  ResNet18 extractor channels: {MultiScaleFeatureExtractor('resnet18').out_channels}")
    print(f"  ResNet50 extractor channels: {MultiScaleFeatureExtractor('resnet50').out_channels}")