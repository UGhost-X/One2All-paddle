#!/usr/bin/env python3
"""
Dinomaly 异常原型库管理模块
支持从异常样本（用户反馈的 false negatives）提取特征构建原型库，
推理时利用异常原型进行分数修正，提升对已知异常模式的检出率。
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


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


class DinomalyFeatureExtractor:
    """从 Dinomaly 模型提取特征向量
    
    使用与推理时相同的预处理：224x224 + letterbox_resize 保持宽高比
    """

    def __init__(self, model, device: str = "cuda", input_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model.eval()
        # 与推理时一致的标准化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """单张图片预处理：保持与推理时一致"""
        # letterbox_resize 保持宽高比到 224x224
        img_padded, _ = letterbox_resize(image, self.input_size)
        # 转为 numpy -> tensor
        roi_np = np.array(img_padded)
        if CV2_AVAILABLE:
            import cv2
            roi_resized = cv2.resize(roi_np, self.input_size, interpolation=cv2.INTER_LINEAR)
        else:
            # 使用 PIL 作为 fallback
            roi_resized = np.array(Image.fromarray(roi_np).resize(self.input_size, Image.BILINEAR))
        roi_normalized = roi_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(roi_normalized).permute(2, 0, 1)
        tensor = self.normalize(tensor)
        return tensor

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """
        提取图片的 encoder 融合特征

        Returns:
            feature: (768,) 特征向量，已 L2 归一化
        """
        tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)

        # 获取 encoder 融合特征 (en)
        en, _ = self.model.model.get_encoder_decoder_outputs(tensor)

        # 两层融合
        feat = (en[0] + en[1]) / 2  # (B, 768, 28, 28)

        # 全局平均池化
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 768)

        # L2 归一化
        feat = F.normalize(feat, p=2, dim=1)

        return feat.squeeze(0)  # (768,)

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """批量提取特征"""
        tensors = torch.stack([self._preprocess_image(img) for img in images]).to(self.device)

        en, _ = self.model.model.get_encoder_decoder_outputs(tensors)
        feat = (en[0] + en[1]) / 2
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
        feat = F.normalize(feat, p=2, dim=1)

        return feat  # (N, 768)


class PrototypeBank:
    """原型库管理：保存和加载异常原型"""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.anomaly_prototypes: Optional[torch.Tensor] = None
        self.metadata: Dict[str, Any] = {}

    def load(self) -> bool:
        """从磁盘加载原型库"""
        anomaly_path = self.save_dir / "anomaly_prototypes.pt"
        meta_path = self.save_dir / "prototype_meta.json"

        if anomaly_path.exists():
            self.anomaly_prototypes = torch.load(anomaly_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded {self.anomaly_prototypes.shape[0]} anomaly prototypes")

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        return self.anomaly_prototypes is not None

    def save(self):
        """保存原型库到磁盘"""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.anomaly_prototypes is not None:
            torch.save(self.anomaly_prototypes, self.save_dir / "anomaly_prototypes.pt")

        with open(self.save_dir / "prototype_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Prototype bank saved to {self.save_dir}")

    def add_anomaly_prototypes(self, features: torch.Tensor):
        """添加异常样本原型"""
        if self.anomaly_prototypes is None:
            self.anomaly_prototypes = features
        else:
            self.anomaly_prototypes = torch.cat([self.anomaly_prototypes, features], dim=0)
        self.metadata["anomaly_count"] = self.anomaly_prototypes.shape[0]

    def has_anomaly_prototypes(self) -> bool:
        return self.anomaly_prototypes is not None and len(self.anomaly_prototypes) > 0

    def get_prototype_count(self) -> int:
        if self.anomaly_prototypes is None:
            return 0
        return self.anomaly_prototypes.shape[0]


class AnomalyScoreRefiner:
    """异常分数修正器：融合 Dinomaly 分数和异常原型对比分数"""

    def __init__(self, prototype_bank: PrototypeBank, device: str = "cuda"):
        self.prototype_bank = prototype_bank
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def refine(
        self,
        dinomaly_score: float,
        image_feature: torch.Tensor,
        weight: float = 0.4,
    ) -> float:
        """
        将 Dinomaly 原始分数与异常原型对比分数融合

        Args:
            dinomaly_score: Dinomaly 输出的原始异常分数 (0~1)
            image_feature: 当前图片的 768 维特征向量
            weight: 原型对比的权重

        Returns:
            final_score: 融合后的异常分数
        """
        if not self.prototype_bank.has_anomaly_prototypes():
            return dinomaly_score

        anomaly_protos = self.prototype_bank.anomaly_prototypes.to(self.device)
        image_feature = image_feature.to(self.device)

        # 与异常原型的最大相似度
        anomaly_sim = torch.cosine_similarity(
            image_feature.unsqueeze(0),
            anomaly_protos,
            dim=1,
        ).max().item()

        # anomaly_sim ∈ [-1, 1]，归一化到 [0, 1]
        prototype_score = (anomaly_sim + 1) / 2

        # 加权融合：异常原型越像 → 分数越高
        final_score = (1 - weight) * dinomaly_score + weight * prototype_score
        return final_score

    def refine_batch(
        self,
        dinomaly_scores: np.ndarray,
        image_features: torch.Tensor,
        weight: float = 0.4,
    ) -> np.ndarray:
        """批量修正分数"""
        if not self.prototype_bank.has_anomaly_prototypes():
            return dinomaly_scores

        anomaly_protos = self.prototype_bank.anomaly_prototypes.to(self.device)
        image_features = image_features.to(self.device)

        # 计算与所有异常原型的相似度 (N_images, N_protos)
        sim_matrix = torch.cosine_similarity(
            image_features.unsqueeze(1),
            anomaly_protos.unsqueeze(0),
            dim=2,
        )  # (N, M)

        # 每张图片与异常原型的最大相似度
        anomaly_sim = sim_matrix.max(dim=1).values  # (N,)

        # 归一化到 [0, 1]
        prototype_scores = (anomaly_sim + 1) / 2

        # 融合
        dinomaly_scores_t = torch.from_numpy(dinomaly_scores).to(self.device)
        final_scores = (1 - weight) * dinomaly_scores_t + weight * prototype_scores

        return final_scores.cpu().numpy()


def build_anomaly_prototypes_from_images(
    model,
    images: List[Image.Image],
    device: str = "cuda",
) -> torch.Tensor:
    """
    从异常样本图片构建异常原型库

    Args:
        model: Dinomaly 模型
        images: 异常样本图片列表
        device: 计算设备

    Returns:
        prototypes: (N, 768) 特征矩阵
    """
    extractor = DinomalyFeatureExtractor(model, device)
    features = extractor.extract_batch(images)
    logger.info(f"Built {features.shape[0]} anomaly prototypes")
    return features.cpu()


def build_anomaly_prototypes_from_paths(
    model,
    image_paths: List[Path],
    device: str = "cuda",
    batch_size: int = 16,
) -> torch.Tensor:
    """
    从异常样本图片路径构建异常原型库

    Args:
        model: Dinomaly 模型
        image_paths: 异常样本图片路径列表
        device: 计算设备
        batch_size: 批量大小

    Returns:
        prototypes: (N, 768) 特征矩阵
    """
    extractor = DinomalyFeatureExtractor(model, device)

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        features = extractor.extract_batch(images)
        all_features.append(features.cpu())

    prototypes = torch.cat(all_features, dim=0)
    logger.info(f"Built {prototypes.shape[0]} anomaly prototypes from {len(image_paths)} images")
    return prototypes
