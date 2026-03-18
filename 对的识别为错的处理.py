#!/usr/bin/env python3
"""
将正常样本添加到 memory_bank 中
用于更新 PatchCore 模型的内存库
"""
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from typing import List, Optional
import datetime
# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"


class PatchCoreModel:
    """PatchCore模型，用于提取特征"""
    def __init__(self, backbone_name="resnet18", device="cuda"):
        self.backbone_name = backbone_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.feature_dim = None
        self._build_backbone()

    def _build_backbone(self):
        """构建预训练的特征提取器"""
        if self.backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif self.backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        else:
            backbone = models.resnet18(pretrained=True)
            self.feature_dim = 512

        # 移除最后的全连接层和池化层
        self.backbone = nn.Sequential(*list(backbone.children())[:-2]).to(self.device)
        self.backbone.eval()

        # 冻结参数
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取patch级别的特征"""
        with torch.no_grad():
            features = self.backbone(x)
            B, C, H_p, W_p = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(B, H_p * W_p, C)
        return patches

    def extract_features_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """从单张图片提取特征"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图片: {image_path}")
                return None

            # 预处理
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img[np.newaxis]).float().to(self.device)

            # 提取特征
            with torch.no_grad():
                features = self._extract_features(tensor)
                features = features.cpu().numpy()[0]  # [N_patches, C]

            return features
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return None


def load_memory_bank(memory_bank_path: str) -> Optional[np.ndarray]:
    """加载现有的 memory bank"""
    try:
        memory_data = np.load(memory_bank_path)
        if 'memory_bank' in memory_data:
            return memory_data['memory_bank']
        else:
            return memory_data
    except Exception as e:
        print(f"加载 memory bank 失败: {e}")
        return None


def save_memory_bank(memory_bank: np.ndarray, save_path: str):
    """保存 memory bank"""
    try:
        np.savez_compressed(save_path, memory_bank=memory_bank)
        print(f"Memory bank 已保存到: {save_path}")
    except Exception as e:
        print(f"保存 memory bank 失败: {e}")


def add_images_to_memory_bank(
    image_paths: List[str],
    memory_bank_path: str,
    backbone_name: str = "resnet18",
    save_backup: bool = True
):
    """
    将图片添加到 memory bank

    Args:
        image_paths: 图片路径列表
        memory_bank_path: memory bank 文件路径
        backbone_name: 骨干网络名称
        save_backup: 是否保存备份
    """
    # 加载现有 memory bank
    existing_memory = load_memory_bank(memory_bank_path)
    if existing_memory is None:
        print("无法加载现有 memory bank，将创建新的")
        existing_memory = np.empty((0, 512), dtype=np.float32)

    print(f"现有 memory bank 形状: {existing_memory.shape}")

    # 创建模型
    model = PatchCoreModel(backbone_name=backbone_name)

    # 提取新图片的特征
    new_features_list = []
    for img_path in image_paths:
        print(f"处理: {img_path}")
        features = model.extract_features_from_image(img_path)
        if features is not None:
            new_features_list.append(features)
            print(f"  提取特征形状: {features.shape}")

    if not new_features_list:
        print("没有成功提取到任何特征")
        return

    # 合并新特征
    new_features = np.vstack(new_features_list)
    print(f"新特征总形状: {new_features.shape}")

    # 合并到现有 memory bank
    updated_memory = np.vstack([existing_memory, new_features])
    print(f"更新后 memory bank 形状: {updated_memory.shape}")

    # 保存备份
    if save_backup:
        backup_path = memory_bank_path.replace('.npz', f'_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
        save_memory_bank(existing_memory, backup_path)

    # 保存更新后的 memory bank
    save_memory_bank(updated_memory, memory_bank_path)


def main():
    # 配置
    category = "矩形孔洞"
    roi_dir = f"/home/software/One2All-paddle/roi_screenshots/{category}/对的识别为错"
    memory_bank_path = f"/home/software/One2All-paddle/output/5/d34c8061/{category}/memory_bank.npz"

    # 获取所有 ROI 图片
    roi_path = Path(roi_dir)
    if not roi_path.exists():
        print(f"ROI 目录不存在: {roi_dir}")
        return

    image_paths = sorted([str(p) for p in roi_path.glob("*.jpg")])
    if not image_paths:
        print(f"在 {roi_dir} 中没有找到图片")
        return

    print(f"找到 {len(image_paths)} 张图片:")
    for p in image_paths:
        print(f"  - {p}")

    # 添加到 memory bank
    add_images_to_memory_bank(
        image_paths=image_paths,
        memory_bank_path=memory_bank_path,
        backbone_name="resnet18",
        save_backup=True
    )

    print("\n完成！")
    print(f"注意：这些图片现在仍留在 {roi_dir} 目录中")
    print(f"如果确认添加成功，可以手动将这些图片移动到 ok 目录或删除")


if __name__ == "__main__":
    main()
