#!/usr/bin/env python3
"""
建立异常记忆库（Anomaly Memory Bank）
用于双库推理：正常库 + 异常库
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
from typing import List, Optional, Dict
import json

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"


class PatchCoreFeatureExtractor:
    """PatchCore特征提取器"""
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

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取patch级别的特征"""
        with torch.no_grad():
            features = self.backbone(x)
            B, C, H_p, W_p = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(B, H_p * W_p, C)
        return patches

    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """从单张图片提取特征"""
        try:
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

            with torch.no_grad():
                features = self.extract_features(tensor)
                features = features.cpu().numpy()[0]

            return features
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return None


def build_anomaly_memory_bank(
    anomaly_image_paths: List[str],
    save_path: str,
    backbone_name: str = "resnet18",
    category: str = "unknown"
) -> Optional[np.ndarray]:
    """
    建立异常记忆库

    Args:
        anomaly_image_paths: 异常图片路径列表
        save_path: 保存路径
        backbone_name: 骨干网络名称
        category: 类别名称

    Returns:
        异常记忆库特征数组
    """
    if not anomaly_image_paths:
        print("没有提供异常图片")
        return None

    print(f"开始为类别 '{category}' 建立异常记忆库...")
    print(f"图片数量: {len(anomaly_image_paths)}")

    # 创建特征提取器
    extractor = PatchCoreFeatureExtractor(backbone_name=backbone_name)

    # 提取所有异常图片的特征
    all_features = []
    for img_path in anomaly_image_paths:
        print(f"处理: {img_path}")
        features = extractor.extract_from_image(img_path)
        if features is not None:
            all_features.append(features)
            print(f"  特征形状: {features.shape}")

    if not all_features:
        print("没有成功提取到任何特征")
        return None

    # 合并所有特征
    anomaly_memory_bank = np.vstack(all_features)
    print(f"\n异常记忆库形状: {anomaly_memory_bank.shape}")

    # 保存异常记忆库
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(save_path, memory_bank=anomaly_memory_bank)
    print(f"异常记忆库已保存到: {save_path}")

    # 保存元数据
    metadata = {
        "category": category,
        "num_samples": len(anomaly_image_paths),
        "memory_bank_shape": list(anomaly_memory_bank.shape),
        "backbone": backbone_name,
        "created_at": str(np.datetime64('now')),
        "image_paths": anomaly_image_paths
    }

    metadata_path = str(save_path).replace('.npz', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存到: {metadata_path}")

    return anomaly_memory_bank


def add_to_anomaly_memory_bank(
    new_anomaly_paths: List[str],
    existing_bank_path: str,
    backbone_name: str = "resnet18"
) -> Optional[np.ndarray]:
    """
    向现有异常记忆库添加新的异常样本

    Args:
        new_anomaly_paths: 新的异常图片路径列表
        existing_bank_path: 现有异常记忆库路径
        backbone_name: 骨干网络名称

    Returns:
        更新后的异常记忆库
    """
    # 加载现有异常记忆库
    try:
        data = np.load(existing_bank_path)
        existing_bank = data['memory_bank']
        print(f"加载现有异常记忆库: {existing_bank.shape}")
    except Exception as e:
        print(f"无法加载现有异常记忆库: {e}")
        existing_bank = np.empty((0, 512), dtype=np.float32)

    # 提取新样本特征
    extractor = PatchCoreFeatureExtractor(backbone_name=backbone_name)
    new_features = []

    for img_path in new_anomaly_paths:
        print(f"处理新异常样本: {img_path}")
        features = extractor.extract_from_image(img_path)
        if features is not None:
            new_features.append(features)

    if not new_features:
        print("没有成功提取到新特征")
        return existing_bank

    # 合并
    new_bank = np.vstack(new_features)
    updated_bank = np.vstack([existing_bank, new_bank])

    print(f"更新前: {existing_bank.shape}")
    print(f"新增: {new_bank.shape}")
    print(f"更新后: {updated_bank.shape}")

    # 保存备份
    backup_path = existing_bank_path.replace('.npz', '_backup.npz')
    np.savez_compressed(backup_path, memory_bank=existing_bank)
    print(f"备份已保存: {backup_path}")

    # 保存更新后的记忆库
    np.savez_compressed(existing_bank_path, memory_bank=updated_bank)
    print(f"异常记忆库已更新: {existing_bank_path}")

    return updated_bank


def main():
    """主函数：从 roi_screenshots 的 false_negative 目录建立异常记忆库"""

    # 配置
    category = "圆形孔洞"

    # 可能的异常图片来源
    sources = [
        # 从 ok 目录中人工确认的异常（false negatives）
        f"/home/software/One2All-paddle/roi_screenshots/{category}/fn",
        # 可以添加其他来源
    ]

    # 收集所有图片
    all_anomaly_images = []
    for source_dir in sources:
        source_path = Path(source_dir)
        if source_path.exists():
            images = list(source_path.glob("*.jpg"))
            all_anomaly_images.extend([str(p) for p in images])
            print(f"从 {source_dir} 找到 {len(images)} 张图片")

    if not all_anomaly_images:
        print("没有找到异常图片")
        print("\n提示：请确认以下之一：")
        print(f"1. 将确认的异常图片放入: /home/software/One2All-paddle/roi_screenshots/{category}/错的识别为对/")
        print(f"2. 或者修改脚本指定其他图片来源")
        return

    print(f"\n总共找到 {len(all_anomaly_images)} 张异常图片")

    # 建立异常记忆库
    anomaly_bank_path = f"/home/software/One2All-paddle/output/5/d34c8061/{category}/anomaly_memory_bank.npz"

    build_anomaly_memory_bank(
        anomaly_image_paths=all_anomaly_images,
        save_path=anomaly_bank_path,
        backbone_name="resnet18",
        category=category
    )

    print("\n" + "="*60)
    print("异常记忆库建立完成！")
    print("="*60)
    print(f"\n下一步：")
    print(f"1. 确认异常记忆库文件: {anomaly_bank_path}")
    print(f"2. 更新推理服务以支持双库推理")
    print(f"3. 重启服务测试双库检测效果")


if __name__ == "__main__":
    main()
