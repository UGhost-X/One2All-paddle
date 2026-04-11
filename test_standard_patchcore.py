#!/usr/bin/env python3
"""
测试标准 PatchCore 实现 - 验证训练数据应该得到低异常分数
"""
import numpy as np
import torch
import torch.nn.functional as F

def compute_anomaly_score_standard(patch_features, memory_tensor):
    """
    标准 PatchCore 异常分数计算
    图像级异常分数 = 最大 patch 距离
    """
    # 计算每个 patch 到内存库的最近邻距离
    similarity = torch.matmul(patch_features, memory_tensor.T)
    distances = 1.0 - similarity
    patch_distances = torch.min(distances, dim=-1)[0]

    # 标准 PatchCore: 图像级异常分数 = 最大 patch 距离
    anomaly_score = torch.max(patch_distances).item()

    return anomaly_score

def test_standard_patchcore():
    """测试标准 PatchCore 实现"""
    print("=" * 60)
    print("测试标准 PatchCore 实现")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    dim = 196
    n_memory = 1000
    patches_per_roi = 256

    # 创建内存库（已归一化）
    memory_bank = torch.randn(n_memory, dim)
    memory_bank = F.normalize(memory_bank, dim=-1)

    print(f"\n内存库大小: {n_memory}")
    print(f"每个 ROI patches: {patches_per_roi}")

    # 场景1: 训练样本（应该与内存库中的某些特征匹配）
    print("\n" + "-" * 60)
    print("场景1: 训练样本（从内存库附近采样）")
    print("-" * 60)

    # 从内存库附近采样 patches（模拟训练样本）
    test_patches = []
    for i in range(patches_per_roi):
        # 随机选择一个内存库特征并添加少量噪声
        idx = np.random.randint(0, n_memory)
        noise = torch.randn(dim) * 0.01
        patch = memory_bank[idx] + noise
        patch = F.normalize(patch.unsqueeze(0), dim=-1).squeeze(0)
        test_patches.append(patch)

    test_patches = torch.stack(test_patches)

    score = compute_anomaly_score_standard(test_patches, memory_bank)
    print(f"异常分数: {score:.6f}")
    print(f"预期: 应该接近 0（因为 patches 都来自内存库附近）")

    # 场景2: 异常样本（随机特征）
    print("\n" + "-" * 60)
    print("场景2: 异常样本（随机特征）")
    print("-" * 60)

    anomaly_patches = torch.randn(patches_per_roi, dim)
    anomaly_patches = F.normalize(anomaly_patches, dim=-1)

    anomaly_score = compute_anomaly_score_standard(anomaly_patches, memory_bank)
    print(f"异常分数: {anomaly_score:.6f}")
    print(f"预期: 应该较高（因为随机特征与内存库不匹配）")

    # 场景3: 部分异常（部分 patches 异常）
    print("\n" + "-" * 60)
    print("场景3: 部分异常（10% patches 异常）")
    print("-" * 60)

    mixed_patches = test_patches.clone()
    n_anomaly = int(patches_per_roi * 0.1)
    anomaly_feat = torch.randn(n_anomaly, dim)
    anomaly_feat = F.normalize(anomaly_feat, dim=-1)
    mixed_patches[:n_anomaly] = anomaly_feat

    mixed_score = compute_anomaly_score_standard(mixed_patches, memory_bank)
    print(f"异常分数: {mixed_score:.6f}")
    print(f"预期: 应该与纯异常样本接近（因为 max 只关心最远的那个）")

    # 场景4: 完全匹配（有一个 patch 完全在内存库中）
    print("\n" + "-" * 60)
    print("场景4: 完全匹配（有一个 patch 完全匹配内存库）")
    print("-" * 60)

    perfect_patches = test_patches.clone()
    perfect_patches[0] = memory_bank[0]  # 完全匹配

    perfect_score = compute_anomaly_score_standard(perfect_patches, memory_bank)
    print(f"异常分数: {perfect_score:.6f}")
    print(f"预期: 应该与场景1类似（max 不关心匹配的 patch）")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"训练样本分数: {score:.6f}")
    print(f"异常样本分数: {anomaly_score:.6f}")
    print(f"比例: {anomaly_score / max(score, 0.001):.2f}x")
    print("\n标准 PatchCore 使用 max 距离，能够区分正常和异常样本！")

if __name__ == "__main__":
    test_standard_patchcore()
