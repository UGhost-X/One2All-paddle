#!/usr/bin/env python3
"""
同步测试 PatchCore 训练 - 按类别分组
"""
import sys
sys.path.insert(0, '/home/software/One2All-paddle')

import os
import json
import time
from pathlib import Path

import numpy as np
import torch

from utils.patchcore_trainer import PatchCoreTrainer

def train_group_sync(trainer, dataset_dir, config, group_id, group_annotations):
    """同步训练一个组"""
    task_id = f"test_{group_id}_{int(time.time())}"
    
    print(f"\n{'='*60}")
    print(f"Training group: {group_id}")
    print(f"Samples: {len(group_annotations)}")
    print(f"{'='*60}")
    
    save_dir = os.path.join(
        trainer.output_dir,
        config.get("project_id", "default"),
        config.get("task_uuid", "unknown"),
        str(group_id).replace('/', '_')
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取ROI图像
    print("[1/4] Extracting ROI images...")
    roi_images = trainer._extract_roi_images(task_id, dataset_dir, group_annotations)
    print(f"  Extracted {len(roi_images)} ROI images")
    
    if not roi_images:
        print("  ERROR: No ROI images extracted!")
        return None
    
    # 构建模型
    print("[2/4] Building PatchCore model...")
    backbone_name = config.get("backbone", "resnet18")
    input_size = config.get("input_size", [224, 224])
    coreset_ratio = config.get("coreset_ratio", 0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = trainer._build_feature_extractor(backbone_name, device)
    
    # 提取特征
    print("[3/4] Extracting features...")
    features = trainer._extract_features(task_id, feature_extractor, roi_images, input_size, device)
    print(f"  Extracted {len(features)} feature patches")
    
    # 核心集采样
    print("  Applying coreset sampling...")
    memory_bank = trainer._coreset_sampling(features, coreset_ratio)
    print(f"  Memory bank size: {memory_bank.shape}")
    
    # 保存内存库
    memory_bank_path = os.path.join(save_dir, "memory_bank.npz")
    np.savez_compressed(memory_bank_path, memory_bank=memory_bank)
    print(f"  Saved to {memory_bank_path}")
    
    # 阈值校准
    print("[4/4] Calibrating threshold...")
    threshold = trainer._calibrate_threshold(
        task_id, feature_extractor, memory_bank, 
        roi_images, input_size, config, device
    )
    
    # 保存配置
    first_ann = group_annotations[0] if group_annotations else {}
    config_data = {
        "group_id": group_id,
        "category": first_ann.get('label', 'unknown'),
        "category_id": first_ann.get('category_id', 0),
        "threshold": threshold,
        "input_size": input_size,
        "num_samples": len(roi_images),
        "model_name": "PatchCore",
        "backbone": backbone_name,
        "memory_bank_shape": list(memory_bank.shape),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Training complete for group '{group_id}'")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Save dir: {save_dir}")
    
    return save_dir

def main():
    print("=" * 60)
    print("PatchCore 训练测试 (同步模式)")
    print("=" * 60)
    
    # 初始化
    trainer = PatchCoreTrainer(output_dir="test_output")
    
    config = {
        "project_id": "5",
        "task_uuid": "d34c8061",
        "model_name": "patchcore",
        "backbone": "resnet18",
        "coreset_ratio": 0.1,
        "input_size": [224, 224],
        "threshold_percentile": 95,
        "group_by": "label",
    }
    
    dataset_dir = "/home/software/One2All-paddle/5/train/d34c8061"
    
    # 加载数据
    print(f"\nLoading data from {dataset_dir}")
    annotation_file = Path(dataset_dir) / "annotations.json"
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    print(f"Total annotations: {len(annotations)}")
    
    # 按 label 分组
    groups = {}
    for ann in annotations:
        label = ann.get('label', 'unknown')
        if label not in groups:
            groups[label] = []
        groups[label].append(ann)
    
    print(f"Groups: {list(groups.keys())}")
    for label, anns in groups.items():
        print(f"  {label}: {len(anns)} annotations")
    
    # 训练每个组
    results = {}
    start_time = time.time()
    
    for group_id in sorted(groups.keys()):
        group_annotations = groups[group_id]
        try:
            save_dir = train_group_sync(trainer, dataset_dir, config, group_id, group_annotations)
            results[group_id] = {"status": "success", "save_dir": save_dir}
        except Exception as e:
            print(f"\n✗ Training failed for group '{group_id}': {e}")
            import traceback
            traceback.print_exc()
            results[group_id] = {"status": "failed", "error": str(e)}
    
    elapsed = time.time() - start_time
    
    # 总结
    print("\n" + "=" * 60)
    print("训练完成总结")
    print("=" * 60)
    print(f"总耗时: {elapsed:.1f} 秒")
    
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    failed_count = len(results) - success_count
    
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {failed_count}/{len(results)}")
    
    for group_id, result in results.items():
        status = result.get("status")
        if status == "success":
            print(f"\n✓ {group_id}:")
            print(f"  保存目录: {result.get('save_dir')}")
            # 列出文件
            save_dir = result.get('save_dir')
            if save_dir and os.path.exists(save_dir):
                for f in os.listdir(save_dir):
                    print(f"    - {f}")
        else:
            print(f"\n✗ {group_id}: {result.get('error')}")

if __name__ == "__main__":
    main()
