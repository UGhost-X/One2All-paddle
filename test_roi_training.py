#!/usr/bin/env python3
"""
测试 ROI 级别的 PatchCore 训练
"""
import sys
import time
sys.path.insert(0, '/home/software/One2All-paddle')

from utils.patchcore_trainer import PatchCoreTrainer

def main():
    print("=" * 60)
    print("ROI PatchCore 训练测试")
    print("=" * 60)

    # 初始化训练器
    trainer = PatchCoreTrainer(output_dir="test_roi_output")
    print("\n[1/4] 训练器初始化完成")

    # 配置
    config = {
        "project_id": "5",
        "task_uuid": "d34c8061",
        "model_name": "patchcore",
        "backbone": "resnet18",
        "coreset_ratio": 0.1,
        "input_size": [224, 224],
        "threshold_percentile": 95,
    }

    dataset_dir = "/home/software/One2All-paddle/5/train/d34c8061"

    print(f"[2/4] 数据集目录: {dataset_dir}")
    print(f"[2/4] 配置: backbone={config['backbone']}, coreset_ratio={config['coreset_ratio']}")

    # 启动训练
    print("\n[3/4] 启动训练...")
    task_ids = trainer.run_training_async(dataset_dir, config)

    print(f"[3/4] 创建了 {len(task_ids)} 个训练任务")
    print(f"任务ID列表 (前5个): {task_ids[:5]}")

    # 监控训练进度
    print("\n[4/4] 监控训练进度...")
    print("-" * 60)

    completed = set()
    failed = set()

    while len(completed) + len(failed) < len(task_ids):
        for task_id in task_ids:
            if task_id in completed or task_id in failed:
                continue

            status = trainer.get_training_status(task_id)
            if status:
                state = status.get('status', 'unknown')
                progress = status.get('progress', 0)
                roi_id = status.get('roi_id', 'unknown')

                if state == 'completed':
                    print(f"✓ ROI {roi_id}: 完成 (进度: {progress}%)")
                    completed.add(task_id)
                elif state == 'failed':
                    error = status.get('error', 'unknown error')
                    print(f"✗ ROI {roi_id}: 失败 - {error}")
                    failed.add(task_id)
                elif state == 'training':
                    print(f"  ROI {roi_id}: 训练中 (进度: {progress}%)")

        time.sleep(2)

    print("-" * 60)
    print(f"\n训练完成统计:")
    print(f"  成功: {len(completed)}/{len(task_ids)}")
    print(f"  失败: {len(failed)}/{len(task_ids)}")

    # 显示一个成功的结果
    if completed:
        sample_task = list(completed)[0]
        status = trainer.get_training_status(sample_task)
        print(f"\n示例结果 (ROI {status.get('roi_id')}):")
        print(f"  保存目录: {status.get('save_dir')}")
        print(f"  样本数: {status.get('num_samples')}")
        if 'calibration_stats' in status:
            stats = status['calibration_stats']
            print(f"  阈值: {stats.get('threshold', 'N/A'):.6f}")
            print(f"  校准样本数: {stats.get('n', 'N/A')}")

        # 检查生成的文件
        import os
        save_dir = status.get('save_dir')
        if save_dir and os.path.exists(save_dir):
            print(f"\n生成的文件:")
            for f in os.listdir(save_dir):
                print(f"  - {f}")

if __name__ == "__main__":
    main()
