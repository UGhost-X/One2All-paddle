#!/usr/bin/env python3
"""
测试 PatchCore 训练 - 按类别分组
"""
import sys
import time
sys.path.insert(0, '/home/software/One2All-paddle')

from utils.patchcore_trainer import PatchCoreTrainer

def main():
    print("=" * 60)
    print("PatchCore 训练测试 (按类别分组)")
    print("=" * 60)

    # 初始化训练器
    trainer = PatchCoreTrainer(output_dir="test_output")
    print("\n[1/3] 训练器初始化完成")

    # 配置 - 按 label 分组
    config = {
        "project_id": "5",
        "task_uuid": "d34c8061",
        "model_name": "patchcore",
        "backbone": "resnet18",
        "coreset_ratio": 0.1,
        "input_size": [224, 224],
        "threshold_percentile": 95,
        "group_by": "label",  # 按 label 分组
    }

    dataset_dir = "/home/software/One2All-paddle/5/train/d34c8061"

    print(f"[2/3] 数据集目录: {dataset_dir}")
    print(f"配置: backbone={config['backbone']}, group_by={config['group_by']}")

    # 启动训练
    print("\n[3/3] 启动训练...")
    print("-" * 60)

    try:
        task_ids = trainer.run_training_async(dataset_dir, config)
        print(f"创建了 {len(task_ids)} 个训练任务")

        # 监控训练进度
        completed = set()
        failed = set()
        max_wait = 300  # 最多等待300秒
        start_time = time.time()

        while len(completed) + len(failed) < len(task_ids):
            if time.time() - start_time > max_wait:
                print("\n超时，停止监控")
                break

            for task_id in task_ids:
                if task_id in completed or task_id in failed:
                    continue

                status = trainer.get_training_status(task_id)
                if status:
                    state = status.get('status', 'unknown')
                    progress = status.get('progress', 0)
                    group_id = status.get('group_id', 'unknown')
                    logs = status.get('logs', [])

                    if state == 'completed':
                        print(f"✓ 类别 '{group_id}': 完成")
                        completed.add(task_id)
                    elif state == 'failed':
                        error = status.get('error', 'unknown error')
                        print(f"✗ 类别 '{group_id}': 失败 - {error}")
                        failed.add(task_id)
                    elif state == 'training':
                        if logs:
                            print(f"  类别 '{group_id}': 训练中 (进度: {progress}%) - {logs[-1]}")

            time.sleep(2)

        print("-" * 60)
        print(f"\n训练完成统计:")
        print(f"  成功: {len(completed)}/{len(task_ids)}")
        print(f"  失败: {len(failed)}/{len(task_ids)}")

        # 显示结果
        for task_id in completed:
            status = trainer.get_training_status(task_id)
            group_id = status.get('group_id', 'unknown')
            print(f"\n类别 '{group_id}':")
            print(f"  保存目录: {status.get('save_dir')}")
            print(f"  样本数: {status.get('num_samples')}")
            if 'calibration_stats' in status:
                stats = status['calibration_stats']
                print(f"  阈值: {stats.get('threshold', 'N/A'):.6f}")

            # 检查生成的文件
            import os
            save_dir = status.get('save_dir')
            if save_dir and os.path.exists(save_dir):
                print(f"  生成的文件:")
                for f in os.listdir(save_dir):
                    print(f"    - {f}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
