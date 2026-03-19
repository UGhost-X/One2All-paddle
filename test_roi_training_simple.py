#!/usr/bin/env python3
"""
简化版 ROI PatchCore 训练测试 - 只训练前5个ROI
"""
import sys
import time
sys.path.insert(0, '/home/software/One2All-paddle')

from utils.patchcore_trainer import PatchCoreTrainer

def main():
    print("=" * 60)
    print("ROI PatchCore 训练测试 (简化版 - 只训练5个ROI)")
    print("=" * 60)

    # 初始化训练器
    trainer = PatchCoreTrainer(output_dir="test_roi_output")
    print("\n[1/3] 训练器初始化完成")

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

    print(f"[2/3] 数据集目录: {dataset_dir}")
    print(f"配置: backbone={config['backbone']}, coreset_ratio={config['coreset_ratio']}")

    # 启动训练
    print("\n[3/3] 启动训练...")
    print("-" * 60)

    try:
        task_ids = trainer.run_training_async(dataset_dir, config)
        print(f"创建了 {len(task_ids)} 个训练任务")
        print(f"任务ID列表 (前5个): {task_ids[:5]}")

        # 只监控前5个任务
        test_tasks = task_ids[:5]
        print(f"\n监控前 {len(test_tasks)} 个任务...")
        print("-" * 60)

        completed = set()
        failed = set()
        max_wait = 120  # 最多等待120秒
        start_time = time.time()

        while len(completed) + len(failed) < len(test_tasks):
            if time.time() - start_time > max_wait:
                print("\n超时，停止监控")
                break

            for task_id in test_tasks:
                if task_id in completed or task_id in failed:
                    continue

                status = trainer.get_training_status(task_id)
                if status:
                    state = status.get('status', 'unknown')
                    progress = status.get('progress', 0)
                    roi_id = status.get('roi_id', 'unknown')
                    logs = status.get('logs', [])

                    if state == 'completed':
                        print(f"✓ ROI {roi_id}: 完成 (进度: {progress}%)")
                        if logs:
                            print(f"  最后日志: {logs[-1]}")
                        completed.add(task_id)
                    elif state == 'failed':
                        error = status.get('error', 'unknown error')
                        print(f"✗ ROI {roi_id}: 失败 - {error}")
                        failed.add(task_id)
                    elif state == 'training':
                        if logs and len(logs) > 0:
                            print(f"  ROI {roi_id}: 训练中 (进度: {progress}%) - {logs[-1]}")

            time.sleep(3)

        print("-" * 60)
        print(f"\n训练完成统计:")
        print(f"  成功: {len(completed)}/{len(test_tasks)}")
        print(f"  失败: {len(failed)}/{len(test_tasks)}")

        # 显示成功的结果
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
                for root, dirs, files in os.walk(save_dir):
                    level = root.replace(save_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f'{indent}{os.path.basename(root)}/')
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # 只显示前5个文件
                        print(f'{subindent}{file}')

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
