#!/usr/bin/env python3
"""
PatchCore 模型训练脚本 (PyTorch版本)
使用 /home/software/One2All-paddle/2/train/c56700f8 数据集
"""
import sys
import os
import time
import logging

# 添加项目路径
sys.path.insert(0, '/home/software/One2All-paddle')

from utils.patchcore_trainer_torch import trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_category(category_name, dataset_dir):
    """训练单个类别"""
    config = {
        'project_id': '2',
        'task_uuid': 'c56700f8',
        'label_name': category_name,
        'model_name': 'PatchCore',
        'backbone': 'resnet18',
        'input_size': [224, 224],
        'coreset_ratio': 0.1,
        'threshold_percentile': 95,
    }

    logger.info(f"Starting training for {category_name}...")
    logger.info(f"Dataset: {dataset_dir}")

    task_id = trainer.run_training_async(dataset_dir, config)
    logger.info(f"Task ID: {task_id}")

    # 等待训练完成
    last_log_count = 0
    while True:
        status = trainer.get_status(task_id)
        current_status = status.get('status')
        progress = status.get('progress', 0)

        logger.info(f"Status: {current_status}, Progress: {progress}%")

        # 打印最近的日志
        logs = status.get('logs', [])
        if len(logs) > last_log_count:
            for log in logs[last_log_count:]:
                logger.info(f"  {log}")
            last_log_count = len(logs)

        if current_status in ['completed', 'failed', 'cancelled']:
            logger.info(f"Training finished with status: {current_status}")
            if status.get('error'):
                logger.error(f"Error: {status.get('error')}")
            if status.get('calibration_stats'):
                stats = status.get('calibration_stats')
                logger.info(f"Calibration stats: threshold={stats.get('threshold', 0):.6f}, "
                           f"n={stats.get('n', 0)}, mean={stats.get('mean', 0):.6f}")
            break

        time.sleep(3)

    return task_id, status


def main():
    base_dir = '/home/software/One2All-paddle/2/train/c56700f8'

    # 定义要训练的类别
    categories = [
        ('孔洞', os.path.join(base_dir, '孔洞')),
        ('螺丝', os.path.join(base_dir, '螺丝')),
        ('轴', os.path.join(base_dir, '轴')),
    ]

    results = {}

    for category_name, dataset_dir in categories:
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset not found: {dataset_dir}, skipping...")
            continue

        logger.info("=" * 60)
        logger.info(f"Training category: {category_name}")
        logger.info("=" * 60)

        try:
            task_id, status = train_category(category_name, dataset_dir)
            results[category_name] = {
                'task_id': task_id,
                'status': status.get('status'),
                'save_dir': status.get('save_dir'),
            }
        except Exception as e:
            logger.error(f"Training failed for {category_name}: {e}")
            import traceback
            traceback.print_exc()
            results[category_name] = {
                'status': 'failed',
                'error': str(e),
            }

    # 打印最终结果
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    for category_name, result in results.items():
        logger.info(f"{category_name}: {result.get('status')}")
        if result.get('save_dir'):
            logger.info(f"  Save dir: {result.get('save_dir')}")


if __name__ == "__main__":
    main()
