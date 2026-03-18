#!/usr/bin/env python3
"""
PatchCore 模型训练脚本 - 只训练圆形孔洞类别
"""
import sys
import os
import time
import logging

sys.path.insert(0, '/home/software/One2All-paddle')

from utils.patchcore_trainer_torch import trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    category_name = '矩形孔洞'
    dataset_dir = '/home/software/One2All-paddle/5/train/d34c8061/矩形孔洞'

    config = {
        'project_id': '5',
        'task_uuid': 'd34c8061',
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
                logger.info(f"Calibration stats: threshold={stats.get('threshold', 0):.6f}")
            break

        time.sleep(3)

    return task_id, status


if __name__ == '__main__':
    main()
