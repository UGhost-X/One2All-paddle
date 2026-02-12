import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import base64
import json
import os
import cv2
import time
import threading
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from main import app
from utils.trainer import AnomalyTrainer
import types

client = TestClient(app)

def load_test_dataset():
    dataset_dir = Path(__file__).parent / "dataset"
    ann_path = dataset_dir / "annotations.json"
    
    with open(ann_path, "r") as f:
        coco_data = json.load(f)
    
    # 补充 categories 如果缺失 (虽然 annotations.json 里可能已经有了)
    if "categories" not in coco_data or not coco_data["categories"]:
        coco_data["categories"] = [{"id": 1, "name": "bolt"}]
    
    # 将 annotation 中的 category_id 映射到 label 名 (如果需要的话)
    for ann in coco_data["annotations"]:
        if "label" not in ann:
            ann["label"] = "bolt" # 默认给一个 label

    images_b64 = []
    # 只取前 2 张图片进行测试，加快速度
    for img_info in coco_data["images"][:2]:
        img_path = dataset_dir / img_info["file_name"]
        if img_path.exists():
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
                images_b64.append(img_b64)
    
    return images_b64, coco_data

def test_train_anomaly_integration():
    """
    集成测试：验证真实的数据处理和训练启动流程
    """
    # 1. 加载真实测试数据集
    images_b64, coco_data = load_test_dataset()

    payload = {
        "project_id": "integration_test_pdx",
        "version": "v1",
        "data_version": "dv1",
        "run_count": 1,
        "base_path": "ignored_in_linux", # Linux 下会自动使用 os.getcwd()
        "model_name": "STFPM",
        "images": images_b64,
        "coco_data": coco_data,
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001
    }

    # 2. 仅 Mock 掉最耗时的 trainer.train() 物理过程
    # 我们要确保 UadTrainer 的初始化过程是真实的，以验证模型注册和配置
    with patch("paddlex.modules.anomaly_detection.UadTrainer.train") as mock_pdx_train:
        mock_pdx_train.return_value = None
        
        # 3. 发送真实请求
        response = client.post("/train/anomaly", json=payload)
        
        # 4. 验证 API 响应
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        task_id = data["tasks"][0]["task_id"]
        
        # 5. 验证文件系统：检查裁剪后的图片是否真的生成了
        storage_path = Path(data["storage_path"])
        crop_dir = storage_path / "bolt" / "images"
        assert crop_dir.exists(), f"裁剪目录未创建: {crop_dir}"
        crops = list(crop_dir.glob("*.png"))
        assert len(crops) > 0, "没有生成裁剪后的图片"
        print(f"\n[验证成功] 已生成 {len(crops)} 张裁剪图片在 {crop_dir}")

        # 6. 验证训练状态：等待几秒让后台线程初始化 PaddleX
        time_waited = 0
        while time_waited < 10:
            status_resp = client.get(f"/train/status/{task_id}")
            status_data = status_resp.json()
            if status_data["status"] == "training":
                print(f"[验证成功] PaddleX 训练任务已成功进入 'training' 状态")
                break
            if status_data["status"] == "failed":
                pytest.fail(f"训练启动失败: {status_data.get('error')}")
            import time
            time.sleep(1)
            time_waited += 1
        else:
            pytest.fail("训练任务初始化超时")

def test_get_train_status_mock():
    # 模拟 trainer.get_status 返回
    mock_status = {
        "status": "training",
        "progress": 50,
        "label": "bolt",
        "logs": ["Epoch 10 finished"]
    }
    
    with patch("main.trainer.get_status") as mock_get_status:
        mock_get_status.return_value = mock_status
        
        response = client.get("/train/status/mock_task_123")
        
        assert response.status_code == 200
        assert response.json() == mock_status
        mock_get_status.assert_called_with("mock_task_123")

def test_find_latest_resume_path(tmp_path: Path):
    trainer = AnomalyTrainer(output_dir=str(tmp_path / "out"))
    save_dir = tmp_path / "out" / "p" / "label" / "task_1"
    iter_1 = save_dir / "iter_1"
    iter_2 = save_dir / "iter_2"
    iter_1.mkdir(parents=True)
    iter_2.mkdir(parents=True)
    (iter_1 / "model.pdparams").write_text("a")
    (iter_1 / "model.pdopt").write_text("b")
    (iter_2 / "model.pdparams").write_text("c")
    (iter_2 / "model.pdopt").write_text("d")

    older = time.time() - 10
    newer = time.time()
    os.utime(iter_1 / "model.pdparams", (older, older))
    os.utime(iter_2 / "model.pdparams", (newer, newer))

    resume_path = trainer._find_latest_resume_path(str(save_dir))
    assert resume_path.endswith("iter_2/model.pdparams")

def test_run_training_async_resumes_existing_task(tmp_path: Path):
    trainer = AnomalyTrainer(output_dir=str(tmp_path / "out"))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.txt").write_text("images/a.png masks/a.png\n")

    config = {
        "project_id": "p",
        "version": "v1",
        "data_version": "dv1",
        "run_count": 1,
        "model_name": "STFPM",
        "label_name": "label",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
    }

    task_id = "task_1"
    save_dir = tmp_path / "out" / "p" / "label" / task_id
    iter_1 = save_dir / "iter_1"
    iter_1.mkdir(parents=True)
    (iter_1 / "model.pdparams").write_text("a")
    (iter_1 / "model.pdopt").write_text("b")

    task_key = trainer._make_task_key(str(dataset_dir), config)
    trainer.training_status[task_id] = {
        "status": "failed",
        "progress": 50,
        "label": "label",
        "group_id": None,
        "logs": [],
        "metrics": [],
        "total_epochs": 1,
        "start_time": time.time(),
        "dataset_dir": str(dataset_dir),
        "save_dir": str(save_dir),
        "config": config,
        "task_key": task_key,
    }
    trainer.task_key_index[task_key] = task_id

    started = threading.Event()

    def fake_train_process(self, task_id_arg, dataset_dir_arg, config_arg, resume_arg):
        assert task_id_arg == task_id
        assert resume_arg is True
        started.set()

    trainer._train_process = types.MethodType(fake_train_process, trainer)
    returned = trainer.run_training_async(str(dataset_dir), config, group_id="g1")
    assert returned == task_id
    assert started.wait(2)

if __name__ == "__main__":
    pytest.main([__file__])
