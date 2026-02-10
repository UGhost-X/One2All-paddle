import requests
import os
import base64
from pathlib import Path

def test_filter_training():
    base_url = "http://127.0.0.1:8000"
    
    # 1. 准备图片数据
    images_b64 = []
    img_path = "/home/software/One2All-paddle/tests/dataset/image_1.jpg"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
    
    # 2. 测试场景 1：训练特定 label ("bolt")
    print("\n--- 测试场景 1：仅训练 'bolt' 标签 ---")
    payload_bolt_only = {
        "images": images_b64,
        "coco_data": {
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "label": "bolt"},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 50, 50], "label": "nut"}
            ],
            "categories": [
                {"id": 1, "name": "bolt"},
                {"id": 2, "name": "nut"}
            ]
        },
        "base_path": "/home/software/One2All-paddle",
        "project_id": "test_filter",
        "version": "v1",
        "data_version": "dv1",
        "run_count": 1,
        "label_names": ["bolt"], # 仅训练 bolt
        "epochs": 1,
        "batch_size": 1
    }
    
    resp = requests.post(f"{base_url}/train/anomaly", json=payload_bolt_only)
    data = resp.json()
    print(f"返回任务: {[t['label'] for t in data.get('tasks', [])]}")
    assert len(data.get("tasks", [])) == 1
    assert data["tasks"][0]["label"] == "bolt"
    print("验证成功：仅启动了 bolt 训练任务")

    # 3. 测试场景 2：训练全部 label (不传 label_names)
    print("\n--- 测试场景 2：训练全部标签 ---")
    payload_all = {
        "images": images_b64,
        "coco_data": {
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "label": "bolt"},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 50, 50], "label": "nut"}
            ],
            "categories": [
                {"id": 1, "name": "bolt"},
                {"id": 2, "name": "nut"}
            ]
        },
        "base_path": "/home/software/One2All-paddle",
        "project_id": "test_filter",
        "version": "v1",
        "data_version": "dv1",
        "run_count": 2,
        "epochs": 1,
        "batch_size": 1
    }
    
    resp = requests.post(f"{base_url}/train/anomaly", json=payload_all)
    data = resp.json()
    labels = sorted([t['label'] for t in data.get('tasks', [])])
    print(f"返回任务: {labels}")
    assert len(data.get("tasks", [])) == 2
    assert "bolt" in labels
    assert "nut" in labels
    print("验证成功：启动了所有标签的训练任务")

if __name__ == "__main__":
    test_filter_training()
