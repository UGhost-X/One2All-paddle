import requests
import time
import json
import os
import base64
from pathlib import Path

def test_full_training():
    # 1. 准备测试数据和请求
    base_url = "http://127.0.0.1:8000"
    
    # 读取真实图片并转换为 base64
    images_b64 = []
    for i in range(1, 4):
        img_path = f"/home/software/One2All-paddle/tests/dataset/image_{i}.jpg"
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
    
    # 模拟前端传入的训练请求
    train_payload = {
        "images": images_b64,
        "coco_data": {
            "images": [
                {"id": 1, "file_name": "image_1.jpg", "width": 1000, "height": 1000},
                {"id": 2, "file_name": "image_2.jpg", "width": 1000, "height": 1000},
                {"id": 3, "file_name": "image_3.jpg", "width": 1000, "height": 1000}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 200], "area": 40000, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [150, 150, 220, 220], "area": 48400, "iscrowd": 0},
                {"id": 3, "image_id": 3, "category_id": 1, "bbox": [50, 50, 180, 180], "area": 32400, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "bolt"}
            ]
        },
        "base_path": "/home/software/One2All-paddle",
        "project_id": "test_proj_full",
        "version": "v1",
        "data_version": "dv1",
        "run_count": 1,
        "model_name": "STFPM",
        "epochs": 2,
        "batch_size": 1,
        "learning_rate": 0.001
    }

    print(f"\n[1/4] 发送训练请求到 {base_url}/train/anomaly ...")
    response = requests.post(f"{base_url}/train/anomaly", json=train_payload)
    
    if response.status_code != 200:
        print(f"请求失败: {response.text}")
        return

    resp_json = response.json()
    tasks = resp_json.get("tasks", [])
    if not tasks:
        print("未返回任何训练任务")
        return
        
    task_id = tasks[0].get("task_id")
    print(f"任务已启动, Task ID: {task_id}")

    # 2. 开始轮询训练状态
    print("\n[2/4] 开始轮询训练状态 (预计运行 2-5 分钟)...")
    last_progress = -1
    while True:
        status_resp = requests.get(f"{base_url}/train/status/{task_id}")
        status_data = status_resp.json()
        
        curr_progress = status_data.get("progress", 0)
        status = status_data.get("status", "unknown")
        
        if curr_progress != last_progress:
            print(f"进度: {curr_progress}% | 状态: {status}")
            if "metrics" in status_data and status_data["metrics"]:
                last_metric = status_data["metrics"][-1]
                print(f"  最新训练指标: Epoch {last_metric.get('epoch')}, Iter {last_metric.get('iter')}, Loss {last_metric.get('loss')}")
            if "eval_metrics" in status_data and status_data["eval_metrics"]:
                last_eval = status_data["eval_metrics"][-1]
                print(f"  最新评估指标: mIoU {last_eval.get('miou')}")
            last_progress = curr_progress
            
        if status in ["completed", "failed"]:
            if status == "failed":
                print(f"\n[错误] 训练失败: {status_data.get('error')}")
                return
            print("\n[成功] 训练任务已圆满完成！")
            break
        time.sleep(5)

    # 3. 检查输出文件
    print("\n[3/4] 检查输出目录...")
    output_path = Path(f"/home/software/One2All-paddle/output/test_proj_full/bolt/{task_id}")
    if output_path.exists():
        print(f"输出目录已创建: {output_path}")
        # 查找权重文件
        weights = list(output_path.glob("**/*.pdparams"))
        if weights:
            print(f"找到权重文件: {[w.name for w in weights]}")
        else:
            print("未找到权重文件 (.pdparams)")
    else:
        print(f"输出目录未找到: {output_path}")

    # 4. 检查裁剪后的数据集
    print("\n[4/4] 检查裁剪数据集目录...")
    # 根据 main.py 中的逻辑，数据集路径应该是 项目id/train/数据版本/运行次数/标注类别/
    crop_dir = Path(f"/home/software/One2All-paddle/test_proj_full/train/dv1/1/bolt")
    if crop_dir.exists():
        img_dir = crop_dir / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.png")))
            print(f"裁剪后的训练图片数量: {count}")
        else:
            print(f"未找到裁剪图片目录: {img_dir}")
        
        for f_name in ["train.txt", "val.txt"]:
            if (crop_dir / f_name).exists():
                print(f"列表文件已生成: {f_name}")
            else:
                print(f"未找到列表文件: {f_name}")
    else:
        print(f"未找到裁剪根目录: {crop_dir}")

if __name__ == "__main__":
    # 确保 API 服务器正在运行
    # 这里我们假设用户已经在另一个终端运行了 uvicorn main:app --host 0.0.0.0 --port 8000
    # 或者我们可以尝试在这里启动它，但通常建议手动启动以便观察输出
    test_full_training()
