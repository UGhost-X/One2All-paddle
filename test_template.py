#!/usr/bin/env python3
"""
测试模板匹配功能
"""
import requests
import cv2
import numpy as np
import base64
import json

BASE_URL = "http://localhost:8001"

def create_test_image(width=400, height=300, color=(100, 150, 200)):
    """创建一个测试图片"""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # 添加一些特征点（角点）
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(img, (250, 100), (350, 200), (0, 0, 0), -1)
    cv2.circle(img, (200, 250), 30, (255, 0, 0), -1)
    return img

def image_to_base64(img):
    """图片转base64"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def test_list_templates():
    """测试列出模板"""
    print("\n=== 测试列出模板 ===")
    response = requests.get(f"{BASE_URL}/templates")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_create_template():
    """测试创建模板"""
    print("\n=== 测试创建模板 ===")

    # 创建测试模板图片
    template_img = create_test_image(400, 300, (100, 150, 200))
    template_base64 = image_to_base64(template_img)

    data = {
        "name": "test_workpiece",
        "image_base64": template_base64
    }

    response = requests.post(f"{BASE_URL}/templates", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_get_template_info(template_name):
    """测试获取模板信息"""
    print(f"\n=== 测试获取模板信息: {template_name} ===")
    response = requests.get(f"{BASE_URL}/templates/{template_name}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_match_template(template_name):
    """测试模板匹配"""
    print(f"\n=== 测试模板匹配: {template_name} ===")

    # 创建一个旋转/平移后的测试图片
    test_img = create_test_image(400, 300, (100, 150, 200))
    # 稍微旋转图片
    center = (200, 150)
    angle = 15
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    test_img = cv2.warpAffine(test_img, M, (400, 300))

    test_base64 = image_to_base64(test_img)

    data = {
        "image_base64": test_base64
    }

    response = requests.post(f"{BASE_URL}/templates/{template_name}/match", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    # 如果匹配成功，保存对齐后的图片
    if result.get("success") and "aligned_image" in result:
        aligned_data = base64.b64decode(result["aligned_image"])
        aligned_nparr = np.frombuffer(aligned_data, np.uint8)
        aligned_img = cv2.imdecode(aligned_nparr, cv2.IMREAD_COLOR)
        cv2.imwrite("aligned_result.png", aligned_img)
        print("对齐后的图片已保存到: aligned_result.png")

    return result

def test_train_with_template():
    """测试带模板的训练"""
    print("\n=== 测试带模板的训练 ===")

    # 创建测试图片
    img1 = create_test_image(400, 300, (100, 150, 200))
    img2 = create_test_image(400, 300, (110, 160, 210))  # 稍微不同的颜色

    # 添加一些bbox标注区域
    cv2.rectangle(img1, (80, 80), (120, 120), (255, 0, 0), 2)
    cv2.rectangle(img2, (280, 120), (320, 160), (0, 255, 0), 2)

    # 构建训练请求
    train_data = {
        "images": [image_to_base64(img1), image_to_base64(img2)],
        "coco_data": {
            "images": [
                {"id": 1, "width": 400, "height": 300, "file_name": "test1.png"},
                {"id": 2, "width": 400, "height": 300, "file_name": "test2.png"}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [80, 80, 40, 40],
                    "label": "screw"
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [280, 120, 40, 40],
                    "label": "hole"
                }
            ],
            "categories": [
                {"id": 1, "name": "screw"},
                {"id": 2, "name": "hole"}
            ]
        },
        "base_path": "/tmp/test",
        "project_id": "test_project",
        "model_name": "STFPM",
        "train_iters": 10,  # 只训练10次迭代用于测试
        "batch_size": 2,
        "template_name": "test_workpiece"  # 使用刚才创建的模板
    }

    response = requests.post(f"{BASE_URL}/train/anomaly", json=train_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result

def test_delete_template(template_name):
    """测试删除模板"""
    print(f"\n=== 测试删除模板: {template_name} ===")
    response = requests.delete(f"{BASE_URL}/templates/{template_name}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

if __name__ == "__main__":
    print("开始测试模板匹配功能...")

    # 1. 列出当前模板
    test_list_templates()

    # 2. 创建模板
    create_result = test_create_template()

    if create_result.get("success"):
        template_name = create_result.get("template_name")

        # 3. 获取模板信息
        test_get_template_info(template_name)

        # 4. 测试模板匹配
        test_match_template(template_name)

        # 5. 测试训练（带模板对齐）
        train_result = test_train_with_template()

        # 6. 再次列出模板
        test_list_templates()

        # 7. 删除模板（可选）
        # test_delete_template(template_name)

    print("\n测试完成!")
