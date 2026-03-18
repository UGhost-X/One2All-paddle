#!/usr/bin/env python3
"""
测试推理可视化功能的脚本
"""
import requests
import base64
import json
from pathlib import Path


def test_visualization_api(image_path: str, api_url: str = "http://localhost:9397"):
    """
    测试简化版可视化推理API（直接套用模板坐标）

    Args:
        image_path: 测试图片路径
        api_url: API服务地址
    """
    # 读取测试图片
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    print(f"Testing SIMPLE visualization API with image: {image_path}")
    print(f"Image size: {len(image_bytes)} bytes")

    # 调用简化版可视化API
    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    data = {'return_viz': 'true'}

    try:
        response = requests.post(
            f"{api_url}/predict_simple",
            files=files,
            data=data,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Processing time: {result['processing_time']:.3f}s")
        print(f"Total detections: {result['total_detections']}")
        print(f"Anomaly count: {result['anomaly_count']}")
        print(f"Infer failed count: {result['infer_failed_count']}")
        print(f"Alignment strategy: {result['alignment_strategy']}")

        # 打印每个ROI的检测结果
        print("\n" + "-"*60)
        print("ROI DETAILS")
        print("-"*60)
        for i, r in enumerate(result['results']):
            status = "ANOMALY" if r.get('is_anomaly') else "NORMAL"
            print(f"ROI {i+1}: {r.get('category', '?'):10s} | "
                  f"{status:7s} | Score: {r.get('anomaly_score', 0):.3f} | "
                  f"Error: {r.get('error', 0):.3f}")

        # 保存可视化图像
        if result.get('visualization_b64'):
            viz_path = Path(image_path).stem + "_visualization_simple.png"
            with open(viz_path, 'wb') as f:
                f.write(base64.b64decode(result['visualization_b64']))
            print(f"\nVisualization saved to: {viz_path}")
            print(f"Visualization size: {len(result['visualization_b64'])} bytes (base64)")
        else:
            print("\nNo visualization generated")

        return result

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {api_url}")
        print("Please make sure the inference service is running.")
        return None
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_regular_api(image_path: str, api_url: str = "http://localhost:9397"):
    """
    测试普通推理API（不带可视化）

    Args:
        image_path: 测试图片路径
        api_url: API服务地址
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    print(f"\nTesting regular API with image: {image_path}")

    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}

    try:
        response = requests.post(
            f"{api_url}/predict",
            files=files,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        print(f"Regular API - Processing time: {result['processing_time']:.3f}s")
        print(f"Regular API - Anomalies: {result['anomaly_count']}/{result['total_detections']}")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    import sys

    # 默认测试图片路径
    default_image = "/home/software/One2All-paddle/test/demo-2-13.jpg"

    # 从命令行参数获取图片路径
    image_path = sys.argv[1] if len(sys.argv) > 1 else default_image
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9397"

    print("="*60)
    print("Inference Visualization Test")
    print("="*60)
    print(f"API URL: {api_url}")
    print(f"Image: {image_path}")
    print("="*60)

    # 先测试普通API
    test_regular_api(image_path, api_url)

    # 再测试可视化API
    print("\n" + "="*60)
    result = test_visualization_api(image_path, api_url)

    if result:
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
