#!/usr/bin/env python3
"""
测试推理可视化功能的脚本
用于测试 service_svc_1774311522_9f1c0e91 的可视化功能
"""
import requests
import base64
import json
from pathlib import Path


def test_visualization_api(image_path: str, api_url: str = "http://localhost:9395"):
    """
    测试可视化推理API

    Args:
        image_path: 测试图片路径
        api_url: API服务地址
    """
    # 读取测试图片
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    print(f"Testing visualization API with image: {image_path}")
    print(f"Image size: {len(image_bytes)} bytes")

    # 调用可视化API
    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    data = {'return_viz': 'true'}

    try:
        response = requests.post(
            f"{api_url}/predict",
            files=files,
            data=data,
            timeout=120
        )

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # 保存可视化结果
            if result.get('visualization_b64'):
                viz_bytes = base64.b64decode(result['visualization_b64'])
                output_path = Path("visualization_result.png")
                with open(output_path, 'wb') as f:
                    f.write(viz_bytes)
                print(f"✓ Visualization saved to: {output_path.absolute()}")
                print(f"  Size: {len(viz_bytes)} bytes")
            else:
                print("✗ No visualization in response")

            # 打印推理结果摘要
            print(f"\n=== Inference Results ===")
            print(f"Total detections: {result.get('total_detections', 0)}")
            print(f"Anomaly count: {result.get('anomaly_count', 0)}")
            print(f"Infer failed count: {result.get('infer_failed_count', 0)}")
            print(f"Alignment strategy: {result.get('alignment_strategy', 'none')}")
            print(f"Processing time: {result.get('processing_time', 0):.3f}s")

            # 打印每个ROI的结果
            print(f"\n=== ROI Details ===")
            for i, roi_result in enumerate(result.get('results', [])):
                is_anomaly = roi_result.get('is_anomaly', False)
                status = "ANOMALY" if is_anomaly else "NORMAL"
                score = roi_result.get('anomaly_score', 0)
                category = roi_result.get('category', 'unknown')
                print(f"  ROI {i+1}: {category:10s} | {status:7s} | Score: {score:.3f}")

            return True
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to API at {api_url}")
        print(f"  Please ensure the inference service is running")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_regular_api(image_path: str, api_url: str = "http://localhost:9395"):
    """
    测试普通推理API（不带可视化）

    Args:
        image_path: 测试图片路径
        api_url: API服务地址
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    print(f"\nTesting regular API (no visualization)")

    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    # 不提供 return_viz 参数，默认为 False

    try:
        response = requests.post(
            f"{api_url}/predict",
            files=files,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Regular API works")
            print(f"  Detections: {result.get('total_detections', 0)}")
            print(f"  Has visualization: {result.get('visualization_b64') is not None}")
            return True
        else:
            print(f"✗ Regular API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_health_check(api_url: str = "http://localhost:9395"):
    """测试健康检查接口"""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✓ Health check: OK")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


if __name__ == "__main__":
    import sys

    # 默认测试图片路径
    default_image = "/home/software/One2All-paddle/test/demo-2-18.jpg"

    # 从命令行参数获取图片路径
    image_path = sys.argv[1] if len(sys.argv) > 1 else default_image
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9395"

    print("="*60)
    print("Inference Service Visualization Test")
    print("="*60)
    print(f"API URL: {api_url}")
    print(f"Image: {image_path}")
    print("="*60)

    # 先测试健康检查
    if not test_health_check(api_url):
        print("\n✗ Service is not available. Please start the service first.")
        sys.exit(1)

    # 测试普通API
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
