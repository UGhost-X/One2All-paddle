#!/usr/bin/env python3
"""
测试脚本 - 验证 single-workpiece.py 推理服务的 API 端点
服务端口: 9795

支持接口:
- /health: 健康检查
- /predict: 单工件检测
- /predict_multiple: 多工件检测
"""

import requests
import base64
import cv2
import json
from pathlib import Path
from typing import Optional, Dict, Any


def encode_image_to_base64(image_path: str) -> str:
    """将图像编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_health_check(api_url: str = "http://localhost:9795"):
    """测试健康检查端点"""
    print("=" * 70)
    print("测试 /health 端点")
    print("=" * 70)
    
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"服务状态: {result.get('status', 'unknown')}")
            print(f"服务 ID: {result.get('service_id', 'unknown')}")
            print(f"项目 ID: {result.get('project_id', 'unknown')}")
            print(f"任务 UUID: {result.get('task_uuid', 'unknown')}")
            print(f"已加载模型: {result.get('loaded_models', [])}")
            print(f"GPU 内存: {result.get('gpu_memory', {})}")
            return True
        else:
            print(f"错误: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到服务 - {api_url}")
        print("请确保服务已启动: python3 single-workpiece.py")
        return False
    except Exception as e:
        print(f"请求错误: {e}")
        return False


def test_predict_api(
    image_path: str,
    api_url: str = "http://localhost:9795/predict",
    return_viz: bool = True
):
    """
    测试单工件 predict 端点

    Args:
        image_path: 测试图像路径
        api_url: predict 端点 URL
        return_viz: 是否返回可视化图像
    """
    print("\n" + "=" * 70)
    print("测试 /predict 端点 (单工件检测)")
    print("=" * 70)
    print(f"图像路径: {image_path}")
    print(f"API URL: {api_url}")
    print(f"返回可视化: {return_viz}")
    print("-" * 70)

    if not Path(image_path).exists():
        print(f"错误: 图像不存在 - {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 - {image_path}")
        return None

    print(f"图像尺寸: {img.shape}")
    print()

    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/png")}
            data = {"return_viz": str(return_viz).lower()}

            print("发送请求...")
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=120
            )
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到服务 - {api_url}")
        print("请确保服务已启动: python3 single-workpiece.py")
        return None
    except Exception as e:
        print(f"请求错误: {e}")
        return None

    print(f"响应状态码: {response.status_code}")
    print()

    if response.status_code != 200:
        print(f"错误响应: {response.text}")
        return None

    result = response.json()

    print("=" * 70)
    print("响应结果摘要")
    print("=" * 70)

    print(f"\n【基本信息】")
    print(f"  处理时间: {result.get('processing_time', 0):.3f} 秒")
    print(f"  对齐策略: {result.get('alignment_strategy', 'none')}")

    print(f"\n【检测结果】")
    print(f"  检测总数: {result.get('total_detections', 0)}")
    print(f"  异常数量: {result.get('anomaly_count', 0)}")
    print(f"  失败数量: {result.get('infer_failed_count', 0)}")

    results = result.get('results', [])
    if results:
        print(f"\n  ROI 检测结果:")
        for i, r in enumerate(results[:10]):
            status = "异常" if r.get('is_anomaly') else "正常"
            failed = " [失败]" if r.get('infer_failed') else ""
            print(f"    {i+1}. {r.get('category', '?'):10s} | "
                  f"{status:6s} | "
                  f"分数: {r.get('anomaly_score', 0):.3f}{failed}")

        if len(results) > 10:
            print(f"      ... 还有 {len(results) - 10} 个 ROI")

    if return_viz:
        viz_b64 = result.get('visualization_b64')
        if viz_b64:
            output_dir = Path("/home/software/One2All-paddle/test")
            output_dir.mkdir(exist_ok=True)
            viz_path = output_dir / "single_workpiece_viz.png"
            with open(viz_path, "wb") as f:
                f.write(base64.b64decode(viz_b64))
            print(f"\n  可视化图像已保存: {viz_path}")
        else:
            print("\n  警告: 响应中没有可视化数据")

    print("\n" + "=" * 70)
    print("单工件测试完成")
    print("=" * 70)

    return result


def test_predict_multiple_api(
    image_path: str,
    api_url: str = "http://localhost:9795/predict",
    return_viz: bool = True
):
    """
    测试多工件 predict_multiple 端点

    Args:
        image_path: 测试图像路径
        api_url: predict_multiple 端点 URL
        return_viz: 是否返回可视化图像
    """
    print("\n" + "=" * 70)
    print("测试 /predict_multiple 端点 (多工件检测)")
    print("=" * 70)
    print(f"图像路径: {image_path}")
    print(f"API URL: {api_url}")
    print(f"返回可视化: {return_viz}")
    print("-" * 70)

    if not Path(image_path).exists():
        print(f"错误: 图像不存在 - {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 - {image_path}")
        return None

    print(f"图像尺寸: {img.shape}")
    print()

    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/png")}
            data = {"return_viz": str(return_viz).lower()}

            print("发送请求...")
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=180  # 多工件检测超时时间更长
            )
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到服务 - {api_url}")
        print("请确保服务已启动: python3 single-workpiece.py")
        return None
    except Exception as e:
        print(f"请求错误: {e}")
        return None

    print(f"响应状态码: {response.status_code}")
    print()

    if response.status_code != 200:
        print(f"错误响应: {response.text}")
        return None

    result = response.json()

    print("=" * 70)
    print("响应结果摘要")
    print("=" * 70)

    print(f"\n【基本信息】")
    print(f"  处理时间: {result.get('processing_time', 0):.3f} 秒")
    print(f"  工件总数: {result.get('total_workpieces', 0)}")
    print(f"  异常总数: {result.get('total_anomalies', 0)}")

    workpiece_results = result.get('workpiece_results', [])
    if workpiece_results:
        print(f"\n【各工件详情】")
        
        for wp in workpiece_results:
            wp_id = wp.get('workpiece_id', '?')
            bbox = wp.get('bbox', [])
            anomaly_count = wp.get('anomaly_count', 0)
            strategy = wp.get('strategy', 'none')
            error = wp.get('error')
            
            print(f"\n  --- 工件 {wp_id} ---")
            print(f"    工件位置 (bbox): {bbox}")
            print(f"    对齐策略: {strategy}")
            print(f"    异常数量: {anomaly_count}")
            
            if error:
                print(f"    错误信息: {error}")
                continue
            
            results = wp.get('results', [])
            if results:
                print(f"    ROI 检测结果 ({len(results)} 个):")
                for i, r in enumerate(results[:5]):
                    status = "异常" if r.get('is_anomaly') else "正常"
                    failed = " [失败]" if r.get('infer_failed') else ""
                    print(f"      {i+1}. {r.get('category', '?'):10s} | "
                          f"{status:6s} | "
                          f"分数: {r.get('anomaly_score', 0):.3f}{failed}")
                
                if len(results) > 5:
                    print(f"      ... 还有 {len(results) - 5} 个 ROI")

    if return_viz:
        viz_b64 = result.get('visualization_b64')
        if viz_b64:
            output_dir = Path("/home/software/One2All-paddle/test")
            output_dir.mkdir(exist_ok=True)
            viz_path = output_dir / "multi_workpiece_viz.png"
            with open(viz_path, "wb") as f:
                f.write(base64.b64decode(viz_b64))
            print(f"\n  可视化图像已保存: {viz_path}")
        else:
            print("\n  警告: 响应中没有可视化数据")

    print("\n" + "=" * 70)
    print("多工件测试完成")
    print("=" * 70)

    return result


def test_with_base64(
    image_path: str, 
    api_url: str = "http://localhost:9795/predict",
    use_multiple: bool = False
):
    """使用 base64 编码测试"""
    print("\n" + "=" * 70)
    print("测试 Base64 编码方式上传")
    if use_multiple:
        print("接口: /predict_multiple")
    else:
        print("接口: /predict")
    print("=" * 70)

    b64_data = encode_image_to_base64(image_path)
    print(f"Base64 编码长度: {len(b64_data)} 字符")

    data = {
        "image": f"data:image/png;base64,{b64_data}",
        "return_viz": "false"
    }

    try:
        print("发送请求...")
        response = requests.post(
            api_url,
            data=data,
            timeout=180
        )

        print(f"响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"处理时间: {result.get('processing_time', 0):.3f} 秒")
            
            if use_multiple:
                print(f"检测到工件数: {result.get('total_workpieces', 0)}")
                print(f"异常总数: {result.get('total_anomalies', 0)}")
            else:
                print(f"检测总数: {result.get('total_detections', 0)}")
                print(f"异常数量: {result.get('anomaly_count', 0)}")
            
            print("Base64 方式测试成功!")
        else:
            print(f"错误: {response.text}")

    except Exception as e:
        print(f"请求错误: {e}")


def compare_single_vs_multiple(image_path: str):
    """对比单工件和多工件检测接口的结果"""
    print("\n" + "=" * 70)
    print("对比单工件 vs 多工件检测")
    print("=" * 70)
    
    if not test_health_check():
        return
    
    # 单工件检测
    # print("\n" + "-" * 70)
    # print("1. 单工件检测 (/predict)")
    # print("-" * 70)
    # single_result = test_predict_api(image_path, return_viz=False)
    
    # 多工件检测
    print("\n" + "-" * 70)
    print("2. 多工件检测 (/predict)")
    print("-" * 70)
    multi_result = test_predict_multiple_api(image_path, return_viz=True)
    
    # 对比结果
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)
    
    if single_result and multi_result:
        single_time = single_result.get('processing_time', 0)
        multi_time = multi_result.get('processing_time', 0)
        
        print(f"\n处理时间:")
        print(f"  单工件: {single_time:.3f} 秒")
        print(f"  多工件: {multi_time:.3f} 秒")
        print(f"  时间差: {abs(multi_time - single_time):.3f} 秒")
        
        print(f"\n检测结果:")
        print(f"  单工件 - 异常数: {single_result.get('anomaly_count', 0)}")
        print(f"  多工件 - 工件数: {multi_result.get('total_workpieces', 0)}, "
              f"异常数: {multi_result.get('total_anomalies', 0)}")


if __name__ == "__main__":

    default_image = "/home/software/One2All-paddle/test/multi-demo-1-h.jpg"
    
    
    test_predict_multiple_api(default_image, return_viz=True)

