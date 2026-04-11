#!/usr/bin/env python3
"""
基于模板匹配的多工件检测脚本 (极速优化版)
优化点：多线程并发、局部+全局NMS双重过滤、图像降采样加速、边缘均值填充(抗背景干扰)
"""

import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
import time

def rotate_image(image, angle):
    """
    旋转图像并调整边界框以确保图像不会被裁剪。
    优化：使用图像均值填充黑边，避免纯黑或纯白边对 CCOEFF_NORMED 相关系数计算造成严重干扰。
    """
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    
    # 计算新图像的边界尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 计算背景均值用于填充（减少边界突变造成的误匹配）
    mean_val = cv2.mean(image)[0]

    # 执行实际旋转
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=mean_val)
    return rotated

def match_single_angle(angle, scene_gray, template_gray, threshold):
    """
    单个角度的匹配任务，供多线程调用。包含局部 NMS 以减轻全局运算压力。
    """
    rotated_template = rotate_image(template_gray, angle)
    h, w = rotated_template.shape[:2]

    # 执行模板匹配
    result = cv2.matchTemplate(scene_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
    
    # 找到所有大于阈值的位置
    locations = np.where(result >= threshold)
    
    bboxes = []
    scores = []
    
    for pt in zip(*locations[::-1]):
        scores.append(float(result[pt[1], pt[0]]))
        bboxes.append([int(pt[0]), int(pt[1]), int(w), int(h)])
        
    # 局部 NMS 去重：直接在单角度内部过滤掉大量重叠框
    if len(bboxes) > 0:
        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=threshold, nms_threshold=0.3)
        if len(indices) > 0:
            filtered_bboxes = [bboxes[i] for i in indices.flatten()]
            filtered_scores = [scores[i] for i in indices.flatten()]
            return filtered_bboxes, filtered_scores
            
    return [], []

def detect_workpieces_by_template(image_path: str, template_path: str, 
                                 threshold: float = 0.65, 
                                 angle_step: int = 10,
                                 scale_factor: float = 0.5,
                                 output_path: str = None):
    """
    多线程极速多工件检测
    
    Args:
        scale_factor: 图像缩放比例 (如 0.5 表示缩小一半进行匹配，大幅提速，最后坐标会还原)
    """
    start_time = time.time()
    
    # 1. 读取图像
    scene_img = cv2.imread(image_path)
    template_img = cv2.imread(template_path)
    
    if scene_img is None or template_img is None:
        print("图像读取失败，请检查路径！")
        return []

    print(f"原图尺寸: {scene_img.shape[:2]}, 模板尺寸: {template_img.shape[:2]}")

    # 2. 转换为灰度图 & 降采样加速
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    if scale_factor != 1.0:
        scene_gray = cv2.resize(scene_gray, (0, 0), fx=scale_factor, fy=scale_factor)
        template_gray = cv2.resize(template_gray, (0, 0), fx=scale_factor, fy=scale_factor)
        print(f"启动降采样加速 ({scale_factor}x)，处理尺寸 -> 原图: {scene_gray.shape}, 模板: {template_gray.shape}")

    all_bboxes = []
    all_scores = []
    angles = list(range(0, 360, angle_step))

    print(f"正在进行多线程多角度模板匹配 (步长 {angle_step}°)... 请稍候...")

    # 3. 多线程并发匹配
    # 使用 ThreadPoolExecutor，根据系统自动分配最佳线程数 (OpenCV 的 cv2.matchTemplate 释放了 GIL)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交所有角度的匹配任务
        futures = {
            executor.submit(match_single_angle, angle, scene_gray, template_gray, threshold): angle 
            for angle in angles
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            bboxes, scores = future.result()
            all_bboxes.extend(bboxes)
            all_scores.extend(scores)

    if not all_bboxes:
        print("未找到匹配的工件，请尝试调低 threshold 参数。")
        return []

    # 4. 全局非极大值抑制 (Global NMS)
    print(f"各角度初步筛选结束，共聚合 {len(all_bboxes)} 个候选框，执行全局 NMS 去重...")
    indices = cv2.dnn.NMSBoxes(all_bboxes, all_scores, score_threshold=threshold, nms_threshold=0.2)
    
    final_bboxes = []
    result_image = scene_img.copy()
    
    if len(indices) > 0:
        for i, idx in enumerate(indices.flatten()):
            # 拿到降采样尺度下的框
            box = all_bboxes[idx]
            score = all_scores[idx]
            
            # 将坐标映射回原图的真实比例
            x = int(box[0] / scale_factor)
            y = int(box[1] / scale_factor)
            w = int(box[2] / scale_factor)
            h = int(box[3] / scale_factor)
            
            final_bboxes.append((x, y, w, h))
            
            # 绘制绿色检测框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # 添加标签和置信度得分
            label = f"OBJ_{i+1} ({score:.2f})"
            cv2.putText(result_image, label, (x, max(10, y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # print(f"检测到工件 {i+1}: 位置=({x},{y}), 置信度={score:.2f}")

    cost_time = time.time() - start_time
    print(f"\n======================================")
    print(f"最终检测到 {len(final_bboxes)} 个工件！")
    print(f"总耗时: {cost_time:.2f} 秒")
    print(f"======================================")

    # 5. 保存结果
    if output_path is None:
        output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_template_detected.jpg")
    
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")

    return final_bboxes

if __name__ == "__main__":
    image_path = "/home/software/One2All-paddle/test/multi-demo-2.jpg"
    template_path = "/home/software/One2All-paddle/product/12/train/5323a521/工件主体/1/raw_image_1_ann1.png"
    
    # 优化点说明：
    # threshold：由于优化了黑边干扰，阈值从你原本的 0.24 提升至 0.6~0.75 也可以匹配到，误检会大幅减少！
    # scale_factor：0.5 表示将图片长宽缩小一半寻找匹配，找出来的框再放大回原图大小。这会带来 4 倍速度提升，对精度几乎无损。如果需要绝对像素级精准，可改回 1.0。
    
    bboxes = detect_workpieces_by_template(
        image_path=image_path, 
        template_path=template_path,
        threshold=0.2,
        angle_step=10,
        scale_factor=1
    )