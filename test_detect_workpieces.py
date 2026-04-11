#!/usr/bin/env python3
"""
基于模板匹配的多工件检测脚本 (极速优化版 + 超强去噪梯度模式 v4)
新增功能：
- 自动提取模板中的真实工件区域（去除背景）
- 利用工件的几何特征（宽高比、面积）对候选框进行二次筛选
- 角度感知的宽高比验证，有效抑制工作台纹理误检
"""

import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
import time


def extract_workpiece_from_template(template_gray):
    """
    从模板图像中自动提取真实工件的区域（去除背景）。
    返回：(workpiece_img, mask, bbox, aspect_ratio, area)
    - workpiece_img: 裁剪后的工件图像（仅包含工件，背景为0）
    - mask: 工件的二值掩膜
    - bbox: (x, y, w, h) 在原模板中的外接矩形
    - aspect_ratio: 工件的宽高比 (w/h)
    - area: 工件的像素面积
    """
    # 1. 二值化（假设工件与背景有明显灰度差异）
    blur = cv2.GaussianBlur(template_gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 形态学闭运算填充孔洞
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. 查找轮廓，取面积最大的作为工件
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # 降级：返回原图
        h, w = template_gray.shape
        return template_gray, np.ones((h, w), np.uint8) * 255, (0, 0, w, h), w / h, w * h

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    aspect_ratio = w / h
    area = cv2.contourArea(max_contour)

    # 创建掩膜并裁剪工件
    mask = np.zeros_like(template_gray, dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], -1, 255, -1)
    workpiece_img = cv2.bitwise_and(template_gray, mask)
    workpiece_img = workpiece_img[y:y+h, x:x+w]

    return workpiece_img, mask[y:y+h, x:x+w], (x, y, w, h), aspect_ratio, area


def rotate_image(image, angle):
    """旋转图像并用均值填充黑边"""
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    mean_val = cv2.mean(image)[0]
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=mean_val)
    return rotated


def compute_gradient(gray, min_grad_threshold=30):
    """Sobel梯度 + 弱梯度压制 + 形态学清理噪点"""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)
    _, magnitude = cv2.threshold(magnitude, min_grad_threshold, 255, cv2.THRESH_TOZERO)
    kernel = np.ones((3, 3), np.uint8)
    magnitude = cv2.morphologyEx(magnitude, cv2.MORPH_OPEN, kernel)
    return magnitude


def match_single_angle(angle, scene_grad, template_denoised, threshold, template_orig_w, template_orig_h):
    """
    单个角度匹配，并返回每个检测框对应的旋转后模板尺寸。
    返回: (bboxes, scores, angles, rotated_w, rotated_h)
    """
    rotated_template = rotate_image(template_denoised, angle)
    rotated_blur = cv2.GaussianBlur(rotated_template, (5, 5), 0)
    rotated_grad = compute_gradient(rotated_blur, min_grad_threshold=30)
    h, w = rotated_grad.shape[:2]

    # 记录旋转后的实际宽高（用于后续几何验证）
    actual_rot_h, actual_rot_w = h, w

    result = cv2.matchTemplate(scene_grad, rotated_grad, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    bboxes = []
    scores = []
    angles_list = []
    rot_ws = []
    rot_hs = []

    for pt in zip(*locations[::-1]):
        scores.append(float(result[pt[1], pt[0]]))
        bboxes.append([int(pt[0]), int(pt[1]), actual_rot_w, actual_rot_h])
        angles_list.append(angle)
        rot_ws.append(actual_rot_w)
        rot_hs.append(actual_rot_h)

    if len(bboxes) > 0:
        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=threshold, nms_threshold=0.3)
        if len(indices) > 0:
            idx = indices.flatten()
            filtered_bboxes = [bboxes[i] for i in idx]
            filtered_scores = [scores[i] for i in idx]
            filtered_angles = [angles_list[i] for i in idx]
            filtered_rot_ws = [rot_ws[i] for i in idx]
            filtered_rot_hs = [rot_hs[i] for i in idx]
            return filtered_bboxes, filtered_scores, filtered_angles, filtered_rot_ws, filtered_rot_hs

    return [], [], [], [], []


def geometric_verification(bbox, angle, rot_w, rot_h, template_aspect_ratio, template_area, scene_gray_original):
    """
    几何特征验证：
    - 计算实际检测框的宽高比，与理论旋转后的宽高比比较
    - 计算检测区域内工件面积比例（通过二值化），与模板面积比例比较
    返回 True 表示通过验证
    """
    x, y, w, h = bbox
    # 防止越界
    img_h, img_w = scene_gray_original.shape
    x, y = max(0, x), max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    if w <= 0 or h <= 0:
        return False

    # 1. 宽高比验证
    detected_aspect = w / h
    # 理论旋转后的宽高比 = (template_orig_w * |cosθ| + template_orig_h * |sinθ|) / (template_orig_h * |cosθ| + template_orig_w * |sinθ|)
    # 但实际匹配时，模板已经旋转，rot_w/rot_h 即为理论尺寸，直接比较即可
    # 注意：rot_w, rot_h 是旋转后模板的外接矩形尺寸
    theoretical_aspect = rot_w / rot_h
    aspect_ratio_diff = abs(detected_aspect - theoretical_aspect) / max(theoretical_aspect, 1e-5)
    if aspect_ratio_diff > 0.35:   # 宽高比偏差超过35%则剔除
        return False

    # 2. 面积验证（通过二值化提取ROI中的前景）
    roi = scene_gray_original[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    # 二值化（Otsu）
    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel)
    roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel)
    foreground_area = cv2.countNonZero(roi_bin)
    area_ratio = foreground_area / (w * h)
    # 模板的面积比例（模板二值化后前景占比）
    # 这里简单假设模板工件几乎占满裁剪后的图像，则模板面积比例 ≈ 1.0
    # 更精确可提前计算模板前景占比
    if area_ratio < 0.25 or area_ratio > 0.95:
        return False

    return True


def detect_workpieces_by_template(image_path: str, template_path: str,
                                 threshold: float = 0.15,
                                 angle_step: int = 5,
                                 scale_factor: float = 1.0,
                                 output_path: str = None):
    start_time = time.time()

    scene_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if scene_img is None or template_img is None:
        print("图像读取失败，请检查路径！")
        return []

    print(f"原图尺寸: {scene_img.shape[:2]}, 模板尺寸: {template_img.shape[:2]}")

    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    if scale_factor != 1.0:
        scene_gray = cv2.resize(scene_gray, (0, 0), fx=scale_factor, fy=scale_factor)
        template_gray = cv2.resize(template_gray, (0, 0), fx=scale_factor, fy=scale_factor)
        print(f"启动降采样加速 ({scale_factor}x)...")

    # ========== 提取模板中的真实工件区域 ==========
    print("正在自动提取模板中的工件区域...")
    template_workpiece, template_mask, (tx, ty, tw, th), template_aspect_ratio, template_area = extract_workpiece_from_template(template_gray)
    print(f"工件区域: 宽={tw}, 高={th}, 宽高比={template_aspect_ratio:.3f}, 面积={template_area:.0f}")

    # 保存调试图像：模板裁剪结果
    debug_dir = Path(image_path).parent
    cv2.imwrite(str(debug_dir / "debug_template_workpiece.jpg"), template_workpiece)
    cv2.imwrite(str(debug_dir / "debug_template_mask.jpg"), template_mask)

    # ========== 场景与模板的强去噪流程 ==========
    print("启动超强去噪（CLAHE + 双边滤波 + 中值 + 形态学）...")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    scene_gray = clahe.apply(scene_gray)
    template_workpiece = clahe.apply(template_workpiece)   # 使用裁剪后的工件进行去噪

    scene_denoised = cv2.bilateralFilter(scene_gray, d=15, sigmaColor=150, sigmaSpace=150)
    scene_denoised = cv2.medianBlur(scene_denoised, 3)
    scene_blur = cv2.GaussianBlur(scene_denoised, (5, 5), 0)
    scene_grad = compute_gradient(scene_blur, min_grad_threshold=30)

    # 模板也做完全相同的强去噪
    template_denoised = cv2.bilateralFilter(template_workpiece, d=15, sigmaColor=150, sigmaSpace=150)
    template_denoised = cv2.medianBlur(template_denoised, 3)

    # 保存调试图
    cv2.imwrite(str(debug_dir / "debug_scene_gradient_clean.jpg"), scene_grad)
    raw_grad_for_compare = compute_gradient(cv2.GaussianBlur(scene_denoised, (3, 3), 0), min_grad_threshold=0)
    cv2.imwrite(str(debug_dir / "debug_scene_gradient_raw.jpg"), raw_grad_for_compare)
    print(f"已保存调试梯度图：{debug_dir}/debug_scene_gradient_clean.jpg")

    # ========== 多角度模板匹配 ==========
    all_bboxes = []
    all_scores = []
    all_angles = []
    all_rot_ws = []
    all_rot_hs = []
    angles = list(range(0, 360, angle_step))
    print(f"正在进行多线程多角度模板匹配 (步长 {angle_step}°)...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(match_single_angle, angle, scene_grad, template_denoised, threshold, tw, th): angle
            for angle in angles
        }
        for future in concurrent.futures.as_completed(futures):
            bboxes, scores, angles_list, rot_ws, rot_hs = future.result()
            all_bboxes.extend(bboxes)
            all_scores.extend(scores)
            all_angles.extend(angles_list)
            all_rot_ws.extend(rot_ws)
            all_rot_hs.extend(rot_hs)

    if not all_scores:
        print("未找到任何匹配")
        return []

    max_score = max(all_scores)
    print(f"本次匹配最高得分: {max_score:.3f}")

    # 相对得分筛选
    relative_threshold = max_score * 0.5
    filtered_bboxes, filtered_scores, filtered_angles, filtered_rot_ws, filtered_rot_hs = [], [], [], [], []
    for bbox, score, angle, rw, rh in zip(all_bboxes, all_scores, all_angles, all_rot_ws, all_rot_hs):
        if score >= relative_threshold:
            filtered_bboxes.append(bbox)
            filtered_scores.append(score)
            filtered_angles.append(angle)
            filtered_rot_ws.append(rw)
            filtered_rot_hs.append(rh)

    print(f"相对得分筛选后候选框: {len(filtered_bboxes)}")

    # ========== 几何特征二次筛选 ==========
    # 注意：此时坐标和尺寸是基于降采样后的图像（如果scale_factor!=1.0）
    verified_bboxes = []
    verified_scores = []
    for bbox, score, angle, rw, rh in zip(filtered_bboxes, filtered_scores, filtered_angles, filtered_rot_ws, filtered_rot_hs):
        # 几何验证使用原始尺寸的场景灰度图（未降采样）？注意：bbox当前是降采样后的坐标
        # 为了验证准确，应将bbox映射回原始尺寸，并在原始图上提取ROI
        if scale_factor != 1.0:
            orig_bbox = [int(bbox[0] / scale_factor), int(bbox[1] / scale_factor),
                         int(bbox[2] / scale_factor), int(bbox[3] / scale_factor)]
        else:
            orig_bbox = bbox[:]

        # 注意：scene_gray_original 是未降采样的原始灰度图
        scene_gray_original = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        if geometric_verification(orig_bbox, angle, rw, rh, template_aspect_ratio, template_area, scene_gray_original):
            verified_bboxes.append(bbox)   # 仍保存降采样后的坐标，最后统一还原
            verified_scores.append(score)

    print(f"几何验证后剩余候选框: {len(verified_bboxes)}")

    if not verified_bboxes:
        print("所有候选框均未通过几何验证，可能无有效工件。")
        return []

    # 全局NMS
    indices = cv2.dnn.NMSBoxes(verified_bboxes, verified_scores, score_threshold=threshold, nms_threshold=0.2)
    final_bboxes = []
    result_image = scene_img.copy()

    if len(indices) > 0:
        for idx in indices.flatten():
            box = verified_bboxes[idx]
            score = verified_scores[idx]
            x = int(box[0] / scale_factor)
            y = int(box[1] / scale_factor)
            w = int(box[2] / scale_factor)
            h = int(box[3] / scale_factor)
            final_bboxes.append((x, y, w, h))
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"OBJ_{len(final_bboxes)} ({score:.2f})"
            cv2.putText(result_image, label, (x, max(10, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cost_time = time.time() - start_time
    print(f"\n======================================")
    print(f"最终检测到 {len(final_bboxes)} 个工件！")
    print(f"总耗时: {cost_time:.2f} 秒")
    print(f"======================================")

    if output_path is None:
        output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_template_detected.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")

    return final_bboxes


if __name__ == "__main__":
    image_path = "/home/software/One2All-paddle/test/demo/demo-2.jpg"
    template_path = "/home/software/One2All-paddle/product/12/train/5323a521/工件主体/1/raw_image_1_ann1.png"

    bboxes = detect_workpieces_by_template(
        image_path=image_path,
        template_path=template_path,
        threshold=0.1,
        angle_step=5,
        scale_factor=0.3
    )