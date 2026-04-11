#!/usr/bin/env python3
"""
基于模板匹配的多工件检测脚本 (极速优化版 + 超强去噪梯度模式 v3)
优化点（本次针对 debug 图纹理仍残留 + 得分卡在0.192）：
- 模板也使用完全一致的强去噪流程（双边+中值），解决模板/场景梯度强度不匹配问题
- compute_gradient 增加形态学 OPEN 清理残留噪点
- min_grad_threshold 提升至 30（更狠压制工作台纹理）
- 双边滤波参数加强（d=15），专门针对你图中那种细碎均匀纹理
- 预期效果：最高得分提升至 0.55~0.80+，背景几乎纯黑，误检消失
"""

import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
import time


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
    
    # 压制工作台弱纹理
    _, magnitude = cv2.threshold(magnitude, min_grad_threshold, 255, cv2.THRESH_TOZERO)
    
    # 形态学 OPEN 去除孤立噪点（关键！）
    kernel = np.ones((3, 3), np.uint8)
    magnitude = cv2.morphologyEx(magnitude, cv2.MORPH_OPEN, kernel)
    
    return magnitude


def extract_edge_features(gray_img, canny_low=50, canny_high=150):
    """
    提取图像的边缘特征
    返回边缘图、边缘点数量和边缘方向直方图
    """
    # Canny边缘检测
    edges = cv2.Canny(gray_img, canny_low, canny_high)
    
    # 计算边缘点数量
    edge_count = np.count_nonzero(edges)
    
    # 计算边缘方向直方图（使用Sobel）
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度方向
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # 只在边缘点上计算方向直方图
    edge_points = edges > 0
    if np.count_nonzero(edge_points) > 0:
        edge_directions = direction[edge_points]
        edge_magnitudes = magnitude[edge_points]
        
        # 创建8个方向区间的直方图
        hist, _ = np.histogram(edge_directions, bins=8, range=(-180, 180), weights=edge_magnitudes)
        # 归一化
        if hist.sum() > 0:
            hist = hist / hist.sum()
    else:
        hist = np.zeros(8)
    
    return edges, edge_count, hist


def compute_edge_similarity(template_gray, candidate_gray, template_mask=None):
    """
    计算模板和候选区域的边缘相似度
    返回相似度得分 (0-1)
    """
    # 提取边缘特征
    _, template_edge_count, template_hist = extract_edge_features(template_gray)
    _, candidate_edge_count, candidate_hist = extract_edge_features(candidate_gray)
    
    if template_edge_count == 0 or candidate_edge_count == 0:
        return 0.0
    
    # 1. 边缘数量比例相似度（边缘密度应该相似）
    template_area = template_gray.shape[0] * template_gray.shape[1]
    candidate_area = candidate_gray.shape[0] * candidate_gray.shape[1]
    
    template_density = template_edge_count / template_area
    candidate_density = candidate_edge_count / candidate_area
    
    # 密度相似度：越接近1越好
    density_ratio = min(template_density, candidate_density) / max(template_density, candidate_density) if max(template_density, candidate_density) > 0 else 0
    
    # 2. 边缘方向直方图相似度（使用余弦相似度）
    hist_sim = np.dot(template_hist, candidate_hist) / (np.linalg.norm(template_hist) * np.linalg.norm(candidate_hist) + 1e-6)
    
    # 3. Hu矩形状相似度（对旋转、缩放、平移不变）
    _, template_thresh = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, candidate_thresh = cv2.threshold(candidate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    template_moments = cv2.moments(template_thresh)
    candidate_moments = cv2.moments(candidate_thresh)
    
    if template_moments['m00'] > 0 and candidate_moments['m00'] > 0:
        template_hu = cv2.HuMoments(template_moments).flatten()
        candidate_hu = cv2.HuMoments(candidate_moments).flatten()
        
        # 对Hu矩取对数并归一化
        template_hu = -np.sign(template_hu) * np.log10(np.abs(template_hu) + 1e-10)
        candidate_hu = -np.sign(candidate_hu) * np.log10(np.abs(candidate_hu) + 1e-10)
        
        # 计算Hu矩距离
        hu_dist = np.linalg.norm(template_hu - candidate_hu)
        hu_sim = 1 / (1 + hu_dist)  # 转换为相似度
    else:
        hu_sim = 0.0
    
    # 综合相似度（加权平均）
    # 方向直方图最重要（对形状描述能力强），其次是Hu矩，最后是边缘密度
    similarity = 0.4 * hist_sim + 0.4 * hu_sim + 0.2 * density_ratio
    
    return similarity


def verify_candidate_by_edge_features(scene_gray, bbox, template_gray, min_edge_sim=0.5):
    """
    使用边缘特征验证候选区域是否真的是工件
    
    Args:
        scene_gray: 场景灰度图
        bbox: 候选框 [x, y, w, h]
        template_gray: 模板灰度图
        min_edge_sim: 最小边缘相似度阈值
    
    Returns:
        (is_valid, edge_similarity): 是否通过验证，以及边缘相似度得分
    """
    x, y, w, h = bbox
    h_img, w_img = scene_gray.shape[:2]
    
    # 确保边界有效
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    
    if w < 10 or h < 10:
        return False, 0.0
    
    # 提取候选区域
    candidate_region = scene_gray[y:y+h, x:x+w]
    
    # 将候选区域缩放到与模板相同大小进行比较
    template_h, template_w = template_gray.shape[:2]
    candidate_resized = cv2.resize(candidate_region, (template_w, template_h))
    
    # 计算边缘相似度
    edge_sim = compute_edge_similarity(template_gray, candidate_resized)
    
    is_valid = edge_sim >= min_edge_sim
    
    return is_valid, edge_sim


def match_single_angle(angle, scene_grad, template_denoised, threshold):
    """单个角度匹配（模板已提前去噪）"""
    rotated_template = rotate_image(template_denoised, angle)
    
    # 模板也走完全一致的轻模糊 + 梯度流程
    rotated_blur = cv2.GaussianBlur(rotated_template, (5, 5), 0)
    rotated_grad = compute_gradient(rotated_blur, min_grad_threshold=30)
    
    h, w = rotated_grad.shape[:2]

    result = cv2.matchTemplate(scene_grad, rotated_grad, cv2.TM_CCOEFF_NORMED)
    
    locations = np.where(result >= threshold)
    
    bboxes = []
    scores = []
    
    for pt in zip(*locations[::-1]):
        scores.append(float(result[pt[1], pt[0]]))
        bboxes.append([int(pt[0]), int(pt[1]), int(w), int(h)])
        
    if len(bboxes) > 0:
        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=threshold, nms_threshold=0.3)
        if len(indices) > 0:
            filtered_bboxes = [bboxes[i] for i in indices.flatten()]
            filtered_scores = [scores[i] for i in indices.flatten()]
            return filtered_bboxes, filtered_scores
            
    return [], []


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

    # ==================== 【超强去噪流程】 ====================
    print("启动超强去噪（CLAHE + 双边滤波 + 中值 + 形态学）...")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    scene_gray = clahe.apply(scene_gray)
    template_gray = clahe.apply(template_gray)

    # 场景强去噪（针对你图中均匀细碎纹理）
    scene_denoised = cv2.bilateralFilter(scene_gray, d=15, sigmaColor=150, sigmaSpace=150)
    scene_denoised = cv2.medianBlur(scene_denoised, 3)
    scene_blur = cv2.GaussianBlur(scene_denoised, (5, 5), 0)
    scene_grad = compute_gradient(scene_blur, min_grad_threshold=30)

    # 模板也提前做完全一致的强去噪（关键！解决得分低的根源）
    template_denoised = cv2.bilateralFilter(template_gray, d=15, sigmaColor=150, sigmaSpace=150)
    template_denoised = cv2.medianBlur(template_denoised, 3)

    # 保存调试图（清理后 + 清理前）
    debug_dir = Path(image_path).parent
    cv2.imwrite(str(debug_dir / "debug_scene_gradient_clean.jpg"), scene_grad)
    raw_grad_for_compare = compute_gradient(cv2.GaussianBlur(scene_denoised, (3, 3), 0), min_grad_threshold=0)
    cv2.imwrite(str(debug_dir / "debug_scene_gradient_raw.jpg"), raw_grad_for_compare)
    
    print(f"已保存调试梯度图：")
    print(f"   • 清理后（强烈建议查看）: {debug_dir}/debug_scene_gradient_clean.jpg")
    print(f"   • 清理前（对比用）     : {debug_dir}/debug_scene_gradient_raw.jpg")
    print("   背景应接近纯黑，工件边缘应清晰连续！")
    # ===========================================================================

    all_bboxes = []
    all_scores = []
    angles = list(range(0, 360, angle_step))

    print(f"正在进行多线程多角度模板匹配 (步长 {angle_step}°，超强去噪梯度模式)...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(match_single_angle, angle, scene_grad, template_denoised, threshold): angle 
            for angle in angles
        }
        
        for future in concurrent.futures.as_completed(futures):
            bboxes, scores = future.result()
            all_bboxes.extend(bboxes)
            all_scores.extend(scores)

    if not all_scores:
        print("未找到任何匹配")
        return []

    max_score = max(all_scores)
    print(f"本次匹配最高得分: {max_score:.3f}  ← 关键指标")

    # 相对得分筛选：保留得分 >= max_score × 0.6 的候选框
    # 原理：多工件场景下所有工件得分都接近max，单工件场景下噪声远低于max×0.6
    relative_threshold = max_score * 0.5
    filtered_bboxes = []
    filtered_scores = []
    for bbox, score in zip(all_bboxes, all_scores):
        if score >= relative_threshold:
            filtered_bboxes.append(bbox)
            filtered_scores.append(score)

    print(f"相对得分筛选：max={max_score:.3f}, 阈值={relative_threshold:.3f}, 筛选后候选框={len(filtered_bboxes)}")

    all_bboxes = filtered_bboxes
    all_scores = filtered_scores

    if not all_bboxes:
        print("未找到匹配的工件，请尝试调低 threshold。")
        return []

    print(f"各角度初步筛选结束，共聚合 {len(all_bboxes)} 个候选框，执行全局 NMS...")
    indices = cv2.dnn.NMSBoxes(all_bboxes, all_scores, score_threshold=threshold, nms_threshold=0.2)
    
    nms_bboxes = []
    nms_scores = []
    
    if len(indices) > 0:
        for idx in indices.flatten():
            box = all_bboxes[idx]
            score = all_scores[idx]
            
            # 转换回原始尺寸
            x = int(box[0] / scale_factor)
            y = int(box[1] / scale_factor)
            w = int(box[2] / scale_factor)
            h = int(box[3] / scale_factor)
            
            nms_bboxes.append([x, y, w, h])
            nms_scores.append(score)
    
    # ==================== 【边缘特征验证】 ====================
    print(f"NMS后剩余 {len(nms_bboxes)} 个候选框，开始边缘特征验证...")
    
    # 使用原始未降采样的灰度图进行边缘验证
    scene_gray_original = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    template_gray_original = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    final_bboxes = []
    final_scores = []
    edge_similarities = []
    
    min_edge_sim = 0.4  # 边缘相似度阈值，可根据实际情况调整
    
    for i, (bbox, score) in enumerate(zip(nms_bboxes, nms_scores)):
        is_valid, edge_sim = verify_candidate_by_edge_features(
            scene_gray_original, 
            bbox, 
            template_gray_original, 
            min_edge_sim=min_edge_sim
        )
        
        print(f"  候选框 {i+1}: 模板匹配得分={score:.3f}, 边缘相似度={edge_sim:.3f}, {'通过' if is_valid else '拒绝'}")
        
        if is_valid:
            final_bboxes.append(tuple(bbox))
            final_scores.append(score)
            edge_similarities.append(edge_sim)
    
    print(f"边缘特征验证后剩余 {len(final_bboxes)} 个有效工件")
    # ===========================================================================
    
    result_image = scene_img.copy()
    
    for i, ((x, y, w, h), score, edge_sim) in enumerate(zip(final_bboxes, final_scores, edge_similarities)):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label = f"OBJ_{i+1} (TM:{score:.2f}, ED:{edge_sim:.2f})"
        cv2.putText(result_image, label, (x, max(10, y - 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
    # image_path = "/home/software/One2All-paddle/test/multi-demo-2.jpg"
    image_path = "/home/software/One2All-paddle/test/demo/demo-1.jpg"
    template_path = "/home/software/One2All-paddle/product/12/train/5323a521/工件主体/1/raw_image_1_ann1.png"
    
    bboxes = detect_workpieces_by_template(
        image_path=image_path, 
        template_path=template_path,
        threshold=0.1,
        angle_step=5,
        scale_factor=0.3
    )