import sys
import os
import platform

# 跨平台：Windows需要添加OpenCV DLL路径

import numpy as np
import cv2
import math
import time

if platform.system() == "Windows":
    opencv_bin = os.environ.get("OPENCV_BIN", r"D:\software\opencv\build\x64\vc16\bin")
    if os.path.exists(opencv_bin):
        os.add_dll_directory(opencv_bin)

# 跨平台：动态确定模块路径
module_paths = [
    os.environ.get("SHAPE_MATCHING_PY_PATH"),
    os.path.join(os.path.dirname(__file__), "build"),
    os.path.join(os.path.dirname(__file__), "build", "Release"),
]

for path in module_paths:
    if path and path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)
        break

import shape_based_matching_py as sbm


prefix = os.path.join(os.path.dirname(__file__), "test")


def normalize_similarity(matches):
    """根据当前匹配结果动态归一化到 0-100
    以最高分为基准，其他分数相对计算百分比
    """
    if not matches:
        return matches
    max_sim = max(m['similarity'] for m in matches)
    if max_sim <= 0:
        return matches
    for m in matches:
        m['similarity_pct'] = m['similarity'] / max_sim * 100.0
    return matches


def compute_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms(matches, detector, infos, iou_threshold=0.1):
    if not matches:
        return []

    sorted_indices = sorted(
        range(len(matches)), key=lambda i: matches[i]["similarity"], reverse=True
    )

    suppressed = [False] * len(matches)
    keep = []

    for i in range(len(sorted_indices)):
        idx_i = sorted_indices[i]
        if suppressed[idx_i]:
            continue

        keep.append(idx_i)
        m_i = matches[idx_i]
        templ_i = detector.getTemplates("test", m_i["template_id"])

        rect_i = (m_i["x"], m_i["y"], templ_i["width"], templ_i["height"])

        for j in range(i + 1, len(sorted_indices)):
            idx_j = sorted_indices[j]
            if suppressed[idx_j]:
                continue

            m_j = matches[idx_j]
            templ_j = detector.getTemplates("test", m_j["template_id"])

            rect_j = (m_j["x"], m_j["y"], templ_j["width"], templ_j["height"])

            iou = compute_iou(rect_i, rect_j)
            if iou > iou_threshold:
                suppressed[idx_j] = True

    return keep


def test():
    detector = sbm.Detector(128, [4, 8])

    ids = ["test"]
    templ_path = os.path.join(prefix, "case4", "%s_templ.yaml")
    detector.readClasses(ids, templ_path)

    info_path = os.path.join(prefix, "case4", "test_info.yaml")
    infos = sbm.ShapeInfoProducer.loadInfos(info_path)

    W0, H0 = None, None
    for idx, info in enumerate(infos):
        if abs(info["angle"]) < 0.5:
            t0 = detector.getTemplates("test", idx)
            W0, H0 = t0["width"], t0["height"]
            break
    assert W0 is not None, "找不到 angle=0 的模板，無法取得原始尺寸"

    test_img_path = os.path.join(prefix, "case4", "multi-demo-4-h.jpg")
    test_img = cv2.imread(test_img_path)
    assert test_img is not None, f"check your img path: {test_img_path}"

    if len(test_img.shape) == 3:
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    stride = 16
    n = test_img.shape[0] // stride
    m = test_img.shape[1] // stride
    test_img = test_img[0 : stride * n, 0 : stride * m].copy()

    start_time = time.time()
    matches = detector.match(test_img, 50, ids)
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.3f}s")

    print(f"matches.size(): {len(matches)}")

    # 动态归一化相似度分数
    matches = normalize_similarity(matches)

    keep = nms(matches, detector, infos, iou_threshold=0.1)
    print(f"keep.size(): {len(keep)}")

    display_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    green = (0, 255, 0)

    
    for idx in keep:
        match = matches[idx]
        templ = detector.getTemplates("test", match["template_id"])
        angle = infos[match["template_id"]]["angle"]

        cx = match["x"] + templ["width"] / 2.0
        cy = match["y"] + templ["height"] / 2.0

        rrect = ((cx, cy), (W0, H0), -angle)
        box = cv2.boxPoints(rrect).astype(np.int32)

        for i in range(4):
            cv2.line(display_img, tuple(box[i]), tuple(box[(i + 1) % 4]), green, 2)

        for feat in templ["features"]:
            fx = feat["x"] + match["x"]
            fy = feat["y"] + match["y"]
            cv2.circle(display_img, (fx, fy), 2, green, -1)

        sim_pct = match.get('similarity_pct', 0)
        text = f"angle:{int(round(angle))} sim:{int(round(sim_pct))}"
        cv2.putText(display_img, text, (int(cx) - 30, int(cy) - 10),
                    cv2.FONT_HERSHEY_PLAIN, 3, green, 2)
        
    
    display_width = 1280
    scale = display_width / display_img.shape[1]
    display_height = int(display_img.shape[0] * scale)
    display_img = cv2.resize(display_img, (display_width, display_height))


    result_path = os.path.join(os.path.dirname(__file__), "result.jpg")
    cv2.imwrite(result_path, display_img)

    print("\ntest end")


if __name__ == "__main__":
    test()
