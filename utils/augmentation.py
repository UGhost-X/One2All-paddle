import cv2
import numpy as np
import random
import copy
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def segmentation_to_rbbox(segmentation: List[float]) -> List[float]:
    """
    将 segmentation 4点格式转换为 rbbox [cx, cy, width, height, angle]
    :param segmentation: [x1, y1, x2, y2, x3, y3, x4, y4] 4个角点
    :return: [cx, cy, width, height, angle] 旋转框
    """
    if not segmentation or len(segmentation) < 8:
        return None

    pts = np.array(segmentation).reshape(-1, 2)
    if len(pts) != 4:
        return None

    # 计算中心点
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    # 计算边长（4边形的对边）
    # 假设点是按顺序排列的：左上、右上、右下、左下
    w1 = np.linalg.norm(pts[1] - pts[0])  # 上边
    w2 = np.linalg.norm(pts[2] - pts[3])  # 下边
    width = (w1 + w2) / 2

    h1 = np.linalg.norm(pts[3] - pts[0])  # 左边
    h2 = np.linalg.norm(pts[2] - pts[1])  # 右边
    height = (h1 + h2) / 2

    # 计算旋转角度（使用上边的中点方向）
    mid_top = (pts[0] + pts[1]) / 2
    dx = mid_top[0] - cx
    dy = mid_top[1] - cy
    angle = np.degrees(np.arctan2(dy, dx)) - 90  # 转换为与垂直方向的夹角

    return [float(cx), float(cy), float(width), float(height), float(angle)]


class DataAugmentor:
    """
    数据增强模块，支持对图像和标注数据进行同步变换。
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化增强器
        :param config: 增强配置
        """
        self.config = config or {}

    def apply(self, image: np.ndarray, annotations: List[Dict[str, Any]], params: Dict[str, Any] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        执行增强
        :param image: numpy 数组格式的图片 (H, W, C)
        :param annotations: 标注数据列表
        :param params: 指定的增强参数（如果不提供则根据 config 随机生成）
        :return: (增强后的图片, 增强后的标注)
        """
        h, w = image.shape[:2]
        new_image = image.copy()
        new_annotations = copy.deepcopy(annotations)
        params = params or {}

        # 预处理：坐标对齐，并将 segmentation 转换为 rbbox
        for ann in new_annotations:
            self._ensure_absolute_coords(ann, w, h)
            # 如果有 segmentation 但没有 rbbox，自动转换
            if "segmentation" in ann and ann["segmentation"] and "rbbox" not in ann:
                if isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0:
                    # 取第一个 segmentation
                    seg = ann["segmentation"][0] if isinstance(ann["segmentation"][0], list) else ann["segmentation"]
                    rbbox = segmentation_to_rbbox(seg)
                    if rbbox:
                        ann["rbbox"] = rbbox

        # 1. 亮度与对比度
        if "brightness" in self.config or "contrast" in self.config:
            b_val = params.get("brightness")
            c_val = params.get("contrast")
            new_image = self._apply_brightness_contrast(new_image, b_val, c_val)

        # 2. 模糊
        if self._should_apply("blur") or "blur" in params:
            blur_val = params.get("blur")
            new_image = self._apply_blur(new_image, blur_val)

        # 3. 水平翻转
        if params.get("horizontal_flip", self._should_apply("horizontal_flip")):
            new_image, new_annotations = self._apply_horizontal_flip(new_image, new_annotations)

        # 4. 垂直翻转
        if params.get("vertical_flip", self._should_apply("vertical_flip")):
            new_image, new_annotations = self._apply_vertical_flip(new_image, new_annotations)

        # 5. 旋转
        if "rotate" in self.config or "rotate" in params:
            rot_val = params.get("rotate")
            if rot_val is not None or self._should_apply("rotate"):
                new_image, new_annotations = self._apply_rotate(new_image, new_annotations, rot_val)

        # 6. 俯视与仰视 (Pitch & Yaw)
        if "pitch" in self.config or "yaw" in self.config or "pitch" in params or "yaw" in params:
            pitch = params.get("pitch")
            yaw = params.get("yaw")
            if pitch is not None or yaw is not None or self._should_apply("pitch") or self._should_apply("yaw"):
                new_image, new_annotations = self._apply_pitch_yaw(new_image, new_annotations, pitch, yaw)

        # 后处理：还原坐标系
        for ann in new_annotations:
            if ann.get("_is_centered"):
                # 处理 rbbox
                if "rbbox" in ann and ann["rbbox"]:
                    ann["rbbox"][0] = ann["rbbox"][0] - (w / 2)
                    ann["rbbox"][1] = ann["rbbox"][1] - (h / 2)
                del ann["_is_centered"]

        return new_image, new_annotations

    def generate_batch(self, image: np.ndarray, annotations: List[Dict[str, Any]], num_results: int) -> List[Dict[str, Any]]:
        """
        按梯度生成一批增强结果
        """
        results = []
        
        # 为每个参数计算梯度值
        param_gradients = {}
        
        # 1. Continuous parameters
        if "brightness" in self.config:
            b_range = self.config["brightness"].get("range", [0.8, 1.2])
            b_range = [float(b_range[0]), float(b_range[1])]
            param_gradients["brightness"] = np.linspace(b_range[0], b_range[1], num_results).tolist()
            
        if "contrast" in self.config:
            c_range = self.config["contrast"].get("range", [0.8, 1.2])
            c_range = [float(c_range[0]), float(c_range[1])]
            param_gradients["contrast"] = np.linspace(c_range[0], c_range[1], num_results).tolist()
            
        if "rotate" in self.config:
            r_cfg = self.config["rotate"]
            r_range = r_cfg.get("range") or [-r_cfg.get("max_angle", 15), r_cfg.get("max_angle", 15)]
            r_range = [float(r_range[0]), float(r_range[1])]
            param_gradients["rotate"] = np.linspace(r_range[0], r_range[1], num_results).tolist()

        if "pitch" in self.config:
            p_cfg = self.config["pitch"]
            p_range = p_cfg.get("range", [-30, 30])
            p_range = [float(p_range[0]), float(p_range[1])]
            param_gradients["pitch"] = np.linspace(p_range[0], p_range[1], num_results).tolist()

        if "yaw" in self.config:
            y_cfg = self.config["yaw"]
            y_range = y_cfg.get("range", [-30, 30])
            y_range = [float(y_range[0]), float(y_range[1])]
            param_gradients["yaw"] = np.linspace(y_range[0], y_range[1], num_results).tolist()
            
        if "blur" in self.config:
            k_range = self.config["blur"].get("ksize_range", [3, 11])
            k_range = [float(k_range[0]), float(k_range[1])]
            # 模糊核必须是奇数
            param_gradients["blur"] = [int(x) | 1 for x in np.linspace(k_range[0], k_range[1], num_results)]

        # 2. Discrete/Boolean parameters (cycle through them)
        if "horizontal_flip" in self.config:
            param_gradients["horizontal_flip"] = [(i % 2 == 1) for i in range(num_results)]
        if "vertical_flip" in self.config:
            param_gradients["vertical_flip"] = [((i // 2) % 2 == 1) for i in range(num_results)]

        for i in range(num_results):
            current_params = {k: v[i] for k, v in param_gradients.items()}
            aug_img, aug_ann = self.apply(image, annotations, params=current_params)
            results.append({
                "image": aug_img,
                "annotations": aug_ann,
                "params": current_params
            })
            
        return results

    def _ensure_absolute_coords(self, ann: Dict[str, Any], w: int, h: int):
        """确保 rbbox 使用左上角 (0,0) 坐标系"""
        # 处理 rbbox [cx, cy, width, height, angle]
        if "rbbox" in ann and ann["rbbox"] and len(ann["rbbox"]) >= 2:
            cx, cy = ann["rbbox"][:2]
            # 如果坐标明显是相对于中心点的（例如 cx < 0）
            if cx < -1e-5:  # 使用微小误差判断
                ann["rbbox"][0] = cx + (w / 2)
                ann["rbbox"][1] = cy + (h / 2)
                ann["_is_centered"] = True

    def _should_apply(self, key: str) -> bool:
        if key not in self.config:
            return False
        prob = self.config[key].get("prob", 1.0)
        result = random.random() < prob
        if result:
            logger.debug(f"[Augmentation] Applying {key}...")
        return result

    def _apply_brightness_contrast(self, image: np.ndarray, brightness: float = None, contrast: float = None) -> np.ndarray:
        brightness_cfg = self.config.get("brightness", {})
        contrast_cfg = self.config.get("contrast", {})
        
        alpha = 1.0
        beta = 0.0
        
        # 亮度处理
        if brightness is not None:
            beta = (brightness - 1.0) * 255
        elif self._should_apply("brightness"):
            b_range = brightness_cfg.get("range", [0.8, 1.2])
            beta = (random.uniform(b_range[0], b_range[1]) - 1.0) * 255
            
        # 对比度处理
        if contrast is not None:
            alpha = contrast
        elif self._should_apply("contrast"):
            c_range = contrast_cfg.get("range", [0.8, 1.2])
            alpha = random.uniform(c_range[0], c_range[1])
            
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _apply_blur(self, image: np.ndarray, ksize: int = None) -> np.ndarray:
        if ksize is None:
            cfg = self.config.get("blur", {})
            ksize = cfg.get("ksize", 5)
        if ksize % 2 == 0: ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def _normalize_angle(self, angle: float) -> float:
        """将角度归一化到 [-180, 180] 范围内"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    def _apply_horizontal_flip(self, image: np.ndarray, annotations: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        image = cv2.flip(image, 1)
        for ann in annotations:
            # 处理旋转框 rbbox [cx, cy, width, height, rotation_angle]
            if "rbbox" in ann and ann["rbbox"]:
                cx, cy, bw, bh, rot = ann["rbbox"]
                # 水平翻转：cx -> w - cx, 旋转角度取反
                ann["rbbox"][0] = w - cx
                ann["rbbox"][4] = self._normalize_angle(-rot)

            # 处理 segmentation（如果有）
            if "segmentation" in ann and ann["segmentation"]:
                for i in range(len(ann["segmentation"])):
                    poly = np.array(ann["segmentation"][i]).reshape(-1, 2)
                    poly[:, 0] = w - poly[:, 0]
                    ann["segmentation"][i] = poly.flatten().tolist()
        return image, annotations

    def _apply_vertical_flip(self, image: np.ndarray, annotations: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        image = cv2.flip(image, 0)
        for ann in annotations:
            # 处理旋转框 rbbox [cx, cy, width, height, rotation_angle]
            if "rbbox" in ann and ann["rbbox"]:
                cx, cy, bw, bh, rot = ann["rbbox"]
                # 垂直翻转：cy -> h - cy, 旋转角度取反
                ann["rbbox"][1] = h - cy
                ann["rbbox"][4] = self._normalize_angle(-rot)

            # 处理 segmentation（如果有）
            if "segmentation" in ann and ann["segmentation"]:
                for i in range(len(ann["segmentation"])):
                    poly = np.array(ann["segmentation"][i]).reshape(-1, 2)
                    poly[:, 1] = h - poly[:, 1]
                    ann["segmentation"][i] = poly.flatten().tolist()
        return image, annotations

    def _apply_rotate(self, image: np.ndarray, annotations: List[Dict[str, Any]], angle: float = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if angle is None:
            cfg = self.config["rotate"]
            max_angle = cfg.get("max_angle", 15)
            angle = random.uniform(-max_angle, max_angle)

        h, w = image.shape[:2]
        # 输出正方形，边长为原始尺寸的最长边
        size = max(w, h)
        center = (size / 2, size / 2)
        
        # 计算缩放比例，使图像在旋转时完整显示
        # 先创建平移矩阵，将图像中心移到新画布中心
        M_translate = np.array([
            [1, 0, (size - w) / 2],
            [0, 1, (size - h) / 2]
        ], dtype=np.float32)
        
        # 旋转矩阵（相对于新画布中心）
        M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 组合变换：先平移，再旋转
        M = M_rotate @ np.vstack([M_translate, [0, 0, 1]])
        M = M[:2, :]

        # 使用白色填充旋转后的边界，输出正方形尺寸
        image = cv2.warpAffine(image, M, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        for ann in annotations:
            # 处理旋转框 rbbox [cx, cy, width, height, rotation_angle]
            if "rbbox" in ann and ann["rbbox"]:
                rbbox = ann["rbbox"]
                cx, cy, bw, bh, rot = rbbox
                # 更新中心点坐标（考虑平移）
                ann["rbbox"][0] = cx + (size - w) / 2
                ann["rbbox"][1] = cy + (size - h) / 2
                # 更新旋转角度（累加）并归一化
                ann["rbbox"][4] = self._normalize_angle(rot + angle)

            # 处理 segmentation（如果有）
            if "segmentation" in ann and ann["segmentation"]:
                for i in range(len(ann["segmentation"])):
                    poly = np.array(ann["segmentation"][i]).reshape(-1, 2)
                    ones = np.ones(shape=(len(poly), 1))
                    poly_ones = np.hstack([poly, ones])
                    transformed_poly = M.dot(poly_ones.T).T
                    ann["segmentation"][i] = transformed_poly.flatten().tolist()

        return image, annotations

    def _apply_pitch_yaw(self, image, annotations, pitch=0, yaw=0):
        """
        使用 3D 投影变换实现俯视 (pitch) 和 侧视 (yaw) 效果，并保持图像内容完整 (自适应缩放)
        输出正方形，边长为原始尺寸的最长边
        """
        h, w = image.shape[:2]
        # 输出正方形，边长为原始尺寸的最长边
        size = max(w, h)
        
        # 1. 转换为弧度
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        
        # 2. 定义 3D 旋转矩阵
        # Rx: 绕 X 轴旋转 (Pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # Ry: 绕 Y 轴旋转 (Yaw)
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        R = Ry @ Rx # 先 Pitch 后 Yaw
        
        # 3. 定义相机内参 (假设主点在中心，焦距为宽度的 1.2 倍)
        f = w * 1.2
        K = np.array([
            [f, 0, w/2],
            [0, f, h/2],
            [0, 0, 1]
        ])
        
        # 4. 计算四个顶点的投影位置
        pts_3d = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0]
        ], dtype=np.float32)
        
        # 将原图中心移动到原点，进行旋转，再移回去
        pts_3d[:, 0] -= w/2
        pts_3d[:, 1] -= h/2
        
        # 应用旋转 (添加一个虚拟的 Z 距离，防止投影到无穷远)
        dist = f
        pts_rotated = (R @ pts_3d.T).T
        pts_rotated[:, 2] += dist
        
        # 投影到 2D 屏幕
        pts_2d = (K @ pts_rotated.T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        
        # 5. 自适应缩放逻辑：计算投影后的包围盒，并调整变换矩阵
        min_x, min_y = np.min(pts_2d[:, :2], axis=0)
        max_x, max_y = np.max(pts_2d[:, :2], axis=0)
        
        proj_w = max_x - min_x
        proj_h = max_y - min_y
        
        # 计算缩放比例，使投影后的图像适应正方形画布
        scale = size / max(proj_w, proj_h)
        
        # 最终的透视变换矩阵 M
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = pts_2d[:, :2].astype(np.float32)
        
        # 应用偏移和缩放，使内容居中并适应正方形画布
        dst_pts[:, 0] = (dst_pts[:, 0] - min_x) * scale + (size - proj_w * scale) / 2
        dst_pts[:, 1] = (dst_pts[:, 1] - min_y) * scale + (size - proj_h * scale) / 2
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 6. 应用变换，输出正方形尺寸
        aug_image = cv2.warpPerspective(image, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # 7. 同步标注数据
        new_annotations = []
        for ann in annotations:
            new_ann = ann.copy()
            
            # 同步分割点
            if "segmentation" in new_ann and new_ann["segmentation"]:
                new_segs = []
                for seg in new_ann["segmentation"]:
                    pts = np.array(seg).reshape(-1, 2)
                    ones = np.ones((pts.shape[0], 1))
                    pts_homo = np.hstack([pts, ones])
                    
                    trans_pts = (M @ pts_homo.T).T
                    trans_pts[:, 0] /= trans_pts[:, 2]
                    trans_pts[:, 1] /= trans_pts[:, 2]
                    
                    new_segs.append(trans_pts[:, :2].flatten().tolist())
                new_ann["segmentation"] = new_segs
            
            # 同步 BBox (根据变换后的分割点或 BBox 角点重新计算)
            if "bbox" in new_ann:
                x, y, bw, bh = new_ann["bbox"]
                bbox_pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
                ones = np.ones((4, 1))
                bbox_homo = np.hstack([bbox_pts, ones])
                
                trans_bbox_pts = (M @ bbox_homo.T).T
                trans_bbox_pts[:, 0] /= trans_bbox_pts[:, 2]
                trans_bbox_pts[:, 1] /= trans_bbox_pts[:, 2]
                
                nx1 = np.min(trans_bbox_pts[:, 0])
                ny1 = np.min(trans_bbox_pts[:, 1])
                nx2 = np.max(trans_bbox_pts[:, 0])
                ny2 = np.max(trans_bbox_pts[:, 1])
                
                new_ann["bbox"] = [float(nx1), float(ny1), float(nx2 - nx1), float(ny2 - ny1)]
                new_ann["area"] = float((nx2 - nx1) * (ny2 - ny1))
            
            new_annotations.append(new_ann)
            
        return aug_image, new_annotations
