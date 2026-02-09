import cv2
import numpy as np
import random
import copy
from typing import List, Dict, Any, Tuple

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

        # 预处理：坐标对齐
        for ann in new_annotations:
            self._ensure_absolute_coords(ann, w, h)

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
                key = "bbox" if "bbox" in ann else "points"
                if key in ann:
                    ann[key][0] = ann[key][0] - (w / 2)
                    ann[key][1] = ann[key][1] - (h / 2)
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
            param_gradients["brightness"] = np.linspace(b_range[0], b_range[1], num_results).tolist()
            
        if "contrast" in self.config:
            c_range = self.config["contrast"].get("range", [0.8, 1.2])
            param_gradients["contrast"] = np.linspace(c_range[0], c_range[1], num_results).tolist()
            
        if "rotate" in self.config:
            r_cfg = self.config["rotate"]
            r_range = r_cfg.get("range") or [-r_cfg.get("max_angle", 15), r_cfg.get("max_angle", 15)]
            param_gradients["rotate"] = np.linspace(r_range[0], r_range[1], num_results).tolist()

        if "pitch" in self.config:
            p_cfg = self.config["pitch"]
            p_range = p_cfg.get("range", [-30, 30])
            param_gradients["pitch"] = np.linspace(p_range[0], p_range[1], num_results).tolist()

        if "yaw" in self.config:
            y_cfg = self.config["yaw"]
            y_range = y_cfg.get("range", [-30, 30])
            param_gradients["yaw"] = np.linspace(y_range[0], y_range[1], num_results).tolist()
            
        if "blur" in self.config:
            k_range = self.config["blur"].get("ksize_range", [3, 11])
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
        """确保 bbox/points 使用左上角 (0,0) 坐标系"""
        # 兼容 points 或 bbox 字段
        key = "bbox" if "bbox" in ann else "points"
        if key in ann and len(ann[key]) >= 4:
            x, y, bw, bh = ann[key][:4]
            # 如果坐标明显是相对于中心点的（例如 x < 0）
            if x < -1e-5: # 使用微小误差判断
                ann[key][0] = x + (w / 2)
                ann[key][1] = y + (h / 2)
                ann["_is_centered"] = True

    def _should_apply(self, key: str) -> bool:
        if key not in self.config:
            return False
        prob = self.config[key].get("prob", 1.0)
        result = random.random() < prob
        if result:
            print(f"[Augmentation] Applying {key}...")
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

    def _apply_horizontal_flip(self, image: np.ndarray, annotations: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        image = cv2.flip(image, 1)
        for ann in annotations:
            key = "bbox" if "bbox" in ann else "points"
            if key in ann:
                x, y, bw, bh = ann[key][:4]
                new_x = w - (x + bw)
                ann[key][0] = new_x
            
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
            key = "bbox" if "bbox" in ann else "points"
            if key in ann:
                x, y, bw, bh = ann[key][:4]
                new_y = h - (y + bh)
                ann[key][1] = new_y
            
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
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 使用白色填充旋转后的边界
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        for ann in annotations:
            key = "bbox" if "bbox" in ann else "points"
            # 1. 优先旋转 Segmentation
            if "segmentation" in ann and ann["segmentation"]:
                for i in range(len(ann["segmentation"])):
                    poly = np.array(ann["segmentation"][i]).reshape(-1, 2)
                    ones = np.ones(shape=(len(poly), 1))
                    poly_ones = np.hstack([poly, ones])
                    transformed_poly = M.dot(poly_ones.T).T
                    ann["segmentation"][i] = transformed_poly.flatten().tolist()
                
                # 基于旋转后的点重新计算 bbox
                all_pts = np.concatenate([np.array(p).reshape(-1, 2) for p in ann["segmentation"]])
                new_x, new_y = np.min(all_pts, axis=0)
                new_w, new_h = np.max(all_pts, axis=0) - [new_x, new_y]
                ann[key] = [float(new_x), float(new_y), float(new_w), float(new_h)]
            
            # 2. 如果只有 bbox/points
            elif key in ann:
                x, y, bw, bh = ann[key][:4]
                pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
                ones = np.ones(shape=(len(pts), 1))
                pts_ones = np.hstack([pts, ones])
                transformed_pts = M.dot(pts_ones.T).T
                
                new_x, new_y = np.min(transformed_pts, axis=0)
                new_w, new_h = np.max(transformed_pts, axis=0) - [new_x, new_y]
                ann[key] = [float(new_x), float(new_y), float(new_w), float(new_h)]
                
        return image, annotations

    def _apply_pitch_yaw(self, image: np.ndarray, annotations: List[Dict[str, Any]], pitch: float = None, yaw: float = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        实现俯视 (Pitch) 和 仰视 (Yaw) 效果
        :param pitch: 俯仰角 (度)
        :param yaw: 偏航角 (度)
        """
        if pitch is None:
            cfg = self.config.get("pitch", {})
            pitch = random.uniform(*cfg.get("range", [-20, 20])) if "range" in cfg else 0
        if yaw is None:
            cfg = self.config.get("yaw", {})
            yaw = random.uniform(*cfg.get("range", [-20, 20])) if "range" in cfg else 0

        h, w = image.shape[:2]
        
        # 1. 计算相机矩阵
        # 焦距近似值
        f = max(w, h)
        # 相机内参矩阵
        K = np.array([
            [f, 0, w/2],
            [0, f, h/2],
            [0, 0, 1]
        ], dtype=np.float32)

        # 2. 计算旋转矩阵 R
        rad_p = np.deg2rad(pitch)
        rad_y = np.deg2rad(yaw)
        
        # Pitch 旋转矩阵 (绕 X 轴)
        R_p = np.array([
            [1, 0, 0],
            [0, np.cos(rad_p), -np.sin(rad_p)],
            [0, np.sin(rad_p), np.cos(rad_p)]
        ], dtype=np.float32)
        
        # Yaw 旋转矩阵 (绕 Y 轴)
        R_y = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ], dtype=np.float32)
        
        R = R_y @ R_p
        
        # 3. 计算单应矩阵 H = K * R * K^-1
        H = K @ R @ np.linalg.inv(K)
        
        # --- 新增：自动缩放以防止图像被裁切 ---
        # 计算原始四个角点变换后的位置
        src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(src_corners, H)
        
        # 获取变换后内容的边界
        xmin, ymin = np.min(dst_corners, axis=0).ravel()
        xmax, ymax = np.max(dst_corners, axis=0).ravel()
        
        # 计算缩放比例，使得变换后的内容能适应原图大小
        content_w = xmax - xmin
        content_h = ymax - ymin
        scale = min(w / content_w, h / content_h) * 0.95 # 留 5% 的边距
        
        # 构建平移和缩放矩阵来修正 H
        # 目标：将变换后的中心移回原图中心，并按比例缩小
        tx = (w / 2) - (xmin + xmax) / 2 * scale
        ty = (h / 2) - (ymin + ymax) / 2 * scale
        
        M_scale = np.array([
            [scale, 0, tx],
            [0, scale, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 最终变换矩阵
        H_final = M_scale @ H
        
        # 4. 执行图片变换
        # 使用纯白色进行填充 (255, 255, 255)
        image = cv2.warpPerspective(image, H_final, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        # 5. 同步标注数据
        for ann in annotations:
            key = "bbox" if "bbox" in ann else "points"
            
            if "segmentation" in ann and ann["segmentation"]:
                for i in range(len(ann["segmentation"])):
                    poly = np.array(ann["segmentation"][i]).reshape(-1, 2).astype(np.float32)
                    poly_h = np.column_stack([poly, np.ones(len(poly))])
                    transformed_poly_h = H_final.dot(poly_h.T).T
                    # 归一化并处理 Z <= 0 的情况
                    transformed_poly = transformed_poly_h[:, :2] / (transformed_poly_h[:, 2:3] + 1e-8)
                    ann["segmentation"][i] = transformed_poly.flatten().tolist()
                
                all_pts = np.concatenate([np.array(p).reshape(-1, 2) for p in ann["segmentation"]])
                new_x, new_y = np.min(all_pts, axis=0)
                new_w, new_h = np.max(all_pts, axis=0) - [new_x, new_y]
                ann[key] = [float(new_x), float(new_y), float(new_w), float(new_h)]
            
            elif key in ann:
                x, y, bw, bh = ann[key][:4]
                pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
                pts_h = np.column_stack([pts, np.ones(len(pts))])
                transformed_pts_h = H_final.dot(pts_h.T).T
                transformed_pts = transformed_pts_h[:, :2] / (transformed_pts_h[:, 2:3] + 1e-8)
                
                new_x, new_y = np.min(transformed_pts, axis=0)
                new_w, new_h = np.max(transformed_pts, axis=0) - [new_x, new_y]
                ann[key] = [float(new_x), float(new_y), float(new_w), float(new_h)]
                
        return image, annotations
