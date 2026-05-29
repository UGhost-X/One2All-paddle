#!/usr/bin/env python3
"""
合成缺陷生成器 — 破坏正常纹理

策略：
- texture_swap: 从另一张正常 ROI 取区域贴过来（模拟异物/材质异常）
- solid_color: 整图填充与背景相近的纯色（模拟完全遮挡/缺失）
"""

import random
from typing import Tuple, List, Optional
import numpy as np
import cv2


def _random_patch_from_pool(
    pool: List[np.ndarray], h: int, w: int
) -> Optional[np.ndarray]:
    if not pool:
        return None
    src = random.choice(pool)
    sh, sw = src.shape[:2]
    if sh < h or sw < w:
        return cv2.resize(src, (w, h))
    y = random.randint(0, sh - h)
    x = random.randint(0, sw - w)
    return src[y:y + h, x:x + w].copy()


class SyntheticDefectGenerator:
    """合成缺陷生成器"""

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        min_defect_size_ratio: float = 0.10,
        max_defect_size_ratio: float = 0.60,
    ):
        self.strategies = strategies or ["texture_swap", "solid_color"]
        self.weights = weights or [0.6, 0.4]
        self.min_defect_ratio = min_defect_size_ratio
        self.max_defect_ratio = max_defect_size_ratio
        self._patch_pool: List[np.ndarray] = []

    def _random_defect_region(self, h: int, w: int) -> Tuple[int, int, int, int]:
        area = h * w
        min_area = int(area * self.min_defect_ratio)
        max_area = int(area * self.max_defect_ratio)
        target = random.randint(min_area, max_area)
        aspect = random.uniform(0.3, 3.0)
        dh = int(np.sqrt(target / aspect))
        dw = int(target / dh)
        dh = max(4, min(dh, h - 2))
        dw = max(4, min(dw, w - 2))
        x = random.randint(1, w - dw - 1)
        y = random.randint(1, h - dh - 1)
        return x, y, dw, dh

    def _to_yolo_bbox(self, x: int, y: int, dw: int, dh: int, w: int, h: int) -> List[float]:
        return [(x + dw / 2) / w, (y + dh / 2) / h, dw / w, dh / h]

    def _natural_mask(self, dh: int, dw: int) -> np.ndarray:
        """椭圆 mask + 随机扰动 + 高斯羽化，模拟自然不规则轮廓。"""
        mask = np.zeros((dh, dw), dtype=np.float32)
        cx, cy = dw / 2, dh / 2
        rx = dw / 2 * random.uniform(0.5, 0.95)
        ry = dh / 2 * random.uniform(0.5, 0.95)
        angle = random.uniform(0, 360)
        y_grid, x_grid = np.ogrid[:dh, :dw]
        xc, yc = x_grid - cx, y_grid - cy
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        x_rot = xc * cos_a + yc * sin_a
        y_rot = -xc * sin_a + yc * cos_a
        ellipse = (x_rot / rx) ** 2 + (y_rot / ry) ** 2
        mask[ellipse <= 1.0] = 1.0
        noise = np.random.uniform(-0.15, 0.15, (dh, dw)).astype(np.float32)
        mask = np.clip(mask + noise, 0, 1)
        k = random.randint(5, 15)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(mask, (k, k), k / 4)

    @staticmethod
    def _to_3d(arr: np.ndarray) -> np.ndarray:
        return arr[:, :, np.newaxis] if arr.ndim == 2 else arr

    def _blend_region(self, result: np.ndarray, y: int, x: int, dh: int, dw: int,
                       mask: np.ndarray, modified: np.ndarray) -> np.ndarray:
        mask_3d = self._to_3d(mask)
        roi = self._to_3d(result[y:y + dh, x:x + dw].astype(np.float32))
        mod = self._to_3d(modified.astype(np.float32))
        blended = roi * (1 - mask_3d) + mod * mask_3d
        if result.ndim == 2:
            blended = blended[:, :, 0]
        result[y:y + dh, x:x + dw] = blended
        return result

    # ── 纹理替换 ─────────────────────────────────────────────

    def _texture_swap(self, img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """从其他正常 ROI 取一块纹理，自然粘贴。"""
        h, w = img.shape[:2]
        x, y, dw, dh = self._random_defect_region(h, w)
        patch = _random_patch_from_pool(self._patch_pool, dh, dw)
        if patch is None:
            return self._solid_color(img)

        result = img.astype(np.float32)
        result = self._blend_region(result, y, x, dh, dw, self._natural_mask(dh, dw), patch)
        return np.clip(result, 0, 255).astype(np.uint8), self._to_yolo_bbox(x, y, dw, dh, w, h)

    # ── 纯色异常 ─────────────────────────────────────────────

    def _solid_color(self, img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """整图填充与背景相近的纯色，模拟完全遮挡/缺失。"""
        h, w = img.shape[:2]
        # 纯色 = 背景均值 + 小偏移（保持与背景相似但略有不同）
        bg_mean = float(img.mean())
        shift = random.uniform(-0.15, 0.15) * 255
        solid_val = np.clip(bg_mean + shift, 0, 255)
        is_color = img.ndim == 3
        solid_img = np.full((h, w, 3) if is_color else (h, w), solid_val, dtype=np.uint8)

        mask = self._natural_mask(h, w)
        result = img.astype(np.float32)
        result = self._blend_region(result, 0, 0, h, w, mask, solid_img)
        # 整图都是异常，bbox 覆盖全图
        return np.clip(result, 0, 255).astype(np.uint8), [0.5, 0.5, 1.0, 1.0]

    # ── 对外接口 ─────────────────────────────────────────────

    def generate(self, img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        strategy = random.choices(self.strategies, weights=self.weights, k=1)[0]
        if strategy == "texture_swap":
            return self._texture_swap(img)
        elif strategy == "solid_color":
            return self._solid_color(img)
        return self._texture_swap(img)

    def generate_multiple(self, img: np.ndarray, n: int = 3) -> List[Tuple[np.ndarray, List[float]]]:
        return [self.generate(img) for _ in range(n)]

    def set_patch_pool(self, pool: List[np.ndarray]):
        self._patch_pool = pool


def load_synthetic_config(config_path: Optional[str] = None) -> dict:
    import yaml
    from pathlib import Path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "synthetic_defects.yaml"
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
