"""
ROI 预检测模块 - 增加深色遮挡检测
在 PatchCore 推理前拦截：完全遮挡 + 部分遮挡（如黑色圆盘压住工件）

改进点：
- 原逻辑：std/方差 → 只能检"均匀纯色"遮挡，部分遮挡时 std 反而高，漏检
- 新增：暗像素占比 + 均值亮度 → 能检"黑色物体局部压住"的情况
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PreCheckResult(Enum):
    PASS              = "pass"
    UNIFORM_OCCLUSION = "uniform_occlusion"   # 纯色遮挡（均匀暗/亮）
    DARK_OCCLUSION    = "dark_occlusion"       # 深色物体遮挡（含部分遮挡）
    EMPTY_ROI         = "empty_roi"
    LOW_VARIANCE      = "low_variance"


@dataclass
class PreCheckReport:
    result: PreCheckResult
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    details: Dict[str, Any]
    message: str


class ROIPreChecker:
    """
    ROI 预检测器

    检测维度（按优先级）：
    1. 均匀遮挡：std < min_contrast             （原有）
    2. 纯色遮挡：var < min_variance             （原有）
    3. 深色遮挡：暗像素占比 > max_dark_ratio    （新增，解决部分遮挡漏检）
    4. 均值过低：mean < min_mean_brightness     （新增，辅助确认）

    不检测（留给 PatchCore）：
    - 边缘 / 形状 / 纹理变化
    """

    def __init__(
        self,
        # ── 原有参数 ──────────────────────────────
        min_contrast: float = 10.0,
        min_variance: float = 50.0,

        # ── 新增：深色遮挡检测 ────────────────────
        # 像素亮度低于此值视为"极暗像素"
        # 推荐值：30~50（黑色圆盘约 15~45，金属表面约 100~180）
        dark_pixel_threshold: int = 40,

        # 极暗像素占比超过此值 → 判定为深色遮挡
        # 推荐值：0.20~0.30（设 0.20 可捕捉 20% 面积遮挡）
        max_dark_ratio: float = 0.20,

        # 全图均值低于此值 → 辅助确认深色遮挡
        # 推荐值：60~80
        min_mean_brightness: float = 70.0,

        # ── 白名单：本来就是深色的类别，跳过暗色检测 ──
        # 例如：某些孔洞内壁 / 密封圈 ROI 本身就是黑色
        skip_dark_check_categories: List[str] = None,

        debug: bool = False,
    ):
        self.min_contrast = min_contrast
        self.min_variance = min_variance
        self.dark_pixel_threshold = dark_pixel_threshold
        self.max_dark_ratio = max_dark_ratio
        self.min_mean_brightness = min_mean_brightness
        self.skip_dark_check_categories = set(skip_dark_check_categories or [])
        self.debug = debug

    def check(self, roi: np.ndarray, category: str = "") -> PreCheckReport:
        # ── 基础校验 ──────────────────────────────────────────────────────
        if roi is None or roi.size == 0:
            return PreCheckReport(
                result=PreCheckResult.EMPTY_ROI,
                is_anomaly=True, anomaly_score=1.0, confidence=1.0,
                details={}, message="ROI 为空或无效"
            )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

        # ── 基础统计量 ────────────────────────────────────────────────────
        std_val  = float(np.std(gray))
        var_val  = float(np.var(gray))
        mean_val = float(np.mean(gray))

        # 新增：暗像素占比
        dark_mask  = gray < self.dark_pixel_threshold
        dark_ratio = float(dark_mask.mean())

        details = {
            "std": std_val,
            "variance": var_val,
            "mean": mean_val,
            "dark_ratio": dark_ratio,
            "dark_pixel_threshold": self.dark_pixel_threshold,
        }

        if self.debug:
            logger.info(
                f"[ROIPreChecker] {category} "
                f"std={std_val:.1f} var={var_val:.1f} "
                f"mean={mean_val:.1f} dark_ratio={dark_ratio:.3f}"
            )

        # ── 检测 1：均匀遮挡（对比度极低，原逻辑）────────────────────────
        if std_val < self.min_contrast:
            score = min(1.0, 1.0 - std_val / self.min_contrast)
            return PreCheckReport(
                result=PreCheckResult.UNIFORM_OCCLUSION,
                is_anomaly=True, anomaly_score=score, confidence=0.9,
                details=details,
                message=f"均匀遮挡: std={std_val:.1f} < {self.min_contrast}"
            )

        # ── 检测 2：纯色块（方差极低，原逻辑）───────────────────────────
        if var_val < self.min_variance:
            score = min(1.0, 1.0 - var_val / self.min_variance)
            return PreCheckReport(
                result=PreCheckResult.LOW_VARIANCE,
                is_anomaly=True, anomaly_score=score, confidence=0.8,
                details=details,
                message=f"纯色块: var={var_val:.1f} < {self.min_variance}"
            )

        # ── 检测 3：深色遮挡（新增，解决部分遮挡漏检）───────────────────
        if category not in self.skip_dark_check_categories:
            dark_anomaly = False
            dark_score   = 0.0
            reason_parts = []

            # 3a. 暗像素占比过高（主判据）
            if dark_ratio > self.max_dark_ratio:
                dark_anomaly = True
                # 分数随占比线性增长，占比=阈值时约 0.5，占比=1 时为 1.0
                dark_score = max(dark_score, min(1.0, dark_ratio / (self.max_dark_ratio * 2)))
                reason_parts.append(
                    f"暗像素占比={dark_ratio:.2%} > {self.max_dark_ratio:.0%}"
                )

            # 3b. 均值过低（辅助判据，单独也可触发）
            if mean_val < self.min_mean_brightness:
                dark_anomaly = True
                brightness_score = min(1.0, 1.0 - mean_val / self.min_mean_brightness)
                dark_score = max(dark_score, brightness_score)
                reason_parts.append(
                    f"均值亮度={mean_val:.1f} < {self.min_mean_brightness}"
                )

            if dark_anomaly:
                return PreCheckReport(
                    result=PreCheckResult.DARK_OCCLUSION,
                    is_anomaly=True,
                    anomaly_score=dark_score,
                    confidence=0.85,
                    details=details,
                    message="深色遮挡: " + ", ".join(reason_parts)
                )

        # ── 通过预检 ──────────────────────────────────────────────────────
        return PreCheckReport(
            result=PreCheckResult.PASS,
            is_anomaly=False, anomaly_score=0.0, confidence=0.5,
            details=details,
            message="通过预检，进入 PatchCore"
        )

    def visualize(self, roi: np.ndarray, report: PreCheckReport) -> np.ndarray:
        vis = roi.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        color  = (0, 0, 255) if report.is_anomaly else (0, 255, 0)
        status = report.result.value.upper()
        h, w   = vis.shape[:2]

        cv2.rectangle(vis, (0, 0), (w-1, h-1), color, 3)
        cv2.putText(vis, status, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        d = report.details
        info = (f"std={d.get('std',0):.0f} "
                f"dark={d.get('dark_ratio',0):.0%} "
                f"mean={d.get('mean',0):.0f}")
        cv2.putText(vis, info, (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return vis


# ─── 单例 & 便捷接口 ──────────────────────────────────────────────────────────

_default_checker: Optional[ROIPreChecker] = None


def get_pre_checker(**kwargs) -> ROIPreChecker:
    global _default_checker
    if _default_checker is None or kwargs:
        _default_checker = ROIPreChecker(**kwargs)
    return _default_checker


def check_roi(roi: np.ndarray, category: str = "", **kwargs) -> PreCheckReport:
    return get_pre_checker(**kwargs).check(roi, category)