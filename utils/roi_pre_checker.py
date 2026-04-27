"""
ROI 预检测模块 - 基于模板对比的遮挡检测
核心原则：所有判断都是"预测ROI 与 模板ROI 的差异"，而非绝对阈值
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
    UNIFORM_OCCLUSION = "uniform_occlusion"
    DARK_OCCLUSION    = "dark_occlusion"      # 相对模板变暗
    BRIGHT_OCCLUSION  = "bright_occlusion"    # 相对模板变亮（反光等）
    EMPTY_ROI         = "empty_roi"
    NO_TEMPLATE       = "no_template"         # 没有注册模板，跳过


@dataclass
class TemplateStats:
    """模板 ROI 的基准统计量"""
    mean: float
    std: float
    dark_ratio: float        # 像素 < dark_threshold 的占比
    bright_ratio: float      # 像素 > bright_threshold 的占比
    dark_threshold: int      # 记录计算时用的阈值，保持一致性
    bright_threshold: int


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
    基于模板对比的 ROI 预检测器

    工作原理：
      1. 启动时从模板图提取每个 category 的基准统计量（mean/std/dark_ratio）
      2. 推理时：pred_dark_ratio - tmpl_dark_ratio > max_dark_delta → 遮挡
         这样无论模板本身多暗（孔洞/橡胶/涂层），都不会误报

    不检测（留给 PatchCore）：边缘/形状/纹理变化
    """

    def __init__(
        self,
        # 像素亮度分界点（两个方向）
        dark_pixel_threshold: int = 40,
        bright_pixel_threshold: int = 220,

        # 相对模板的暗像素占比增量阈值 → 判定深色遮挡
        # 例：模板 dark_ratio=0.05，pred=0.45，Δ=0.40 > 0.25 → 遮挡
        max_dark_delta: float = 0.5,

        # 相对模板的均值亮度下降阈值（绝对值，0~255）→ 辅助确认
        max_mean_drop: float = 175,

        # 相对模板的 std 下降阈值 → 判定均匀遮挡
        # 只有当预测 std 比模板 std 下降超过此值时才触发
        max_std_drop: float = 15.0,

        # 无模板时的兜底绝对阈值（退化为原始逻辑）
        fallback_min_contrast: float = 10.0,
        fallback_max_dark_ratio: float = 0.70,   # 没有模板时才用绝对值

        debug: bool = False,
    ):
        self.dark_pixel_threshold  = dark_pixel_threshold
        self.bright_pixel_threshold = bright_pixel_threshold
        self.max_dark_delta        = max_dark_delta
        self.max_mean_drop         = max_mean_drop
        self.max_std_drop          = max_std_drop
        self.fallback_min_contrast = fallback_min_contrast
        self.fallback_max_dark_ratio = fallback_max_dark_ratio
        self.debug                 = debug

        # category -> TemplateStats
        self._template_stats: Dict[str, TemplateStats] = {}

    # ──────────────────────────────────────────────────────────────────────
    # 模板注册
    # ──────────────────────────────────────────────────────────────────────

    def register_template_roi(self, category: str, template_roi: np.ndarray):
        """
        注册某个 category 的模板 ROI，提取基准统计量。
        在服务启动 load_models() 之后、第一次推理之前调用。
        """
        if template_roi is None or template_roi.size == 0:
            logger.warning(f"[PreChecker] 模板ROI为空，跳过注册: {category}")
            return

        gray = (cv2.cvtColor(template_roi, cv2.COLOR_BGR2GRAY)
                if len(template_roi.shape) == 3 else template_roi.copy())

        stats = TemplateStats(
            mean         = float(np.mean(gray)),
            std          = float(np.std(gray)),
            dark_ratio   = float((gray < self.dark_pixel_threshold).mean()),
            bright_ratio = float((gray > self.bright_pixel_threshold).mean()),
            dark_threshold   = self.dark_pixel_threshold,
            bright_threshold = self.bright_pixel_threshold,
        )
        self._template_stats[category] = stats

        logger.info(
            f"[PreChecker] 注册模板 [{category}]: "
            f"mean={stats.mean:.1f} std={stats.std:.1f} "
            f"dark_ratio={stats.dark_ratio:.3f} bright_ratio={stats.bright_ratio:.3f}"
        )

    def has_template(self, category: str) -> bool:
        return category in self._template_stats

    # ──────────────────────────────────────────────────────────────────────
    # 推理检测
    # ──────────────────────────────────────────────────────────────────────

    def check(self, roi: np.ndarray, category: str = "") -> PreCheckReport:
        if roi is None or roi.size == 0:
            return PreCheckReport(
                result=PreCheckResult.EMPTY_ROI,
                is_anomaly=True, anomaly_score=1.0, confidence=1.0,
                details={}, message="ROI 为空或无效"
            )

        gray = (cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if len(roi.shape) == 3 else roi.copy())

        pred_mean        = float(np.mean(gray))
        pred_std         = float(np.std(gray))
        pred_dark_ratio  = float((gray < self.dark_pixel_threshold).mean())
        pred_bright_ratio = float((gray > self.bright_pixel_threshold).mean())

        details = {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_dark_ratio": pred_dark_ratio,
        }

        tmpl = self._template_stats.get(category)

        # ── 有模板：做差值对比 ────────────────────────────────────────────
        if tmpl is not None:
            dark_delta  = pred_dark_ratio - tmpl.dark_ratio   # 正值=变暗
            mean_drop   = tmpl.mean - pred_mean                # 正值=变暗

            details.update({
                "tmpl_mean": tmpl.mean,
                "tmpl_dark_ratio": tmpl.dark_ratio,
                "dark_delta": dark_delta,
                "mean_drop": mean_drop,
            })

            if self.debug:
                logger.info(
                    f"[PreChecker] [{category}] "
                    f"pred(mean={pred_mean:.1f} std={pred_std:.1f} dark={pred_dark_ratio:.3f}) "
                    f"tmpl(mean={tmpl.mean:.1f} std={tmpl.std:.1f} dark={tmpl.dark_ratio:.3f}) "
                    f"Δdark={dark_delta:+.3f} Δmean={mean_drop:+.1f} Δstd={std_drop:+.1f}"
                )

            # 均匀遮挡：std 相对模板显著下降（与模板对比，而非绝对值）
            # 如果模板本身 std 就低（如黑色区域），预测 std 也低是正常的，不应触发
            std_drop = tmpl.std - pred_std  # 正值 = 对比度下降
            details["std_drop"] = std_drop

            # 只有当 std 相对模板显著下降，且预测 std 本身也很低时才触发
            if std_drop > self.max_std_drop and pred_std < self.fallback_min_contrast:
                score = min(1.0, std_drop / (self.max_std_drop * 2))
                return PreCheckReport(
                    result=PreCheckResult.UNIFORM_OCCLUSION,
                    is_anomaly=True, anomaly_score=score, confidence=0.92,
                    details=details,
                    message=f"均匀遮挡: std_drop={std_drop:.1f} (tmpl={tmpl.std:.1f} -> pred={pred_std:.1f})"
                )

            # 深色遮挡：暗像素占比相对模板显著增加
            triggered = []
            score = 0.0

            if dark_delta > self.max_dark_delta:
                triggered.append(f"Δdark={dark_delta:+.2%} > {self.max_dark_delta:.0%}")
                score = max(score, min(1.0, dark_delta / (self.max_dark_delta * 2)))

            if mean_drop > self.max_mean_drop:
                triggered.append(f"Δmean={mean_drop:+.1f} > {self.max_mean_drop:.0f}")
                score = max(score, min(1.0, mean_drop / (self.max_mean_drop * 2)))

            if triggered:
                return PreCheckReport(
                    result=PreCheckResult.DARK_OCCLUSION,
                    is_anomaly=True, anomaly_score=score, confidence=0.88,
                    details=details,
                    message="深色遮挡(对比模板): " + ", ".join(triggered)
                )

            return PreCheckReport(
                result=PreCheckResult.PASS,
                is_anomaly=False, anomaly_score=0.0, confidence=0.6,
                details=details,
                message="通过预检（对比模板），进入 PatchCore"
            )

        # ── 无模板：退化为原始绝对阈值逻辑 ──────────────────────────────
        else:
            if self.debug:
                logger.info(
                    f"[PreChecker] [{category}] 无模板，使用绝对阈值兜底 "
                    f"std={pred_std:.1f} dark={pred_dark_ratio:.3f}"
                )

            if pred_std < self.fallback_min_contrast:
                score = min(1.0, 1.0 - pred_std / self.fallback_min_contrast)
                return PreCheckReport(
                    result=PreCheckResult.UNIFORM_OCCLUSION,
                    is_anomaly=True, anomaly_score=score, confidence=0.9,
                    details=details,
                    message=f"均匀遮挡(无模板兜底): std={pred_std:.1f}"
                )

            if pred_dark_ratio > self.fallback_max_dark_ratio:
                score = min(1.0, pred_dark_ratio / self.fallback_max_dark_ratio - 1.0 + 0.5)
                return PreCheckReport(
                    result=PreCheckResult.DARK_OCCLUSION,
                    is_anomaly=True, anomaly_score=score, confidence=0.6,
                    details=details,
                    message=f"深色遮挡(无模板兜底): dark_ratio={pred_dark_ratio:.2%}"
                )

            return PreCheckReport(
                result=PreCheckResult.NO_TEMPLATE,
                is_anomaly=False, anomaly_score=0.0, confidence=0.3,
                details=details,
                message=f"[{category}] 未注册模板，跳过对比检测"
            )

    def visualize(self, roi: np.ndarray, report: PreCheckReport) -> np.ndarray:
        vis = roi.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        color = (0, 0, 255) if report.is_anomaly else (0, 255, 0)
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w-1, h-1), color, 3)
        cv2.putText(vis, report.result.value, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        d = report.details
        info = (f"Δdark={d.get('dark_delta', d.get('pred_dark_ratio', 0)):+.2f} "
                f"Δmean={d.get('mean_drop', 0):+.0f}")
        cv2.putText(vis, info, (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
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
