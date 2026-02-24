"""
模板匹配模块，用于工件方向对齐
支持基于特征点的模板匹配和透视变换对齐
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TemplateInfo:
    """模板信息"""
    name: str
    image: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    width: int
    height: int


@dataclass
class MatchResult:
    """匹配结果"""
    success: bool
    aligned_image: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    match_score: float = 0.0
    error_message: str = ""


class TemplateMatcher:
    """
    基于ORB特征点的模板匹配器
    支持旋转、缩放、平移的工件对齐
    """

    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, TemplateInfo] = {}
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.min_match_count = 10
        self.match_ratio = 0.75

    def load_template(self, name: str, image: np.ndarray) -> bool:
        """
        加载模板图片
        :param name: 模板名称
        :param image: 模板图片 (BGR格式)
        :return: 是否成功
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < self.min_match_count:
                logger.error(f"Template {name}: Not enough keypoints detected ({len(keypoints)})")
                return False

            self.templates[name] = TemplateInfo(
                name=name,
                image=image.copy(),
                keypoints=keypoints,
                descriptors=descriptors,
                width=image.shape[1],
                height=image.shape[0]
            )
            logger.info(f"Template {name} loaded: {len(keypoints)} keypoints")
            return True
        except Exception as e:
            logger.error(f"Failed to load template {name}: {e}")
            return False

    def save_template_to_disk(self, name: str, image: np.ndarray) -> bool:
        """
        保存模板图片到磁盘
        :param name: 模板名称
        :param image: 模板图片
        :return: 是否成功
        """
        try:
            template_path = self.template_dir / f"{name}.png"
            cv2.imwrite(str(template_path), image)
            logger.info(f"Template saved to {template_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save template {name}: {e}")
            return False

    def load_template_from_disk(self, name: str) -> bool:
        """
        从磁盘加载模板
        :param name: 模板名称（不含扩展名）
        :return: 是否成功
        """
        try:
            template_path = self.template_dir / f"{name}.png"
            if not template_path.exists():
                logger.error(f"Template file not found: {template_path}")
                return False

            image = cv2.imread(str(template_path))
            if image is None:
                logger.error(f"Failed to read template image: {template_path}")
                return False

            return self.load_template(name, image)
        except Exception as e:
            logger.error(f"Failed to load template from disk {name}: {e}")
            return False

    def delete_template(self, name: str) -> bool:
        """
        删除模板
        :param name: 模板名称
        :return: 是否成功
        """
        try:
            # 从内存中删除
            if name in self.templates:
                del self.templates[name]

            # 从磁盘删除
            template_path = self.template_dir / f"{name}.png"
            if template_path.exists():
                template_path.unlink()

            logger.info(f"Template {name} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete template {name}: {e}")
            return False

    def list_templates(self) -> List[str]:
        """
        列出所有可用的模板
        :return: 模板名称列表
        """
        templates = []
        for template_file in self.template_dir.glob("*.png"):
            templates.append(template_file.stem)
        return templates

    def match_and_align(self, image: np.ndarray, template_name: str,
                        min_match_count: int = None) -> MatchResult:
        """
        匹配并对齐图片
        :param image: 输入图片 (BGR格式)
        :param template_name: 模板名称
        :param min_match_count: 最小匹配点数
        :return: 匹配结果
        """
        if template_name not in self.templates:
            return MatchResult(
                success=False,
                error_message=f"Template {template_name} not found"
            )

        min_count = min_match_count or self.min_match_count
        template = self.templates[template_name]

        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检测特征点
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < min_count:
                return MatchResult(
                    success=False,
                    error_message=f"Not enough keypoints in input image ({len(keypoints) if keypoints else 0})"
                )

            # 特征匹配
            matches = self.matcher.knnMatch(descriptors, template.descriptors, k=2)

            # 应用比率测试
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)

            if len(good_matches) < min_count:
                return MatchResult(
                    success=False,
                    error_message=f"Not enough good matches ({len(good_matches)} < {min_count})"
                )

            # 提取匹配点坐标
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([template.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算单应性矩阵
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if homography is None:
                return MatchResult(
                    success=False,
                    error_message="Failed to compute homography matrix"
                )

            # 计算匹配分数
            match_score = np.sum(mask) / len(mask) if mask is not None else 0

            # 应用透视变换对齐图片
            aligned_image = cv2.warpPerspective(image, homography, (template.width, template.height))

            return MatchResult(
                success=True,
                aligned_image=aligned_image,
                homography=homography,
                match_score=float(match_score)
            )

        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return MatchResult(
                success=False,
                error_message=str(e)
            )

    def batch_match_and_align(self, images: List[np.ndarray], template_name: str
                             ) -> List[MatchResult]:
        """
        批量匹配并对齐图片
        :param images: 输入图片列表
        :param template_name: 模板名称
        :return: 匹配结果列表
        """
        results = []
        for image in images:
            result = self.match_and_align(image, template_name)
            results.append(result)
        return results

    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        获取模板信息
        :param template_name: 模板名称
        :return: 模板信息字典
        """
        if template_name not in self.templates:
            return None

        template = self.templates[template_name]
        return {
            "name": template.name,
            "width": template.width,
            "height": template.height,
            "keypoints_count": len(template.keypoints)
        }

    def auto_load_templates(self):
        """
        自动加载所有磁盘上的模板
        """
        templates = self.list_templates()
        loaded_count = 0
        for name in templates:
            if self.load_template_from_disk(name):
                loaded_count += 1
        logger.info(f"Auto-loaded {loaded_count}/{len(templates)} templates")
        return loaded_count


# 全局模板匹配器实例
template_matcher = TemplateMatcher()
