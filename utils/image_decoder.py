import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional

# 尝试导入 TurboJPEG
try:
    from turbojpeg import TurboJPEG
    _turbo_jpeg = TurboJPEG()
    _TURBOJPEG_AVAILABLE = True
except Exception:
    # 导入失败或版本不兼容
    _TURBOJPEG_AVAILABLE = False
    _turbo_jpeg = None


def is_turbo_jpeg_available() -> bool:
    """检查 TurboJPEG 是否可用"""
    return _TURBOJPEG_AVAILABLE


def decode_image(data: Union[bytes, str, Path], use_turbo: bool = True) -> np.ndarray:
    """
    解码图像数据
    
    Args:
        data: 图像数据（字节、文件路径）
        use_turbo: 是否尝试使用 TurboJPEG（如果可用）
    
    Returns:
        解码后的图像数组 (numpy array)
    """
    # 如果是路径，读取文件
    if isinstance(data, (str, Path)):
        with open(data, 'rb') as f:
            data = f.read()
    
    # 尝试使用 TurboJPEG 解码（如果可用且启用）
    if use_turbo and _TURBOJPEG_AVAILABLE and _turbo_jpeg is not None:
        try:
            # 检查是否是 JPEG 格式
            if data[:2] == b'\xff\xd8':  # JPEG magic number
                return _turbo_jpeg.decode(data)
        except Exception:
            pass  # 解码失败，回退到 OpenCV
    
    # 使用 OpenCV 解码
    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("无法解码图像数据")
    
    return image
