"""
相机驱动接口抽象基类
定义所有相机驱动必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class DriverCapability:
    """驱动能力描述"""
    supports_exposure_manual: bool = True
    supports_exposure_auto: bool = True
    supports_gain_manual: bool = True
    supports_gain_auto: bool = True
    supports_offset: bool = True
    supports_white_balance: bool = False
    supports_trigger_mode: bool = True
    supports_frame_rate: bool = True
    exposure_time_range: Tuple[Optional[int], Optional[int]] = (None, None)  # (min, max) in us
    gain_range: Tuple[Optional[int], Optional[int]] = (None, None)  # (min, max)
    exposure_time_step: int = 1  # 曝光时间调整步长
    offset_x_step: int = 1  # 水平偏移调整步长
    offset_y_step: int = 1  # 垂直偏移调整步长


@dataclass
class DiscoveredDevice:
    """发现的设备信息"""
    ip_address: str
    model: str
    serial: str
    device_class: str
    vendor: str
    device_id: Optional[str] = None  # 驱动特定的设备标识


class ICameraDriver(ABC):
    """
    相机驱动接口抽象基类
    所有品牌相机驱动必须实现此接口
    """

    @property
    @abstractmethod
    def vendor_name(self) -> str:
        """返回相机品牌名称，如 'Basler', 'Hikrobot'"""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """驱动SDK是否可用（是否已安装）"""
        pass

    @property
    @abstractmethod
    def capability(self) -> DriverCapability:
        """返回驱动能力描述"""
        pass

    @abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        连接相机
        
        Args:
            connection_params: 连接参数，包含ip_address等
            
        Returns:
            (success: bool, message: str)
        """
        pass

    @abstractmethod
    def disconnect(self) -> Tuple[bool, str]:
        """
        断开相机连接
        
        Returns:
            (success: bool, message: str)
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查相机是否已连接"""
        pass

    @abstractmethod
    def capture(self, timeout_ms: int = 5000) -> Tuple[bool, Any]:
        """
        采集单帧图像
        
        Args:
            timeout_ms: 超时时间（毫秒）
            
        Returns:
            (success: bool, result: np.ndarray or error_message: str)
        """
        pass

    @abstractmethod
    def start_grabbing(self) -> Tuple[bool, str]:
        """开始连续采集"""
        pass

    @abstractmethod
    def stop_grabbing(self) -> Tuple[bool, str]:
        """停止连续采集"""
        pass

    @abstractmethod
    def is_grabbing(self) -> bool:
        """检查是否正在采集"""
        pass

    # ========== 参数获取/设置接口 ==========

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """获取当前分辨率 (width, height)"""
        pass

    @abstractmethod
    def set_resolution(self, width: int, height: int) -> Tuple[bool, str]:
        """设置分辨率"""
        pass

    @abstractmethod
    def get_max_resolution(self) -> Tuple[int, int]:
        """获取最大分辨率 (max_width, max_height)"""
        pass

    @abstractmethod
    def get_exposure_time(self) -> Optional[int]:
        """获取曝光时间（微秒）"""
        pass

    @abstractmethod
    def set_exposure_time(self, exposure_us: int) -> Tuple[bool, str]:
        """设置曝光时间（微秒）"""
        pass

    @abstractmethod
    def get_gain(self) -> Optional[int]:
        """获取增益值"""
        pass

    @abstractmethod
    def set_gain(self, gain: int) -> Tuple[bool, str]:
        """设置增益值"""
        pass

    @abstractmethod
    def get_offset(self) -> Tuple[Optional[int], Optional[int]]:
        """获取偏移量 (offset_x, offset_y)"""
        pass

    @abstractmethod
    def set_offset(self, offset_x: int, offset_y: int) -> Tuple[bool, str]:
        """设置偏移量"""
        pass

    @abstractmethod
    def set_center_alignment(self, enabled: bool) -> Tuple[bool, str]:
        """设置中心对齐"""
        pass

    @abstractmethod
    def get_auto_exposure(self) -> str:
        """获取自动曝光模式"""
        pass

    @abstractmethod
    def set_auto_exposure(self, mode: str) -> Tuple[bool, str]:
        """设置自动曝光模式 (Off, Once, Continuous)"""
        pass

    @abstractmethod
    def get_auto_gain(self) -> str:
        """获取自动增益模式"""
        pass

    @abstractmethod
    def set_auto_gain(self, mode: str) -> Tuple[bool, str]:
        """设置自动增益模式 (Off, Once, Continuous)"""
        pass

    @abstractmethod
    def get_frame_rate(self) -> Optional[float]:
        """获取帧率"""
        pass

    @abstractmethod
    def set_frame_rate(self, fps: float) -> Tuple[bool, str]:
        """设置帧率"""
        pass

    @abstractmethod
    def get_pixel_format(self) -> str:
        """获取像素格式"""
        pass

    @abstractmethod
    def set_pixel_format(self, pixel_format: str) -> Tuple[bool, str]:
        """设置像素格式"""
        pass

    # ========== 网络参数接口 ==========

    @abstractmethod
    def get_packet_size(self) -> int:
        """获取数据包大小"""
        pass

    @abstractmethod
    def set_packet_size(self, packet_size: int) -> Tuple[bool, str]:
        """设置数据包大小"""
        pass

    @abstractmethod
    def get_payload_size(self) -> int:
        """获取负载大小（字节）"""
        pass

    # ========== 状态信息接口 ==========

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        pass

    @abstractmethod
    def get_temperature(self) -> Optional[float]:
        """获取相机温度（摄氏度）"""
        pass

    # ========== 静态方法 ==========

    @staticmethod
    @abstractmethod
    def discover_devices() -> List[DiscoveredDevice]:
        """发现网络中的设备"""
        pass

    @staticmethod
    @abstractmethod
    def check_sdk_available() -> bool:
        """检查SDK是否可用"""
        pass
