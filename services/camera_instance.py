"""
品牌无关的相机实例包装类
通过驱动接口与具体相机品牌解耦
"""

import os
import time
import base64
import logging
import threading
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from services.camera_driver_interface import ICameraDriver, DriverCapability
from services.drivers import get_driver, list_available_vendors

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """相机配置"""
    camera_id: str
    vendor: str = "Basler"  # 品牌: Basler, Hikrobot 等
    ip_address: str = "192.168.110.10"
    width: Optional[int] = None
    height: Optional[int] = None
    packet_size: int = 1500
    frame_retention: int = 2000000000
    timeout_ms: int = 120000
    max_retry: int = 5
    # 图像参数
    exposure_time: Optional[int] = None  # 曝光时间（微秒）
    gain: Optional[int] = None  # 增益
    offset_x: Optional[int] = None  # 水平偏移
    offset_y: Optional[int] = None  # 垂直偏移


@dataclass
class CameraStatus:
    """相机状态"""
    camera_id: str
    vendor: str = ""
    connected: bool = False
    width: int = 0
    height: int = 0
    max_width: int = 0
    max_height: int = 0
    payload_size_mb: float = 0.0
    packet_size: int = 0
    frame_retention_ms: float = 0.0
    last_capture_time: Optional[float] = None
    last_error: Optional[str] = None
    # 图像参数
    exposure_time: Optional[int] = None
    gain: Optional[int] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    exposure_auto: str = ""
    gain_auto: str = ""


class CameraInstance:
    """
    品牌无关的相机实例
    通过驱动接口与具体相机品牌解耦
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._driver: Optional[ICameraDriver] = None
        self._status = CameraStatus(camera_id=config.camera_id, vendor=config.vendor)
        self._lock = threading.RLock()
        self._capability: Optional[DriverCapability] = None
        
        self._init_driver()

    def _init_driver(self):
        """初始化驱动"""
        try:
            driver_class = get_driver(self.config.vendor)
            self._driver = driver_class()
            self._capability = self._driver.capability
            self._status.vendor = self.config.vendor
        except Exception as e:
            logger.error(f"Failed to initialize driver for {self.config.vendor}: {e}")
            self._driver = None
            self._status.last_error = f"Driver initialization failed: {e}"

    @property
    def is_available(self) -> bool:
        """驱动是否可用"""
        return self._driver is not None and self._driver.is_available

    @property
    def status(self) -> CameraStatus:
        """获取当前状态（返回副本）"""
        with self._lock:
            return CameraStatus(**asdict(self._status))

    @property
    def capability(self) -> Optional[DriverCapability]:
        """获取驱动能力"""
        return self._capability

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    def connect(self) -> Tuple[bool, str]:
        """连接相机"""
        if not self.is_available:
            return False, f"Driver for {self.config.vendor} not available"

        with self._lock:
            if self._driver.is_connected():
                return True, "Camera already connected"

            try:
                # 构建连接参数
                connection_params = {
                    "ip_address": self.config.ip_address,
                    # frame_retention 必须在 connect() 阶段通过 stream grabber 设置，
                    # 因为 _apply_config() 中没有设置它的入口
                    "frame_retention": self.config.frame_retention,
                }

                success, message = self._driver.connect(connection_params)
                if not success:
                    self._status.last_error = message
                    return False, message

                # 应用配置
                self._apply_config()

                # 更新状态
                self._update_status()
                self._status.connected = True
                self._status.last_error = None

                logger.info(f"Camera {self.config.camera_id} connected: {self._status.width}x{self._status.height}")
                return True, f"Connected at {self._status.width}x{self._status.height}"

            except Exception as e:
                error_msg = str(e)
                self._status.last_error = error_msg
                logger.error(f"Failed to connect camera {self.config.camera_id}: {error_msg}")
                return False, error_msg

    def _apply_config(self):
        """应用配置到相机"""
        if not self._driver.is_connected():
            return

        # 设置分辨率
        if self.config.width is not None and self.config.height is not None:
            max_w, max_h = self._driver.get_max_resolution()
            target_w = min(self.config.width, max_w)
            target_h = min(self.config.height, max_h)
            
            is_full_resolution = (target_w == max_w and target_h == max_h)
            
            if not is_full_resolution:
                # 启用中心对齐
                self._driver.set_center_alignment(True)
            
            self._driver.set_resolution(target_w, target_h)
            
            if not is_full_resolution:
                # 关闭中心对齐以便手动调整偏移
                self._driver.set_center_alignment(False)

        # 设置曝光时间和增益
        self._driver.set_auto_exposure("Off")
        self._driver.set_auto_gain("Off")

        if self.config.exposure_time is not None:
            # 根据驱动能力调整步长
            if self._capability:
                step = self._capability.exposure_time_step
                exposure = (self.config.exposure_time // step) * step
            else:
                exposure = self.config.exposure_time
            self._driver.set_exposure_time(exposure)

        if self.config.gain is not None:
            self._driver.set_gain(self.config.gain)

        # 设置偏移
        if self.config.offset_x is not None or self.config.offset_y is not None:
            current_x, current_y = self._driver.get_offset()
            new_x = self.config.offset_x if self.config.offset_x is not None else (current_x or 0)
            new_y = self.config.offset_y if self.config.offset_y is not None else (current_y or 0)
            self._driver.set_offset(new_x, new_y)

        # 设置数据包大小
        self._driver.set_packet_size(self.config.packet_size)

    def _update_status(self):
        """从驱动更新状态"""
        if not self._driver.is_connected():
            self._status.connected = False
            return

        self._status.width, self._status.height = self._driver.get_resolution()
        self._status.max_width, self._status.max_height = self._driver.get_max_resolution()
        self._status.exposure_time = self._driver.get_exposure_time()
        self._status.gain = self._driver.get_gain()
        self._status.offset_x, self._status.offset_y = self._driver.get_offset()
        self._status.exposure_auto = self._driver.get_auto_exposure()
        self._status.gain_auto = self._driver.get_auto_gain()
        self._status.packet_size = self._driver.get_packet_size()
        
        payload = self._driver.get_payload_size()
        self._status.payload_size_mb = payload / 1024 / 1024

    def disconnect(self) -> Tuple[bool, str]:
        """断开相机连接"""
        with self._lock:
            if self._driver:
                success, message = self._driver.disconnect()
                self._status.connected = False
                logger.info(f"Camera {self.config.camera_id} disconnected")
                return success, message
            return True, "Camera not connected"

    def capture(self, save_path: Optional[str] = None, return_base64: bool = True) -> Tuple[bool, Any]:
        """
        采集单帧图像
        
        Args:
            save_path: 保存路径（可选）
            return_base64: 是否返回base64编码
            
        Returns:
            (success, result)
        """
        if not self.is_available:
            return False, f"Driver for {self.config.vendor} not available"

        with self._lock:
            # 自动连接
            if not self._driver.is_connected():
                success, msg = self.connect()
                if not success:
                    return False, msg

            for attempt in range(self.config.max_retry):
                try:
                    success, result = self._driver.capture(self.config.timeout_ms)
                    
                    if not success:
                        logger.warning(f"Capture attempt {attempt+1} failed: {result}")
                        time.sleep(0.5)
                        continue

                    img = result  # numpy array
                    self._status.last_capture_time = time.time()

                    # 保存文件
                    if save_path:
                        cv2.imwrite(save_path, img)
                        logger.info(f"Image saved to {save_path}")

                    # 返回结果
                    if return_base64:
                        _, buffer = cv2.imencode('.jpg', img)
                        base64_str = base64.b64encode(buffer).decode('utf-8')
                        return True, {
                            "image_base64": base64_str,
                            "width": img.shape[1],
                            "height": img.shape[0],
                            "save_path": save_path
                        }
                    else:
                        return True, {
                            "width": img.shape[1],
                            "height": img.shape[0],
                            "save_path": save_path,
                            "message": "Image captured successfully"
                        }

                except Exception as e:
                    logger.warning(f"Capture attempt {attempt+1} failed: {e}")
                    time.sleep(0.5)

            error_msg = f"Failed to capture after {self.config.max_retry} attempts"
            self._status.last_error = error_msg
            return False, error_msg

    def get_supported_resolutions(self) -> List[Dict[str, Any]]:
        """获取支持的分辨率列表"""
        common_resolutions = [
            {"width": 640, "height": 480},
            {"width": 800, "height": 600},
            {"width": 1024, "height": 768},
            {"width": 1280, "height": 720},
            {"width": 1280, "height": 1024},
            {"width": 1600, "height": 1200},
            {"width": 1920, "height": 1080},
            {"width": 2048, "height": 1536},
            {"width": 2560, "height": 1440},
            {"width": 2592, "height": 1944},
            {"width": 3840, "height": 2160},
            {"width": 4096, "height": 3072},
        ]

        with self._lock:
            if self._driver and self._driver.is_connected():
                max_w, max_h = self._driver.get_max_resolution()
            elif self._status.max_width > 0 and self._status.max_height > 0:
                max_w, max_h = self._status.max_width, self._status.max_height
            else:
                return common_resolutions

        supported = []
        for res in common_resolutions:
            if res["width"] <= max_w and res["height"] <= max_h:
                supported.append(res)

        supported.append({"width": max_w, "height": max_h, "label": "Maximum"})
        return supported

    def update_config(self, **kwargs) -> Tuple[bool, str]:
        """
        更新配置（需要断开重连）
        
        Args:
            **kwargs: 配置参数
            
        Returns:
            (success, message)
        """
        with self._lock:
            was_connected = self._driver and self._driver.is_connected()

            if was_connected:
                self.disconnect()

            # 更新配置
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Camera {self.config.camera_id} config updated: {key} = {value}")

            # 如果品牌改变，重新初始化驱动
            if 'vendor' in kwargs:
                self._init_driver()

            if was_connected:
                return self.connect()

            return True, "Configuration updated"

    def update_parameters(self, **kwargs) -> Tuple[bool, str]:
        """
        动态更新相机参数（不需要断开重连）
        
        Args:
            exposure_time: 曝光时间（微秒）
            gain: 增益
            offset_x: 水平偏移
            offset_y: 垂直偏移
            
        Returns:
            (success, message)
        """
        with self._lock:
            if not self._driver or not self._driver.is_connected():
                return False, "Camera not connected"

            updated = []

            if 'exposure_time' in kwargs:
                exposure = kwargs['exposure_time']
                if self._capability:
                    step = self._capability.exposure_time_step
                    exposure = (exposure // step) * step
                success, msg = self._driver.set_exposure_time(exposure)
                if success:
                    self.config.exposure_time = exposure
                    updated.append(f"exposure_time={exposure}")
                else:
                    return False, msg

            if 'gain' in kwargs:
                gain = kwargs['gain']
                success, msg = self._driver.set_gain(gain)
                if success:
                    self.config.gain = gain
                    updated.append(f"gain={gain}")
                else:
                    return False, msg

            if 'offset_x' in kwargs or 'offset_y' in kwargs:
                current_x, current_y = self._driver.get_offset()
                new_x = kwargs.get('offset_x', current_x or 0)
                new_y = kwargs.get('offset_y', current_y or 0)
                success, msg = self._driver.set_offset(new_x, new_y)
                if success:
                    self.config.offset_x = new_x
                    self.config.offset_y = new_y
                    updated.append(f"offset=({new_x}, {new_y})")
                else:
                    return False, msg

            # 更新状态
            self._update_status()

            if updated:
                logger.info(f"Camera {self.config.camera_id} parameters updated: {', '.join(updated)}")
                return True, f"Parameters updated: {', '.join(updated)}"
            else:
                return True, "No parameters to update"

    def get_driver_info(self) -> Dict[str, Any]:
        """获取驱动信息"""
        return {
            "vendor": self.config.vendor,
            "available": self.is_available,
            "connected": self._driver.is_connected() if self._driver else False,
            "capability": {
                "supports_exposure_manual": self._capability.supports_exposure_manual if self._capability else False,
                "supports_exposure_auto": self._capability.supports_exposure_auto if self._capability else False,
                "supports_gain_manual": self._capability.supports_gain_manual if self._capability else False,
                "supports_offset": self._capability.supports_offset if self._capability else False,
                "exposure_time_range": self._capability.exposure_time_range if self._capability else (None, None),
                "gain_range": self._capability.gain_range if self._capability else (None, None),
            } if self._capability else None
        }