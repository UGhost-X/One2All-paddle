"""
多品牌相机管理器 V2
支持 Basler、Hikrobot 等多种品牌相机
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

from services.camera_instance import CameraInstance, CameraConfig, CameraStatus
from services.drivers import (
    discover_all_cameras, 
    list_available_vendors, 
    check_vendor_available,
    AVAILABLE_DRIVERS
)

logger = logging.getLogger(__name__)


class CameraManager:
    """
    多品牌相机管理器（支持多相机）
    """

    _instance: Optional['CameraManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._cameras: Dict[str, CameraInstance] = {}
        self._lock = threading.RLock()

    def create_camera(self, camera_id: str, 
                      vendor: str = "Basler",
                      ip_address: str = "192.168.110.10",
                      width: Optional[int] = None, 
                      height: Optional[int] = None,
                      packet_size: int = 1500,
                      exposure_time: Optional[int] = None,
                      gain: Optional[int] = None,
                      offset_x: Optional[int] = None,
                      offset_y: Optional[int] = None) -> Tuple[bool, str]:
        """
        创建新的相机实例
        
        Args:
            camera_id: 相机唯一标识
            vendor: 品牌，如 "Basler", "Hikrobot"
            ip_address: IP地址
            width/height: 分辨率
            packet_size: 数据包大小
            exposure_time: 曝光时间
            gain: 增益
            offset_x/y: 偏移量
            
        Returns:
            (success, message)
        """
        with self._lock:
            if camera_id in self._cameras:
                return False, f"Camera {camera_id} already exists"

            # 检查品牌是否可用
            if vendor not in list_available_vendors():
                return False, f"Unknown vendor: {vendor}. Available: {list_available_vendors()}"

            config = CameraConfig(
                camera_id=camera_id,
                vendor=vendor,
                ip_address=ip_address,
                width=width,
                height=height,
                packet_size=packet_size,
                exposure_time=exposure_time,
                gain=gain,
                offset_x=offset_x,
                offset_y=offset_y
            )

            camera = CameraInstance(config)
            
            if not camera.is_available:
                return False, f"Driver for {vendor} is not available. Please install the SDK."

            self._cameras[camera_id] = camera
            logger.info(f"Camera {camera_id} created with vendor {vendor}, IP {ip_address}")
            return True, f"Camera {camera_id} created"

    def remove_camera(self, camera_id: str) -> Tuple[bool, str]:
        """移除相机实例"""
        with self._lock:
            if camera_id not in self._cameras:
                return False, f"Camera {camera_id} not found"

            camera = self._cameras[camera_id]
            if camera.status.connected:
                camera.disconnect()

            del self._cameras[camera_id]
            logger.info(f"Camera {camera_id} removed")
            return True, f"Camera {camera_id} removed"

    def get_camera(self, camera_id: str) -> Optional[CameraInstance]:
        """获取指定相机实例"""
        with self._lock:
            return self._cameras.get(camera_id)

    def list_cameras(self) -> List[Dict[str, Any]]:
        """列出所有相机"""
        with self._lock:
            return [
                {
                    "camera_id": cid,
                    "vendor": cam.config.vendor,
                    "ip_address": cam.config.ip_address,
                    "connected": cam.status.connected,
                    "resolution": f"{cam.status.width}x{cam.status.height}" if cam.status.connected else None
                }
                for cid, cam in self._cameras.items()
            ]

    def get_all_status(self) -> List[CameraStatus]:
        """获取所有相机状态"""
        with self._lock:
            return [cam.status for cam in self._cameras.values()]

    def discover_cameras(self, vendor: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        发现网络中的相机
        
        Args:
            vendor: 指定品牌，None表示发现所有品牌
            
        Returns:
            发现的相机列表
        """
        discovered = []
        
        if vendor:
            # 发现指定品牌
            if vendor in AVAILABLE_DRIVERS:
                driver_class = AVAILABLE_DRIVERS[vendor]
                if driver_class.check_sdk_available():
                    devices = driver_class.discover_devices()
                    for device in devices:
                        discovered.append({
                            "vendor": vendor,
                            "ip_address": device.ip_address,
                            "model": device.model,
                            "serial": device.serial,
                            "device_class": device.device_class
                        })
        else:
            # 发现所有品牌
            all_devices = discover_all_cameras()
            for vnd, devices in all_devices.items():
                for device in devices:
                    discovered.append({
                        "vendor": vnd,
                        "ip_address": device.ip_address,
                        "model": device.model,
                        "serial": device.serial,
                        "device_class": device.device_class
                    })
        
        return discovered

    def get_available_vendors(self) -> List[str]:
        """获取所有可用的品牌列表"""
        return list_available_vendors()

    def check_vendor_status(self) -> Dict[str, bool]:
        """检查各品牌SDK可用状态"""
        return {
            vendor: check_vendor_available(vendor)
            for vendor in list_available_vendors()
        }

    def switch_camera_vendor(self, camera_id: str, new_vendor: str) -> Tuple[bool, str]:
        """
        切换相机品牌（保留配置）
        
        Args:
            camera_id: 相机ID
            new_vendor: 新品牌
            
        Returns:
            (success, message)
        """
        with self._lock:
            if camera_id not in self._cameras:
                return False, f"Camera {camera_id} not found"

            old_camera = self._cameras[camera_id]
            old_config = old_camera.config

            # 断开旧连接
            if old_camera.status.connected:
                old_camera.disconnect()

            # 创建新配置（保留IP、分辨率等）
            new_config = CameraConfig(
                camera_id=camera_id,
                vendor=new_vendor,
                ip_address=old_config.ip_address,
                width=old_config.width,
                height=old_config.height,
                packet_size=old_config.packet_size,
                exposure_time=old_config.exposure_time,
                gain=old_config.gain,
                offset_x=old_config.offset_x,
                offset_y=old_config.offset_y
            )

            # 创建新实例
            new_camera = CameraInstance(new_config)
            
            if not new_camera.is_available:
                # 恢复旧实例
                self._cameras[camera_id] = old_camera
                return False, f"Driver for {new_vendor} is not available"

            self._cameras[camera_id] = new_camera
            logger.info(f"Camera {camera_id} switched from {old_config.vendor} to {new_vendor}")
            return True, f"Camera switched to {new_vendor}"


# 全局管理器实例
camera_manager = CameraManager()
