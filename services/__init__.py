"""
相机服务模块
支持多品牌相机：Basler, Hikrobot
"""

# V2 API (推荐) - 多品牌支持
from services.camera_manager import CameraManager, camera_manager
from services.camera_instance import CameraInstance, CameraConfig, CameraStatus

# 驱动相关
from services.camera_driver_interface import ICameraDriver, DriverCapability, DiscoveredDevice
from services.drivers import (
    BaslerDriver,
    HikrobotDriver,
    AVAILABLE_DRIVERS,
    get_driver,
    list_available_vendors,
    check_vendor_available,
    discover_all_cameras,
)

# 为了保持向后兼容，V1 API仍然可用
# 但建议使用V2 API以获得多品牌支持
try:
    from services.camera_manager import CameraManager as CameraManagerV1
    from services.camera_manager import camera_manager as camera_manager_v1
    from services.camera_service import CameraService, camera_service
except ImportError:
    CameraManagerV1 = None
    camera_manager_v1 = None
    CameraService = None
    camera_service = None

__all__ = [
    # V2 API (推荐)
    "CameraManager",
    "camera_manager",
    "CameraInstance",
    "CameraConfig",
    "CameraStatus",
    
    # 驱动接口
    "ICameraDriver",
    "DriverCapability",
    "DiscoveredDevice",
    "BaslerDriver",
    "HikrobotDriver",
    "AVAILABLE_DRIVERS",
    "get_driver",
    "list_available_vendors",
    "check_vendor_available",
    "discover_all_cameras",
    
    # V1 API (兼容)
    "CameraManagerV1",
    "camera_manager_v1",
    "CameraService",
    "camera_service",
]
