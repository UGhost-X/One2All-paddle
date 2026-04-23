"""
相机驱动模块
"""

from services.camera_driver_interface import ICameraDriver, DriverCapability, DiscoveredDevice
from services.drivers.basler_driver import BaslerDriver
from services.drivers.hikrobot_driver import HikrobotDriver

# 注册所有可用的驱动
AVAILABLE_DRIVERS = {
    "Basler": BaslerDriver,
    "Hikrobot": HikrobotDriver,
}


def get_driver(vendor: str) -> type:
    """
    获取指定品牌的驱动类
    
    Args:
        vendor: 品牌名称，如 "Basler", "Hikrobot"
        
    Returns:
        驱动类
        
    Raises:
        ValueError: 如果品牌不存在
    """
    if vendor not in AVAILABLE_DRIVERS:
        raise ValueError(f"Unknown vendor: {vendor}. Available: {list(AVAILABLE_DRIVERS.keys())}")
    return AVAILABLE_DRIVERS[vendor]


def list_available_vendors() -> list:
    """列出所有可用的相机品牌"""
    return list(AVAILABLE_DRIVERS.keys())


def check_vendor_available(vendor: str) -> bool:
    """检查指定品牌的SDK是否可用"""
    try:
        driver_class = get_driver(vendor)
        return driver_class.check_sdk_available()
    except:
        return False


def discover_all_cameras() -> dict:
    """
    发现所有品牌的相机
    
    Returns:
        dict: {vendor: [DiscoveredDevice, ...]}
    """
    results = {}
    for vendor, driver_class in AVAILABLE_DRIVERS.items():
        try:
            if driver_class.check_sdk_available():
                devices = driver_class.discover_devices()
                if devices:
                    results[vendor] = devices
        except Exception as e:
            pass
    return results


__all__ = [
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
]
