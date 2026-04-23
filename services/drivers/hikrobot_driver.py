"""
Hikrobot (海康机器人) 相机驱动实现
基于 MvCameraControl_class.py 的SDK封装
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import ctypes
from ctypes import *

from services.camera_driver_interface import ICameraDriver, DriverCapability, DiscoveredDevice

logger = logging.getLogger(__name__)

# 添加MvImport路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "dependencies", "MvImport"))


class HikrobotDriver(ICameraDriver):
    """Hikrobot (海康机器人) 相机驱动"""

    def __init__(self):
        self._mv_camera = None  # MvCamera实例
        self._handle = None
        self._is_connected = False
        self._is_grabbing = False
        self._capability = DriverCapability(
            supports_exposure_manual=True,
            supports_exposure_auto=True,
            supports_gain_manual=True,
            supports_gain_auto=True,
            supports_offset=True,
            supports_white_balance=True,
            supports_trigger_mode=True,
            supports_frame_rate=True,
            exposure_time_range=(14, 10000000),  # 海康典型范围
            gain_range=(0, 255),
            exposure_time_step=1,
            offset_x_step=4,
            offset_y_step=4
        )
        self._current_width = 0
        self._current_height = 0
        self._pixel_format = "BGR8"
        self._try_import_sdk()

    def _try_import_sdk(self):
        try:
            import MvCameraControl_class
            import MvErrorDefine_const
            import CameraParams_header
            self._sdk_module = {
                'MvCamera': MvCameraControl_class.MvCamera,
                'MV_CC_DEVICE_INFO_LIST': CameraParams_header.MV_CC_DEVICE_INFO_LIST,
                'MV_CC_DEVICE_INFO': CameraParams_header.MV_CC_DEVICE_INFO,
                'MV_FRAME_OUT': CameraParams_header.MV_FRAME_OUT,
                'MV_GIGE_DEVICE': CameraParams_header.MV_GIGE_DEVICE,
                'MV_USB_DEVICE': CameraParams_header.MV_USB_DEVICE,
                'MV_GENTL_CAMERALINK_DEVICE': CameraParams_header.MV_GENTL_CAMERALINK_DEVICE,
                'MV_GENTL_CXP_DEVICE': CameraParams_header.MV_GENTL_CXP_DEVICE,
                'MV_GENTL_XOF_DEVICE': CameraParams_header.MV_GENTL_XOF_DEVICE,
            }
            logger.info("Hikrobot SDK imported successfully")
        except ImportError as e:
            self._sdk_module = None
            logger.warning(f"Hikrobot SDK not available: {e}")

    @property
    def vendor_name(self) -> str:
        return "Hikrobot"

    @property
    def is_available(self) -> bool:
        return self._sdk_module is not None

    @property
    def capability(self) -> DriverCapability:
        return self._capability

    def connect(self, connection_params: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.is_available:
            return False, "Hikrobot SDK not installed"

        try:
            ip_address = connection_params.get("ip_address", "192.168.110.10")
            
            MvCamera = self._sdk_module['MvCamera']
            MV_CC_DEVICE_INFO_LIST = self._sdk_module['MV_CC_DEVICE_INFO_LIST']
            MV_CC_DEVICE_INFO = self._sdk_module['MV_CC_DEVICE_INFO']
            MV_GIGE_DEVICE = self._sdk_module['MV_GIGE_DEVICE']
            MV_USB_DEVICE = self._sdk_module['MV_USB_DEVICE']
            
            # 初始化SDK
            MvCamera.MV_CC_Initialize()
            
            # 枚举设备
            device_list = MV_CC_DEVICE_INFO_LIST()
            tlayer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | 
                          self._sdk_module['MV_GENTL_CAMERALINK_DEVICE'] |
                          self._sdk_module['MV_GENTL_CXP_DEVICE'] |
                          self._sdk_module['MV_GENTL_XOF_DEVICE'])
            
            ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
            if ret != 0:
                return False, f"Enum devices failed: 0x{ret:08x}"
            
            if device_list.nDeviceNum == 0:
                return False, "No device found"
            
            # 查找匹配IP的设备
            target_device_index = -1
            for i in range(device_list.nDeviceNum):
                mvcc_dev_info = ctypes.cast(
                    device_list.pDeviceInfo[i], 
                    ctypes.POINTER(MV_CC_DEVICE_INFO)
                ).contents
                
                if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                    # 获取GigE设备的IP
                    ip = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
                    device_ip = f"{(ip >> 24) & 0xff}.{(ip >> 16) & 0xff}.{(ip >> 8) & 0xff}.{ip & 0xff}"
                    if device_ip == ip_address:
                        target_device_index = i
                        break
            
            if target_device_index == -1:
                # 如果没有匹配IP，使用第一个设备
                target_device_index = 0
                logger.warning(f"Device with IP {ip_address} not found, using first device")
            
            # 创建相机实例
            self._mv_camera = MvCamera()
            
            # 选择设备
            mvcc_dev_info = ctypes.cast(
                device_list.pDeviceInfo[target_device_index],
                ctypes.POINTER(MV_CC_DEVICE_INFO)
            ).contents
            
            ret = self._mv_camera.MV_CC_CreateHandle(mvcc_dev_info)
            if ret != 0:
                return False, f"Create handle failed: 0x{ret:08x}"
            
            # 打开设备
            ret = self._mv_camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                return False, f"Open device failed: 0x{ret:08x}"
            
            self._is_connected = True
            
            # 获取当前分辨率
            self._current_width, self._current_height = self.get_resolution()
            
            logger.info(f"Hikrobot camera connected: {ip_address}")
            return True, f"Connected to {ip_address}"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect Hikrobot camera: {error_msg}")
            self._cleanup()
            return False, error_msg

    def disconnect(self) -> Tuple[bool, str]:
        self._cleanup()
        self._is_connected = False
        logger.info("Hikrobot camera disconnected")
        return True, "Camera disconnected"

    def _cleanup(self):
        if self._mv_camera:
            try:
                if self._is_grabbing:
                    self._mv_camera.MV_CC_StopGrabbing()
                self._mv_camera.MV_CC_CloseDevice()
                self._mv_camera.MV_CC_DestroyHandle()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            finally:
                self._mv_camera = None
                self._is_grabbing = False

    def is_connected(self) -> bool:
        return self._is_connected

    def capture(self, timeout_ms: int = 5000) -> Tuple[bool, Any]:
        if not self.is_connected():
            return False, "Camera not connected"

        was_grabbing = self._is_grabbing
        
        try:
            MV_FRAME_OUT = self._sdk_module['MV_FRAME_OUT']
            
            if not was_grabbing:
                ret = self._mv_camera.MV_CC_StartGrabbing()
                if ret != 0:
                    return False, f"Start grabbing failed: 0x{ret:08x}"
                self._is_grabbing = True

            stOutFrame = MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
            
            ret = self._mv_camera.MV_CC_GetImageBuffer(stOutFrame, timeout_ms)
            if ret != 0:
                return False, f"Get image buffer failed: 0x{ret:08x}"
            
            if stOutFrame.pBufAddr is None:
                return False, "Image buffer is null"
            
            # 转换为numpy数组
            width = stOutFrame.stFrameInfo.nWidth
            height = stOutFrame.stFrameInfo.nHeight
            pixel_type = stOutFrame.stFrameInfo.enPixelType
            
            # 根据像素格式转换
            if pixel_type == 0x1080001:  # PixelType_Gvsp_Mono8
                img = np.ctypeslib.as_array(
                    ctypes.cast(stOutFrame.pBufAddr, ctypes.POINTER(ctypes.c_ubyte)),
                    shape=(height, width)
                ).copy()
                # 转为BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif pixel_type == 0x02180014:  # PixelType_Gvsp_BGR8_Packed
                img = np.ctypeslib.as_array(
                    ctypes.cast(stOutFrame.pBufAddr, ctypes.POINTER(ctypes.c_ubyte)),
                    shape=(height, width, 3)
                ).copy()
            elif pixel_type == 0x02180015:  # PixelType_Gvsp_RGB8_Packed
                img = np.ctypeslib.as_array(
                    ctypes.cast(stOutFrame.pBufAddr, ctypes.POINTER(ctypes.c_ubyte)),
                    shape=(height, width, 3)
                ).copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                # 其他格式，尝试通用转换
                img = np.ctypeslib.as_array(
                    ctypes.cast(stOutFrame.pBufAddr, ctypes.POINTER(ctypes.c_ubyte)),
                    shape=(height, width, 3)
                ).copy()
            
            # 释放缓冲区
            self._mv_camera.MV_CC_FreeImageBuffer(stOutFrame)
            
            if not was_grabbing:
                self._mv_camera.MV_CC_StopGrabbing()
                self._is_grabbing = False
            
            return True, img

        except Exception as e:
            return False, str(e)

    def start_grabbing(self) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            ret = self._mv_camera.MV_CC_StartGrabbing()
            if ret != 0:
                return False, f"Start grabbing failed: 0x{ret:08x}"
            self._is_grabbing = True
            return True, "Grabbing started"
        except Exception as e:
            return False, str(e)

    def stop_grabbing(self) -> Tuple[bool, str]:
        if not self.is_connected():
            return True, "Camera not connected"
        
        try:
            ret = self._mv_camera.MV_CC_StopGrabbing()
            if ret != 0:
                return False, f"Stop grabbing failed: 0x{ret:08x}"
            self._is_grabbing = False
            return True, "Grabbing stopped"
        except Exception as e:
            return False, str(e)

    def is_grabbing(self) -> bool:
        return self._is_grabbing

    def get_resolution(self) -> Tuple[int, int]:
        if not self.is_connected():
            return (0, 0)
        try:
            # 使用IntValue获取整数类型的节点值
            width = ctypes.c_int()
            height = ctypes.c_int()
            self._mv_camera.MV_CC_GetIntValue("Width", ctypes.byref(width))
            self._mv_camera.MV_CC_GetIntValue("Height", ctypes.byref(height))
            return (width.value, height.value)
        except:
            return (self._current_width, self._current_height)

    def set_resolution(self, width: int, height: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            # 停止采集
            if self._is_grabbing:
                self._mv_camera.MV_CC_StopGrabbing()
            
            # 设置分辨率
            ret = self._mv_camera.MV_CC_SetIntValue("Width", width)
            if ret != 0:
                return False, f"Set width failed: 0x{ret:08x}"
            
            ret = self._mv_camera.MV_CC_SetIntValue("Height", height)
            if ret != 0:
                return False, f"Set height failed: 0x{ret:08x}"
            
            self._current_width = width
            self._current_height = height
            
            return True, f"Resolution set to {width}x{height}"
        except Exception as e:
            return False, str(e)

    def get_max_resolution(self) -> Tuple[int, int]:
        if not self.is_connected():
            return (0, 0)
        try:
            # 获取最大值
            stParam = MVCC_INTVALUE()
            self._mv_camera.MV_CC_GetIntValue("Width", ctypes.byref(stParam))
            max_w = stParam.nMax
            self._mv_camera.MV_CC_GetIntValue("Height", ctypes.byref(stParam))
            max_h = stParam.nMax
            return (max_w, max_h)
        except:
            return (self._current_width, self._current_height)

    def get_exposure_time(self) -> Optional[int]:
        if not self.is_connected():
            return None
        try:
            exposure = ctypes.c_float()
            self._mv_camera.MV_CC_GetFloatValue("ExposureTime", ctypes.byref(exposure))
            return int(exposure.value)
        except:
            return None

    def set_exposure_time(self, exposure_us: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            ret = self._mv_camera.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
            if ret != 0:
                return False, f"Set exposure time failed: 0x{ret:08x}"
            return True, f"Exposure time set to {exposure_us} μs"
        except Exception as e:
            return False, str(e)

    def get_gain(self) -> Optional[int]:
        if not self.is_connected():
            return None
        try:
            gain = ctypes.c_float()
            self._mv_camera.MV_CC_GetFloatValue("Gain", ctypes.byref(gain))
            return int(gain.value)
        except:
            return None

    def set_gain(self, gain: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            ret = self._mv_camera.MV_CC_SetFloatValue("Gain", float(gain))
            if ret != 0:
                return False, f"Set gain failed: 0x{ret:08x}"
            return True, f"Gain set to {gain}"
        except Exception as e:
            return False, str(e)

    def get_offset(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.is_connected():
            return (None, None)
        try:
            offset_x = ctypes.c_int()
            offset_y = ctypes.c_int()
            self._mv_camera.MV_CC_GetIntValue("OffsetX", ctypes.byref(offset_x))
            self._mv_camera.MV_CC_GetIntValue("OffsetY", ctypes.byref(offset_y))
            return (offset_x.value, offset_y.value)
        except:
            return (None, None)

    def set_offset(self, offset_x: int, offset_y: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            # OffsetX 必须是 4 的倍数
            offset_x = (offset_x // 4) * 4
            # OffsetY 必须是 4 的倍数
            offset_y = (offset_y // 4) * 4
            
            ret = self._mv_camera.MV_CC_SetIntValue("OffsetX", offset_x)
            if ret != 0:
                return False, f"Set offset X failed: 0x{ret:08x}"
            
            ret = self._mv_camera.MV_CC_SetIntValue("OffsetY", offset_y)
            if ret != 0:
                return False, f"Set offset Y failed: 0x{ret:08x}"
            
            return True, f"Offset set to ({offset_x}, {offset_y})"
        except Exception as e:
            return False, str(e)

    def set_center_alignment(self, enabled: bool) -> Tuple[bool, str]:
        # 海康相机没有直接的CenterX/CenterY参数
        # 通过计算偏移量实现
        if not self.is_connected():
            return False, "Camera not connected"
        
        if enabled:
            try:
                max_w, max_h = self.get_max_resolution()
                cur_w, cur_h = self.get_resolution()
                
                offset_x = (max_w - cur_w) // 2
                offset_y = (max_h - cur_h) // 2
                
                return self.set_offset(offset_x, offset_y)
            except Exception as e:
                return False, str(e)
        return True, "Center alignment not supported"

    def get_auto_exposure(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            mode = ctypes.c_int()
            self._mv_camera.MV_CC_GetEnumValue("ExposureAuto", ctypes.byref(mode))
            modes = {0: "Off", 1: "Once", 2: "Continuous"}
            return modes.get(mode.value, "Unknown")
        except:
            return "Unknown"

    def set_auto_exposure(self, mode: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            mode_map = {"Off": 0, "Once": 1, "Continuous": 2}
            mode_value = mode_map.get(mode, 0)
            
            ret = self._mv_camera.MV_CC_SetEnumValue("ExposureAuto", mode_value)
            if ret != 0:
                return False, f"Set auto exposure failed: 0x{ret:08x}"
            return True, f"Auto exposure set to {mode}"
        except Exception as e:
            return False, str(e)

    def get_auto_gain(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            mode = ctypes.c_int()
            self._mv_camera.MV_CC_GetEnumValue("GainAuto", ctypes.byref(mode))
            modes = {0: "Off", 1: "Once", 2: "Continuous"}
            return modes.get(mode.value, "Unknown")
        except:
            return "Unknown"

    def set_auto_gain(self, mode: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            mode_map = {"Off": 0, "Once": 1, "Continuous": 2}
            mode_value = mode_map.get(mode, 0)
            
            ret = self._mv_camera.MV_CC_SetEnumValue("GainAuto", mode_value)
            if ret != 0:
                return False, f"Set auto gain failed: 0x{ret:08x}"
            return True, f"Auto gain set to {mode}"
        except Exception as e:
            return False, str(e)

    def get_frame_rate(self) -> Optional[float]:
        if not self.is_connected():
            return None
        try:
            fps = ctypes.c_float()
            self._mv_camera.MV_CC_GetFloatValue("AcquisitionFrameRate", ctypes.byref(fps))
            return float(fps.value)
        except:
            return None

    def set_frame_rate(self, fps: float) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            ret = self._mv_camera.MV_CC_SetFloatValue("AcquisitionFrameRate", fps)
            if ret != 0:
                return False, f"Set frame rate failed: 0x{ret:08x}"
            return True, f"Frame rate set to {fps}"
        except Exception as e:
            return False, str(e)

    def get_pixel_format(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            pixel_format = ctypes.c_int()
            self._mv_camera.MV_CC_GetEnumValue("PixelFormat", ctypes.byref(pixel_format))
            # 转换像素格式代码为字符串
            formats = {
                0x1080001: "Mono8",
                0x02180014: "BGR8",
                0x02180015: "RGB8",
            }
            return formats.get(pixel_format.value, f"0x{pixel_format.value:08x}")
        except:
            return "Unknown"

    def set_pixel_format(self, pixel_format: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            format_map = {
                "Mono8": 0x1080001,
                "BGR8": 0x02180014,
                "RGB8": 0x02180015,
            }
            format_value = format_map.get(pixel_format, 0x02180014)
            
            ret = self._mv_camera.MV_CC_SetEnumValue("PixelFormat", format_value)
            if ret != 0:
                return False, f"Set pixel format failed: 0x{ret:08x}"
            self._pixel_format = pixel_format
            return True, f"Pixel format set to {pixel_format}"
        except Exception as e:
            return False, str(e)

    def get_packet_size(self) -> int:
        if not self.is_connected():
            return 0
        try:
            packet_size = ctypes.c_int()
            self._mv_camera.MV_CC_GetIntValue("GevSCPSPacketSize", ctypes.byref(packet_size))
            return packet_size.value
        except:
            return 0

    def set_packet_size(self, packet_size: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            ret = self._mv_camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
            if ret != 0:
                return False, f"Set packet size failed: 0x{ret:08x}"
            return True, f"Packet size set to {packet_size}"
        except Exception as e:
            return False, str(e)

    def get_payload_size(self) -> int:
        if not self.is_connected():
            return 0
        try:
            payload = ctypes.c_int()
            self._mv_camera.MV_CC_GetIntValue("PayloadSize", ctypes.byref(payload))
            return payload.value
        except:
            return 0

    def get_device_info(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {}
        
        try:
            # 获取设备信息
            stDevInfo = MV_CC_DEVICE_INFO()
            # 这里需要通过其他方式获取设备信息
            return {
                "vendor": "Hikrobot",
                "model": "Unknown",  # 需要通过其他接口获取
                "serial": "Unknown",
                "version": "Unknown",
            }
        except Exception as e:
            return {"error": str(e)}

    def get_temperature(self) -> Optional[float]:
        if not self.is_connected():
            return None
        try:
            temp = ctypes.c_float()
            self._mv_camera.MV_CC_GetFloatValue("DeviceTemperature", ctypes.byref(temp))
            return float(temp.value)
        except:
            return None

    @staticmethod
    def discover_devices() -> List[DiscoveredDevice]:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "dependencies", "MvImport"))
            from MvCameraControl_class import MvCamera
            from CameraParams_header import MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO
            from CameraParams_const import MV_GIGE_DEVICE, MV_USB_DEVICE
            
            # 初始化SDK
            MvCamera.MV_CC_Initialize()
            
            device_list = MV_CC_DEVICE_INFO_LIST()
            tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
            
            ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
            if ret != 0 or device_list.nDeviceNum == 0:
                return []
            
            discovered = []
            for i in range(device_list.nDeviceNum):
                mvcc_dev_info = ctypes.cast(
                    device_list.pDeviceInfo[i],
                    ctypes.POINTER(MV_CC_DEVICE_INFO)
                ).contents
                
                ip = "Unknown"
                model = "Unknown"
                serial = "Unknown"
                
                if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                    ip_val = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
                    ip = f"{(ip_val >> 24) & 0xff}.{(ip_val >> 16) & 0xff}.{(ip_val >> 8) & 0xff}.{ip_val & 0xff}"
                    # 获取型号和序列号
                    try:
                        model_bytes = bytes(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                        model = model_bytes.decode('utf-8', errors='ignore').strip('\x00')
                        serial_bytes = bytes(mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber)
                        serial = serial_bytes.decode('utf-8', errors='ignore').strip('\x00')
                    except:
                        pass
                
                discovered.append(DiscoveredDevice(
                    ip_address=ip,
                    model=model,
                    serial=serial,
                    device_class="GigE" if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE else "USB3",
                    vendor="Hikrobot"
                ))
            
            return discovered
        except Exception as e:
            logger.error(f"Failed to discover Hikrobot cameras: {e}")
            return []

    @staticmethod
    def check_sdk_available() -> bool:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "dependencies", "MvImport"))
            from MvCameraControl_class import MvCamera
            return True
        except ImportError:
            return False
