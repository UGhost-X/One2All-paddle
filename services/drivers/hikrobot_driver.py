"""
Hikrobot (海康机器人) 相机驱动实现
基于官方SDK示例重写
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import cv2
import ctypes
from ctypes import *

from services.camera_driver_interface import ICameraDriver, DriverCapability, DiscoveredDevice

logger = logging.getLogger(__name__)

# 添加MvImport路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "MvImport"))


class HikrobotDriver(ICameraDriver):
    """Hikrobot (海康机器人) 相机驱动"""

    def __init__(self):
        self._mv_camera = None
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
            exposure_time_range=(14, 10000000),
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
        """导入2D相机SDK"""
        try:
            import MvCameraControl_class
            import MvErrorDefine_const
            import CameraParams_header
            import PixelType_header
            
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
                'MV_GENTL_GIGE_DEVICE': CameraParams_header.MV_GENTL_GIGE_DEVICE,
                'MV_ACCESS_Exclusive': CameraParams_header.MV_ACCESS_Exclusive,
                'MV_TRIGGER_MODE_OFF': CameraParams_header.MV_TRIGGER_MODE_OFF,
                'PixelType': PixelType_header,
                'MVCC_INTVALUE': CameraParams_header.MVCC_INTVALUE,
                'MVCC_FLOATVALUE': CameraParams_header.MVCC_FLOATVALUE,
                'MVCC_ENUMVALUE': CameraParams_header.MVCC_ENUMVALUE,
            }
            
            MvCameraControl_class.MvCamera.MV_CC_Initialize()
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
        """连接相机"""
        if not self.is_available:
            return False, "Hikrobot SDK not installed"

        ip_address = connection_params.get("ip_address")
        return self._connect_2d_camera(ip_address)

    def _connect_2d_camera(self, ip_address: str = None) -> Tuple[bool, str]:
        """2D相机连接流程"""
        try:
            MvCamera = self._sdk_module['MvCamera']
            MV_CC_DEVICE_INFO_LIST = self._sdk_module['MV_CC_DEVICE_INFO_LIST']
            MV_CC_DEVICE_INFO = self._sdk_module['MV_CC_DEVICE_INFO']
            MV_GIGE_DEVICE = self._sdk_module['MV_GIGE_DEVICE']
            MV_USB_DEVICE = self._sdk_module['MV_USB_DEVICE']
            MV_GENTL_GIGE_DEVICE = self._sdk_module['MV_GENTL_GIGE_DEVICE']
            MV_ACCESS_Exclusive = self._sdk_module['MV_ACCESS_Exclusive']
            MV_TRIGGER_MODE_OFF = self._sdk_module['MV_TRIGGER_MODE_OFF']
            
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
            
            target_device_index = -1
            for i in range(device_list.nDeviceNum):
                mvcc_dev_info = cast(
                    device_list.pDeviceInfo[i], 
                    POINTER(MV_CC_DEVICE_INFO)
                ).contents
                
                if mvcc_dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                    ip = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
                    device_ip = f"{(ip >> 24) & 0xff}.{(ip >> 16) & 0xff}.{(ip >> 8) & 0xff}.{ip & 0xff}"
                    if ip_address and device_ip == ip_address:
                        target_device_index = i
                        break
                    elif not ip_address and target_device_index == -1:
                        target_device_index = i
            
            if target_device_index == -1:
                target_device_index = 0
                logger.warning(f"Device with IP {ip_address} not found, using first device")
            
            self._mv_camera = MvCamera()
            
            mvcc_dev_info = cast(
                device_list.pDeviceInfo[target_device_index],
                POINTER(MV_CC_DEVICE_INFO)
            ).contents
            
            ret = self._mv_camera.MV_CC_CreateHandle(mvcc_dev_info)
            if ret != 0:
                return False, f"Create handle failed: 0x{ret:08x}"
            
            ret = self._mv_camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                return False, f"Open device failed: 0x{ret:08x}"
            
            if mvcc_dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                nPacketSize = self._mv_camera.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self._mv_camera.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        logger.warning(f"Set Packet Size fail: 0x{ret:08x}")
                else:
                    logger.warning(f"Get Packet Size fail: 0x{nPacketSize:08x}")
            
            ret = self._mv_camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                logger.warning(f"Set trigger mode fail: 0x{ret:08x}")
            
            self._is_connected = True
            self._current_width, self._current_height = self.get_resolution()
            
            return True, "Connected"

        except Exception as e:
            import traceback
            logger.error(f"Camera connect failed: {e}\n{traceback.format_exc()}")
            self._cleanup()
            return False, str(e)

    def disconnect(self) -> Tuple[bool, str]:
        self._cleanup()
        self._is_connected = False
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

    def start_grabbing(self) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            if not self._is_grabbing:
                ret = self._mv_camera.MV_CC_StartGrabbing()
                if ret != 0:
                    return False, f"Start grabbing failed: 0x{ret:08x}"
                self._is_grabbing = True
            return True, "Grabbing started"
        except Exception as e:
            return False, str(e)

    def stop_grabbing(self) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            if self._is_grabbing:
                ret = self._mv_camera.MV_CC_StopGrabbing()
                if ret != 0:
                    return False, f"Stop grabbing failed: 0x{ret:08x}"
                self._is_grabbing = False
            return True, "Grabbing stopped"
        except Exception as e:
            return False, str(e)

    def is_grabbing(self) -> bool:
        return self._is_grabbing

    def get_frame_rate(self) -> Optional[float]:
        if not self.is_connected():
            return None
        try:
            stValue = self._sdk_module['MVCC_FLOATVALUE']()
            ret = self._mv_camera.MV_CC_GetFloatValue("AcquisitionFrameRate", stValue)
            if ret != 0:
                return None
            return float(stValue.fCurValue)
        except:
            return None

    def set_frame_rate(self, fps: float) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            ret = self._mv_camera.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
            if ret != 0:
                return False, f"Set frame rate failed: 0x{ret:08x}"
            return True, f"Frame rate set to {fps}"
        except Exception as e:
            return False, str(e)

    def get_pixel_format(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            stValue = self._sdk_module['MVCC_ENUMVALUE']()
            ret = self._mv_camera.MV_CC_GetEnumValue("PixelFormat", stValue)
            if ret != 0:
                return "Unknown"
            format_map = {
                0x01080001: "Mono8",
                0x01080008: "BayerGR8",
                0x01080009: "BayerRG8",
                0x0108000A: "BayerGB8",
                0x0108000B: "BayerBG8",
                0x02180014: "RGB8",
                0x02180015: "BGR8",
            }
            return format_map.get(stValue.nCurValue, f"0x{stValue.nCurValue:08x}")
        except:
            return "Unknown"

    def set_pixel_format(self, pixel_format: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            # 常见像素格式反向映射
            format_map = {
                "Mono8": 0x01080001,
                "BayerGR8": 0x01080008,
                "BayerRG8": 0x01080009,
                "BayerGB8": 0x0108000A,
                "BayerBG8": 0x0108000B,
                "RGB8": 0x02180014,
                "BGR8": 0x02180015,
            }
            if pixel_format in format_map:
                fmt_value = format_map[pixel_format]
            elif pixel_format.startswith("0x"):
                fmt_value = int(pixel_format, 16)
            else:
                return False, f"Unsupported pixel format: {pixel_format}"
            
            ret = self._mv_camera.MV_CC_SetEnumValue("PixelFormat", fmt_value)
            if ret != 0:
                return False, f"Set pixel format failed: 0x{ret:08x}"
            return True, f"Pixel format set to {pixel_format}"
        except Exception as e:
            return False, str(e)

    def get_device_info(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {}
        try:
            info = {
                "vendor": "Hikrobot",
                "connected": self._is_connected,
                "resolution": self.get_resolution(),
                "pixel_format": self.get_pixel_format(),
                "exposure_time": self.get_exposure_time(),
                "gain": self.get_gain(),
            }
            return info
        except:
            return {}

    def get_temperature(self) -> Optional[float]:
        if not self.is_connected():
            return None
        try:
            stValue = self._sdk_module['MVCC_FLOATVALUE']()
            ret = self._mv_camera.MV_CC_GetFloatValue("DeviceTemperature", stValue)
            if ret != 0:
                return None
            return float(stValue.fCurValue)
        except:
            return None

    def capture(self, timeout_ms: int = 5000) -> Tuple[bool, Any]:
        if not self.is_connected():
            return False, "Camera not connected"

        try:
            return self._capture_2d(timeout_ms)
        except Exception as e:
            import traceback
            logger.error(f"Capture error: {e}\n{traceback.format_exc()}")
            return False, str(e)

    def _capture_2d(self, timeout_ms: int = 5000) -> Tuple[bool, Any]:
        """使用2D相机SDK采集图像"""
        import time
        was_grabbing = self._is_grabbing
        max_retries = 3

        try:
            MV_FRAME_OUT = self._sdk_module['MV_FRAME_OUT']
            PixelType = self._sdk_module['PixelType']

            if not was_grabbing:
                ret = self._mv_camera.MV_CC_StartGrabbing()
                if ret != 0:
                    return False, f"Start grabbing failed: 0x{ret:08x}"
                self._is_grabbing = True
                time.sleep(0.2)

            stOutFrame = MV_FRAME_OUT()
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))

            last_ret = 0
            for attempt in range(max_retries):
                ret = self._mv_camera.MV_CC_GetImageBuffer(stOutFrame, timeout_ms // max_retries)
                if ret == 0 and stOutFrame.pBufAddr is not None:
                    break
                last_ret = ret
                logger.warning(f"Get image buffer attempt {attempt + 1}/{max_retries} failed: 0x{ret:08x}")
                if attempt < max_retries - 1:
                    time.sleep(0.1)
            else:
                return False, f"Get image buffer failed after {max_retries} attempts: 0x{last_ret:08x}"

            width = stOutFrame.stFrameInfo.nWidth
            height = stOutFrame.stFrameInfo.nHeight
            pixel_type = stOutFrame.stFrameInfo.enPixelType

            img = self._convert_pixel_format(stOutFrame, width, height, pixel_type, PixelType)

            self._mv_camera.MV_CC_FreeImageBuffer(stOutFrame)

            if not was_grabbing:
                self._mv_camera.MV_CC_StopGrabbing()
                self._is_grabbing = False

            return True, img

        except Exception as e:
            import traceback
            logger.error(f"Capture error: {e}\n{traceback.format_exc()}")
            return False, str(e)

    def _convert_pixel_format(self, stOutFrame, width, height, pixel_type, PixelType):
        """使用SDK转换RGB，失败时fallback手动解码"""
        from MvCameraControl_class import MV_CC_PIXEL_CONVERT_PARAM_EX
        
        nRGBSize = width * height * 3
        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        stConvertParam.nWidth = width
        stConvertParam.nHeight = height
        stConvertParam.pSrcData = stOutFrame.pBufAddr
        stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
        stConvertParam.enSrcPixelType = pixel_type
        stConvertParam.enDstPixelType = PixelType.PixelType_Gvsp_RGB8_Packed
        stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
        stConvertParam.nDstBufferSize = nRGBSize
        
        ret = self._mv_camera.MV_CC_ConvertPixelTypeEx(stConvertParam)
        if ret == 0:
            img = np.ctypeslib.as_array(
                cast(stConvertParam.pDstBuffer, POINTER(c_ubyte)),
                shape=(height, width, 3)
            ).copy()
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        logger.warning(f"ConvertPixelTypeEx failed (0x{ret:08x}), using fallback for 0x{pixel_type:08x}")
        return self._fallback_convert(stOutFrame, width, height, pixel_type)
    
    def _fallback_convert(self, stOutFrame, width, height, pixel_type) -> np.ndarray:
        """手动fallback转换，支持Mono8/16, BayerXX8/10/12/16"""
        frame_len = stOutFrame.stFrameInfo.nFrameLen
        raw = np.ctypeslib.as_array(
            cast(stOutFrame.pBufAddr, POINTER(c_ubyte)),
            shape=(frame_len,)
        ).copy()
        
        # Mono8
        if pixel_type == 0x01080001:
            return cv2.cvtColor(raw.reshape(height, width), cv2.COLOR_GRAY2BGR)
        
        # Mono10/12/16 packed
        if pixel_type in (0x01100003, 0x01100005, 0x01100007):
            bits = {0x01100003: 10, 0x01100005: 12, 0x01100007: 16}[pixel_type]
            raw_uint16 = raw.view(np.uint16)
            img = (raw_uint16 >> (bits - 8)).astype(np.uint8).reshape(height, width)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Bayer8
        bayer_map = {
            0x01080008: (cv2.COLOR_BAYER_GR2BGR, "GR"),
            0x01080009: (cv2.COLOR_BAYER_RG2BGR, "RG"),
            0x0108000A: (cv2.COLOR_BAYER_GB2BGR, "GB"),
            0x0108000B: (cv2.COLOR_BAYER_BG2BGR, "BG"),
        }
        if pixel_type in bayer_map:
            code, name = bayer_map[pixel_type]
            img = raw.reshape(height, width)
            try:
                return cv2.cvtColor(img, code)
            except cv2.error:
                logger.warning(f"OpenCV Bayer conversion failed for {name}, returning grayscale")
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Unknown
        logger.error(f"Unsupported pixel format: 0x{pixel_type:08x}, returning empty image")
        return np.zeros((height, width, 3), dtype=np.uint8)

    def set_resolution(self, width: int, height: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        def _set_resolution_impl():
            self._mv_camera.MV_CC_SetIntValue("Width", width)
            self._mv_camera.MV_CC_SetIntValue("Height", height)
            self._current_width = width
            self._current_height = height
            return True, f"Resolution set to {width}x{height}"
        
        return self._pause_grabbing_for_config(_set_resolution_impl)

    def get_resolution(self) -> Tuple[int, int]:
        if not self.is_connected():
            return (0, 0)
        try:
            MVCC_INTVALUE = self._sdk_module['MVCC_INTVALUE']
            stWidth = MVCC_INTVALUE()
            stHeight = MVCC_INTVALUE()
            ret = self._mv_camera.MV_CC_GetIntValue("Width", stWidth)
            ret = self._mv_camera.MV_CC_GetIntValue("Height", stHeight)
            return (int(stWidth.nCurValue), int(stHeight.nCurValue))
        except Exception as e:
            logger.warning(f"get_resolution error: {e}")
            return (self._current_width, self._current_height)

    def get_max_resolution(self) -> Tuple[int, int]:
        if not self.is_connected():
            return (0, 0)
        try:
            MVCC_INTVALUE = self._sdk_module['MVCC_INTVALUE']
            stWidth = MVCC_INTVALUE()
            stHeight = MVCC_INTVALUE()
            self._mv_camera.MV_CC_GetIntValue("WidthMax", stWidth)
            self._mv_camera.MV_CC_GetIntValue("HeightMax", stHeight)
            return (int(stWidth.nCurValue), int(stHeight.nCurValue))
        except:
            return (self._current_width, self._current_height)

    def set_exposure_time(self, exposure_time: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            self._mv_camera.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
            return True, f"Exposure time set to {exposure_time}"
        except Exception as e:
            return False, str(e)

    def get_exposure_time(self) -> Optional[int]:
        if not self.is_connected():
            return None
        try:
            stExposure = self._sdk_module['MVCC_FLOATVALUE']()
            self._mv_camera.MV_CC_GetFloatValue("ExposureTime", stExposure)
            return int(stExposure.fCurValue)
        except:
            return None

    def set_gain(self, gain: float) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            self._mv_camera.MV_CC_SetFloatValue("Gain", float(gain))
            return True, f"Gain set to {gain}"
        except Exception as e:
            return False, str(e)

    def get_gain(self) -> Optional[float]:
        if not self.is_connected():
            return None
        try:
            stGain = self._sdk_module['MVCC_FLOATVALUE']()
            self._mv_camera.MV_CC_GetFloatValue("Gain", stGain)
            return float(stGain.fCurValue)
        except:
            return None

    def set_offset(self, offset_x: int, offset_y: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            self._mv_camera.MV_CC_SetIntValue("OffsetX", offset_x)
            self._mv_camera.MV_CC_SetIntValue("OffsetY", offset_y)
            return True, f"Offset set to ({offset_x}, {offset_y})"
        except Exception as e:
            return False, str(e)

    def get_offset(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.is_connected():
            return (None, None)
        try:
            MVCC_INTVALUE = self._sdk_module['MVCC_INTVALUE']
            stOffsetX = MVCC_INTVALUE()
            stOffsetY = MVCC_INTVALUE()
            self._mv_camera.MV_CC_GetIntValue("OffsetX", stOffsetX)
            self._mv_camera.MV_CC_GetIntValue("OffsetY", stOffsetY)
            return (int(stOffsetX.nCurValue), int(stOffsetY.nCurValue))
        except:
            return (None, None)

    def set_center_alignment(self, enable: bool) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            value = 1 if enable else 0
            self._mv_camera.MV_CC_SetBoolValue("CenterAlignment", value)
            return True, f"Center alignment set to {enable}"
        except Exception as e:
            return False, str(e)

    def set_auto_exposure(self, mode: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            modes = {"Off": 0, "Once": 1, "Continuous": 2}
            value = modes.get(mode, 0)
            ret = self._mv_camera.MV_CC_SetEnumValue("ExposureAuto", value)
            if ret != 0:
                return False, f"Set auto exposure failed: 0x{ret:08x}"
            return True, f"Auto exposure set to {mode}"
        except Exception as e:
            return False, str(e)

    def get_auto_exposure(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            stValue = self._sdk_module['MVCC_ENUMVALUE']()
            ret = self._mv_camera.MV_CC_GetEnumValue("ExposureAuto", stValue)
            if ret != 0:
                return "Unknown"
            modes = {0: "Off", 1: "Once", 2: "Continuous"}
            return modes.get(stValue.nCurValue, "Unknown")
        except:
            return "Unknown"

    def set_auto_gain(self, mode: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            modes = {"Off": 0, "Once": 1, "Continuous": 2}
            value = modes.get(mode, 0)
            ret = self._mv_camera.MV_CC_SetEnumValue("GainAuto", value)
            if ret != 0:
                return False, f"Set auto gain failed: 0x{ret:08x}"
            return True, f"Auto gain set to {mode}"
        except Exception as e:
            return False, str(e)

    def get_auto_gain(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            stValue = self._sdk_module['MVCC_ENUMVALUE']()
            ret = self._mv_camera.MV_CC_GetEnumValue("GainAuto", stValue)
            if ret != 0:
                return "Unknown"
            modes = {0: "Off", 1: "Once", 2: "Continuous"}
            return modes.get(stValue.nCurValue, "Unknown")
        except:
            return "Unknown"

    def set_packet_size(self, packet_size: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        try:
            self._mv_camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
            return True, f"Packet size set to {packet_size}"
        except Exception as e:
            return False, str(e)

    def get_packet_size(self) -> int:
        if not self.is_connected():
            return 0
        try:
            stValue = self._sdk_module['MVCC_INTVALUE']()
            self._mv_camera.MV_CC_GetIntValue("GevSCPSPacketSize", stValue)
            return int(stValue.nCurValue)
        except:
            return 0

    def get_payload_size(self) -> int:
        if not self.is_connected():
            return 0
        try:
            stValue = self._sdk_module['MVCC_INTVALUE']()
            self._mv_camera.MV_CC_GetIntValue("PayloadSize", stValue)
            return int(stValue.nCurValue)
        except:
            return 0

    def _pause_grabbing_for_config(self, func):
        """暂停取流执行配置操作，完成后恢复"""
        was_grabbing = self._is_grabbing
        if was_grabbing:
            try:
                self._mv_camera.MV_CC_StopGrabbing()
                self._is_grabbing = False
            except Exception as e:
                logger.warning(f"Failed to stop grabbing: {e}")
        try:
            result = func()
        finally:
            if was_grabbing:
                try:
                    self._mv_camera.MV_CC_StartGrabbing()
                    self._is_grabbing = True
                except Exception as e:
                    logger.warning(f"Failed to restart grabbing: {e}")
        return result

    @staticmethod
    def discover_devices() -> List[DiscoveredDevice]:
        devices = []
        try:
            import MvCameraControl_class
            import CameraParams_header
            
            MvCameraControl_class.MvCamera.MV_CC_Initialize()
            
            device_list = CameraParams_header.MV_CC_DEVICE_INFO_LIST()
            tlayer_type = (CameraParams_header.MV_GIGE_DEVICE | 
                          CameraParams_header.MV_USB_DEVICE |
                          CameraParams_header.MV_GENTL_CAMERALINK_DEVICE |
                          CameraParams_header.MV_GENTL_CXP_DEVICE |
                          CameraParams_header.MV_GENTL_XOF_DEVICE)
            
            ret = MvCameraControl_class.MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
            if ret != 0 or device_list.nDeviceNum == 0:
                MvCameraControl_class.MvCamera.MV_CC_Finalize()
                return devices
            
            for i in range(device_list.nDeviceNum):
                mvcc_dev_info = ctypes.cast(
                    device_list.pDeviceInfo[i],
                    ctypes.POINTER(CameraParams_header.MV_CC_DEVICE_INFO)
                ).contents
                
                if mvcc_dev_info.nTLayerType == CameraParams_header.MV_GIGE_DEVICE:
                    ip = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
                    ip_address = f"{(ip >> 24) & 0xff}.{(ip >> 16) & 0xff}.{(ip >> 8) & 0xff}.{ip & 0xff}"
                    
                    model_name = ""
                    for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                        if per == 0:
                            break
                        model_name += chr(per)
                    
                    serial = ""
                    for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                        if per == 0:
                            break
                        serial += chr(per)
                    
                    devices.append(DiscoveredDevice(
                        vendor="Hikrobot",
                        ip_address=ip_address,
                        model=model_name,
                        serial=serial,
                        device_class="GigE"
                    ))
            
            MvCameraControl_class.MvCamera.MV_CC_Finalize()
            
        except Exception as e:
            logger.error(f"Failed to discover Hikrobot devices: {e}")
        
        return devices

    @staticmethod
    def check_sdk_available() -> bool:
        try:
            import MvCameraControl_class
            return True
        except ImportError:
            return False
