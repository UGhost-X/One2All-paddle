import os
import time
import base64
import logging
import threading
from io import BytesIO
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

os.environ["PYLON_GIGE_HEARTBEAT"] = "3000"
os.environ["PYLON_GIGE_FRAMERETENTION"] = "2000000000"


@dataclass
class CameraConfig:
    ip_address: str = "192.168.110.10"
    width: Optional[int] = None
    height: Optional[int] = None
    packet_size: int = 1500
    frame_retention: int = 2000000000
    timeout_ms: int = 120000
    max_retry: int = 5
    # 新增参数
    exposure_time: Optional[int] = None  # 曝光时间（微秒），必须是52的倍数
    gain: Optional[int] = None  # 增益（0-1957）
    offset_x: Optional[int] = None  # 水平偏移，全尺寸时失效
    offset_y: Optional[int] = None  # 垂直偏移，全尺寸时失效


@dataclass
class CameraStatus:
    connected: bool = False
    width: int = 0
    height: int = 0
    max_width: int = 0
    max_height: int = 0
    payload_size_mb: float = 0.0
    packet_size: int = 0
    scpd: int = 0
    frame_retention_ms: float = 0.0
    last_capture_time: Optional[float] = None
    last_error: Optional[str] = None
    # 新增状态
    exposure_time: Optional[int] = None
    gain: Optional[int] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    exposure_auto: str = ""
    gain_auto: str = ""


class CameraService:
    _instance: Optional['CameraService'] = None
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

        self._camera = None
        self._converter = None
        self._config = CameraConfig()
        self._status = CameraStatus()
        self._lock = threading.RLock()
        self._pylon = None
        self._pylon_available = False

        self._try_import_pylon()

    def _try_import_pylon(self):
        try:
            from pypylon import pylon
            self._pylon = pylon
            self._pylon_available = True
            logger.info("pypylon imported successfully")
        except ImportError as e:
            self._pylon_available = False
            logger.warning(f"pypylon not available: {e}")

    @property
    def is_available(self) -> bool:
        return self._pylon_available

    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                connected=self._status.connected,
                width=self._status.width,
                height=self._status.height,
                max_width=self._status.max_width,
                max_height=self._status.max_height,
                payload_size_mb=self._status.payload_size_mb,
                packet_size=self._status.packet_size,
                scpd=self._status.scpd,
                frame_retention_ms=self._status.frame_retention_ms,
                last_capture_time=self._status.last_capture_time,
                last_error=self._status.last_error,
                exposure_time=self._status.exposure_time,
                gain=self._status.gain,
                offset_x=self._status.offset_x,
                offset_y=self._status.offset_y,
                exposure_auto=self._status.exposure_auto,
                gain_auto=self._status.gain_auto
            )

    def connect(self, config: Optional[CameraConfig] = None) -> Tuple[bool, str]:
        if not self._pylon_available:
            return False, "pypylon not installed"

        with self._lock:
            if self._camera is not None and self._camera.IsOpen():
                return True, "Camera already connected"

            if config:
                self._config = config

            try:
                tl_factory = self._pylon.TlFactory.GetInstance()
                device_info = self._pylon.DeviceInfo()
                device_info.SetPropertyValue("IpAddress", self._config.ip_address)
                device_info.SetPropertyValue("DeviceClass", "BaslerGigE")

                self._camera = self._pylon.InstantCamera(tl_factory.CreateDevice(device_info))
                self._camera.MaxNumBuffer = 50
                self._camera.Open()

                nodemap = self._camera.GetNodeMap()

                max_w = nodemap.GetNode("Width").GetMax()
                max_h = nodemap.GetNode("Height").GetMax()
                target_w = self._config.width if self._config.width else max_w
                target_h = self._config.height if self._config.height else max_h

                # 判断是否为全尺寸
                is_full_resolution = (target_w == max_w and target_h == max_h)

                # 如果不是全尺寸，启用中心对齐
                if not is_full_resolution:
                    try:
                        self._camera.CenterX.Value = True
                        self._camera.CenterY.Value = True
                        logger.info("Center alignment enabled")
                    except Exception as e:
                        logger.warning(f"Failed to enable center alignment: {e}")

                # 设置分辨率
                nodemap.GetNode("Width").SetValue(target_w)
                nodemap.GetNode("Height").SetValue(target_h)

                # 如果不是全尺寸，应用偏移量
                if not is_full_resolution:
                    # 关闭中心对齐以便手动调整
                    try:
                        self._camera.CenterX.Value = False
                        self._camera.CenterY.Value = False
                    except:
                        pass

                    # 获取当前居中偏移量
                    try:
                        current_offset_x = nodemap.GetNode("OffsetX").GetValue()
                        current_offset_y = nodemap.GetNode("OffsetY").GetValue()
                    except:
                        current_offset_x = 0
                        current_offset_y = 0

                    # 应用配置的偏移量
                    if self._config.offset_x is not None:
                        # OffsetX 必须是 4 的倍数
                        new_offset_x = current_offset_x + self._config.offset_x
                        new_offset_x = (new_offset_x // 4) * 4
                        # 确保在有效范围内
                        offset_x_max = nodemap.GetNode("OffsetX").GetMax()
                        new_offset_x = max(0, min(new_offset_x, offset_x_max))
                        nodemap.GetNode("OffsetX").SetValue(new_offset_x)
                        self._status.offset_x = new_offset_x
                        logger.info(f"OffsetX set to {new_offset_x}")
                    else:
                        self._status.offset_x = current_offset_x

                    if self._config.offset_y is not None:
                        # OffsetY 必须是 2 的倍数
                        new_offset_y = current_offset_y + self._config.offset_y
                        new_offset_y = (new_offset_y // 2) * 2
                        # 确保在有效范围内
                        offset_y_max = nodemap.GetNode("OffsetY").GetMax()
                        new_offset_y = max(0, min(new_offset_y, offset_y_max))
                        nodemap.GetNode("OffsetY").SetValue(new_offset_y)
                        self._status.offset_y = new_offset_y
                        logger.info(f"OffsetY set to {new_offset_y}")
                    else:
                        self._status.offset_y = current_offset_y
                else:
                    # 全尺寸时，偏移量失效，重置为0
                    nodemap.GetNode("OffsetX").SetValue(0)
                    nodemap.GetNode("OffsetY").SetValue(0)
                    self._status.offset_x = 0
                    self._status.offset_y = 0
                    logger.info("Full resolution mode, offsets disabled")

                # 设置曝光时间和增益
                # 先关闭自动模式
                try:
                    nodemap.GetNode("ExposureAuto").SetValue("Off")
                    self._status.exposure_auto = "Off"
                except Exception as e:
                    logger.warning(f"Failed to set ExposureAuto: {e}")

                try:
                    nodemap.GetNode("GainAuto").SetValue("Off")
                    self._status.gain_auto = "Off"
                except Exception as e:
                    logger.warning(f"Failed to set GainAuto: {e}")

                # 设置曝光时间（必须是52的倍数）
                if self._config.exposure_time is not None:
                    try:
                        # 确保是52的倍数
                        exposure_time = (self._config.exposure_time // 52) * 52
                        # 确保在有效范围内
                        exposure_min = nodemap.GetNode("ExposureTimeRaw").GetMin()
                        exposure_max = nodemap.GetNode("ExposureTimeRaw").GetMax()
                        exposure_time = max(exposure_min, min(exposure_time, exposure_max))
                        nodemap.GetNode("ExposureTimeRaw").SetValue(exposure_time)
                        self._status.exposure_time = exposure_time
                        logger.info(f"Exposure time set to {exposure_time} μs")
                    except Exception as e:
                        logger.warning(f"Failed to set exposure time: {e}")
                        self._status.exposure_time = None
                else:
                    try:
                        current_exposure = nodemap.GetNode("ExposureTimeRaw").GetValue()
                        self._status.exposure_time = current_exposure
                        self._config.exposure_time = current_exposure
                        logger.info(f"Exposure time read from camera: {current_exposure} μs, saved to config")
                    except:
                        self._status.exposure_time = None

                # 设置增益
                if self._config.gain is not None:
                    try:
                        gain_min = nodemap.GetNode("GainRaw").GetMin()
                        gain_max = nodemap.GetNode("GainRaw").GetMax()
                        gain = max(gain_min, min(self._config.gain, gain_max))
                        nodemap.GetNode("GainRaw").SetValue(gain)
                        self._status.gain = gain
                        logger.info(f"Gain set to {gain}")
                    except Exception as e:
                        logger.warning(f"Failed to set gain: {e}")
                        self._status.gain = None
                else:
                    try:
                        current_gain = nodemap.GetNode("GainRaw").GetValue()
                        self._status.gain = current_gain
                        self._config.gain = current_gain
                        logger.info(f"Gain read from camera: {current_gain}, saved to config")
                    except:
                        self._status.gain = None

                payload = nodemap.GetNode("PayloadSize").GetValue()
                payload_mb = payload / 1024 / 1024

                nodemap.GetNode("GevStreamChannelSelector").SetValue("StreamChannel0")
                nodemap.GetNode("GevSCPSPacketSize").SetValue(self._config.packet_size)

                scpd = max(5000, int(payload_mb * 3000))
                nodemap.GetNode("GevSCPD").SetValue(scpd)
                nodemap.GetNode("GevSCFTD").SetValue(0)

                try:
                    sg_nodemap = self._camera.GetStreamGrabberNodeMap()
                    fr_node = sg_nodemap.GetNode("FrameRetention")
                    fr_node.SetValue(self._config.frame_retention)
                    frame_retention_ms = fr_node.GetValue() / 1e6
                except Exception as e:
                    frame_retention_ms = self._config.frame_retention / 1e6
                    logger.warning(f"Failed to set FrameRetention: {e}")

                self._converter = self._pylon.ImageFormatConverter()
                self._converter.OutputPixelFormat = self._pylon.PixelType_BGR8packed
                self._converter.OutputBitAlignment = self._pylon.OutputBitAlignment_MsbAligned

                self._status.connected = True
                self._status.width = target_w
                self._status.height = target_h
                self._status.max_width = max_w
                self._status.max_height = max_h
                self._status.payload_size_mb = payload_mb
                self._status.packet_size = self._config.packet_size
                self._status.scpd = scpd
                self._status.frame_retention_ms = frame_retention_ms
                self._status.last_error = None

                logger.info(f"Camera connected: {target_w}x{target_h}, payload={payload_mb:.1f}MB")
                return True, f"Connected at {target_w}x{target_h}"

            except Exception as e:
                error_msg = str(e)
                self._status.last_error = error_msg
                logger.error(f"Failed to connect camera: {error_msg}")
                self._cleanup()
                return False, error_msg

    def disconnect(self) -> Tuple[bool, str]:
        with self._lock:
            self._cleanup()
            self._status.connected = False
            logger.info("Camera disconnected")
            return True, "Camera disconnected"

    def _cleanup(self):
        if self._camera:
            try:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                if self._camera.IsOpen():
                    self._camera.Close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            finally:
                self._camera = None
                self._converter = None

    def capture(self, save_path: Optional[str] = None, return_base64: bool = True) -> Tuple[bool, Any]:
        if not self._pylon_available:
            return False, "pypylon not installed"

        with self._lock:
            if self._camera is None or not self._camera.IsOpen():
                success, msg = self.connect()
                if not success:
                    return False, msg

            try:
                self._camera.StartGrabbing(self._pylon.GrabStrategy_OneByOne)

                for attempt in range(self._config.max_retry):
                    try:
                        res = self._camera.RetrieveResult(
                            self._config.timeout_ms,
                            self._pylon.TimeoutHandling_ThrowException
                        )

                        if res.GrabSucceeded():
                            img = self._converter.Convert(res).GetArray()
                            res.Release()

                            self._status.last_capture_time = time.time()

                            if save_path:
                                cv2.imwrite(save_path, img)
                                logger.info(f"Image saved to {save_path}")

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
                        else:
                            error_code = hex(res.ErrorCode)
                            error_desc = res.ErrorDescription
                            res.Release()
                            logger.warning(f"Grab failed ({attempt+1}/{self._config.max_retry}): [{error_code}] {error_desc}")
                            time.sleep(0.5)

                    except Exception as e:
                        logger.warning(f"Capture attempt {attempt+1} failed: {e}")
                        time.sleep(0.5)

                return False, f"Failed to capture after {self._config.max_retry} attempts"

            except Exception as e:
                error_msg = str(e)
                self._status.last_error = error_msg
                logger.error(f"Capture error: {error_msg}")
                return False, error_msg

            finally:
                try:
                    if self._camera.IsGrabbing():
                        self._camera.StopGrabbing()
                except:
                    pass

    def get_supported_resolutions(self) -> List[Dict[str, int]]:
        if not self._pylon_available:
            return []

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
            if self._status.max_width > 0 and self._status.max_height > 0:
                max_w = self._status.max_width
                max_h = self._status.max_height
            else:
                return common_resolutions

        supported = []
        for res in common_resolutions:
            if res["width"] <= max_w and res["height"] <= max_h:
                supported.append(res)

        supported.append({"width": max_w, "height": max_h, "label": "Maximum"})
        return supported

    def update_config(self, **kwargs) -> Tuple[bool, str]:
        with self._lock:
            was_connected = self._camera is not None and self._camera.IsOpen()

            if was_connected:
                self.disconnect()

            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                    logger.info(f"Camera config updated: {key} = {value}")

            if was_connected:
                return self.connect()

            return True, "Configuration updated"


camera_service = CameraService()
