import os
import time
import base64
import logging
import threading
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

os.environ["PYLON_GIGE_HEARTBEAT"] = "3000"
os.environ["PYLON_GIGE_FRAMERETENTION"] = "2000000000"


@dataclass
class CameraConfig:
    camera_id: str
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
    camera_id: str
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


class CameraInstance:
    """单个相机实例"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._camera = None
        self._converter = None
        self._status = CameraStatus(camera_id=config.camera_id)
        self._lock = threading.RLock()
        self._pylon = None
        self._pylon_available = False

        self._try_import_pylon()

    def _try_import_pylon(self):
        try:
            from pypylon import pylon
            self._pylon = pylon
            self._pylon_available = True
        except ImportError:
            self._pylon_available = False

    @property
    def is_available(self) -> bool:
        return self._pylon_available

    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                camera_id=self._status.camera_id,
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

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    def connect(self) -> Tuple[bool, str]:
        if not self._pylon_available:
            return False, "pypylon not installed"

        with self._lock:
            if self._camera is not None and self._camera.IsOpen():
                return True, "Camera already connected"

            try:
                tl_factory = self._pylon.TlFactory.GetInstance()
                device_info = self._pylon.DeviceInfo()
                device_info.SetPropertyValue("IpAddress", self.config.ip_address)
                device_info.SetPropertyValue("DeviceClass", "BaslerGigE")

                self._camera = self._pylon.InstantCamera(tl_factory.CreateDevice(device_info))
                self._camera.MaxNumBuffer = 50
                self._camera.Open()

                nodemap = self._camera.GetNodeMap()

                max_w = nodemap.GetNode("Width").GetMax()
                max_h = nodemap.GetNode("Height").GetMax()
                target_w = self.config.width if self.config.width else max_w
                target_h = self.config.height if self.config.height else max_h

                # 判断是否为全尺寸
                is_full_resolution = (target_w == max_w and target_h == max_h)

                # 如果不是全尺寸，启用中心对齐
                if not is_full_resolution:
                    try:
                        self._camera.CenterX.Value = True
                        self._camera.CenterY.Value = True
                        logger.info(f"Camera {self.config.camera_id}: Center alignment enabled")
                    except Exception as e:
                        logger.warning(f"Camera {self.config.camera_id}: Failed to enable center alignment: {e}")

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
                    # self.config.offset_x/y 存储的是相对于居中位置的调整值
                    if self.config.offset_x is not None:
                        # OffsetX 必须是 4 的倍数
                        new_offset_x = current_offset_x + self.config.offset_x
                        new_offset_x = (new_offset_x // 4) * 4
                        # 确保在有效范围内
                        offset_x_max = nodemap.GetNode("OffsetX").GetMax()
                        new_offset_x = max(0, min(new_offset_x, offset_x_max))
                        nodemap.GetNode("OffsetX").SetValue(new_offset_x)
                        self._status.offset_x = new_offset_x
                        logger.info(f"Camera {self.config.camera_id}: OffsetX set to {new_offset_x} (center={current_offset_x}, adjust={self.config.offset_x})")
                    else:
                        # 用户未指定偏移量调整，使用居中偏移量，并保存相对调整值为0
                        self._status.offset_x = current_offset_x
                        self.config.offset_x = 0
                        logger.info(f"Camera {self.config.camera_id}: OffsetX using center position {current_offset_x}, saved offset_x=0 to config")

                    if self.config.offset_y is not None:
                        # OffsetY 必须是 2 的倍数
                        new_offset_y = current_offset_y + self.config.offset_y
                        new_offset_y = (new_offset_y // 2) * 2
                        # 确保在有效范围内
                        offset_y_max = nodemap.GetNode("OffsetY").GetMax()
                        new_offset_y = max(0, min(new_offset_y, offset_y_max))
                        nodemap.GetNode("OffsetY").SetValue(new_offset_y)
                        self._status.offset_y = new_offset_y
                        logger.info(f"Camera {self.config.camera_id}: OffsetY set to {new_offset_y} (center={current_offset_y}, adjust={self.config.offset_y})")
                    else:
                        # 用户未指定偏移量调整，使用居中偏移量，并保存相对调整值为0
                        self._status.offset_y = current_offset_y
                        self.config.offset_y = 0
                        logger.info(f"Camera {self.config.camera_id}: OffsetY using center position {current_offset_y}, saved offset_y=0 to config")
                else:
                    # 全尺寸时，偏移量失效，重置为0
                    nodemap.GetNode("OffsetX").SetValue(0)
                    nodemap.GetNode("OffsetY").SetValue(0)
                    self._status.offset_x = 0
                    self._status.offset_y = 0
                    self.config.offset_x = 0
                    self.config.offset_y = 0
                    logger.info(f"Camera {self.config.camera_id}: Full resolution mode, offsets disabled and saved to config")

                # 设置曝光时间和增益
                # 先关闭自动模式
                try:
                    nodemap.GetNode("ExposureAuto").SetValue("Off")
                    self._status.exposure_auto = "Off"
                except Exception as e:
                    logger.warning(f"Camera {self.config.camera_id}: Failed to set ExposureAuto: {e}")

                try:
                    nodemap.GetNode("GainAuto").SetValue("Off")
                    self._status.gain_auto = "Off"
                except Exception as e:
                    logger.warning(f"Camera {self.config.camera_id}: Failed to set GainAuto: {e}")

                # 设置曝光时间（必须是52的倍数）
                if self.config.exposure_time is not None:
                    try:
                        # 确保是52的倍数
                        exposure_time = (self.config.exposure_time // 52) * 52
                        # 确保在有效范围内
                        exposure_min = nodemap.GetNode("ExposureTimeRaw").GetMin()
                        exposure_max = nodemap.GetNode("ExposureTimeRaw").GetMax()
                        exposure_time = max(exposure_min, min(exposure_time, exposure_max))
                        nodemap.GetNode("ExposureTimeRaw").SetValue(exposure_time)
                        self._status.exposure_time = exposure_time
                        logger.info(f"Camera {self.config.camera_id}: Exposure time set to {exposure_time} μs")
                    except Exception as e:
                        logger.warning(f"Camera {self.config.camera_id}: Failed to set exposure time: {e}")
                        self._status.exposure_time = None
                else:
                    try:
                        current_exposure = nodemap.GetNode("ExposureTimeRaw").GetValue()
                        self._status.exposure_time = current_exposure
                        self.config.exposure_time = current_exposure
                        logger.info(f"Camera {self.config.camera_id}: Exposure time read from camera: {current_exposure} μs, saved to config")
                    except:
                        self._status.exposure_time = None

                # 设置增益
                if self.config.gain is not None:
                    try:
                        gain_min = nodemap.GetNode("GainRaw").GetMin()
                        gain_max = nodemap.GetNode("GainRaw").GetMax()
                        gain = max(gain_min, min(self.config.gain, gain_max))
                        nodemap.GetNode("GainRaw").SetValue(gain)
                        self._status.gain = gain
                        logger.info(f"Camera {self.config.camera_id}: Gain set to {gain}")
                    except Exception as e:
                        logger.warning(f"Camera {self.config.camera_id}: Failed to set gain: {e}")
                        self._status.gain = None
                else:
                    try:
                        current_gain = nodemap.GetNode("GainRaw").GetValue()
                        self._status.gain = current_gain
                        self.config.gain = current_gain
                        logger.info(f"Camera {self.config.camera_id}: Gain read from camera: {current_gain}, saved to config")
                    except:
                        self._status.gain = None

                payload = nodemap.GetNode("PayloadSize").GetValue()
                payload_mb = payload / 1024 / 1024

                nodemap.GetNode("GevStreamChannelSelector").SetValue("StreamChannel0")
                nodemap.GetNode("GevSCPSPacketSize").SetValue(self.config.packet_size)

                # 计算SCPD并限制在相机允许的最大值内
                scpd_calculated = max(5000, int(payload_mb * 3000))
                scpd_max = nodemap.GetNode("GevSCPD").GetMax()
                scpd = min(scpd_calculated, scpd_max)
                nodemap.GetNode("GevSCPD").SetValue(scpd)
                nodemap.GetNode("GevSCFTD").SetValue(0)

                try:
                    sg_nodemap = self._camera.GetStreamGrabberNodeMap()
                    fr_node = sg_nodemap.GetNode("FrameRetention")
                    fr_node.SetValue(self.config.frame_retention)
                    frame_retention_ms = fr_node.GetValue() / 1e6
                except Exception:
                    frame_retention_ms = self.config.frame_retention / 1e6

                self._converter = self._pylon.ImageFormatConverter()
                self._converter.OutputPixelFormat = self._pylon.PixelType_BGR8packed
                self._converter.OutputBitAlignment = self._pylon.OutputBitAlignment_MsbAligned

                self._status.connected = True
                self._status.width = target_w
                self._status.height = target_h
                self._status.max_width = max_w
                self._status.max_height = max_h
                self._status.payload_size_mb = payload_mb
                self._status.packet_size = self.config.packet_size
                self._status.scpd = scpd
                self._status.frame_retention_ms = frame_retention_ms
                self._status.last_error = None

                logger.info(f"Camera {self.config.camera_id} connected: {target_w}x{target_h}")
                return True, f"Connected at {target_w}x{target_h}"

            except Exception as e:
                error_msg = str(e)
                self._status.last_error = error_msg
                logger.error(f"Failed to connect camera {self.config.camera_id}: {error_msg}")
                self._cleanup()
                return False, error_msg

    def disconnect(self) -> Tuple[bool, str]:
        with self._lock:
            self._cleanup()
            self._status.connected = False
            logger.info(f"Camera {self.config.camera_id} disconnected")
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

                for attempt in range(self.config.max_retry):
                    try:
                        res = self._camera.RetrieveResult(
                            self.config.timeout_ms,
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
                            logger.warning(f"Grab failed ({attempt+1}/{self.config.max_retry}): [{error_code}] {error_desc}")
                            time.sleep(0.5)

                    except Exception as e:
                        logger.warning(f"Capture attempt {attempt+1} failed: {e}")
                        time.sleep(0.5)

                return False, f"Failed to capture after {self.config.max_retry} attempts"

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

    def get_supported_resolutions(self) -> List[Dict[str, Any]]:
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
        """更新配置，如果相机已连接会断开重连"""
        with self._lock:
            was_connected = self._camera is not None and self._camera.IsOpen()

            if was_connected:
                self.disconnect()

            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Camera {self.config.camera_id} config updated: {key} = {value}")

            if was_connected:
                return self.connect()

            return True, "Configuration updated"

    def update_parameters(self, **kwargs) -> Tuple[bool, str]:
        """
        动态更新相机参数（不需要断开重连）
        支持: exposure_time, gain, offset_x, offset_y
        """
        with self._lock:
            if self._camera is None or not self._camera.IsOpen():
                return False, "Camera not connected"

            try:
                nodemap = self._camera.GetNodeMap()
                updated_params = []

                # 更新曝光时间
                if 'exposure_time' in kwargs and kwargs['exposure_time'] is not None:
                    try:
                        exposure_time = (kwargs['exposure_time'] // 52) * 52
                        exposure_min = nodemap.GetNode("ExposureTimeRaw").GetMin()
                        exposure_max = nodemap.GetNode("ExposureTimeRaw").GetMax()
                        exposure_time = max(exposure_min, min(exposure_time, exposure_max))
                        nodemap.GetNode("ExposureTimeRaw").SetValue(exposure_time)
                        self._status.exposure_time = exposure_time
                        self.config.exposure_time = exposure_time
                        updated_params.append(f"exposure_time={exposure_time}")
                        logger.info(f"Camera {self.config.camera_id}: Exposure time updated to {exposure_time} μs")
                    except Exception as e:
                        logger.warning(f"Camera {self.config.camera_id}: Failed to update exposure time: {e}")

                # 更新增益
                if 'gain' in kwargs and kwargs['gain'] is not None:
                    try:
                        gain_min = nodemap.GetNode("GainRaw").GetMin()
                        gain_max = nodemap.GetNode("GainRaw").GetMax()
                        gain = max(gain_min, min(kwargs['gain'], gain_max))
                        nodemap.GetNode("GainRaw").SetValue(gain)
                        self._status.gain = gain
                        self.config.gain = gain
                        updated_params.append(f"gain={gain}")
                        logger.info(f"Camera {self.config.camera_id}: Gain updated to {gain}")
                    except Exception as e:
                        logger.warning(f"Camera {self.config.camera_id}: Failed to update gain: {e}")

                # 更新偏移量（仅非全尺寸时有效）
                is_full_resolution = (self._status.width == self._status.max_width and 
                                     self._status.height == self._status.max_height)
                
                if not is_full_resolution:
                    # 计算居中基准偏移量
                    center_offset_x = (self._status.max_width - self._status.width) // 2
                    center_offset_y = (self._status.max_height - self._status.height) // 2
                    
                    if 'offset_x' in kwargs and kwargs['offset_x'] is not None:
                        try:
                            # 基于居中位置计算新的绝对偏移量
                            # 传入的 offset_x 是相对于居中位置的调整（可正可负）
                            new_offset_x = center_offset_x + kwargs['offset_x']
                            new_offset_x = (new_offset_x // 4) * 4  # 必须是4的倍数
                            offset_x_max = nodemap.GetNode("OffsetX").GetMax()
                            new_offset_x = max(0, min(new_offset_x, offset_x_max))
                            nodemap.GetNode("OffsetX").SetValue(new_offset_x)
                            self._status.offset_x = new_offset_x
                            self.config.offset_x = kwargs['offset_x']  # 保存相对调整值
                            updated_params.append(f"offset_x={new_offset_x}")
                            logger.info(f"Camera {self.config.camera_id}: OffsetX set to {new_offset_x} (center={center_offset_x}, adjust={kwargs['offset_x']})")
                        except Exception as e:
                            logger.warning(f"Camera {self.config.camera_id}: Failed to update offset_x: {e}")

                    if 'offset_y' in kwargs and kwargs['offset_y'] is not None:
                        try:
                            # 基于居中位置计算新的绝对偏移量
                            new_offset_y = center_offset_y + kwargs['offset_y']
                            new_offset_y = (new_offset_y // 2) * 2  # 必须是2的倍数
                            offset_y_max = nodemap.GetNode("OffsetY").GetMax()
                            new_offset_y = max(0, min(new_offset_y, offset_y_max))
                            nodemap.GetNode("OffsetY").SetValue(new_offset_y)
                            self._status.offset_y = new_offset_y
                            self.config.offset_y = kwargs['offset_y']
                            updated_params.append(f"offset_y={new_offset_y}")
                            logger.info(f"Camera {self.config.camera_id}: OffsetY set to {new_offset_y} (center={center_offset_y}, adjust={kwargs['offset_y']})")
                        except Exception as e:
                            logger.warning(f"Camera {self.config.camera_id}: Failed to update offset_y: {e}")
                else:
                    if 'offset_x' in kwargs or 'offset_y' in kwargs:
                        logger.warning(f"Camera {self.config.camera_id}: Cannot update offsets in full resolution mode")

                if updated_params:
                    return True, f"Parameters updated: {', '.join(updated_params)}"
                else:
                    return True, "No parameters were updated"

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Camera {self.config.camera_id}: Failed to update parameters: {error_msg}")
                return False, error_msg


class CameraManager:
    """多相机管理器"""

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

    def create_camera(self, camera_id: str, ip_address: str = "192.168.110.10",
                      width: Optional[int] = None, height: Optional[int] = None,
                      packet_size: int = 1500,
                      exposure_time: Optional[int] = None,
                      gain: Optional[int] = None,
                      offset_x: Optional[int] = None,
                      offset_y: Optional[int] = None) -> Tuple[bool, str]:
        """创建新的相机实例"""
        with self._lock:
            if camera_id in self._cameras:
                return False, f"Camera {camera_id} already exists"

            config = CameraConfig(
                camera_id=camera_id,
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
            self._cameras[camera_id] = camera
            logger.info(f"Camera {camera_id} created with IP {ip_address}")
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

    def discover_cameras(self) -> List[Dict[str, str]]:
        """发现网络中的Basler相机"""
        try:
            from pypylon import pylon
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()

            discovered = []
            for device in devices:
                discovered.append({
                    "ip_address": device.GetIpAddress(),
                    "model": device.GetModelName(),
                    "serial": device.GetSerialNumber(),
                    "device_class": device.GetDeviceClass()
                })
            return discovered
        except Exception as e:
            logger.error(f"Failed to discover cameras: {e}")
            return []


camera_manager = CameraManager()
