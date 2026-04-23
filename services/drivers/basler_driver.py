"""
Basler Pylon 相机驱动实现
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from services.camera_driver_interface import ICameraDriver, DriverCapability, DiscoveredDevice

logger = logging.getLogger(__name__)

# 设置Basler环境变量
os.environ["PYLON_GIGE_HEARTBEAT"] = "3000"
os.environ["PYLON_GIGE_FRAMERETENTION"] = "2000000000"


class BaslerDriver(ICameraDriver):
    """Basler Pylon 相机驱动"""

    def __init__(self):
        self._pylon = None
        self._camera = None
        self._converter = None
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
            exposure_time_range=(52, 10000016),
            gain_range=(0, 1957),
            exposure_time_step=52,
            offset_x_step=4,
            offset_y_step=2
        )
        self._try_import_pylon()

    def _try_import_pylon(self):
        try:
            from pypylon import pylon
            self._pylon = pylon
            logger.info("pypylon imported successfully")
        except ImportError as e:
            logger.warning(f"pypylon not available: {e}")

    @property
    def vendor_name(self) -> str:
        return "Basler"

    @property
    def is_available(self) -> bool:
        return self._pylon is not None

    @property
    def capability(self) -> DriverCapability:
        return self._capability

    def connect(self, connection_params: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.is_available:
            return False, "pypylon not installed"

        try:
            ip_address = connection_params.get("ip_address", "192.168.110.10")

            tl_factory = self._pylon.TlFactory.GetInstance()

            # Prefer an enumerated DeviceInfo so pylon knows which NIC the
            # camera lives on and binds the GigE stream to the correct
            # interface.  A partial DeviceInfo (IP only) causes pylon to
            # guess the interface and often picks the wrong one, leading to
            # "Failed to allocate resources / 参数错误 (0xC0070057)".
            device_info = None
            try:
                all_devices = tl_factory.EnumerateDevices()
                for dev in all_devices:
                    try:
                        if dev.GetIpAddress() == ip_address:
                            device_info = dev
                            logger.info(
                                f"Found camera {ip_address} via enumeration "
                                f"(interface: {dev.GetInterface() if hasattr(dev, 'GetInterface') else 'unknown'})"
                            )
                            break
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Device enumeration failed, falling back to direct connect: {e}")

            if device_info is None:
                logger.warning(
                    f"Camera {ip_address} not found via enumeration; "
                    "using partial DeviceInfo — stream interface may be incorrect"
                )
                device_info = self._pylon.DeviceInfo()
                device_info.SetPropertyValue("IpAddress", ip_address)
                device_info.SetPropertyValue("DeviceClass", "BaslerGigE")

            self._camera = self._pylon.InstantCamera(tl_factory.CreateDevice(device_info))
            self._camera.MaxNumBuffer = 50
            self._camera.Open()
            self._is_connected = True

            # NOTE: AutoPacketSize and GevSCPD are intentionally NOT set here.
            # They must be applied AFTER set_resolution() changes the PayloadSize,
            # which happens in CameraInstance._apply_config() → set_packet_size().
            # Setting them here would use the wrong (max-resolution) PayloadSize
            # and would be overwritten anyway by _apply_config().

            # Set FrameRetention via stream grabber to prevent incomplete-grab errors
            # when the network is under load or the NIC buffer is small.
            frame_retention = connection_params.get("frame_retention", 2000000000)
            try:
                sg_nodemap = self._camera.GetStreamGrabberNodeMap()
                sg_nodemap.GetNode("FrameRetention").SetValue(frame_retention)
                logger.info(f"FrameRetention set to {frame_retention / 1e6:.0f} ms")
            except Exception as e:
                logger.warning(f"Failed to set FrameRetention via stream grabber: {e}")

            # 创建图像转换器
            self._converter = self._pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = self._pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = self._pylon.OutputBitAlignment_MsbAligned

            logger.info(f"Basler camera connected: {ip_address}")
            return True, f"Connected to {ip_address}"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect Basler camera: {error_msg}")
            self._cleanup()
            return False, error_msg

    def disconnect(self) -> Tuple[bool, str]:
        self._cleanup()
        self._is_connected = False
        logger.info("Basler camera disconnected")
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
                self._is_grabbing = False

    def is_connected(self) -> bool:
        if self._camera is None:
            return False
        try:
            return self._camera.IsOpen()
        except:
            return False

    def capture(self, timeout_ms: int = 5000) -> Tuple[bool, Any]:
        if not self.is_connected():
            return False, "Camera not connected"

        was_grabbing = self._is_grabbing

        try:
            if not was_grabbing:
                self._camera.StartGrabbing(self._pylon.GrabStrategy_OneByOne)

            res = self._camera.RetrieveResult(
                timeout_ms,
                self._pylon.TimeoutHandling_ThrowException
            )

            if res.GrabSucceeded():
                img = self._converter.Convert(res).GetArray()
                res.Release()
                return True, img
            else:
                error_code = hex(res.ErrorCode)
                error_desc = res.ErrorDescription
                res.Release()
                return False, f"Grab failed: [{error_code}] {error_desc}"

        except Exception as e:
            error_msg = str(e)
            # 0xC0070057 = "Failed to allocate resources" — usually means
            # the stream channel is bound to the wrong NIC.  Log clearly
            # so the operator knows it is a network interface problem.
            if "0xC0070057" in error_msg or "allocate resources" in error_msg:
                logger.error(
                    "Stream resource allocation failed (0xC0070057). "
                    "The GigE stream is likely bound to the wrong network interface. "
                    "Ensure the host NIC connected to the camera is on the same "
                    "subnet as the camera IP, and that no firewall blocks UDP traffic."
                )
            return False, error_msg
        finally:
            if not was_grabbing:
                try:
                    self._camera.StopGrabbing()
                except Exception:
                    pass

    def start_grabbing(self) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            self._camera.StartGrabbing(self._pylon.GrabStrategy_OneByOne)
            self._is_grabbing = True
            return True, "Grabbing started"
        except Exception as e:
            return False, str(e)

    def stop_grabbing(self) -> Tuple[bool, str]:
        if not self.is_connected():
            return True, "Camera not connected"
        
        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
            self._is_grabbing = False
            return True, "Grabbing stopped"
        except Exception as e:
            return False, str(e)

    def is_grabbing(self) -> bool:
        if self._camera is None:
            return False
        try:
            return self._camera.IsGrabbing()
        except:
            return False

    def get_resolution(self) -> Tuple[int, int]:
        if not self.is_connected():
            return (0, 0)
        nodemap = self._camera.GetNodeMap()
        width = nodemap.GetNode("Width").GetValue()
        height = nodemap.GetNode("Height").GetValue()
        return (int(width), int(height))

    def set_resolution(self, width: int, height: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            nodemap = self._camera.GetNodeMap()
            max_w = nodemap.GetNode("Width").GetMax()
            max_h = nodemap.GetNode("Height").GetMax()
            
            # 如果是全尺寸，重置偏移
            if width == max_w and height == max_h:
                nodemap.GetNode("OffsetX").SetValue(0)
                nodemap.GetNode("OffsetY").SetValue(0)
            else:
                # 启用中心对齐
                try:
                    self._camera.CenterX.Value = True
                    self._camera.CenterY.Value = True
                except:
                    pass
            
            nodemap.GetNode("Width").SetValue(width)
            nodemap.GetNode("Height").SetValue(height)
            return True, f"Resolution set to {width}x{height}"
        except Exception as e:
            return False, str(e)

    def get_max_resolution(self) -> Tuple[int, int]:
        if not self.is_connected():
            return (0, 0)
        nodemap = self._camera.GetNodeMap()
        max_w = nodemap.GetNode("Width").GetMax()
        max_h = nodemap.GetNode("Height").GetMax()
        return (int(max_w), int(max_h))

    def get_exposure_time(self) -> Optional[int]:
        if not self.is_connected():
            return None
        try:
            nodemap = self._camera.GetNodeMap()
            return int(nodemap.GetNode("ExposureTimeRaw").GetValue())
        except:
            return None

    def set_exposure_time(self, exposure_us: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            # 确保是52的倍数
            exposure_us = (exposure_us // 52) * 52
            nodemap = self._camera.GetNodeMap()
            exposure_min = nodemap.GetNode("ExposureTimeRaw").GetMin()
            exposure_max = nodemap.GetNode("ExposureTimeRaw").GetMax()
            exposure_us = max(exposure_min, min(exposure_us, exposure_max))
            nodemap.GetNode("ExposureTimeRaw").SetValue(exposure_us)
            return True, f"Exposure time set to {exposure_us} μs"
        except Exception as e:
            return False, str(e)

    def get_gain(self) -> Optional[int]:
        if not self.is_connected():
            return None
        try:
            nodemap = self._camera.GetNodeMap()
            return int(nodemap.GetNode("GainRaw").GetValue())
        except:
            return None

    def set_gain(self, gain: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            nodemap = self._camera.GetNodeMap()
            gain_min = nodemap.GetNode("GainRaw").GetMin()
            gain_max = nodemap.GetNode("GainRaw").GetMax()
            gain = max(gain_min, min(gain, gain_max))
            nodemap.GetNode("GainRaw").SetValue(gain)
            return True, f"Gain set to {gain}"
        except Exception as e:
            return False, str(e)

    def get_offset(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.is_connected():
            return (None, None)
        try:
            nodemap = self._camera.GetNodeMap()
            offset_x = int(nodemap.GetNode("OffsetX").GetValue())
            offset_y = int(nodemap.GetNode("OffsetY").GetValue())
            return (offset_x, offset_y)
        except:
            return (None, None)

    def set_offset(self, offset_x: int, offset_y: int) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            nodemap = self._camera.GetNodeMap()
            # OffsetX 必须是 4 的倍数
            offset_x = (offset_x // 4) * 4
            # OffsetY 必须是 2 的倍数
            offset_y = (offset_y // 2) * 2
            
            offset_x_max = nodemap.GetNode("OffsetX").GetMax()
            offset_y_max = nodemap.GetNode("OffsetY").GetMax()
            
            offset_x = max(0, min(offset_x, offset_x_max))
            offset_y = max(0, min(offset_y, offset_y_max))
            
            nodemap.GetNode("OffsetX").SetValue(offset_x)
            nodemap.GetNode("OffsetY").SetValue(offset_y)
            return True, f"Offset set to ({offset_x}, {offset_y})"
        except Exception as e:
            return False, str(e)

    def set_center_alignment(self, enabled: bool) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            self._camera.CenterX.Value = enabled
            self._camera.CenterY.Value = enabled
            return True, f"Center alignment {'enabled' if enabled else 'disabled'}"
        except Exception as e:
            return False, str(e)

    def get_auto_exposure(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            nodemap = self._camera.GetNodeMap()
            return str(nodemap.GetNode("ExposureAuto").GetValue())
        except:
            return "Unknown"

    def set_auto_exposure(self, mode: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            nodemap = self._camera.GetNodeMap()
            nodemap.GetNode("ExposureAuto").SetValue(mode)
            return True, f"Auto exposure set to {mode}"
        except Exception as e:
            return False, str(e)

    def get_auto_gain(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            nodemap = self._camera.GetNodeMap()
            return str(nodemap.GetNode("GainAuto").GetValue())
        except:
            return "Unknown"

    def set_auto_gain(self, mode: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            nodemap = self._camera.GetNodeMap()
            nodemap.GetNode("GainAuto").SetValue(mode)
            return True, f"Auto gain set to {mode}"
        except Exception as e:
            return False, str(e)

    def get_frame_rate(self) -> Optional[float]:
        if not self.is_connected():
            return None
        try:
            nodemap = self._camera.GetNodeMap()
            return float(nodemap.GetNode("AcquisitionFrameRateAbs").GetValue())
        except:
            return None

    def set_frame_rate(self, fps: float) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            nodemap = self._camera.GetNodeMap()
            nodemap.GetNode("AcquisitionFrameRateAbs").SetValue(fps)
            return True, f"Frame rate set to {fps}"
        except Exception as e:
            return False, str(e)

    def get_pixel_format(self) -> str:
        if not self.is_connected():
            return "Unknown"
        try:
            nodemap = self._camera.GetNodeMap()
            return str(nodemap.GetNode("PixelFormat").GetValue())
        except:
            return "Unknown"

    def set_pixel_format(self, pixel_format: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Camera not connected"
        
        try:
            nodemap = self._camera.GetNodeMap()
            nodemap.GetNode("PixelFormat").SetValue(pixel_format)
            return True, f"Pixel format set to {pixel_format}"
        except Exception as e:
            return False, str(e)

    def get_packet_size(self) -> int:
        if not self.is_connected():
            return 0
        try:
            nodemap = self._camera.GetNodeMap()
            return int(nodemap.GetNode("GevSCPSPacketSize").GetValue())
        except:
            return 0

    def set_packet_size(self, packet_size: int) -> Tuple[bool, str]:
        """
        设置数据包大小，并在设置后动态计算 GevSCPD（包间延迟）。

        必须在 set_resolution() 之后调用，这样 PayloadSize 已经反映了目标分辨率，
        GevSCPD 才能根据正确的负载量计算出合适的延迟值，避免 [0xe1000014] 错误。

        当 packet_size <= 1500（默认值）时，自动尝试 AutoPacketSize() 协商最优帧大小
        （如网络支持巨型帧则会协商到 8228 字节），否则直接使用指定值。
        """
        if not self.is_connected():
            return False, "Camera not connected"

        try:
            nodemap = self._camera.GetNodeMap()
            nodemap.GetNode("GevStreamChannelSelector").SetValue("StreamChannel0")

            # 若使用默认 1500，先尝试 AutoPacketSize 协商最优值（巨型帧 etc.）
            if packet_size <= 1500:
                try:
                    self._camera.GigECamera.AutoPacketSize()
                    actual_size = int(nodemap.GetNode("GevSCPSPacketSize").GetValue())
                    logger.info(f"AutoPacketSize negotiated: {actual_size} bytes")
                except Exception as e:
                    logger.warning(f"AutoPacketSize failed, falling back to {packet_size}: {e}")
                    nodemap.GetNode("GevSCPSPacketSize").SetValue(packet_size)
                    actual_size = packet_size
            else:
                nodemap.GetNode("GevSCPSPacketSize").SetValue(packet_size)
                actual_size = packet_size

            # 根据当前（已应用目标分辨率后的）PayloadSize 动态计算包间延迟
            # 公式与 CameraService 保持一致：max(5000, payload_mb * 3000) ns
            try:
                payload = int(nodemap.GetNode("PayloadSize").GetValue())
                payload_mb = payload / 1024 / 1024
                scpd = max(5000, int(payload_mb * 3000))
                nodemap.GetNode("GevSCPD").SetValue(scpd)
                nodemap.GetNode("GevSCFTD").SetValue(0)
                logger.info(
                    f"GevSCPD set to {scpd} ns "
                    f"(payload={payload_mb:.2f} MB, packet={actual_size} B)"
                )
            except Exception as e:
                logger.warning(f"Could not set GevSCPD/GevSCFTD: {e}")

            return True, f"Packet size set to {actual_size}"
        except Exception as e:
            return False, str(e)

    def get_payload_size(self) -> int:
        if not self.is_connected():
            return 0
        try:
            nodemap = self._camera.GetNodeMap()
            return int(nodemap.GetNode("PayloadSize").GetValue())
        except:
            return 0

    def get_device_info(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {}
        
        try:
            nodemap = self._camera.GetNodeMap()
            return {
                "vendor": "Basler",
                "model": str(nodemap.GetNode("DeviceModelName").GetValue()),
                "serial": str(nodemap.GetNode("DeviceSerialNumber").GetValue()),
                "version": str(nodemap.GetNode("DeviceFirmwareVersion").GetValue()),
                "ip_address": str(nodemap.GetNode("GevCurrentIPAddress").GetValue()),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_temperature(self) -> Optional[float]:
        if not self.is_connected():
            return None
        try:
            nodemap = self._camera.GetNodeMap()
            return float(nodemap.GetNode("DeviceTemperature").GetValue())
        except:
            return None

    @staticmethod
    def discover_devices() -> List[DiscoveredDevice]:
        try:
            from pypylon import pylon
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()

            discovered = []
            for device in devices:
                discovered.append(DiscoveredDevice(
                    ip_address=device.GetIpAddress(),
                    model=device.GetModelName(),
                    serial=device.GetSerialNumber(),
                    device_class=device.GetDeviceClass(),
                    vendor="Basler"
                ))
            return discovered
        except Exception as e:
            logger.error(f"Failed to discover Basler cameras: {e}")
            return []

    @staticmethod
    def check_sdk_available() -> bool:
        try:
            from pypylon import pylon
            return True
        except ImportError:
            return False