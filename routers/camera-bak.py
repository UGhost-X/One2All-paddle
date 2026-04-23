import os
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse

from services.camera_manager import camera_manager, CameraConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/camera", tags=["Camera"])


class CreateCameraRequest(BaseModel):
    camera_id: str = Field(..., description="相机唯一标识ID")
    ip_address: str = Field(default="192.168.110.10", description="相机IP地址")
    width: Optional[int] = Field(default=None, description="目标宽度，None表示使用最大分辨率")
    height: Optional[int] = Field(default=None, description="目标高度，None表示使用最大分辨率")
    packet_size: int = Field(default=1500, description="数据包大小")
    # 新增参数
    exposure_time: Optional[int] = Field(default=None, description="曝光时间（微秒），必须是52的倍数，范围52-10000016")
    gain: Optional[int] = Field(default=None, description="增益，范围0-1957")
    offset_x: Optional[int] = Field(default=None, description="水平偏移（像素），必须是4的倍数，全尺寸时失效")
    offset_y: Optional[int] = Field(default=None, description="垂直偏移（像素），必须是2的倍数，全尺寸时失效")


class ConnectRequest(BaseModel):
    ip_address: Optional[str] = Field(default="192.168.110.10", description="相机IP地址")
    width: Optional[int] = Field(default=None, description="目标宽度")
    height: Optional[int] = Field(default=None, description="目标高度")
    packet_size: int = Field(default=1500, description="数据包大小")
    # 新增参数
    exposure_time: Optional[int] = Field(default=None, description="曝光时间（微秒），必须是52的倍数")
    gain: Optional[int] = Field(default=None, description="增益，范围0-1957")
    offset_x: Optional[int] = Field(default=None, description="水平偏移（像素），必须是4的倍数，全尺寸时失效")
    offset_y: Optional[int] = Field(default=None, description="垂直偏移（像素），必须是2的倍数，全尺寸时失效")


class ConfigRequest(BaseModel):
    ip_address: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    packet_size: Optional[int] = None
    timeout_ms: Optional[int] = None
    max_retry: Optional[int] = None
    # 新增参数
    exposure_time: Optional[int] = Field(default=None, description="曝光时间（微秒），必须是52的倍数")
    gain: Optional[int] = Field(default=None, description="增益，范围0-1957")
    offset_x: Optional[int] = Field(default=None, description="水平偏移（像素），必须是4的倍数，全尺寸时失效")
    offset_y: Optional[int] = Field(default=None, description="垂直偏移（像素），必须是2的倍数，全尺寸时失效")


class CaptureResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


class CameraInfo(BaseModel):
    camera_id: str
    ip_address: str
    connected: bool
    resolution: Optional[str] = None


class DiscoveredCamera(BaseModel):
    ip_address: str
    model: str
    serial: str
    device_class: str


@router.post("/create", response_model=CaptureResponse)
async def create_camera(
    camera_id: str = Form(..., description="相机唯一标识ID"),
    ip_address: str = Form(default="192.168.110.10", description="相机IP地址"),
    width: Optional[int] = Form(default=None, description="目标宽度"),
    height: Optional[int] = Form(default=None, description="目标高度"),
    packet_size: int = Form(default=1500, description="数据包大小"),
    exposure_time: Optional[int] = Form(default=None, description="曝光时间（微秒），必须是52的倍数，范围52-10000016"),
    gain: Optional[int] = Form(default=None, description="增益，范围0-1957"),
    offset_x: Optional[int] = Form(default=None, description="水平偏移（像素），必须是4的倍数，全尺寸时失效"),
    offset_y: Optional[int] = Form(default=None, description="垂直偏移（像素），必须是2的倍数，全尺寸时失效")
):
    """
    创建新的相机实例

    - camera_id: 唯一标识，如 "camera_1", "left_camera" 等
    - ip_address: 相机IP地址
    - width/height: 可选，None表示使用最大分辨率
    - exposure_time: 曝光时间（微秒），必须是52的倍数
    - gain: 增益，范围0-1957
    - offset_x: 水平偏移（像素），必须是4的倍数，全尺寸时失效
    - offset_y: 垂直偏移（像素），必须是2的倍数，全尺寸时失效
    """
    success, message = camera_manager.create_camera(
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

    if success:
        return CaptureResponse(success=True, message=message, data={"camera_id": camera_id})
    else:
        raise HTTPException(status_code=400, detail=message)


@router.post("/{camera_id}/remove", response_model=CaptureResponse)
async def remove_camera(camera_id: str):
    """移除指定相机实例"""
    success, message = camera_manager.remove_camera(camera_id)

    if success:
        return CaptureResponse(success=True, message=message)
    else:
        raise HTTPException(status_code=404, detail=message)


@router.get("/list")
async def list_cameras():
    """列出所有已创建的相机"""
    cameras = camera_manager.list_cameras()
    return {
        "success": True,
        "count": len(cameras),
        "cameras": cameras
    }


@router.get("/discover")
async def discover_cameras():
    """发现网络中的Basler相机"""
    discovered = camera_manager.discover_cameras()
    return {
        "success": True,
        "count": len(discovered),
        "cameras": discovered
    }


@router.post("/{camera_id}/connect", response_model=CaptureResponse)
async def connect_camera(
    camera_id: str,
    ip_address: Optional[str] = Form(default=None, description="相机IP地址（可选，覆盖创建时的配置）"),
    width: Optional[int] = Form(default=None, description="目标宽度"),
    height: Optional[int] = Form(default=None, description="目标高度"),
    packet_size: Optional[int] = Form(default=None, description="数据包大小"),
    exposure_time: Optional[int] = Form(default=None, description="曝光时间（微秒），必须是52的倍数"),
    gain: Optional[int] = Form(default=None, description="增益，范围0-1957"),
    offset_x: Optional[int] = Form(default=None, description="水平偏移（像素），必须是4的倍数，全尺寸时失效"),
    offset_y: Optional[int] = Form(default=None, description="垂直偏移（像素），必须是2的倍数，全尺寸时失效")
):
    """
    连接指定相机

    - 如果相机不存在，会自动创建（使用camera_id作为标识）
    - 支持临时覆盖IP地址、分辨率、曝光、增益、偏移等配置
    """
    camera = camera_manager.get_camera(camera_id)

    if camera is None:
        success, msg = camera_manager.create_camera(
            camera_id=camera_id,
            ip_address=ip_address or "192.168.110.10",
            width=width,
            height=height,
            packet_size=packet_size or 1500,
            exposure_time=exposure_time,
            gain=gain,
            offset_x=offset_x,
            offset_y=offset_y
        )
        if not success:
            raise HTTPException(status_code=400, detail=msg)
        camera = camera_manager.get_camera(camera_id)

    # 只在传入非None值时更新配置，避免覆盖创建时的设置
    updates = {}
    if ip_address is not None:
        updates['ip_address'] = ip_address
    if width is not None:
        updates['width'] = width
    if height is not None:
        updates['height'] = height
    if packet_size is not None:
        updates['packet_size'] = packet_size
    if exposure_time is not None:
        updates['exposure_time'] = exposure_time
    if gain is not None:
        updates['gain'] = gain
    if offset_x is not None:
        updates['offset_x'] = offset_x
    if offset_y is not None:
        updates['offset_y'] = offset_y
    if updates:
        camera.update_config(**updates)

    success, message = camera.connect()

    if success:
        status = camera.status
        return CaptureResponse(
            success=True,
            message=message,
            data={
                "camera_id": camera_id,
                "resolution": f"{status.width}x{status.height}",
                "max_resolution": f"{status.max_width}x{status.max_height}",
                "payload_size_mb": status.payload_size_mb,
                "exposure_time": status.exposure_time,
                "gain": status.gain,
                "offset_x": status.offset_x,
                "offset_y": status.offset_y
            }
        )
    else:
        raise HTTPException(status_code=500, detail=message)


@router.post("/{camera_id}/disconnect", response_model=CaptureResponse)
async def disconnect_camera(camera_id: str):
    """断开指定相机连接"""
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    success, message = camera.disconnect()
    return CaptureResponse(success=success, message=message)


@router.get("/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """获取指定相机状态"""
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    status = camera.status
    return {
        "success": True,
        "camera_id": camera_id,
        "status": {
            "connected": status.connected,
            "width": status.width,
            "height": status.height,
            "max_width": status.max_width,
            "max_height": status.max_height,
            "payload_size_mb": status.payload_size_mb,
            "packet_size": status.packet_size,
            "scpd": status.scpd,
            "frame_retention_ms": status.frame_retention_ms,
            "last_capture_time": status.last_capture_time,
            "last_error": status.last_error,
            "exposure_time": status.exposure_time,
            "gain": status.gain,
            "offset_x": status.offset_x,
            "offset_y": status.offset_y,
            "exposure_auto": status.exposure_auto,
            "gain_auto": status.gain_auto
        }
    }


@router.get("/status/all")
async def get_all_cameras_status():
    """获取所有相机状态"""
    statuses = camera_manager.get_all_status()
    return {
        "success": True,
        "count": len(statuses),
        "cameras": [
            {
                "camera_id": s.camera_id,
                "connected": s.connected,
                "resolution": f"{s.width}x{s.height}" if s.connected else None,
                "last_error": s.last_error
            }
            for s in statuses
        ]
    }


@router.post("/{camera_id}/capture")
async def capture_image(
    camera_id: str,
    save_path: Optional[str] = Query(None, description="保存路径，None则不保存"),
    return_base64: bool = Query(True, description="是否返回base64编码的图片")
):
    """
    使用指定相机拍摄单张照片

    - 自动连接相机（如果未连接）
    - 返回base64编码的图片数据
    - 可选保存到指定路径
    """
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    success, result = camera.capture(save_path=save_path, return_base64=return_base64)

    if success:
        return {
            "success": True,
            "message": "Image captured successfully",
            "camera_id": camera_id,
            "data": result
        }
    else:
        raise HTTPException(status_code=500, detail=result)


@router.post("/{camera_id}/capture/save")
async def capture_and_save(
    camera_id: str,
    filename: Optional[str] = Query(None, description="文件名，默认自动生成"),
    directory: Optional[str] = Query("captured_images", description="保存目录")
):
    """
    使用指定相机拍照并保存

    - 自动创建目录
    - 自动生成文件名（如果未指定）
    - 返回保存的文件路径
    """
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    os.makedirs(directory, exist_ok=True)

    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{camera_id}_{timestamp}.jpg"

    save_path = os.path.join(directory, filename)

    success, result = camera.capture(save_path=save_path, return_base64=False)

    if success:
        return {
            "success": True,
            "message": "Image saved successfully",
            "camera_id": camera_id,
            "file_path": save_path,
            "width": result["width"],
            "height": result["height"]
        }
    else:
        raise HTTPException(status_code=500, detail=result)


@router.get("/{camera_id}/resolutions")
async def get_supported_resolutions(camera_id: str):
    """获取指定相机支持的分辨率列表"""
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    resolutions = camera.get_supported_resolutions()
    return {
        "success": True,
        "camera_id": camera_id,
        "resolutions": resolutions
    }


@router.post("/{camera_id}/parameters")
async def update_camera_parameters(
    camera_id: str,
    exposure_time: Optional[int] = Form(None, description="曝光时间（微秒），必须是52的倍数"),
    gain: Optional[int] = Form(None, description="增益，范围0-1957"),
    offset_x: Optional[int] = Form(None, description="水平偏移调整（像素），在居中基础上调整，必须是4的倍数"),
    offset_y: Optional[int] = Form(None, description="垂直偏移调整（像素），在居中基础上调整，必须是2的倍数")
):
    """
    动态更新相机参数（不需要断开重连）

    - 在相机连接状态下直接更新参数
    - 适用于实时调试曝光、增益、偏移等参数
    - 修改立即生效，无需重新连接相机
    """
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    updates = {}
    if exposure_time is not None:
        updates['exposure_time'] = exposure_time
    if gain is not None:
        updates['gain'] = gain
    if offset_x is not None:
        updates['offset_x'] = offset_x
    if offset_y is not None:
        updates['offset_y'] = offset_y

    if not updates:
        return {"success": True, "message": "No parameters to update"}

    success, message = camera.update_parameters(**updates)

    if success:
        status = camera.status
        return {
            "success": True,
            "message": message,
            "camera_id": camera_id,
            "current_values": {
                "exposure_time": status.exposure_time,
                "gain": status.gain,
                "offset_x": status.offset_x,
                "offset_y": status.offset_y
            }
        }
    else:
        raise HTTPException(status_code=500, detail=message)


@router.post("/{camera_id}/config")
async def update_camera_config(
    camera_id: str,
    ip_address: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    packet_size: Optional[int] = Form(None),
    exposure_time: Optional[int] = Form(None, description="曝光时间（微秒），必须是52的倍数"),
    gain: Optional[int] = Form(None, description="增益，范围0-1957"),
    offset_x: Optional[int] = Form(None, description="水平偏移（像素），必须是4的倍数，全尺寸时失效"),
    offset_y: Optional[int] = Form(None, description="垂直偏移（像素），必须是2的倍数，全尺寸时失效")
):
    """
    更新指定相机配置（需要断开重连）

    - 修改配置参数
    - 如果相机已连接，会自动断开并重新连接
    - 适用于修改分辨率、IP地址等需要重新初始化的参数
    """
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    updates = {}
    if ip_address is not None:
        updates['ip_address'] = ip_address
    if width is not None:
        updates['width'] = width
    if height is not None:
        updates['height'] = height
    if packet_size is not None:
        updates['packet_size'] = packet_size
    if exposure_time is not None:
        updates['exposure_time'] = exposure_time
    if gain is not None:
        updates['gain'] = gain
    if offset_x is not None:
        updates['offset_x'] = offset_x
    if offset_y is not None:
        updates['offset_y'] = offset_y

    if not updates:
        return {"success": True, "message": "No changes made"}

    success, message = camera.update_config(**updates)

    if success:
        return {"success": True, "message": message, "camera_id": camera_id}
    else:
        raise HTTPException(status_code=500, detail=message)


@router.get("/{camera_id}/config")
async def get_camera_config(camera_id: str):
    """获取指定相机当前配置"""
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    status = camera.status
    return {
        "success": True,
        "camera_id": camera_id,
        "config": {
            "ip_address": camera.config.ip_address,
            "width": status.width if status.connected else camera.config.width,
            "height": status.height if status.connected else camera.config.height,
            "packet_size": camera.config.packet_size,
            "frame_retention": camera.config.frame_retention,
            "timeout_ms": camera.config.timeout_ms,
            "max_retry": camera.config.max_retry,
            "exposure_time": status.exposure_time if status.connected else camera.config.exposure_time,
            "gain": status.gain if status.connected else camera.config.gain,
            "offset_x": status.offset_x if status.connected else camera.config.offset_x,
            "offset_y": status.offset_y if status.connected else camera.config.offset_y
        }
    }


@router.get("/{camera_id}/preview")
async def get_preview(camera_id: str):
    """
    获取指定相机的实时预览帧（MJPEG流）

    - 返回multipart/x-mixed-replace流
    - 适合前端<img>标签直接显示
    """
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    async def generate_frames():
        import cv2
        import asyncio

        while True:
            success, result = camera.capture(return_base64=False)
            if success and result.get("save_path"):
                frame = cv2.imread(result["save_path"])
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
