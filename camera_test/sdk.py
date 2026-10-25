"""海康机器人 MVS Python SDK 的定位与封装。

MVS 的 Python 接口是 ctypes 封装（``MvCameraControl_class.py``），
不同 MVS 版本导出的类名/结构体字段略有差异，因此这里全部用
``getattr`` + ``try/except`` 做兼容，采集流程只依赖最小公共子集：

    MV_CC_EnumDevices / MV_CC_CreateHandle / MV_CC_OpenDevice
    MV_CC_SetEnumValue / SetIntValue / SetFloatValue / SetBoolValue
    MV_CC_StartGrabbing / MV_CC_GetImageBuffer / MV_CC_FreeImageBuffer
    MV_CC_StopGrabbing / MV_CC_CloseDevice / MV_CC_DestroyHandle
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

MV_GIGE_DEVICE = 0x00000001
MV_USB_DEVICE = 0x00000004
MV_CAMERALINK_DEVICE = 0x00000008


class SdkNotFound(RuntimeError):
    """没有找到 MVS Python SDK。"""


class CameraError(RuntimeError):
    """相机操作失败。"""


def _default_mvimport_paths() -> List[str]:
    paths: List[str] = []
    runenv = os.environ.get("MVCAM_COMMON_RUNENV")
    if runenv:
        paths.append(os.path.join(runenv, "Samples", "Python", "MvImport"))
    if platform.system() == "Windows":
        bases = [
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("ProgramFiles"),
            r"C:\Program Files (x86)",
            r"C:\Program Files",
        ]
        for base in bases:
            if not base:
                continue
            for sub in (
                os.path.join(base, "MVS", "Development", "Samples", "Python", "MvImport"),
                os.path.join(base, "MVS", "Samples", "Python", "MvImport"),
                os.path.join(base, "MVS", "MVS", "Development", "Samples", "Python", "MvImport"),
            ):
                paths.append(sub)
    else:
        paths.extend(
            [
                "/opt/MVS/Samples/64/Python/MvImport",
                "/opt/MVS/Samples/32/Python/MvImport",
                "/opt/MVS/Development/Samples/Python/MvImport",
                "/opt/MVS/Samples/Python/MvImport",
            ]
        )
    return paths


def find_mvimport(override: Optional[str] = None) -> str:
    """找到 ``MvImport`` 目录（含 MvCameraControl_class.py）。"""
    candidates: List[str] = []
    if override:
        candidates.append(os.path.abspath(override))
    candidates.extend(_default_mvimport_paths())
    if getattr(sys, "frozen", False):  # 打包后仍按原路径查找
        candidates.extend(_default_mvimport_paths())

    for path in candidates:
        module = os.path.join(path, "MvCameraControl_class.py")
        if os.path.exists(module):
            return path
    raise SdkNotFound(
        "未找到 MVS Python SDK（MvCameraControl_class.py）。\n"
        "  1) 安装海康机器人 MVS 客户端（Windows/Linux 安装包自带 Samples/Python/MvImport）；\n"
        "  2) 或使用 --mvs-path 显式指定 MvImport 目录；\n"
        "  3) Linux 下若 SDK 装在非默认路径，可设置环境变量 MVCAM_COMMON_RUNENV。\n"
        "已尝试：\n  " + "\n  ".join(candidates)
    )


def load_sdk(override: Optional[str] = None):
    path = find_mvimport(override)
    if path not in sys.path:
        sys.path.insert(0, path)
    import MvCameraControl_class as module  # noqa: WPS433 - 动态导入

    return module


def sdk_version(sdk) -> str:
    try:
        version = int(sdk.MvCamera.MV_CC_GetSDKVersion())
    except Exception:  # noqa: BLE001
        try:
            version = int(sdk.MV_CC_GetSDKVersion())
        except Exception:  # noqa: BLE001
            return "未知"
    return "%d.%d.%d" % ((version >> 24) & 0xFF, (version >> 16) & 0xFF, (version >> 8) & 0xFF)


def error_text(ret: int) -> str:
    return "0x%08X" % (ret & 0xFFFFFFFF)


def _decode_char_array(array) -> str:
    try:
        raw = bytes(bytearray(array))
    except Exception:  # noqa: BLE001
        return ""
    return raw.split(b"\x00", 1)[0].decode("ascii", "ignore").strip()


@dataclass
class DeviceEntry:
    index: int
    layer: str
    model: str
    serial: str
    user_id: str = ""
    ip: str = ""
    raw: object = None

    def label(self) -> str:
        extra = " @ %s" % self.ip if self.ip else ""
        return "[%d] %s  序列号 %s%s" % (self.index, self.model or "未知型号", self.serial or "-", extra)


def _layer_name(layer: int) -> str:
    if layer == MV_GIGE_DEVICE:
        return "GigE Vision"
    if layer == MV_USB_DEVICE:
        return "USB3 Vision"
    if layer == MV_CAMERALINK_DEVICE:
        return "CameraLink"
    if 0x0100 <= layer <= 0x01FF:
        return "GenTL(0x%04X)" % layer
    return "未知(0x%04X)" % layer


def _camera_class(sdk):
    """不同 MVS 版本里相机类可能叫 MvCamera 或 MvCameraControl。"""
    for name in ("MvCamera", "MvCameraControl"):
        candidate = getattr(sdk, name, None)
        if candidate is not None:
            return candidate
    raise CameraError("当前 MVS Python SDK 未导出相机类（MvCamera/MvCameraControl）")


def _struct(sdk, *names):
    """按候选名称取 ctypes 结构体（兼容 *_EX 命名）。"""
    for name in names:
        candidate = getattr(sdk, name, None)
        if candidate is not None:
            return candidate
    raise CameraError("当前 MVS Python SDK 缺少结构体：%s" % " / ".join(names))


def enumerate_devices(sdk, layer_mask: Optional[int] = None) -> List[DeviceEntry]:
    """枚举相机设备。"""
    if layer_mask is None:
        layer_mask = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_CAMERALINK_DEVICE
    device_list = _struct(sdk, "MV_CC_DEVICE_INFO_LIST")()
    enum = None
    targets = [getattr(sdk, "MvCamera", None), getattr(sdk, "MvCameraControl", None), sdk]
    for target in targets:
        if target is not None and hasattr(target, "MV_CC_EnumDevices"):
            enum = getattr(target, "MV_CC_EnumDevices")
            break
    if enum is None:
        raise CameraError("当前 MVS Python SDK 未导出 MV_CC_EnumDevices")
    try:
        ret = enum(layer_mask, device_list)
    except TypeError:
        # 少数版本导出的是实例方法，退化为模块级函数
        module_enum = getattr(sdk, "MV_CC_EnumDevices", None)
        if module_enum is None:
            raise
        ret = module_enum(layer_mask, device_list)
    if ret != 0:
        raise CameraError("枚举设备失败：%s" % error_text(ret))

    entries: List[DeviceEntry] = []
    for index in range(device_list.nDeviceNum):
        pointer = device_list.pDeviceInfo[index]
        info = ctypes.cast(pointer, ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        layer = int(info.nTLayerType)
        model = serial = user_id = ip = ""
        special = info.SpecialInfo
        if layer == MV_GIGE_DEVICE and hasattr(special, "stGigEInfo"):
            gige = special.stGigEInfo
            model = _decode_char_array(gige.chModelName)
            serial = _decode_char_array(gige.chSerialNumber)
            user_id = _decode_char_array(gige.chUserDefinedName)
            value = int(gige.nCurrentIp)
            ip = "%d.%d.%d.%d" % ((value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
        elif layer == MV_USB_DEVICE and hasattr(special, "stUsb3VInfo"):
            usb = special.stUsb3VInfo
            model = _decode_char_array(usb.chModelName)
            serial = _decode_char_array(usb.chSerialNumber)
            user_id = _decode_char_array(usb.chUserDefinedName)
        entries.append(
            DeviceEntry(
                index=index,
                layer=_layer_name(layer),
                model=model,
                serial=serial,
                user_id=user_id,
                ip=ip,
                raw=info,
            )
        )
    return entries


@dataclass
class RawFrame:
    """一帧原始数据（不含像素拷贝时 ``data`` 为 None）。"""

    host_ns: int
    frame_num: Optional[int] = None
    lost_packet: Optional[int] = None
    dev_tick: Optional[float] = None
    width: int = 0
    height: int = 0
    pixel_type: Optional[int] = None
    n_bytes: int = 0
    data: Optional[bytes] = None


@dataclass
class CaptureConfig:
    """采集与相机侧参数。"""

    duration_s: float = 60.0
    pixel_format: Optional[int] = None   # None 表示保持相机当前设置
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    frame_rate: Optional[float] = None
    image_node_num: int = 6              # 主机侧缓冲帧数
    trigger_off: bool = True
    timeout_ms: int = 1000
    quality_frames: int = 8
    quality_warmup: int = 15
    roi: int = 512
    dark_frames: int = 4                 # 暗场采集帧数（--dark-field 时使用）
    image_quality: str = "diagnostic"    # diagnostic=图像质量只诊断；enforce=坏点率计入判定
    sim_realtime: bool = False


class HikCamera:
    """对 MVS SDK 相机句柄的轻量封装。"""

    def __init__(self, sdk, log=print):
        self.sdk = sdk
        self.log = log
        self.cam = None
        self.device: Optional[DeviceEntry] = None
        self.opened = False
        self.grabbing = False
        self._frame_out = None
        self._legacy_buffer = None
        self._legacy_size = 0
        self.params: Dict[str, object] = {}

    # -- 基础属性节点 -------------------------------------------------------
    def _int(self, node: str) -> Optional[int]:
        try:
            param = _struct(self.sdk, "MVCC_INTVALUE", "MVCC_INTVALUE_EX")()
            ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
            if self.cam.MV_CC_GetIntValue(node, param) == 0:
                return int(param.nCurValue)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _float(self, node: str) -> Optional[float]:
        try:
            param = _struct(self.sdk, "MVCC_FLOATVALUE", "MVCC_FLOATVALUE_EX")()
            ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
            if self.cam.MV_CC_GetFloatValue(node, param) == 0:
                return float(param.fCurValue)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _enum(self, node: str) -> Optional[int]:
        try:
            param = _struct(self.sdk, "MVCC_ENUMVALUE")()
            ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
            if self.cam.MV_CC_GetEnumValue(node, param) == 0:
                return int(param.nCurValue)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _string(self, node: str) -> Optional[str]:
        try:
            param = _struct(self.sdk, "MVCC_STRINGVALUE", "MVCC_STRINGVALUE_EX")()
            ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
            if self.cam.MV_CC_GetStringValue(node, param) == 0:
                return _decode_char_array(param.chCurValue)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _set_enum(self, node: str, value: int) -> bool:
        try:
            return self.cam.MV_CC_SetEnumValue(node, value) == 0
        except Exception:  # noqa: BLE001
            return False

    def _set_int(self, node: str, value: int) -> bool:
        try:
            return self.cam.MV_CC_SetIntValue(node, value) == 0
        except Exception:  # noqa: BLE001
            return False

    def _set_float(self, node: str, value: float) -> bool:
        try:
            return self.cam.MV_CC_SetFloatValue(node, value) == 0
        except Exception:  # noqa: BLE001
            return False

    def _set_bool(self, node: str, value: bool) -> bool:
        try:
            return self.cam.MV_CC_SetBoolValue(node, value) == 0
        except Exception:  # noqa: BLE001
            return False

    # -- 生命周期 -----------------------------------------------------------
    def open(
        self,
        index: Optional[int] = None,
        serial: Optional[str] = None,
        exclusive: bool = True,
    ) -> DeviceEntry:
        devices = enumerate_devices(self.sdk)
        if not devices:
            raise CameraError("未发现相机，请检查 USB3 线缆与供电，确认 MVS 客户端能识别设备")
        chosen: Optional[DeviceEntry] = None
        if serial:
            for device in devices:
                if device.serial == serial:
                    chosen = device
                    break
            if chosen is None:
                raise CameraError("未找到序列号为 %s 的相机" % serial)
        else:
            if index is None:
                index = 0
            if index < 0 or index >= len(devices):
                raise CameraError("设备序号 %d 超出范围（共 %d 台）" % (index, len(devices)))
            chosen = devices[index]
        self.device = chosen
        self.log("打开设备：%s" % chosen.label())

        cam = _camera_class(self.sdk)()
        ret = cam.MV_CC_CreateHandle(chosen.raw)
        if ret != 0:
            raise CameraError("创建句柄失败：%s" % error_text(ret))
        self.cam = cam
        access = getattr(self.sdk, "MV_ACCESS_Exclusive", 1) if exclusive else getattr(
            self.sdk, "MV_ACCESS_Control", 2
        )
        ret = cam.MV_CC_OpenDevice(access, 0)
        if ret != 0:
            self._destroy()
            raise CameraError("打开设备失败：%s（可能被 MVS 客户端占用）" % error_text(ret))
        self.opened = True

        if chosen.layer == "GigE Vision":
            try:
                packet = int(cam.MV_CC_GetOptimalPacketSize())
                if packet > 0:
                    cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet)
            except Exception:  # noqa: BLE001
                pass
        return chosen

    def configure(self, cfg: CaptureConfig) -> List[str]:
        """按配置写入相机参数，返回改动日志。"""
        changes: List[str] = []
        if cfg.trigger_off:
            if self._set_enum("TriggerMode", 0):
                changes.append("TriggerMode=Off")
        if cfg.pixel_format is not None:
            if self._set_enum("PixelFormat", int(cfg.pixel_format)):
                changes.append("PixelFormat=%s" % self.pixel_name(int(cfg.pixel_format)))
        if cfg.exposure_us is not None:
            if self._set_float("ExposureTime", float(cfg.exposure_us)):
                changes.append("ExposureTime=%.1f us" % cfg.exposure_us)
        if cfg.gain_db is not None:
            if self._set_float("Gain", float(cfg.gain_db)):
                changes.append("Gain=%.1f dB" % cfg.gain_db)
        if cfg.frame_rate is not None:
            self._set_bool("AcquisitionFrameRateEnable", True)
            if self._set_float("AcquisitionFrameRate", float(cfg.frame_rate)):
                changes.append("AcquisitionFrameRate=%.1f fps" % cfg.frame_rate)
        if cfg.image_node_num:
            try:
                self.cam.MV_CC_SetImageNodeNum(int(cfg.image_node_num))
                changes.append("ImageNodeNum=%d" % cfg.image_node_num)
            except Exception:  # noqa: BLE001
                pass
        self.params = self.read_params()
        return changes

    @staticmethod
    def pixel_name(value: int) -> str:
        from .pixels import pixel_name  # 延迟导入，避免 numpy 依赖

        return pixel_name(value)

    def read_params(self) -> Dict[str, object]:
        """读取相机参数快照（写入报告与参数文档）。"""
        params: Dict[str, object] = {}
        if self.device is not None:
            params["设备型号"] = self.device.model
            params["序列号"] = self.device.serial
            params["传输层"] = self.device.layer
            if self.device.user_id:
                params["用户自定义名"] = self.device.user_id
        for key, node in (
            ("固件版本", "DeviceVersion"),
            ("制造商", "DeviceManufacturerName"),
        ):
            value = self._string(node)
            if value:
                params[key] = value
        for key, node in (
            ("宽度", "Width"),
            ("高度", "Height"),
            ("OffsetX", "OffsetX"),
            ("OffsetY", "OffsetY"),
        ):
            value = self._int(node)
            if value is not None:
                params[key] = value
        pixel_type = self._enum("PixelFormat")
        if pixel_type is not None:
            params["像素格式"] = self.pixel_name(pixel_type)
            params["像素格式代码"] = "0x%08X" % pixel_type
        for key, node, unit in (
            ("曝光时间", "ExposureTime", "us"),
            ("增益", "Gain", "dB"),
            ("采集帧率", "AcquisitionFrameRate", "fps"),
            ("实际帧率", "ResultingFrameRate", "fps"),
            ("设备温度", "DeviceTemperature", "℃"),
        ):
            value = self._float(node)
            if value is not None:
                params[key] = ("%.2f %s" % (value, unit)).strip()
        return params

    def start(self) -> None:
        if not self.opened:
            raise CameraError("相机尚未打开")
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise CameraError("开始取流失败：%s" % error_text(ret))
        self.grabbing = True
        self._frame_out = _struct(self.sdk, "MV_FRAME_OUT")()

    def stop(self) -> None:
        if self.cam is not None and self.grabbing:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:  # noqa: BLE001
                pass
            self.grabbing = False

    def _destroy(self) -> None:
        if self.cam is not None:
            try:
                self.cam.MV_CC_CloseDevice()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:  # noqa: BLE001
                pass
            self.cam = None
        self.opened = False
        self.grabbing = False

    def close(self) -> None:
        self.stop()
        self._destroy()

    # -- 取帧 ---------------------------------------------------------------
    def _fill_from_info(self, info, frame: RawFrame) -> None:
        frame.width = int(getattr(info, "nWidth", 0) or 0)
        frame.height = int(getattr(info, "nHeight", 0) or 0)
        frame.n_bytes = int(getattr(info, "nFrameLen", 0) or 0)
        pixel_type = getattr(info, "enPixelType", None)
        frame.pixel_type = int(pixel_type) if pixel_type is not None else None
        frame.frame_num = _maybe_int(getattr(info, "nFrameNum", None))
        frame.lost_packet = _maybe_int(getattr(info, "nLostPacket", None))
        second = _maybe_int(getattr(info, "nSecondCount", None))
        cycle = _maybe_int(getattr(info, "nCycleCount", None))
        if second or cycle:
            frame.dev_tick = float((second or 0) * 1_000_000 + (cycle or 0))

    def grab(self, timeout_ms: int = 1000, want_image: bool = False) -> Optional[RawFrame]:
        """取一帧；超时返回 None。"""
        if not self.grabbing:
            return None
        frame = RawFrame(host_ns=time.perf_counter_ns())
        frame_out = self._frame_out
        if frame_out is None:
            frame_out = self._frame_out = _struct(self.sdk, "MV_FRAME_OUT")()
        ctypes.memset(ctypes.byref(frame_out), 0, ctypes.sizeof(frame_out))
        ret = self.cam.MV_CC_GetImageBuffer(frame_out, timeout_ms)
        if ret != 0 or getattr(frame_out, "pBufAddr", None) is None:
            return None
        info = getattr(frame_out, "stFrameInfo", None)
        if info is None:
            info = getattr(frame_out, "stFrameInfoEx", None)
        if info is not None:
            self._fill_from_info(info, frame)
        frame.host_ns = time.perf_counter_ns()
        if want_image and frame.n_bytes > 0:
            try:
                frame.data = ctypes.string_at(frame_out.pBufAddr, frame.n_bytes)
            except Exception:  # noqa: BLE001
                frame.data = None
        try:
            self.cam.MV_CC_FreeImageBuffer(frame_out)
        except Exception:  # noqa: BLE001
            pass
        return frame


def _maybe_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
