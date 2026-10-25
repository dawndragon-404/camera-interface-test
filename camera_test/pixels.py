"""像素格式解析：把相机原始缓冲转成用于质量分析的 8 位灰度图。

只覆盖 MVS 里最常见的几种格式（Mono8、Bayer8、RGB8/BGR8、10/12 位及其打包格式）。
遇到不支持的格式时抛出 ``UnsupportedPixelFormat``，由上层跳过图像质量项并在报告里说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:  # numpy 只在需要做图像分析时才必需
    import numpy as np
except ImportError:  # pragma: no cover - 运行期提示
    np = None  # type: ignore


class UnsupportedPixelFormat(ValueError):
    """无法解析的像素格式。"""


# GenICam PFNC 数值（与 MvGvspPixelType 一致）
MONO8 = 0x01080001
MONO10 = 0x01100003
MONO10_PACKED = 0x010C0004
MONO12 = 0x01100005
MONO12_PACKED = 0x010C0006
MONO16 = 0x01100007

BAYER8 = (0x01080008, 0x01080009, 0x0108000A, 0x0108000B)          # GR/RG/GB/BG
BAYER10 = (0x0110000C, 0x0110000D, 0x0110000E, 0x0110000F)
BAYER12 = (0x01100010, 0x01100011, 0x01100012, 0x01100013)
RGB8 = 0x02180014
BGR8 = 0x02180015

_NAMES: Dict[int, str] = {
    MONO8: "Mono8",
    MONO10: "Mono10",
    MONO10_PACKED: "Mono10p",
    MONO12: "Mono12",
    MONO12_PACKED: "Mono12p",
    MONO16: "Mono16",
    0x01080008: "BayerGR8",
    0x01080009: "BayerRG8",
    0x0108000A: "BayerGB8",
    0x0108000B: "BayerBG8",
    0x0110000C: "BayerGR10",
    0x0110000D: "BayerRG10",
    0x0110000E: "BayerGB10",
    0x0110000F: "BayerBG10",
    0x01100010: "BayerGR12",
    0x01100011: "BayerRG12",
    0x01100012: "BayerGB12",
    0x01100013: "BayerBG12",
    RGB8: "RGB8",
    BGR8: "BGR8",
    0x02200016: "RGBa8",
    0x02200017: "BGRa8",
}


@dataclass(frozen=True)
class PixelFormatInfo:
    name: str
    kind: str      # mono8 / mono16 / rgb8 / bgr8 / packed10 / packed12
    bpp: float     # 每像素字节数（打包格式为小数）

    @property
    def supported(self) -> bool:
        return self.kind in {"mono8", "mono16", "rgb8", "bgr8", "packed10", "packed12"}


def describe(pixel_type: Optional[int]) -> PixelFormatInfo:
    """返回像素格式描述；无法识别的格式保留原始数值。"""
    if pixel_type is None:
        return PixelFormatInfo("未知", "unknown", 0.0)
    value = int(pixel_type)
    if value in BAYER8:
        return PixelFormatInfo(_NAMES.get(value, "Bayer8"), "mono8", 1.0)
    if value in BAYER10:
        return PixelFormatInfo(_NAMES.get(value, "Bayer10"), "mono16", 2.0)
    if value in BAYER12:
        return PixelFormatInfo(_NAMES.get(value, "Bayer12"), "mono16", 2.0)
    if value == MONO8:
        return PixelFormatInfo("Mono8", "mono8", 1.0)
    if value in (MONO10, MONO12, MONO16):
        return PixelFormatInfo(_NAMES[value], "mono16", 2.0)
    if value == MONO10_PACKED:
        return PixelFormatInfo("Mono10p", "packed10", 1.25)
    if value == MONO12_PACKED:
        return PixelFormatInfo("Mono12p", "packed12", 1.5)
    if value == RGB8:
        return PixelFormatInfo("RGB8", "rgb8", 3.0)
    if value == BGR8:
        return PixelFormatInfo("BGR8", "bgr8", 3.0)
    if value in (0x02200016, 0x02200017):
        return PixelFormatInfo(_NAMES[value], "rgba8", 4.0)
    return PixelFormatInfo("0x%08X" % value, "unknown", 0.0)


def pixel_name(pixel_type: Optional[int]) -> str:
    return describe(pixel_type).name


# ---------------------------------------------------------------------------
# 原始缓冲 -> 8 位灰度
# ---------------------------------------------------------------------------
def _unpack10(data: bytes, count: int):
    """Mono10p：5 字节 4 像素。"""
    raw = np.frombuffer(data, dtype=np.uint8)
    usable = (raw.size // 5) * 5
    blocks = raw[:usable].reshape(-1, 5).astype(np.uint16)
    b0, b1, b2, b3, b4 = blocks.T
    pixels = np.empty((blocks.shape[0], 4), dtype=np.uint16)
    pixels[:, 0] = (b0 << 2) | (b1 >> 6)
    pixels[:, 1] = ((b1 & 0x3F) << 4) | (b2 >> 4)
    pixels[:, 2] = ((b2 & 0x0F) << 6) | (b3 >> 2)
    pixels[:, 3] = ((b3 & 0x03) << 8) | b4
    return pixels.reshape(-1)[:count]


def _unpack12(data: bytes, count: int):
    """Mono12p：3 字节 2 像素。"""
    raw = np.frombuffer(data, dtype=np.uint8)
    usable = (raw.size // 3) * 3
    blocks = raw[:usable].reshape(-1, 3).astype(np.uint16)
    b0, b1, b2 = blocks.T
    pixels = np.empty((blocks.shape[0], 2), dtype=np.uint16)
    pixels[:, 0] = (b0 << 4) | (b1 >> 4)
    pixels[:, 1] = ((b1 & 0x0F) << 8) | b2
    return pixels.reshape(-1)[:count]


def to_gray(raw: bytes, width: int, height: int, pixel_type: Optional[int]):
    """把一帧原始数据转成 ``uint8`` 灰度图 (height, width)。"""
    if np is None:
        raise UnsupportedPixelFormat("需要 numpy 才能做图像分析")
    info = describe(pixel_type)
    count = width * height
    expected = int(round(count * info.bpp))
    if len(raw) < expected:
        raise UnsupportedPixelFormat(
            "缓冲区长度不足：需要 %d 字节，实际 %d 字节（格式 %s）" % (expected, len(raw), info.name)
        )
    data = raw[:expected]

    if info.kind == "mono8":
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width).copy()
    if info.kind == "mono16":
        depth = info.name.lower()
        arr = np.frombuffer(data, dtype="<u2").astype(np.uint32)
        if "10" in depth:
            arr = arr >> 2
        elif "12" in depth:
            arr = arr >> 4
        else:  # Mono16
            arr = arr >> 8
        return arr.clip(0, 255).astype(np.uint8).reshape(height, width)
    if info.kind == "packed10":
        arr = (_unpack10(data, count) >> 2).clip(0, 255).astype(np.uint8)
        return arr.reshape(height, width)
    if info.kind == "packed12":
        arr = (_unpack12(data, count) >> 4).clip(0, 255).astype(np.uint8)
        return arr.reshape(height, width)
    if info.kind in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).astype(np.float32)
        if info.kind == "bgr8":
            b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
        else:
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.rint(gray).clip(0, 255).astype(np.uint8)
    raise UnsupportedPixelFormat(
        "暂不支持像素格式 %s，请在 MVS 中把 PixelFormat 设为 Mono8（或 Bayer8）" % info.name
    )


NAME_TO_CODE: Dict[str, int] = {name: value for value, name in _NAMES.items()}


def code_of(name: Optional[str]) -> Optional[int]:
    """按像素格式名称（如 ``Mono8``）取 GenICam 数值。"""
    if not name:
        return None
    return NAME_TO_CODE.get(name.strip())

