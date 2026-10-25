"""视频源：真实相机（MVS SDK）与离线模拟源。

``CaptureResult`` 是采集阶段的产物，交给 ``analyze`` 计算指标。

采集线程只负责“取帧 + 记录元数据 + 拷贝字节”，像素解码放在独立线程里，
避免 3072×2048 的拷贝/转灰度挤占下一帧的接收时间，污染时间戳抖动测量。
"""

from __future__ import annotations

import math
import os
import platform
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import pixels
from .metrics import FrameRecord
from .sdk import CameraError, CaptureConfig, HikCamera


@dataclass
class CaptureResult:
    records: List[FrameRecord] = field(default_factory=list)
    images: List[object] = field(default_factory=list)
    dark_images: List[object] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    resource: Dict[str, float] = field(default_factory=dict)
    pixel_type: Optional[int] = None
    width: int = 0
    height: int = 0
    simulated: bool = False


# ---------------------------------------------------------------------------
# 进程资源采样（补充检查用）
# ---------------------------------------------------------------------------
def _current_rss_mb() -> Optional[float]:
    """当前常驻内存（MB）：Linux 读 /proc，Windows 用 psapi，其它退化为峰值。"""
    try:
        if platform.system() == "Linux":
            with open("/proc/self/statm", "r", encoding="ascii") as handle:
                resident = int(handle.read().split()[1])
            return resident * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
        if platform.system() == "Windows":
            import ctypes
            import ctypes.wintypes as wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return counters.WorkingSetSize / (1024.0 * 1024.0)
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:  # noqa: BLE001 - 资源采样失败不影响测试
        return None


def _current_cpu_seconds() -> Optional[float]:
    try:
        times = os.times()
        return float(times.user + times.system)
    except Exception:  # noqa: BLE001
        return None


class ResourceSampler:
    """采集前后各采一次 RSS/CPU，得到区间内的资源变化。"""

    def __init__(self) -> None:
        self.start_rss = _current_rss_mb()
        self.start_cpu = _current_cpu_seconds()
        self.start_wall = time.perf_counter()

    def finish(self) -> Dict[str, float]:
        wall = time.perf_counter() - self.start_wall
        end_rss = _current_rss_mb()
        end_cpu = _current_cpu_seconds()
        cpu_percent = None
        if self.start_cpu is not None and end_cpu is not None and wall > 0:
            cpu_percent = (end_cpu - self.start_cpu) / wall * 100.0
        data: Dict[str, float] = {"wall_s": wall}
        if self.start_rss is not None and end_rss is not None:
            data["rss_start_mb"] = self.start_rss
            data["rss_end_mb"] = end_rss
            data["rss_growth_mb"] = end_rss - self.start_rss
        if cpu_percent is not None:
            data["cpu_percent"] = cpu_percent
        return data


class MvsSource:
    """通过 MVS Python SDK 采集。"""

    def __init__(self, camera: HikCamera, cfg: CaptureConfig, log=print):
        self.camera = camera
        self.cfg = cfg
        self.log = log

    @staticmethod
    def _decode_worker(work_queue, decoded, errors, lock):
        while True:
            item = work_queue.get()
            if item is None:
                return
            data, width, height, pixel_type = item
            try:
                image = pixels.to_gray(data, width, height, pixel_type)
            except pixels.UnsupportedPixelFormat as exc:
                with lock:
                    errors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(str(exc))
                continue
            with lock:
                decoded.append(image)

    def capture(self) -> CaptureResult:
        cfg = self.cfg
        result = CaptureResult(params=self.camera.params)

        # 质量帧抽取：跳过预热帧后均匀取样，避免每帧都做重处理
        expected = max(int(cfg.duration_s * 50), cfg.quality_frames * 4)
        stride = max(1, expected // max(cfg.quality_frames, 1))
        queued_quality = 0

        decoded: List[object] = []
        errors: List[str] = []
        lock = threading.Lock()
        work_queue: "queue.Queue" = queue.Queue(maxsize=max(4, cfg.quality_frames * 2))
        worker = threading.Thread(
            target=self._decode_worker,
            args=(work_queue, decoded, errors, lock),
            name="decode-worker",
            daemon=True,
        )
        worker.start()

        self.camera.start()
        start = time.perf_counter()
        deadline = start + cfg.duration_s
        index = 0
        last_progress = 0.0
        try:
            while time.perf_counter() < deadline:
                want_image = (
                    cfg.quality_frames > 0
                    and queued_quality < cfg.quality_frames
                    and index >= cfg.quality_warmup
                    and (index - cfg.quality_warmup) % stride == 0
                )
                raw = self.camera.grab(timeout_ms=cfg.timeout_ms, want_image=want_image)
                if raw is None:
                    continue
                record = FrameRecord(
                    index=index,
                    t_ns=raw.host_ns,
                    frame_num=raw.frame_num,
                    lost_packet=raw.lost_packet,
                    dev_ts=raw.dev_tick,
                    width=raw.width,
                    height=raw.height,
                    pixel_type=raw.pixel_type,
                    n_bytes=raw.n_bytes,
                )
                if raw.data is not None:
                    try:
                        work_queue.put_nowait(
                            (raw.data, raw.width, raw.height, raw.pixel_type)
                        )
                        queued_quality += 1
                    except queue.Full:
                        pass
                result.records.append(record)
                index += 1

                now = time.perf_counter()
                if now - last_progress >= 1.0:
                    last_progress = now
                    self.log(
                        "  已采集 %d 帧，用时 %.1f / %.1f s" % (index, now - start, cfg.duration_s)
                    )
        finally:
            self.camera.stop()
            work_queue.put(None)
            worker.join(timeout=60)

        result.images = decoded
        if errors:
            result.notes.append("图像解码问题：%s" % errors[0])
        if result.pixel_type is None:
            result.pixel_type = self._int_or_none(self.camera.params.get("像素格式代码"))
        return result

    def capture_dark(self, count: Optional[int] = None) -> List[object]:
        """盖住镜头后短采若干帧，用于暗场热像素检查。"""
        cfg = self.cfg
        count = count or max(1, cfg.dark_frames)
        images: List[object] = []
        self.camera.start()
        deadline = time.perf_counter() + max(5.0, cfg.duration_s * 0.2)
        try:
            while len(images) < count and time.perf_counter() < deadline:
                raw = self.camera.grab(timeout_ms=cfg.timeout_ms, want_image=True)
                if raw is None or raw.data is None:
                    continue
                try:
                    images.append(
                        pixels.to_gray(raw.data, raw.width, raw.height, raw.pixel_type)
                    )
                except pixels.UnsupportedPixelFormat:
                    break
                except Exception:  # noqa: BLE001
                    continue
        finally:
            self.camera.stop()
        return images

    @staticmethod
    def _int_or_none(value) -> Optional[int]:
        if isinstance(value, str) and value.lower().startswith("0x"):
            try:
                return int(value, 16)
            except ValueError:
                return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# 模拟源
# ---------------------------------------------------------------------------
SIM_PROFILES = {
    "clean": dict(fps=59.6, jitter=0.012, drop_rate=0.0, packet_per_min=0.0, hot=12, dead=3, banding=0.0),
    "jitter": dict(fps=59.6, jitter=0.09, drop_rate=0.0, packet_per_min=0.0, hot=12, dead=3, banding=0.0),
    "drop": dict(fps=59.6, jitter=0.02, drop_rate=0.0017, packet_per_min=0.0, hot=12, dead=3, banding=0.0),
    "packet": dict(fps=59.6, jitter=0.02, drop_rate=0.0, packet_per_min=12.0, hot=12, dead=3, banding=0.0),
    "deadpixel": dict(fps=59.6, jitter=0.02, drop_rate=0.0, packet_per_min=0.0, hot=900, dead=600, banding=0.0),
    "banding": dict(fps=59.6, jitter=0.02, drop_rate=0.0, packet_per_min=0.0, hot=12, dead=3, banding=9.0),
}


class SimulatedSource:
    """没有相机时的离线自检源：合成帧号、时间戳与平场/暗场图像。"""

    def __init__(self, cfg: CaptureConfig, profile: str = "clean", size: int = 640, seed: int = 20261013, log=print):
        if profile not in SIM_PROFILES:
            raise ValueError("未知模拟场景 %s，可选：%s" % (profile, ", ".join(SIM_PROFILES)))
        self.cfg = cfg
        self.profile_name = profile
        self.profile = SIM_PROFILES[profile]
        self.size = size
        self.seed = seed
        self.log = log

    def _make_image(self, np_rng, hot_positions, dead_positions, stripe_phase: float):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise CameraError("模拟模式需要 numpy：%s" % exc)
        size = self.size
        y = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :]
        vignette = 1.0 - 0.25 * (x * x + y * y)          # 模拟镜头暗角
        base = (128.0 * vignette + 6.0 * y).astype(np.float32)
        noise = np_rng.normal(0.0, 3.0, size=(size, size)).astype(np.float32)
        image = base + noise
        banding = float(self.profile["banding"])
        if banding:
            rows = np.arange(size, dtype=np.float32)
            image += banding * np.sin(2.0 * math.pi * (rows / 8.0 + stripe_phase))[:, None]
        for yy, xx in hot_positions:      # 固定坏点位置，模拟传感器缺陷
            image[yy, xx] = 255.0
        for yy, xx in dead_positions:
            image[yy, xx] = 0.0
        return np.clip(image, 0, 255).astype(np.uint8)

    def capture(self) -> CaptureResult:
        cfg = self.cfg
        rng = random.Random(self.seed)
        try:
            import numpy as np

            np_rng = np.random.default_rng(self.seed)
        except ImportError as exc:  # pragma: no cover
            raise CameraError("模拟模式需要 numpy：%s" % exc)
        profile = self.profile
        fps = float(profile["fps"])
        duration = cfg.duration_s
        nominal_count = int(duration * fps)
        result = CaptureResult(simulated=True)
        result.params = {
            "设备型号": "MV-CS060-10UC-PRO（模拟）",
            "序列号": "SIM-000000000",
            "传输层": "模拟源",
            "宽度": self.size,
            "高度": self.size,
            "像素格式": "Mono8",
            "像素格式代码": "0x%08X" % pixels.MONO8,
            "采集帧率": "%.1f fps" % fps,
            "实际帧率": "%.1f fps" % fps,
        }
        result.pixel_type = pixels.MONO8
        result.width = result.height = self.size
        result.notes.append("模拟场景：%s（无真实相机）" % self.profile_name)

        quality_stride = max(1, nominal_count // max(cfg.quality_frames, 1))
        hot_positions = [
            (int(np_rng.integers(2, self.size - 2)), int(np_rng.integers(2, self.size - 2)))
            for _ in range(int(profile["hot"]))
        ]
        dead_positions = [
            (int(np_rng.integers(2, self.size - 2)), int(np_rng.integers(2, self.size - 2)))
            for _ in range(int(profile["dead"]))
        ]
        t0 = time.perf_counter()
        frame_num = 0
        packet_total = 0
        lost_per_frame = 0.0
        drop_rate = float(profile["drop_rate"])
        packet_per_frame = float(profile["packet_per_min"]) / (fps * 60.0)
        jitter = float(profile["jitter"])
        wall_start = time.perf_counter()
        for i in range(nominal_count):
            base_t = i / fps
            # 相邻间隔差值的方差是单帧抖动的 2 倍，故标准差需除以 sqrt(2)
            jitter_s = rng.gauss(0.0, jitter / fps / math.sqrt(2.0))
            t_ns = int((base_t + jitter_s) * 1e9) + t0
            frame_num += 1
            if drop_rate and rng.random() < drop_rate:
                frame_num += rng.randint(1, 3)  # 丢帧：帧号跳变
            lost_per_frame += packet_per_frame
            if lost_per_frame >= 1.0:
                packet_total += int(lost_per_frame)
                lost_per_frame -= int(lost_per_frame)
            record = FrameRecord(
                index=i,
                t_ns=t_ns,
                frame_num=frame_num,
                lost_packet=packet_total,
                dev_ts=float(frame_num) * 1e6 + jitter_s * 1e6,
                width=self.size,
                height=self.size,
                pixel_type=pixels.MONO8,
                n_bytes=self.size * self.size,
            )
            if (
                cfg.quality_frames > 0
                and i >= cfg.quality_warmup
                and i % quality_stride == 0
                and len(result.images) < cfg.quality_frames
            ):
                image = self._make_image(
                    np_rng, hot_positions, dead_positions, i / max(nominal_count, 1)
                )
                result.images.append(image)
            result.records.append(record)
            if cfg.sim_realtime:
                target = wall_start + (t_ns - t0) / 1e9
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        return result

    def capture_dark(self, count: Optional[int] = None) -> List[object]:
        """合成暗场：低电平 + 固定位置的热像素（与真实传感器缺陷一致）。"""
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return []
        count = count or max(1, self.cfg.dark_frames)
        np_rng = np.random.default_rng(self.seed + 1)
        hot_positions = [
            (int(np_rng.integers(2, self.size - 2)), int(np_rng.integers(2, self.size - 2)))
            for _ in range(3)
        ]
        images = []
        for _ in range(count):
            image = np_rng.normal(3.0, 1.5, (self.size, self.size)).astype(np.float32)
            for yy, xx in hot_positions:
                image[yy, xx] = 220.0
            images.append(np.clip(image, 0, 255).astype(np.uint8))
        return images
