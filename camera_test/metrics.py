"""帧级指标：帧号丢帧率、时间戳抖动比、30 秒窗口增长、丢包增长。

所有函数都只依赖 ``FrameRecord`` 列表，不依赖相机 SDK，方便离线自测。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

FRAME_MOD16 = 1 << 16
FRAME_MOD32 = 1 << 32


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class FrameRecord:
    """单帧采集记录（由视频源填充）。"""

    index: int                      # 本机接收顺序号，从 0 开始
    t_ns: int                       # 本机单调时钟，纳秒
    frame_num: Optional[int] = None  # 相机帧号（nFrameNum）
    lost_packet: Optional[int] = None  # 相机报告的丢包计数（nLostPacket）
    dev_ts: Optional[float] = None  # 设备时间戳（若可用，单位未知）
    width: int = 0
    height: int = 0
    pixel_type: Optional[int] = None
    n_bytes: int = 0                # 该帧实际字节数（帧完整性 / 带宽检查用）
    decode_error: str = ""
    image: Optional["object"] = None  # 质量分析用图像（numpy 数组）


@dataclass
class DropEvent:
    index: int      # 该帧在本机序列中的下标
    time_s: float   # 相对首帧的时间（秒）
    count: int      # 该帧之前丢失的帧数


@dataclass
class FrameLossStats:
    expected: int
    received: int
    lost: int
    loss_rate: Optional[float]
    resets: int
    duplicates: int
    events: List[DropEvent] = field(default_factory=list)


@dataclass
class IntervalStats:
    count: int
    mean_ms: float
    std_ms: float
    jitter_ratio: float     # std(dt) / mean(dt)，评判标准中的“时间戳抖动比”
    pp_ratio: float         # (max(dt) - min(dt)) / mean(dt)，辅助参考
    min_ms: float
    max_ms: float
    p50_ms: float
    p99_ms: float


@dataclass
class WindowGrowth:
    window_s: float          # 目标窗口长度（默认 30 s）
    effective_s: float       # 实际使用的窗口跨度（采集不足 30 s 时等于总时长）
    total: int               # 总丢失数
    max_window: int          # 所有窗口中最大值
    mean_window: float
    last_window: int         # 结尾窗口
    windows: List[Tuple[float, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 帧号 / 丢帧
# ---------------------------------------------------------------------------
def detect_modulus(values: Sequence[int]) -> int:
    """根据帧号量级猜测计数器位宽（16 位或 32 位）。"""
    if not values:
        return FRAME_MOD32
    return FRAME_MOD16 if max(values) < FRAME_MOD16 else FRAME_MOD32


def frame_gaps(frame_nums: Sequence[Optional[int]]) -> Tuple[List[Optional[int]], int, int]:
    """计算相邻帧之间丢失的帧数。

    返回值 ``(gaps, resets, duplicates)``：

    * ``gaps[i]``：第 i 帧相对上一帧丢掉的帧数（首帧为 0，帧号缺失为 None）。
    * ``resets``：帧号回退且不是溢出回绕的次数（视为设备重新开始计数）。
    * ``duplicates``：帧号与上一帧相同的次数。
    """
    valid = [v for v in frame_nums if v is not None]
    mod = detect_modulus(valid)
    gaps: List[Optional[int]] = []
    resets = 0
    duplicates = 0
    prev: Optional[int] = None
    for value in frame_nums:
        if value is None:
            gaps.append(None)
            continue
        if prev is None:
            gaps.append(0)
        else:
            delta = value - prev
            if delta > 0:
                gaps.append(delta - 1)
            elif delta == 0:
                gaps.append(0)
                duplicates += 1
            elif (prev - value) > mod // 2:
                # 计数器溢出回绕
                gaps.append((mod - prev + value) - 1)
            else:
                # 设备重新开始计数（掉线重连 / 重新开启采集）
                gaps.append(0)
                resets += 1
        prev = value
    return gaps, resets, duplicates


def frame_loss_stats(records: Sequence[FrameRecord]) -> FrameLossStats:
    """按帧号统计丢帧率与逐帧丢帧事件。"""
    frame_nums = [r.frame_num for r in records]
    gaps, resets, duplicates = frame_gaps(frame_nums)
    expected = 0
    lost = 0
    received = 0
    events: List[DropEvent] = []
    t0 = records[0].t_ns if records else 0
    for record, gap in zip(records, gaps):
        if gap is None:
            continue
        expected += gap + 1
        received += 1
        lost += gap
        if gap > 0:
            events.append(
                DropEvent(
                    index=record.index,
                    time_s=(record.t_ns - t0) / 1e9,
                    count=gap,
                )
            )
    rate = (lost / expected) if expected > 0 else None
    return FrameLossStats(
        expected=expected,
        received=received,
        lost=lost,
        loss_rate=rate,
        resets=resets,
        duplicates=duplicates,
        events=events,
    )


# ---------------------------------------------------------------------------
# 时间戳抖动
# ---------------------------------------------------------------------------
def interval_stats(times_ns: Sequence[int]) -> Optional[IntervalStats]:
    """相邻帧间隔的统计量；抖动比 = 标准差 / 均值。"""
    if len(times_ns) < 2:
        return None
    diffs = [b - a for a, b in zip(times_ns, times_ns[1:]) if b is not None and a is not None]
    diffs = [d for d in diffs if d > 0]
    if len(diffs) < 2:
        return None
    count = len(diffs)
    mean = sum(diffs) / count
    if mean <= 0:
        return None
    var = sum((d - mean) ** 2 for d in diffs) / count
    std = math.sqrt(var)
    ordered = sorted(diffs)
    p50 = ordered[int(0.50 * (count - 1))]
    p99 = ordered[int(0.99 * (count - 1))]
    return IntervalStats(
        count=count,
        mean_ms=mean / 1e6,
        std_ms=std / 1e6,
        jitter_ratio=std / mean,
        pp_ratio=(ordered[-1] - ordered[0]) / mean,
        min_ms=ordered[0] / 1e6,
        max_ms=ordered[-1] / 1e6,
        p50_ms=p50 / 1e6,
        p99_ms=p99 / 1e6,
    )


# ---------------------------------------------------------------------------
# 30 秒窗口增长 / 丢包增长
# ---------------------------------------------------------------------------
def window_growth(
    times_s: Sequence[float],
    counts: Sequence[int],
    window_s: float = 30.0,
    step_s: Optional[float] = None,
) -> Optional[WindowGrowth]:
    """统计滑动窗口（默认 30 s）内的事件总数增长。

    ``times_s`` 与 ``counts`` 是逐事件的时间与数量（例如丢帧事件）。
    采集时长不足 ``window_s`` 时，用整段时长作为窗口，并在结果里标注实际跨度。
    """
    pairs = sorted((float(t), int(c)) for t, c in zip(times_s, counts) if c)
    if not pairs:
        # 没有任何事件：仍然返回全 0 的结果（这是“合格”的常见情况）
        return WindowGrowth(
            window_s=window_s, effective_s=0.0, total=0,
            max_window=0, mean_window=0.0, last_window=0, windows=[],
        )
    times = [t for t, _ in pairs]
    prefix = [0]
    for _, c in pairs:
        prefix.append(prefix[-1] + c)
    span = times[-1] - times[0]
    effective = min(window_s, span) if span > 0 else window_s

    def total_in(low: float, high: float) -> int:
        i = bisect.bisect_left(times, low)
        j = bisect.bisect_left(times, high)
        return prefix[j] - prefix[i]

    step = step_s or max(effective / 20.0, 0.02)
    windows: List[Tuple[float, int]] = []
    start = times[0]
    while start <= times[-1] + 1e-9:
        windows.append((start, total_in(start, start + effective)))
        start += step
    windows.append((times[-1], total_in(times[-1] - effective, times[-1] + 1e-9)))

    values = [v for _, v in windows]
    return WindowGrowth(
        window_s=window_s,
        effective_s=effective,
        total=prefix[-1],
        max_window=max(values),
        mean_window=sum(values) / len(values),
        last_window=values[-1],
        windows=windows,
    )


def packet_loss_series(lost_values: Sequence[Optional[int]]) -> Tuple[Optional[List[int]], str]:
    """把 ``nLostPacket`` 归一化为“每帧新增丢包数”。

    不同固件/传输层对该字段的语义不同（累计计数或单帧计数），
    这里通过单调性自动识别，并返回一段说明文字。
    """
    present = [v for v in lost_values if v is not None]
    if not present:
        return None, "帧信息中未提供丢包字段"
    if all(v == 0 for v in present):
        return [0] * len(lost_values), "SDK 报告的丢包计数恒为 0"

    monotonic = all(b >= a for a, b in zip(present, present[1:]))
    series: List[int] = []
    if monotonic:
        prev = None
        for value in lost_values:
            if value is None:
                series.append(0)
                continue
            if prev is None:
                series.append(0)
            else:
                series.append(max(value - prev, 0))
            prev = value
        return series, "按累计计数（nLostPacket）逐帧求差"
    for value in lost_values:
        series.append(int(value or 0))
    return series, "按单帧计数（nLostPacket）直接累加"
