"""补充检查：不参与接口综合判定的工程性检查。

七项（报告里单独一栏，状态为 通过 / 注意 / 不通过 / 参考）：

    bandwidth  链路带宽    实测 MB/s 与名义需求（宽×高×位深×目标帧率）的比值
    integrity  帧完整性    每帧实际字节数是否等于理论值（抓“半帧/截断帧”）
    fps        帧率偏差    实测帧率 vs 目标帧率
    stall      最长帧间隔  max/中位 比值 + 卡顿帧数（>2×中位间隔）
    segment    长时稳定性  按 60 s 分段统计丢帧率与帧率（看是否随时间恶化）
    resource   进程资源    采集期间 RSS 增长与 CPU 占用（参考值）
    dark_hot   暗场热像素  盖镜头短采后的热像素率（--dark-field 时才有）

设计原则：这些检查 FAIL 不会改变接口综合判定的“合格/警告/不合格”，
只用于暴露“接口项合格但工程上不可用”的情况（比如误插 USB2 口）。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import metrics, pixels, thresholds
from .sdk import CaptureConfig
from .thresholds import Grade, Thresholds


@dataclass
class ExtraCheck:
    key: str
    name: str
    display: str
    status: Optional[Grade]   # None 表示“参考值”，不做判定
    detail: str = ""


def _to_float(value) -> Optional[float]:
    """从参数快照里的字符串（如 ``59.60 fps``）解析数值。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().split()
    if not text:
        return None
    try:
        return float(text[0])
    except ValueError:
        return None


def _frame_geometry(records: Sequence[metrics.FrameRecord]):
    width = height = 0
    pixel_type = None
    for record in records:
        if record.width and record.height:
            width, height = record.width, record.height
            pixel_type = record.pixel_type
            break
    return width, height, pixel_type


def bandwidth_check(
    records: Sequence[metrics.FrameRecord],
    duration_s: float,
    cfg: CaptureConfig,
    params: Dict[str, object],
    fps_measured: float,
    th: Thresholds,
) -> ExtraCheck:
    """实测带宽 = 收到字节总数 / 采集时长；需求 = 宽×高×位深×目标帧率。"""
    name = thresholds.EXTRA_NAMES["bandwidth"]
    width, height, pixel_type = _frame_geometry(records)
    if not (width and height and duration_s > 0 and records):
        return ExtraCheck("bandwidth", name, "—", Grade.NA, "缺少分辨率/时长信息")
    bpp = pixels.describe(pixel_type).bpp or 1.0
    total_bytes = sum(record.n_bytes for record in records)
    if total_bytes <= 0:
        return ExtraCheck("bandwidth", name, "—", Grade.NA, "帧信息中没有字节数")
    measured = total_bytes / duration_s / 1e6
    candidates = [
        cfg.frame_rate,
        _to_float(params.get("实际帧率")),
        _to_float(params.get("采集帧率")),
        fps_measured,
    ]
    target_fps = max([value for value in candidates if value] or [fps_measured])
    required = width * height * bpp * target_fps / 1e6
    ratio = measured / required if required > 0 else None
    status = Grade.NA
    if ratio is not None:
        if ratio >= th.bandwidth_ratio_pass:
            status = Grade.PASS
        elif ratio >= th.bandwidth_ratio_warn:
            status = Grade.WARN
        else:
            status = Grade.FAIL
    detail = "实测 %.1f MB/s，需求 %.1f MB/s（%d×%d，%.1f fps）" % (
        measured,
        required,
        width,
        height,
        target_fps,
    )
    link = params.get("链路速率")
    if link:
        detail += "；相机报告链路速率 %s" % link
    if measured < 100 and required > 200:
        detail += "；带宽低于 100 MB/s，优先怀疑插在 USB2 口或被其它设备占满"
    return ExtraCheck(
        "bandwidth",
        name,
        "%.1f / %.1f MB/s" % (measured, required),
        status,
        detail,
    )


def integrity_check(
    records: Sequence[metrics.FrameRecord],
    th: Thresholds,
) -> ExtraCheck:
    """帧字节数必须等于 宽×高×位深（抓半帧/截断帧）。"""
    name = thresholds.EXTRA_NAMES["integrity"]
    width, height, pixel_type = _frame_geometry(records)
    if not (width and height and records):
        return ExtraCheck("integrity", name, "—", Grade.NA, "缺少分辨率信息")
    bpp = pixels.describe(pixel_type).bpp or 1.0
    expected = int(round(width * height * bpp))
    bad = 0
    zero = 0
    sample = ""
    for record in records:
        if record.n_bytes <= 0:
            zero += 1
            continue
        if record.n_bytes != expected:
            bad += 1
            if not sample:
                sample = "（例如 %d 字节）" % record.n_bytes
    total_bad = bad + zero
    if total_bad == 0:
        status = Grade.PASS
    else:
        status = Grade.FAIL
    display = "0 异常帧" if total_bad == 0 else "%d 异常帧" % total_bad
    detail = "理论单帧 %d 字节%s；%d 帧字节数为 0" % (
        expected,
        sample,
        zero,
    )
    return ExtraCheck("integrity", name, display, status, detail)


def fps_check(
    records: Sequence[metrics.FrameRecord],
    duration_s: float,
    cfg: CaptureConfig,
    params: Dict[str, object],
    th: Thresholds,
) -> ExtraCheck:
    name = thresholds.EXTRA_NAMES["fps"]
    if len(records) < 3 or duration_s <= 0:
        return ExtraCheck("fps", name, "—", Grade.NA, "帧数不足，无法统计帧率")
    measured = (len(records) - 1) / duration_s
    target = (
        cfg.frame_rate
        or _to_float(params.get("实际帧率"))
        or _to_float(params.get("采集帧率"))
        or measured
    )
    deviation = abs(measured - target) / target if target else 0.0
    if deviation <= th.fps_deviation_pass:
        status = Grade.PASS
    elif deviation <= th.fps_deviation_warn:
        status = Grade.WARN
    else:
        status = Grade.FAIL
    detail = "实测 %.2f fps，目标 %.2f fps，偏差 %.2f%%" % (
        measured,
        target,
        deviation * 100.0,
    )
    if cfg.exposure_us and target:
        if cfg.exposure_us > 1e6 / target:
            detail += "；曝光 %.0f us 超过帧周期 %.0f us，必然掉帧" % (
                cfg.exposure_us,
                1e6 / target,
            )
    return ExtraCheck("fps", name, "%.2f fps" % measured, status, detail)


def stall_check(
    records: Sequence[metrics.FrameRecord],
    th: Thresholds,
) -> ExtraCheck:
    """最长帧间隔相对中位间隔的倍数 + 卡顿帧数。"""
    name = thresholds.EXTRA_NAMES["stall"]
    times = [record.t_ns / 1e6 for record in records]
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    if len(diffs) < 10:
        return ExtraCheck("stall", name, "—", Grade.NA, "帧数不足，无法统计间隔")
    diffs.sort()
    median = statistics.median(diffs)
    longest = diffs[-1]
    p999 = diffs[min(len(diffs) - 1, int(0.999 * (len(diffs) - 1)))]
    stalls = sum(1 for value in diffs if value > 2.0 * median)
    ratio = longest / median if median > 0 else float("inf")
    if ratio <= th.stall_ratio_pass and stalls == 0:
        status = Grade.PASS
    elif ratio <= th.stall_ratio_warn:
        status = Grade.WARN
    else:
        status = Grade.FAIL
    display = "%.1f ms（%.2f×）" % (longest, ratio)
    detail = "中位间隔 %.2f ms，p99.9 %.2f ms，卡顿（>2×中位）%d 次" % (
        median,
        p999,
        stalls,
    )
    return ExtraCheck("stall", name, display, status, detail)


def segment_check(
    records: Sequence[metrics.FrameRecord],
    th: Thresholds,
    segment_s: float = 60.0,
) -> ExtraCheck:
    """按 60 s 分段，检查丢帧率与帧率是否随时间恶化。"""
    name = thresholds.EXTRA_NAMES["segment"]
    if len(records) < 120:
        return ExtraCheck("segment", name, "—", Grade.NA, "帧数太少（<120），不做分段")
    t0 = records[0].t_ns
    segments: List[List[metrics.FrameRecord]] = []
    current: List[metrics.FrameRecord] = [records[0]]
    start = 0.0
    for record in records[1:]:
        elapsed = (record.t_ns - t0) / 1e9
        if elapsed - start >= segment_s:
            segments.append(current)
            current = []
            start = elapsed
        current.append(record)
    if current:
        segments.append(current)

    losses = []
    fps_list = []
    for segment in segments:
        stats = metrics.frame_loss_stats(segment)
        losses.append(stats.loss_rate or 0.0)
        span = (segment[-1].t_ns - segment[0].t_ns) / 1e9
        fps_list.append((len(segment) - 1) / span if span > 0 else 0.0)
    if len(segments) < 2:
        return ExtraCheck(
            "segment",
            name,
            "%d 段" % len(segments),
            None,
            "采集时长不足两段（<120 s），分段对比仅作参考",
        )
    worst = max(losses)
    if worst == 0:
        status = Grade.PASS
    elif worst < th.frame_loss_warn:
        status = Grade.WARN
    else:
        status = Grade.FAIL
    display = "%d 段，最大丢帧 %.4f%%" % (len(segments), worst * 100.0)
    detail = "分段丢帧率 %s；分段帧率 %s" % (
        " / ".join("%.4f%%" % (value * 100.0) for value in losses),
        " / ".join("%.1f" % value for value in fps_list),
    )
    return ExtraCheck("segment", name, display, status, detail)


def resource_check(
    resource: Optional[Dict[str, float]],
    th: Thresholds,
) -> ExtraCheck:
    name = thresholds.EXTRA_NAMES["resource"]
    if not resource or resource.get("rss_start_mb") is None or resource.get("rss_end_mb") is None:
        return ExtraCheck("resource", name, "—", None, "未采集到进程资源信息")
    start = float(resource["rss_start_mb"])
    end = float(resource["rss_end_mb"])
    growth = end - start
    cpu = resource.get("cpu_percent")
    status = Grade.PASS if growth < th.rss_growth_warn_mb else Grade.WARN
    display = "+%.0f MB" % growth
    detail = "RSS %.0f → %.0f MB，CPU 平均 %s" % (
        start,
        end,
        "—" if cpu is None else "%.0f%%" % cpu,
    )
    return ExtraCheck("resource", name, display, status, detail)


def dark_hot_check(
    quality,
    th: Thresholds,
) -> ExtraCheck:
    name = thresholds.EXTRA_NAMES["dark_hot"]
    rate = getattr(quality, "dark_hot_rate", None)
    if rate is None:
        return ExtraCheck(
            "dark_hot",
            name,
            "—",
            Grade.NA,
            "未做暗场采集（加 --dark-field 并盖住镜头可采集）",
        )
    if rate < th.dark_hot_pass:
        status = Grade.PASS
    elif rate <= th.dark_hot_warn:
        status = Grade.WARN
    else:
        status = Grade.FAIL
    count = getattr(quality, "dark_hot_count", 0)
    total = getattr(quality, "dead_pixel_total", 0) or max(count, 1)
    display = "%.6f（%d/%d）" % (rate, count, total)
    detail = "暗场 %d 帧，中位 %.1f DN，σ %.2f DN，最亮 %.1f DN" % (
        getattr(quality, "dark_frames", 0),
        getattr(quality, "dark_level", 0.0) or 0.0,
        getattr(quality, "dark_sigma", 0.0) or 0.0,
        getattr(quality, "dark_max", 0.0) or 0.0,
    )
    return ExtraCheck("dark_hot", name, display, status, detail)


def build_extra_checks(
    records: Sequence[metrics.FrameRecord],
    duration_s: float,
    cfg: CaptureConfig,
    params: Dict[str, object],
    th: Thresholds,
    fps_measured: float,
    quality,
    resource: Optional[Dict[str, float]],
) -> List[ExtraCheck]:
    return [
        bandwidth_check(records, duration_s, cfg, params, fps_measured, th),
        integrity_check(records, th),
        fps_check(records, duration_s, cfg, params, th),
        stall_check(records, th),
        segment_check(records, th),
        resource_check(resource, th),
        dark_hot_check(quality, th),
    ]
