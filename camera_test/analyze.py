"""把采集结果换算成接口判定项与图像质量诊断。

接口综合判定只统计“计入”项（B 组四项，以及 --image-quality enforce 时的
坏点率）；FFT 峰值比、PRNU/DSNU 与暗场热像素只作诊断/补充，不改变接口结论。
"""

from __future__ import annotations

import datetime
import platform
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import extra, image_metrics, metrics, thresholds
from .sdk import CaptureConfig
from .sources import CaptureResult
from .thresholds import Grade, MetricResult, Thresholds


@dataclass
class TestReport:
    generated_at: str
    result: CaptureResult
    thresholds: Thresholds
    metrics: List[MetricResult]
    extra_checks: List[extra.ExtraCheck]
    overall: Grade
    image_quality_grade: Grade = Grade.NA
    stats: Dict[str, object] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _fmt_pct(value: Optional[float]) -> str:
    return "—" if value is None else "%.4f%%" % (value * 100.0)


def _fmt_float(value: Optional[float], digits: int = 4) -> str:
    return "—" if value is None else ("%." + str(digits) + "f") % value


def params_int(params: Dict[str, object], key: str) -> Optional[int]:
    """把参数快照里的 ``3072`` / ``3072 像素`` 之类字符串转成整数。"""
    value = params.get(key)
    if value is None:
        return None
    try:
        return int(str(value).strip().split()[0])
    except (TypeError, ValueError):
        return None


def _environment(sdk_version: str = "未加载") -> Dict[str, str]:
    env = {
        "操作系统": "%s %s" % (platform.system(), platform.release()),
        "主机": platform.machine(),
        "Python": sys.version.split()[0],
        "MVS SDK": sdk_version,
    }
    try:
        import numpy

        env["numpy"] = numpy.__version__
    except ImportError:
        env["numpy"] = "未安装"
    return env


def interval_extremes(records: List[metrics.FrameRecord]) -> Dict[str, float]:
    """帧间隔的极值信息：中位、最大、p99.9、卡顿次数（>2×中位）。"""
    times = [record.t_ns / 1e6 for record in records]
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    if len(diffs) < 10:
        return {}
    diffs.sort()
    median = statistics.median(diffs)
    longest = diffs[-1]
    p999 = diffs[min(len(diffs) - 1, int(0.999 * (len(diffs) - 1)))]
    stalls = sum(1 for value in diffs if value > 2.0 * median)
    return {
        "median_ms": median,
        "max_ms": longest,
        "p999_ms": p999,
        "stalls": float(stalls),
        "ratio": longest / median if median > 0 else float("inf"),
    }


def summarize(
    result: CaptureResult,
    cfg: CaptureConfig,
    th: Optional[Thresholds] = None,
    sdk_version: str = "未加载",
) -> TestReport:
    th = th or Thresholds()
    records = result.records
    if not records:
        raise RuntimeError("没有采集到任何帧，无法评估")

    duration = max((records[-1].t_ns - records[0].t_ns) / 1e9, 1e-9)
    effective_fps = (len(records) - 1) / duration if duration > 0 else 0.0

    # 1) 帧号丢帧率 -------------------------------------------------------
    loss = metrics.frame_loss_stats(records)
    loss_rate = loss.loss_rate

    # 2) 时间戳抖动比 -----------------------------------------------------
    host_stats = metrics.interval_stats([r.t_ns for r in records])
    dev_values = [r.dev_ts for r in records if r.dev_ts is not None]
    dev_stats = None
    if len(dev_values) >= 10 and all(b > a for a, b in zip(dev_values, dev_values[1:])):
        dev_stats = metrics.interval_stats([int(round(v * 1e6)) for v in dev_values])
    jitter = host_stats.jitter_ratio if host_stats else None

    # 3) 丢帧增长（30 s） -------------------------------------------------
    drop_growth = metrics.window_growth(
        [event.time_s for event in loss.events],
        [event.count for event in loss.events],
        th_window(cfg, duration),
    )

    # 4) 丢包增长（30 s） -------------------------------------------------
    packet_series, packet_note = metrics.packet_loss_series([r.lost_packet for r in records])
    packet_growth = None
    packet_total = 0
    if packet_series is not None:
        packet_total = sum(packet_series)
        packet_growth = metrics.window_growth(
            [(r.t_ns - records[0].t_ns) / 1e9 for r in records],
            packet_series,
            th_window(cfg, duration),
        )

    # 5) 坏点率 / 6) FFT 峰值比 / 暗场热像素 -------------------------------
    quality = image_metrics.analyse_images(
        result.images, roi=cfg.roi, dark_images=result.dark_images, th=th
    )
    flat = quality.flat
    flat_valid_dead = flat is None or flat.ok_dead
    flat_valid_fft = flat is None or flat.ok_fft
    banding_note = ""
    if flat is not None and flat.verdict in ("row_banding", "col_banding"):
        banding_note = "检测到%s（%s均值谱峰比 %.0f，噪声约 3）" % (
            "全局行条纹" if flat.verdict == "row_banding" else "全局列条纹",
            "行" if flat.verdict == "row_banding" else "列",
            flat.row_profile_ratio if flat.verdict == "row_banding" else flat.col_profile_ratio,
        )

    window = th_window(cfg, duration)
    items: List[MetricResult] = []

    items.append(
        MetricResult(
            key="frame_loss",
            name=thresholds.METRIC_NAMES["frame_loss"],
            value=loss_rate,
            display=_fmt_pct(loss_rate),
            grade=thresholds.grade_frame_loss_rate(loss_rate, th),
            rule=thresholds.JUDGE_RULES["frame_loss"],
            detail="理论 %d 帧，实收 %d 帧，丢失 %d 帧%s" % (
                loss.expected,
                loss.received,
                loss.lost,
                ("，帧号重置 %d 次（重连/重启，仅统计有效区间）" % loss.resets)
                if loss.resets
                else "",
            ),
        )
    )
    jitter_detail = "主机时间戳：均间隔 %.3f ms，σ=%.3f ms" % (
        host_stats.mean_ms,
        host_stats.std_ms,
    ) if host_stats else "帧数不足，无法统计间隔"
    if dev_stats:
        jitter_detail += "；设备时间戳抖动比 %.4f" % dev_stats.jitter_ratio
    items.append(
        MetricResult(
            key="jitter",
            name=thresholds.METRIC_NAMES["jitter"],
            value=jitter,
            display=_fmt_float(jitter),
            grade=thresholds.grade_jitter(jitter, th),
            rule=thresholds.JUDGE_RULES["jitter"],
            detail=jitter_detail,
        )
    )
    items.append(
        MetricResult(
            key="frame_drop_30s",
            name=thresholds.METRIC_NAMES["frame_drop_30s"],
            value=float(drop_growth.max_window) if drop_growth else None,
            display=str(drop_growth.max_window) if drop_growth else "—",
            grade=thresholds.grade_count_growth(
                drop_growth.max_window if drop_growth else None, th.frame_drop_warn
            ),
            rule=thresholds.JUDGE_RULES["frame_drop_30s"],
            detail="窗口 %.0f s，总丢帧 %d，窗口均值 %.2f" % (
                min(window, duration),
                drop_growth.total if drop_growth else 0,
                drop_growth.mean_window if drop_growth else 0.0,
            ),
        )
    )
    if packet_growth is None:
        items.append(
            MetricResult(
                key="packet_loss_30s",
                name=thresholds.METRIC_NAMES["packet_loss_30s"],
                value=None,
                display="—",
                grade=Grade.NA,
                rule=thresholds.JUDGE_RULES["packet_loss_30s"],
                detail="SDK 未提供丢包计数（%s）" % packet_note,
            )
        )
    else:
        packet_detail = "%s，窗口 %.0f s 内新增 %d，累计 %d" % (
            packet_note,
            min(window, duration),
            packet_growth.max_window,
            packet_total,
        )
        if packet_total > 0 and loss.lost == 0:
            packet_detail += "；丢包未造成掉帧（重传或统计口径），可继续观察"
        items.append(
            MetricResult(
                key="packet_loss_30s",
                name=thresholds.METRIC_NAMES["packet_loss_30s"],
                value=float(packet_growth.max_window),
                display=str(packet_growth.max_window),
                grade=thresholds.grade_count_growth(packet_growth.max_window, th.packet_loss_warn),
                rule=thresholds.JUDGE_RULES["packet_loss_30s"],
                detail=packet_detail,
            )
        )
    dead_rate = quality.dead_pixel_rate
    dead_display = "—"
    if dead_rate is not None:
        count = int(round(dead_rate * max(quality.dead_pixel_total, 1)))
        dead_display = "%.6f（%d/%d）" % (dead_rate, count, max(quality.dead_pixel_total, 1))
    dead_grade = thresholds.grade_dead_pixel_rate(dead_rate, th)
    if not flat_valid_dead:
        dead_grade = Grade.NA
        dead_display = "—"
    dead_detail = quality.skipped or "平场 %d 帧，热像素 %d，死像素 %d（判据 max(6σ, max(12%%×邻域中值, 8 DN))）" % (
        quality.dead_pixel_frames,
        quality.dead_pixel_hot,
        quality.dead_pixel_dead,
    )
    if not flat_valid_dead and flat is not None:
        dead_detail = flat.reason
    items.append(
        MetricResult(
            key="dead_pixel",
            name=thresholds.METRIC_NAMES["dead_pixel"],
            value=dead_rate,
            display=dead_display,
            grade=dead_grade,
            rule=thresholds.JUDGE_RULES["dead_pixel"],
            detail=dead_detail,
            diagnostic=cfg.image_quality != "enforce",
        )
    )
    direction_text = {
        "row": "行方向条纹（横带）",
        "col": "列方向条纹（竖带）",
        "scene": "散点结构峰",
        "": "未知",
    }.get(quality.fft_axis, "未知")
    fft_display = "—" if quality.fft_ratio is None else "%.2f" % quality.fft_ratio
    fft_grade = thresholds.grade_fft_ratio(quality.fft_ratio, th)
    if not flat_valid_fft:
        fft_grade = Grade.NA
    fft_detail = quality.skipped or "峰值位于 %s（%s，周期 %s px），原始峰值/中位数 %s，行均值谱峰比 %s" % (
        quality.fft_frequency or "—",
        direction_text,
        "—" if not quality.fft_period_px else "%.1f" % quality.fft_period_px,
        "—" if quality.fft_raw_ratio is None else "%.1f" % quality.fft_raw_ratio,
        "—" if quality.profile_ratio is None else "%.2f" % quality.profile_ratio,
    )
    if not flat_valid_fft and flat is not None:
        fft_detail = flat.reason + "；" + fft_detail
    if banding_note:
        fft_detail = banding_note + "；" + fft_detail
    items.append(
        MetricResult(
            key="fft_ratio",
            name=thresholds.METRIC_NAMES["fft_ratio"],
            value=quality.fft_ratio,
            display=fft_display,
            grade=fft_grade,
            rule=thresholds.JUDGE_RULES["fft_ratio"],
            detail=fft_detail,
            diagnostic=True,
        )
    )

    # 补充检查（不参与上面的判定）-----------------------------------------
    extra_checks = extra.build_extra_checks(
        records=records,
        duration_s=duration,
        cfg=cfg,
        params=result.params,
        th=th,
        fps_measured=effective_fps,
        quality=quality,
        resource=result.resource,
    )
    extremes = interval_extremes(records)
    bandwidth_mbps = (
        sum(record.n_bytes for record in records) / duration / 1e6 if duration > 0 else None
    )

    counted = [item for item in items if not item.diagnostic]
    diagnostics = [item for item in items if item.diagnostic]
    overall = thresholds.overall_grade([item.grade for item in counted])
    image_quality_grade = thresholds.overall_grade([item.grade for item in diagnostics])
    stats = {
        "duration_s": duration,
        "frames": len(records),
        "effective_fps": effective_fps,
        "expected": loss.expected,
        "lost": loss.lost,
        "resets": loss.resets,
        "duplicates": loss.duplicates,
        "image_quality_grade": image_quality_grade.value,
        "jitter_ratio": jitter,
        "dev_jitter_ratio": dev_stats.jitter_ratio if dev_stats else None,
        "mean_interval_ms": host_stats.mean_ms if host_stats else None,
        "std_interval_ms": host_stats.std_ms if host_stats else None,
        "p99_interval_ms": host_stats.p99_ms if host_stats else None,
        "p999_interval_ms": extremes.get("p999_ms"),
        "max_interval_ms": extremes.get("max_ms"),
        "stall_count": int(extremes.get("stalls", 0)),
        "frame_drop_max_30s": drop_growth.max_window if drop_growth else None,
        "packet_loss_max_30s": packet_growth.max_window if packet_growth else None,
        "packet_loss_total": packet_total,
        "dead_pixel_rate": dead_rate,
        "dead_pixel_hot": quality.dead_pixel_hot,
        "dead_pixel_dead": quality.dead_pixel_dead,
        "fft_ratio": quality.fft_ratio,
        "fft_frequency": quality.fft_frequency,
        "profile_ratio": quality.profile_ratio,
        "dark_hot_rate": quality.dark_hot_rate,
        "bandwidth_mbps": bandwidth_mbps,
        "rss_growth_mb": result.resource.get("rss_growth_mb"),
        "cpu_percent": result.resource.get("cpu_percent"),
        "flat_verdict": flat.verdict if flat else None,
        "flat_label": flat.label if flat else None,
        "flat_reason": flat.reason if flat else None,
        "flat_saturation_ratio": flat.saturation_ratio if flat else None,
        "flat_structure_score": flat.structure_score if flat else None,
        "flat_row_alt": flat.row_alt_score if flat else None,
        "flat_col_alt": flat.col_alt_score if flat else None,
        "fft_axis": quality.fft_axis or None,
        "fft_axial_ratio": quality.fft_axial_ratio,
        "fft_period_px": quality.fft_period_px,
    }
    config = {
        "采集时长(s)": round(cfg.duration_s, 1),
        "统计窗口(s)": round(min(window, duration), 1),
        "质量帧数": len(result.images),
        "FFT ROI": cfg.roi,
        "图像质量判定": "计入" if cfg.image_quality == "enforce" else "仅诊断",
    }
    if cfg.exposure_us is not None:
        config["曝光时间(us)"] = cfg.exposure_us
    if cfg.gain_db is not None:
        config["增益(dB)"] = cfg.gain_db
    if cfg.frame_rate is not None:
        config["目标帧率(fps)"] = cfg.frame_rate
    if result.dark_images:
        config["暗场帧数"] = len(result.dark_images)
    if flat is not None:
        config["图像输入"] = flat.label

    notes = list(result.notes)
    notes.append("接口综合判定只统计计入项；FFT 频谱质量与图像质量项按诊断处理，不改变接口结论。")
    if loss.resets:
        notes.append(
            "检测到 %d 次帧号重置（相机重连或重新开始取流），丢帧率只统计有效区间。"
            % loss.resets
        )
    if duration < 180:
        notes.append("采集时长 %.0f s，建议正式测试用 ≥180 s（长时稳定性更有说服力）。" % duration)
    if flat is not None:
        if flat.verdict in ("scene", "overexposed"):
            notes.append("图像质量诊断不可用：%s" % flat.reason)
        elif flat.verdict in ("row_banding", "col_banding"):
            notes.append(
                "%s；属条纹类缺陷，需排查照明频闪、滚动快门或读出异常（FFT 按实际值记录为诊断项）。"
                % banding_note
            )
        elif flat.verdict == "dark":
            notes.append("质量帧为暗场：坏点率需要平场（均匀面光）后重测；暗场热像素见补充检查。")
    width = params_int(result.params, "宽度")
    height = params_int(result.params, "高度")
    if width and height and width * height != 3072 * 2048:
        notes.append(
            "当前分辨率 %d×%d（%.2f MP）不是原生 3072×2048，链路带宽与帧率结论请按实际分辨率理解。"
            % (width, height, width * height / 1e6)
        )

    report = TestReport(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result=result,
        thresholds=th,
        metrics=items,
        extra_checks=extra_checks,
        overall=overall,
        image_quality_grade=image_quality_grade,
        stats=stats,
        environment=_environment(sdk_version),
        config=config,
        notes=notes,
    )
    return report


def th_window(cfg: CaptureConfig, duration: float) -> float:
    """30 s 统计窗口；采集不足 30 s 时退化为整段时长。"""
    return 30.0 if duration >= 30.0 else max(duration, 1.0)
