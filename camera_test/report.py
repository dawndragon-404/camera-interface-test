"""报告输出：控制台表格、Markdown、JSON、逐帧 CSV、相机参数快照。"""

from __future__ import annotations

import csv
import datetime
import json
import os
import unicodedata
from typing import Dict, List

from . import metrics, thresholds
from .analyze import TestReport
from .thresholds import Grade

ANSI = {
    Grade.PASS: "\033[32m",
    Grade.WARN: "\033[33m",
    Grade.FAIL: "\033[31m",
    Grade.INCOMPLETE: "\033[36m",
    Grade.NA: "\033[90m",
}
RESET = "\033[0m"

EXTRA_STATUS_TEXT = {
    Grade.PASS: "通过",
    Grade.WARN: "注意",
    Grade.FAIL: "不通过",
    Grade.NA: "不可用",
    None: "参考",
}


def extra_status_text(status) -> str:
    return EXTRA_STATUS_TEXT.get(status, "参考")


def _width(text: str) -> int:
    """按终端显示宽度计算字符串宽度（中文占 2 列）。"""
    total = 0
    for char in text:
        total += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return total


def pad(text: str, width: int, align: str = "left") -> str:
    space = max(width - _width(text), 0)
    if align == "right":
        return " " * space + text
    return text + " " * space


def _png_gray_bytes(array) -> bytes:
    """把 8 位灰度数组编码成 PNG（仅用标准库，不额外依赖 Pillow）。"""
    import struct
    import zlib

    import numpy as np

    data = np.ascontiguousarray(array, dtype=np.uint8)
    height, width = data.shape
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(height))

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def save_quality_images(images, out_dir: str, tag: str = "quality") -> List[str]:
    """把质量帧 / 暗场帧保存为 PNG，并写一张统计索引，便于目视判断条纹/场景/过曝。"""
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []
    rows = []
    for index, image in enumerate(images, start=1):
        array = np.asarray(image)
        if array.ndim != 2:
            continue
        array = np.clip(array, 0, 255).astype(np.uint8)
        name = "%s_%02d.png" % (tag, index)
        path = os.path.join(out_dir, name)
        with open(path, "wb") as handle:
            handle.write(_png_gray_bytes(array))
        saved.append(path)
        rows.append(
            [
                name,
                "%.1f" % float(array.mean()),
                int(array.min()),
                int(array.max()),
                "%.4f" % float((array >= 250).mean()),
                "%.4f" % float((array <= 5).mean()),
            ]
        )
    if rows:
        with open(os.path.join(out_dir, "index.csv"), "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "mean", "min", "max", "saturated_ratio", "dark_ratio"])
            writer.writerows(rows)
    return saved


def _render_extra_console(report: TestReport, color: bool) -> List[str]:
    """补充检查表（不参与接口综合判定）。"""
    if not report.extra_checks:
        return []
    headers = ["检查项", "实测", "状态"]
    rows = [
        [check.name, check.display, "[%s]" % extra_status_text(check.status)]
        for check in report.extra_checks
    ]
    widths = [
        max(_width(row[col]) for row in rows + [headers]) + 2 for col in range(len(headers))
    ]
    line_width = sum(widths)
    lines = ["补充检查（不参与接口综合判定）："]
    lines.append("".join(pad(h, w) for h, w in zip(headers, widths)).rstrip())
    lines.append("-" * line_width)
    for check, row in zip(report.extra_checks, rows):
        parts = []
        for index, (cell, width) in enumerate(zip(row, widths)):
            text = pad(cell, width)
            if index == 2 and color and os.environ.get("NO_COLOR") is None:
                text = ANSI.get(check.status, "") + text + RESET
            parts.append(text)
        lines.append("".join(parts).rstrip())
    lines.append("-" * line_width)
    for check in report.extra_checks:
        if check.detail:
            lines.append("  · %s：%s" % (check.name, check.detail))
        if check.status in (Grade.WARN, Grade.FAIL):
            tip = thresholds.EXTRA_ADVICE.get(check.key)
            if tip:
                lines.append("    → %s" % tip)
    lines.append("")
    return lines


def render_console(report: TestReport, color: bool = True) -> str:
    stats = report.stats
    params = report.result.params
    lines: List[str] = []
    title = "%s 相机接口测试报告" % (params.get("设备型号") or "相机")
    lines.append("=" * 78)
    lines.append(" " + title + ("（模拟数据）" if report.result.simulated else ""))
    lines.append("=" * 78)
    lines.append("测试时间 : %s" % report.generated_at)
    lines.append(
        "设备型号 : %s    序列号 : %s"
        % (params.get("设备型号", "—"), params.get("序列号", "—"))
    )
    lines.append(
        "接口类型 : %s    分辨率 : %s×%s    像素格式 : %s"
        % (
            params.get("传输层", "—"),
            params.get("宽度", report.result.width or "—"),
            params.get("高度", report.result.height or "—"),
            params.get("像素格式", "—"),
        )
    )
    lines.append(
        "采集时长 : %.1f s    帧数 : %d    平均帧率 : %.2f fps"
        % (
            float(stats["duration_s"]),
            int(stats["frames"]),
            float(stats["effective_fps"]),
        )
    )
    env = "  ".join("%s %s" % (k, v) for k, v in report.environment.items())
    lines.append("运行环境 : %s" % env)
    flat_label = stats.get("flat_label")
    if flat_label:
        lines.append("图像输入 : %s" % flat_label)
    lines.append("")

    headers = ["指标", "实测值", "类别", "判定", "判定标准（合格 / 警告 / 不合格）"]
    rows = [
        [item.name, item.display, "诊断" if item.diagnostic else "判定",
         "[%s]" % item.grade.value, item.rule]
        for item in report.metrics
    ]
    widths = [max(_width(row[col]) for row in rows + [headers]) + 2 for col in range(len(headers))]
    line_width = sum(widths)
    lines.append("".join(pad(h, w) for h, w in zip(headers, widths)).rstrip())
    lines.append("-" * line_width)
    for item, row in zip(report.metrics, rows):
        parts = []
        for index, (cell, width) in enumerate(zip(row, widths)):
            text = pad(cell, width)
            if index == 3 and color and os.environ.get("NO_COLOR") is None:
                text = ANSI.get(item.grade, "") + text + RESET
            parts.append(text)
        lines.append("".join(parts).rstrip())
    lines.append("-" * line_width)

    counted = [item for item in report.metrics if not item.diagnostic]
    diagnostics = [item for item in report.metrics if item.diagnostic]
    counts = {grade: 0 for grade in Grade}
    for item in counted:
        counts[item.grade] += 1
    overall_text = report.overall.value
    if color and os.environ.get("NO_COLOR") is None:
        overall_text = ANSI.get(report.overall, "") + overall_text + RESET
    lines.append(
        "接口综合判定 : %s    （%d 项计入：%d 合格 / %d 警告 / %d 不合格 / %d 不可用）"
        % (
            overall_text,
            len(counted),
            counts[Grade.PASS],
            counts[Grade.WARN],
            counts[Grade.FAIL],
            counts[Grade.NA],
        )
    )
    if diagnostics:
        dcounts = {grade: 0 for grade in Grade}
        for item in diagnostics:
            dcounts[item.grade] += 1
        diag_text = report.image_quality_grade.value
        if color and os.environ.get("NO_COLOR") is None:
            diag_text = ANSI.get(report.image_quality_grade, "") + diag_text + RESET
        lines.append(
            "图像质量诊断 : %s    （%d 项：%d 合格 / %d 警告 / %d 不合格 / %d 不可用；不计入接口判定）"
            % (
                diag_text,
                len(diagnostics),
                dcounts[Grade.PASS],
                dcounts[Grade.WARN],
                dcounts[Grade.FAIL],
                dcounts[Grade.NA],
            )
        )
    lines.append("")

    lines.extend(_render_extra_console(report, color))

    details = [item for item in report.metrics if item.detail]
    if details:
        lines.append("明细：")
        for item in details:
            lines.append("  · %s：%s" % (item.name, item.detail))
        lines.append("")

    advice = []
    for item in report.metrics:
        tip = thresholds.advice(item.key, item.grade)
        if tip:
            advice.append("  · %s（%s）：%s" % (item.name, item.grade.value, tip))
    if advice:
        lines.append("排查建议：")
        lines.extend(advice)
        lines.append("")
    for note in report.notes:
        lines.append("备注：%s" % note)
    return "\n".join(lines)


def render_markdown(report: TestReport) -> str:
    params = report.result.params
    stats = report.stats
    lines: List[str] = []
    lines.append("# %s 相机接口测试报告" % (params.get("设备型号") or "相机"))
    lines.append("")
    if report.result.simulated:
        lines.append("> 本报告来自**模拟数据**（未连接真实相机），仅用于验证流程与判定逻辑。")
        lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| 测试时间 | %s |" % report.generated_at)
    lines.append("| 设备型号 | %s |" % params.get("设备型号", "—"))
    lines.append("| 序列号 | %s |" % params.get("序列号", "—"))
    lines.append("| 接口类型 | %s |" % params.get("传输层", "—"))
    lines.append(
        "| 分辨率 / 像素格式 | %s×%s / %s |"
        % (
            params.get("宽度", report.result.width or "—"),
            params.get("高度", report.result.height or "—"),
            params.get("像素格式", "—"),
        )
    )
    lines.append(
        "| 采集时长 / 帧数 | %.1f s / %d 帧 |"
        % (float(stats["duration_s"]), int(stats["frames"]))
    )
    lines.append("| 平均帧率 | %.2f fps |" % float(stats["effective_fps"]))
    for key, value in report.environment.items():
        lines.append("| %s | %s |" % (key, value))
    for key, value in report.config.items():
        lines.append("| 配置：%s | %s |" % (key, value))
    lines.append("")

    lines.append("## 判定结果")
    lines.append("")
    lines.append("| 指标 | 实测值 | 类别 | 判定 | 判定标准（合格 / 警告 / 不合格） |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in report.metrics:
        lines.append(
            "| %s | %s | %s | **%s** | %s |"
            % (
                item.name,
                item.display,
                "诊断" if item.diagnostic else "判定",
                item.grade.value,
                item.rule,
            )
        )
    lines.append("")
    lines.append("**接口综合判定：%s**" % report.overall.value)
    lines.append("")
    lines.append("**图像质量诊断：%s**（不计入接口综合判定）" % report.image_quality_grade.value)
    lines.append("")

    if report.extra_checks:
        lines.append("## 补充检查（不参与接口综合判定）")
        lines.append("")
        lines.append("| 检查项 | 实测 | 状态 | 说明 |")
        lines.append("| --- | --- | --- | --- |")
        for check in report.extra_checks:
            status = extra_status_text(check.status)
            if check.status in (Grade.WARN, Grade.FAIL):
                status = "**%s**" % status
            lines.append(
                "| %s | %s | %s | %s |"
                % (check.name, check.display, status, check.detail)
            )
        lines.append("")

    lines.append("## 明细")
    lines.append("")
    for item in report.metrics:
        if item.detail:
            lines.append("- **%s**：%s" % (item.name, item.detail))
    lines.append("")

    advice = []
    for item in report.metrics:
        tip = thresholds.advice(item.key, item.grade)
        if tip:
            advice.append("- **%s（%s）**：%s" % (item.name, item.grade.value, tip))
    if advice:
        lines.append("## 排查建议")
        lines.append("")
        lines.extend(advice)
        lines.append("")
    if report.notes:
        lines.append("## 备注")
        lines.append("")
        for note in report.notes:
            lines.append("- %s" % note)
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "指标定义见 `README.md`；FFT 峰值比与图像质量项默认只作诊断，"
        "接口综合判定只统计计入项。"
    )
    return "\n".join(lines)


def report_to_dict(report: TestReport) -> Dict[str, object]:
    stats = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in report.stats.items()}
    return {
        "generated_at": report.generated_at,
        "simulated": report.result.simulated,
        "overall": report.overall.value,
        "image_quality_grade": report.image_quality_grade.value,
        "environment": report.environment,
        "config": report.config,
        "camera": report.result.params,
        "stats": stats,
        "metrics": [
            {
                "key": item.key,
                "name": item.name,
                "value": item.value,
                "display": item.display,
                "grade": item.grade.value,
                "diagnostic": item.diagnostic,
                "rule": item.rule,
                "detail": item.detail,
            }
            for item in report.metrics
        ],
        "extra_checks": [
            {
                "key": check.key,
                "name": check.name,
                "display": check.display,
                "status": None if check.status is None else check.status.value,
                "detail": check.detail,
            }
            for check in report.extra_checks
        ],
        "notes": report.notes,
    }


def write_bundle(report: TestReport, out_dir: str) -> Dict[str, str]:
    """写出完整报告包，返回各文件路径。"""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "模拟" if report.result.simulated else "实测"
    paths = {
        "markdown": os.path.join(out_dir, "测试报告_%s_%s.md" % (prefix, stamp)),
        "json": os.path.join(out_dir, "测试数据_%s_%s.json" % (prefix, stamp)),
        "csv": os.path.join(out_dir, "逐帧数据_%s_%s.csv" % (prefix, stamp)),
        "params": os.path.join(out_dir, "相机参数快照.md"),
    }
    with open(paths["markdown"], "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(report_to_dict(report), handle, ensure_ascii=False, indent=2)
    _write_frames_csv(report, paths["csv"])
    with open(paths["params"], "w", encoding="utf-8") as handle:
        handle.write(render_params(report))
    return paths


def _write_frames_csv(report: TestReport, path: str) -> None:
    records = report.result.records
    gaps, _, _ = metrics.frame_gaps([r.frame_num for r in records])
    t0 = records[0].t_ns if records else 0
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "time_s",
                "frame_num",
                "frame_gap",
                "lost_packet",
                "dev_ts",
                "width",
                "height",
                "pixel_format",
            ]
        )
        for record, gap in zip(records, gaps):
            writer.writerow(
                [
                    record.index,
                    "%.6f" % ((record.t_ns - t0) / 1e9),
                    "" if record.frame_num is None else record.frame_num,
                    "" if gap is None else gap,
                    "" if record.lost_packet is None else record.lost_packet,
                    "" if record.dev_ts is None else "%.3f" % record.dev_ts,
                    record.width,
                    record.height,
                    "" if record.pixel_type is None else "0x%08X" % record.pixel_type,
                ]
            )


def render_params(report: TestReport) -> str:
    """生成“相机参数快照”，可直接粘贴到参数记录文档。"""
    lines = ["# %s 参数快照" % (report.result.params.get("设备型号") or "相机"), ""]
    lines.append("采集时间：%s" % report.generated_at)
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("| --- | --- |")
    for key, value in report.result.params.items():
        lines.append("| %s | %s |" % (key, value))
    lines.append("")
    lines.append("## 采集配置")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("| --- | --- |")
    for key, value in report.config.items():
        lines.append("| %s | %s |" % (key, value))
    lines.append("")
    lines.append("## 帧统计")
    lines.append("")
    stats = report.stats
    for key in (
        "frames",
        "duration_s",
        "effective_fps",
        "expected",
        "lost",
        "resets",
        "jitter_ratio",
        "dev_jitter_ratio",
        "mean_interval_ms",
        "std_interval_ms",
        "p99_interval_ms",
        "frame_drop_max_30s",
        "packet_loss_max_30s",
        "dead_pixel_rate",
        "fft_ratio",
    ):
        if key in stats:
            lines.append("- %s：%s" % (key, stats[key]))
    lines.append("")
    return "\n".join(lines)
