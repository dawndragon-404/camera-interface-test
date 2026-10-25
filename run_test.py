"""MV-CS060-10UC-PRO 相机接口测试 —— 命令行入口。

用法示例::

    python run_test.py                     # 真实相机，默认采集 60 s
    python run_test.py --list              # 只枚举设备
    python run_test.py --duration 30       # 采集 30 s
    python run_test.py --serial 0A12345    # 按序列号选择相机
    python run_test.py --exposure 2000 --fps 30
    python run_test.py --dark-field        # 结束后提示盖镜头，再采暗场热像素
    python run_test.py --save-quality 质量帧   # 同时保存质量帧 PNG 便于目视检查
    python run_test.py --simulate          # 无相机离线自检（模拟数据）
    python run_test.py --simulate --sim-profile packet   # 模拟丢包场景

    接口判定按 A 方案：A/B/C/F 与计入项参与综合判定，FFT/图像质量默认只作诊断；退出码：
0 合格 / 1 警告 / 2 不合格 / 3 环境错误 / 4 数据不完整。
"""

from __future__ import annotations

import argparse
import os
import sys

from camera_test import pixels, report as report_mod, thresholds
from camera_test.analyze import summarize
from camera_test.sdk import (
    CameraError,
    CaptureConfig,
    HikCamera,
    SdkNotFound,
    enumerate_devices,
    load_sdk,
    sdk_version,
)
from camera_test.sources import (
    SIM_PROFILES,
    MvsSource,
    ResourceSampler,
    SimulatedSource,
)
from camera_test.thresholds import Grade

DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "报告")


def log(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="海康机器人 MV-CS060-10UC-PRO 相机接口测试（A 方案：接口判定 + 图像诊断）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="仅枚举相机设备后退出")
    parser.add_argument("--index", type=int, default=0, help="设备序号（默认 0）")
    parser.add_argument("--serial", default=None, help="按相机序列号选择设备")
    parser.add_argument("--duration", type=float, default=60.0, help="采集时长（秒，默认 60）")
    parser.add_argument("--mvs-path", default=None, help="MvImport 目录（含 MvCameraControl_class.py）")
    parser.add_argument("--pixel-format", default=None, help="设置像素格式，如 Mono8 / BayerRG8")
    parser.add_argument("--exposure", type=float, default=None, help="曝光时间（微秒）")
    parser.add_argument("--gain", type=float, default=None, help="增益（dB）")
    parser.add_argument("--fps", type=float, default=None, help="限制采集帧率（fps）")
    parser.add_argument("--quality-frames", type=int, default=8, help="用于坏点/FFT 的图像帧数（默认 8）")
    parser.add_argument("--roi", type=int, default=512, help="FFT 分析区域边长（默认 512）")
    parser.add_argument(
        "--image-quality",
        choices=("diagnostic", "enforce"),
        default="diagnostic",
        help="图像质量项（坏点率/FFT）是否计入总判定：diagnostic=只诊断（默认），enforce=坏点率计入",
    )
    parser.add_argument(
        "--dark-field",
        action="store_true",
        help="采集结束后追加暗场短采（会提示盖住镜头），用于暗场热像素检查",
    )
    parser.add_argument("--dark-frames", type=int, default=4, help="暗场采集帧数（默认 4）")
    parser.add_argument("--no-wait", action="store_true", help="暗场采集前不等待回车（自动化场景）")
    parser.add_argument(
        "--save-quality",
        default=None,
        metavar="DIR",
        help="把质量帧与暗场帧保存为 PNG（便于目视确认条纹/场景/过曝）",
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="报告输出目录")
    parser.add_argument("--no-color", action="store_true", help="控制台不使用颜色")
    parser.add_argument("--simulate", action="store_true", help="使用模拟源离线运行")
    parser.add_argument(
        "--sim-profile",
        default="clean",
        choices=sorted(SIM_PROFILES),
        help="模拟场景（clean/jitter/drop/packet/deadpixel/banding）",
    )
    parser.add_argument("--sim-size", type=int, default=640, help="模拟图像边长（默认 640）")
    parser.add_argument("--sim-realtime", action="store_true", help="模拟时按真实帧间隔等待")
    return parser


def _setup_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def cmd_list(args) -> int:
    sdk = load_sdk(args.mvs_path)
    log("MVS SDK 版本：%s" % sdk_version(sdk))
    devices = enumerate_devices(sdk)
    if not devices:
        log("未发现相机。请检查 USB3 线缆/供电，并确认 MVS 客户端能识别设备。")
        return 3
    for device in devices:
        log(device.label())
    return 0


def _wait_for_dark(args) -> None:
    """提示操作者盖住镜头，再开始暗场采集。"""
    if args.no_wait:
        log("--no-wait：直接开始暗场采集。")
        return
    if not sys.stdin or not sys.stdin.isatty():
        log("未检测到交互终端，直接开始暗场采集（等同 --no-wait）。")
        return
    print("请盖住镜头（或关闭光源），按回车开始暗场采集……", end="", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print()


def cmd_real(args, cfg: CaptureConfig):
    sdk = load_sdk(args.mvs_path)
    version = sdk_version(sdk)
    log("MVS SDK 版本：%s" % version)
    camera = HikCamera(sdk, log=log)
    camera.open(index=args.index, serial=args.serial)
    try:
        changes = camera.configure(cfg)
        if changes:
            log("已设置：" + "，".join(changes))
        log("开始采集，时长 %.1f s ……" % cfg.duration_s)
        source = MvsSource(camera, cfg, log=log)
        result = source.capture()
        if args.dark_field:
            _wait_for_dark(args)
            log("暗场采集中（%d 帧）……" % cfg.dark_frames)
            result.dark_images = source.capture_dark()
            if not result.dark_images:
                log("暗场采集没有拿到图像，跳过暗场热像素检查。")
            else:
                log("暗场采集完成：%d 帧" % len(result.dark_images))
    finally:
        camera.close()
    return result, version


def cmd_simulate(args, cfg: CaptureConfig):
    log("模拟源：%s（时长 %.1f s）" % (args.sim_profile, cfg.duration_s))
    source = SimulatedSource(cfg, profile=args.sim_profile, size=args.sim_size, log=log)
    result = source.capture()
    if args.dark_field:
        result.dark_images = source.capture_dark()
        if result.dark_images:
            log("模拟暗场采集完成：%d 帧" % len(result.dark_images))
    return result, "模拟"


def main(argv=None) -> int:
    _setup_stdout()
    args = build_parser().parse_args(argv)

    if args.list:
        try:
            return cmd_list(args)
        except SdkNotFound as exc:
            log(str(exc))
            return 3

    pixel_code = None
    if args.pixel_format:
        pixel_code = pixels.code_of(args.pixel_format)
        if pixel_code is None:
            log("未知像素格式：%s（可用：%s）" % (args.pixel_format, ", ".join(sorted(pixels.NAME_TO_CODE))))
            return 3

    cfg = CaptureConfig(
        duration_s=args.duration,
        pixel_format=pixel_code,
        exposure_us=args.exposure,
        gain_db=args.gain,
        frame_rate=args.fps,
        quality_frames=args.quality_frames,
        roi=args.roi,
        dark_frames=args.dark_frames,
        image_quality=args.image_quality,
        sim_realtime=args.sim_realtime,
    )
    th = thresholds.Thresholds()

    result = None
    version = "模拟"
    sampler = ResourceSampler()
    try:
        if args.simulate:
            result, version = cmd_simulate(args, cfg)
        else:
            result, version = cmd_real(args, cfg)
    except SdkNotFound as exc:
        log(str(exc))
        return 3
    except CameraError as exc:
        log("相机错误：%s" % exc)
        return 3
    except KeyboardInterrupt:
        log("\n已中断采集，使用已收到的帧出报告。")
        if result is None:
            return 3
    if result is None:
        return 3
    result.resource = sampler.finish()
    if not result.records:
        log("没有采集到任何帧，请检查相机连接与带宽设置。")
        return 3

    log("采集结束：%d 帧" % len(result.records))
    log("正在计算图像质量与补充检查……")
    try:
        report = summarize(result, cfg, th=th, sdk_version=version)
    except Exception as exc:  # noqa: BLE001
        log("指标计算失败：%s" % exc)
        return 3

    print()
    print(report_mod.render_console(report, color=not args.no_color))
    if args.save_quality:
        try:
            saved = report_mod.save_quality_images(result.images, args.save_quality, tag="flat")
            if result.dark_images:
                saved += report_mod.save_quality_images(
                    result.dark_images, args.save_quality, tag="dark"
                )
            log("已保存 %d 张质量帧（含 index.csv）到 %s" % (len(saved), args.save_quality))
        except Exception as exc:  # noqa: BLE001
            log("保存质量帧失败：%s" % exc)
    paths = report_mod.write_bundle(report, args.out)
    print()
    for name, path in paths.items():
        print("已生成 %s：%s" % (name, path))

    if report.overall is Grade.PASS:
        return 0
    if report.overall is Grade.WARN:
        return 1
    if report.overall is Grade.INCOMPLETE:
        return 4
    return 2


if __name__ == "__main__":
    sys.exit(main())
