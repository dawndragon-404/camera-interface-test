"""补充检查与口径收紧的单元测试（无需相机、无需真实图像）。

运行方式（在项目根目录）::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from camera_test import extra, image_metrics, metrics, thresholds  # noqa: E402
from camera_test.analyze import summarize  # noqa: E402
from camera_test.sdk import CaptureConfig  # noqa: E402
from camera_test.sources import CaptureResult, SimulatedSource  # noqa: E402
from camera_test.thresholds import Grade, Thresholds  # noqa: E402


def make_records(count, fps=60.0, n_bytes=1024, gap_at=None, gap=3):
    """构造帧记录；gap_at 位置插入一次帧号跳变（丢 gap 帧）。"""
    records = []
    frame_num = 0
    for index in range(count):
        if gap_at is not None and index == gap_at:
            frame_num += gap
        records.append(
            metrics.FrameRecord(
                index=index,
                t_ns=int(index * 1e9 / fps),
                frame_num=frame_num,
                n_bytes=n_bytes,
                width=32,
                height=32,
                pixel_type=0x01080001,
            )
        )
        frame_num += 1
    return records


class TestOverallGrade(unittest.TestCase):
    def test_incomplete_when_na(self):
        self.assertIs(
            thresholds.overall_grade([Grade.PASS, Grade.NA, Grade.PASS]),
            Grade.INCOMPLETE,
        )

    def test_fail_beats_incomplete(self):
        self.assertIs(
            thresholds.overall_grade([Grade.NA, Grade.FAIL]), Grade.FAIL
        )

    def test_warn_beats_incomplete(self):
        self.assertIs(
            thresholds.overall_grade([Grade.NA, Grade.WARN]), Grade.WARN
        )

    def test_all_pass(self):
        self.assertIs(thresholds.overall_grade([Grade.PASS, Grade.PASS]), Grade.PASS)

    def test_all_na(self):
        self.assertIs(thresholds.overall_grade([Grade.NA, Grade.NA]), Grade.NA)


class TestDeadPixelThreshold(unittest.TestCase):
    def test_relative_threshold_flags_bright_defect(self):
        level = 100.0
        image = np.full((64, 64), level, dtype=np.float32)
        image[30, 30] = 140.0  # 相对偏差 40% > 12%
        mask = image_metrics.dead_pixel_mask(image.astype(np.uint8))
        self.assertEqual(int(mask.sum()), 1)

    def test_absolute_floor_ignores_small_deviation_in_dark(self):
        level = 10.0
        image = np.full((64, 64), level, dtype=np.float32)
        image[30, 30] = 16.0  # 偏差 6 DN < 绝对下限 8 DN
        mask = image_metrics.dead_pixel_mask(image.astype(np.uint8))
        self.assertEqual(int(mask.sum()), 0)


class TestDarkHotPixels(unittest.TestCase):
    def test_detect_fixed_hot_pixels(self):
        rng = np.random.default_rng(5)
        frames = []
        for _ in range(4):
            image = rng.normal(3.0, 1.5, (64, 64)).astype(np.float32)
            image[10, 10] = 220.0
            image[40, 20] = 200.0
            frames.append(np.clip(image, 0, 255).astype(np.uint8))
        result = image_metrics.dark_hot_pixel_analysis(frames)
        self.assertEqual(result.count, 2)
        self.assertAlmostEqual(result.rate, 2 / (64 * 64), places=9)
        self.assertGreater(result.max_value, 200)

    def test_clean_dark_field(self):
        rng = np.random.default_rng(6)
        frames = [rng.normal(3.0, 1.0, (64, 64)).astype(np.uint8) for _ in range(4)]
        result = image_metrics.dark_hot_pixel_analysis(frames)
        self.assertLess(result.rate, 0.001)


class TestBandwidthCheck(unittest.TestCase):
    def setUp(self):
        self.th = Thresholds()
        self.cfg = CaptureConfig(duration_s=10.0)

    def test_full_rate_passes(self):
        records = make_records(600, n_bytes=32 * 32)
        check = extra.bandwidth_check(
            records, 9.98, self.cfg, {"采集帧率": "60 fps"}, 60.0, self.th
        )
        self.assertIs(check.status, Grade.PASS)

    def test_half_bandwidth_fails(self):
        records = make_records(600, n_bytes=32 * 32 // 2)
        check = extra.bandwidth_check(
            records, 9.98, self.cfg, {"采集帧率": "60 fps"}, 60.0, self.th
        )
        self.assertIs(check.status, Grade.FAIL)


class TestIntegrityCheck(unittest.TestCase):
    def setUp(self):
        self.th = Thresholds()

    def test_ok(self):
        records = make_records(10, n_bytes=32 * 32)
        self.assertIs(extra.integrity_check(records, self.th).status, Grade.PASS)

    def test_truncated_frame_fails(self):
        records = make_records(10, n_bytes=32 * 32)
        records[5].n_bytes = 100
        self.assertIs(extra.integrity_check(records, self.th).status, Grade.FAIL)

    def test_zero_length_frame_fails(self):
        records = make_records(10, n_bytes=32 * 32)
        records[3].n_bytes = 0
        self.assertIs(extra.integrity_check(records, self.th).status, Grade.FAIL)


class TestFpsCheck(unittest.TestCase):
    def setUp(self):
        self.th = Thresholds()

    def test_exact_fps_passes(self):
        records = make_records(600)
        check = extra.fps_check(records, 9.98, CaptureConfig(duration_s=10.0), {}, self.th)
        self.assertIs(check.status, Grade.PASS)

    def test_target_mismatch_fails(self):
        records = make_records(600)
        cfg = CaptureConfig(duration_s=10.0, frame_rate=30.0)
        check = extra.fps_check(records, 9.98, cfg, {}, self.th)
        self.assertIs(check.status, Grade.FAIL)


class TestStallCheck(unittest.TestCase):
    def setUp(self):
        self.th = Thresholds()

    def test_smooth_stream_passes(self):
        records = make_records(200)
        self.assertIs(extra.stall_check(records, self.th).status, Grade.PASS)

    def test_single_big_stall_detected(self):
        records = make_records(200)
        for record in records[120:]:
            record.t_ns += int(0.2 * 1e9)  # 200 ms 卡顿
        check = extra.stall_check(records, self.th)
        self.assertIs(check.status, Grade.FAIL)
        self.assertIn("卡顿", check.detail)


class TestSegmentCheck(unittest.TestCase):
    def setUp(self):
        self.th = Thresholds()

    def test_too_few_frames_is_na(self):
        records = make_records(60)
        self.assertIs(extra.segment_check(records, self.th).status, Grade.NA)

    def test_single_segment_is_reference(self):
        records = make_records(300)  # 5 s，只够一段
        self.assertIsNone(extra.segment_check(records, self.th).status)

    def test_loss_in_second_segment(self):
        records = make_records(7200, fps=60.0)
        records[4000].frame_num += 5  # 后半段丢 5 帧
        check = extra.segment_check(records, self.th)
        self.assertIn(check.status, (Grade.WARN, Grade.FAIL))
        self.assertIn("分段", check.detail)


class TestResourceCheck(unittest.TestCase):
    def test_reference_when_missing(self):
        self.assertIsNone(extra.resource_check({}, Thresholds()).status)

    def test_growth_judged(self):
        data = {"rss_start_mb": 100.0, "rss_end_mb": 120.0, "cpu_percent": 55.0}
        self.assertIs(extra.resource_check(data, Thresholds()).status, Grade.PASS)


class TestSummarizeWithExtras(unittest.TestCase):
    def test_clean_simulation_has_extra_checks(self):
        cfg = CaptureConfig(duration_s=6.0, quality_frames=3)
        result = SimulatedSource(cfg, profile="clean", size=128).capture()
        report = summarize(result, cfg)
        self.assertEqual(len(report.extra_checks), 7)
        by_key = {check.key: check for check in report.extra_checks}
        self.assertIs(by_key["integrity"].status, Grade.PASS)
        self.assertIs(by_key["bandwidth"].status, Grade.PASS)
        self.assertIs(by_key["fps"].status, Grade.PASS)
        self.assertIs(by_key["stall"].status, Grade.PASS)
        self.assertIs(report.overall, Grade.PASS)

    def test_missing_images_marks_incomplete(self):
        records = make_records(300)
        result = CaptureResult(records=records)
        report = summarize(result, CaptureConfig(duration_s=5.0, quality_frames=0))
        self.assertIs(report.overall, Grade.INCOMPLETE)
        self.assertIs(report.image_quality_grade, Grade.NA)

    def test_dark_field_checked(self):
        cfg = CaptureConfig(duration_s=5.0, quality_frames=2)
        source = SimulatedSource(cfg, profile="clean", size=128)
        result = source.capture()
        result.dark_images = source.capture_dark(3)
        report = summarize(result, cfg)
        by_key = {check.key: check for check in report.extra_checks}
        self.assertIn(by_key["dark_hot"].status, (Grade.PASS, Grade.WARN, Grade.FAIL))
        self.assertIsNotNone(report.stats["dark_hot_rate"])
        self.assertGreater(report.stats["dark_hot_rate"], 0)

    def test_extra_fail_does_not_change_six_item_verdict(self):
        # 帧率偏差只进补充检查，不影响接口计入项
        cfg = CaptureConfig(duration_s=5.0, quality_frames=2, frame_rate=30.0)
        result = SimulatedSource(cfg, profile="clean", size=128).capture()
        report = summarize(result, cfg)
        self.assertIs(report.overall, Grade.PASS)
        by_key = {check.key: check for check in report.extra_checks}
        self.assertIs(by_key["fps"].status, Grade.FAIL)


if __name__ == "__main__":
    unittest.main()
