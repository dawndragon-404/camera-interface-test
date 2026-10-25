"""单元测试：指标计算、判定边界、像素格式、图像质量与报告输出。

运行方式（在项目根目录）::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from camera_test import image_metrics, metrics, pixels, report as report_mod, thresholds  # noqa: E402
from camera_test.analyze import summarize  # noqa: E402
from camera_test.sdk import CaptureConfig  # noqa: E402
from camera_test.sources import CaptureResult, SimulatedSource  # noqa: E402
from camera_test.thresholds import Grade  # noqa: E402


def make_records(count, fps=60.0, frame_nums=None, lost=None):
    records = []
    for i in range(count):
        records.append(
            metrics.FrameRecord(
                index=i,
                t_ns=int(i * 1e9 / fps),
                frame_num=i if frame_nums is None else frame_nums[i],
                lost_packet=None if lost is None else lost[i],
            )
        )
    return records


class TestFrameGaps(unittest.TestCase):
    def test_continuous(self):
        gaps, resets, dups = metrics.frame_gaps(list(range(10)))
        self.assertEqual(gaps, [0] * 10)
        self.assertEqual((resets, dups), (0, 0))

    def test_single_gap(self):
        gaps, _, _ = metrics.frame_gaps([0, 1, 2, 5, 6])
        self.assertEqual(gaps[3], 2)  # 3、4 两帧丢失
        self.assertEqual(sum(gaps), 2)

    def test_wrap_32bit(self):
        base = 2 ** 32 - 2
        gaps, resets, _ = metrics.frame_gaps([base, base + 1, 0, 1])
        self.assertEqual(gaps, [0, 0, 0, 0])
        self.assertEqual(resets, 0)

    def test_wrap_16bit(self):
        values = [65534, 65535, 0, 1]
        gaps, resets, _ = metrics.frame_gaps(values)
        self.assertEqual(gaps, [0, 0, 0, 0])
        self.assertEqual(resets, 0)

    def test_wrap_with_lost_frame(self):
        gaps, resets, _ = metrics.frame_gaps([2 ** 32 - 3, 2 ** 32 - 2, 0, 1])
        self.assertEqual(gaps[2], 1)  # 2^32-1 这一帧丢失
        self.assertEqual(resets, 0)

    def test_reset_detected(self):
        gaps, resets, _ = metrics.frame_gaps([100, 101, 102, 0, 1])
        self.assertEqual(resets, 1)
        self.assertEqual(gaps[3], 0)

    def test_duplicates(self):
        gaps, _, dups = metrics.frame_gaps([0, 1, 1, 2])
        self.assertEqual(dups, 1)
        self.assertEqual(gaps, [0, 0, 0, 0])

    def test_missing_frame_numbers(self):
        gaps, _, _ = metrics.frame_gaps([0, None, 2])
        self.assertIsNone(gaps[1])


class TestFrameLossRate(unittest.TestCase):
    def test_perfect(self):
        stats = metrics.frame_loss_stats(make_records(600))
        self.assertEqual(stats.loss_rate, 0.0)
        self.assertEqual(stats.expected, 600)
        self.assertEqual(stats.events, [])

    def test_dropped_frames(self):
        records = make_records(100)
        for index in range(50, 100):
            records[index].frame_num += 5  # 50 之前丢 5 帧
        stats = metrics.frame_loss_stats(records)
        self.assertEqual(stats.lost, 5)
        self.assertAlmostEqual(stats.loss_rate, 5 / 105.0, places=10)
        self.assertEqual(len(stats.events), 1)
        self.assertEqual(stats.events[0].count, 5)


class TestIntervalStats(unittest.TestCase):
    def test_constant_interval(self):
        times = [i * 16_666_667 for i in range(100)]
        stats = metrics.interval_stats(times)
        self.assertAlmostEqual(stats.jitter_ratio, 0.0, places=12)
        self.assertAlmostEqual(stats.mean_ms, 16.666667, places=4)

    def test_jitter_ratio(self):
        rng = np.random.default_rng(7)
        intervals = 16.6667e6 * (1 + rng.normal(0, 0.02, 500))
        times = np.concatenate([[0.0], np.cumsum(intervals)]).astype(np.int64)
        stats = metrics.interval_stats(times.tolist())
        self.assertAlmostEqual(stats.jitter_ratio, 0.02, delta=0.004)

    def test_too_few(self):
        self.assertIsNone(metrics.interval_stats([1]))


class TestWindowGrowth(unittest.TestCase):
    def test_no_events(self):
        growth = metrics.window_growth([], [])
        self.assertEqual(growth.max_window, 0)
        self.assertEqual(growth.total, 0)

    def test_events_spread(self):
        times = [1.0, 2.0, 20.0, 40.0, 41.0]
        counts = [1, 1, 1, 1, 1]
        growth = metrics.window_growth(times, counts, window_s=30.0)
        self.assertEqual(growth.total, 5)
        self.assertEqual(growth.max_window, 3)  # 1,2,20 落在 0~30 窗口

    def test_short_duration_uses_span(self):
        growth = metrics.window_growth([1.0, 5.0], [2, 3], window_s=30.0)
        self.assertLess(growth.effective_s, 30.0)
        self.assertEqual(growth.max_window, 5)


class TestPacketLoss(unittest.TestCase):
    def test_cumulative(self):
        series, note = metrics.packet_loss_series([0, 0, 2, 2, 5])
        self.assertEqual(series, [0, 0, 2, 0, 3])
        self.assertIn("累计", note)

    def test_all_zero(self):
        series, _ = metrics.packet_loss_series([0, 0, 0])
        self.assertEqual(series, [0, 0, 0])

    def test_per_frame_values(self):
        series, _ = metrics.packet_loss_series([1, 0, 2, 0])
        self.assertEqual(series, [1, 0, 2, 0])

    def test_missing(self):
        series, note = metrics.packet_loss_series([None, None])
        self.assertIsNone(series)
        self.assertIn("未提供", note)


class TestGrading(unittest.TestCase):
    def setUp(self):
        self.th = thresholds.Thresholds()

    def test_frame_loss_boundaries(self):
        self.assertIs(thresholds.grade_frame_loss_rate(0.0, self.th), Grade.PASS)
        self.assertIs(thresholds.grade_frame_loss_rate(0.00005, self.th), Grade.WARN)  # 0.005%
        self.assertIs(thresholds.grade_frame_loss_rate(0.0001, self.th), Grade.FAIL)   # 0.01%

    def test_jitter_boundaries(self):
        self.assertIs(thresholds.grade_jitter(0.0499, self.th), Grade.PASS)
        self.assertIs(thresholds.grade_jitter(0.05, self.th), Grade.WARN)
        self.assertIs(thresholds.grade_jitter(0.10, self.th), Grade.WARN)
        self.assertIs(thresholds.grade_jitter(0.1001, self.th), Grade.FAIL)

    def test_drop_growth_boundaries(self):
        self.assertIs(thresholds.grade_count_growth(0, 5), Grade.PASS)
        self.assertIs(thresholds.grade_count_growth(1, 5), Grade.WARN)
        self.assertIs(thresholds.grade_count_growth(5, 5), Grade.WARN)
        self.assertIs(thresholds.grade_count_growth(6, 5), Grade.FAIL)

    def test_packet_growth_boundaries(self):
        self.assertIs(thresholds.grade_count_growth(0, 10), Grade.PASS)
        self.assertIs(thresholds.grade_count_growth(10, 10), Grade.WARN)
        self.assertIs(thresholds.grade_count_growth(11, 10), Grade.FAIL)

    def test_dead_pixel_boundaries(self):
        self.assertIs(thresholds.grade_dead_pixel_rate(0.0009, self.th), Grade.PASS)
        self.assertIs(thresholds.grade_dead_pixel_rate(0.001, self.th), Grade.WARN)
        self.assertIs(thresholds.grade_dead_pixel_rate(0.01, self.th), Grade.WARN)
        self.assertIs(thresholds.grade_dead_pixel_rate(0.0101, self.th), Grade.FAIL)

    def test_fft_boundaries(self):
        self.assertIs(thresholds.grade_fft_ratio(2.99, self.th), Grade.PASS)
        self.assertIs(thresholds.grade_fft_ratio(3.0, self.th), Grade.WARN)
        self.assertIs(thresholds.grade_fft_ratio(5.0, self.th), Grade.WARN)
        self.assertIs(thresholds.grade_fft_ratio(5.01, self.th), Grade.FAIL)

    def test_worst_grade(self):
        self.assertIs(
            thresholds.worst_grade([Grade.PASS, Grade.WARN, Grade.FAIL]), Grade.FAIL
        )
        self.assertIs(thresholds.worst_grade([Grade.NA, Grade.PASS]), Grade.PASS)


class TestPixels(unittest.TestCase):
    def test_mono8(self):
        raw = bytes(range(12))
        gray = pixels.to_gray(raw, 4, 3, pixels.MONO8)
        self.assertEqual(gray.shape, (3, 4))
        self.assertEqual(int(gray[1, 2]), 6)

    def test_mono10_to_8bit(self):
        values = np.array([0, 1, 256, 1023], dtype="<u2")
        gray = pixels.to_gray(values.tobytes(), 2, 2, pixels.MONO10)
        self.assertEqual(gray.ravel().tolist(), [0, 0, 64, 255])

    def test_packed10_roundtrip(self):
        values = [0, 1, 512, 1023]
        b0 = values[0] >> 2
        b1 = ((values[0] & 3) << 6) | (values[1] >> 4)
        b2 = ((values[1] & 15) << 4) | (values[2] >> 6)
        b3 = ((values[2] & 63) << 2) | (values[3] >> 8)
        b4 = values[3] & 0xFF
        gray = pixels.to_gray(bytes([b0, b1, b2, b3, b4]), 4, 1, pixels.MONO10_PACKED)
        self.assertEqual(gray.ravel().tolist(), [v >> 2 for v in values])

    def test_packed12_roundtrip(self):
        values = [0, 1, 4095, 2048]
        data = bytearray()
        for i in range(0, 4, 2):
            p0, p1 = values[i], values[i + 1]
            data += bytes([p0 >> 4, ((p0 & 15) << 4) | (p1 >> 8), p1 & 0xFF])
        gray = pixels.to_gray(bytes(data), 4, 1, 0x010C0006)
        self.assertEqual(gray.ravel().tolist(), [v >> 4 for v in values])

    def test_rgb8_luminance(self):
        raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
        gray = pixels.to_gray(raw, 3, 1, pixels.RGB8)
        self.assertEqual(gray.ravel().tolist(), [76, 150, 29])

    def test_unsupported(self):
        with self.assertRaises(pixels.UnsupportedPixelFormat):
            pixels.to_gray(b"\x00" * 16, 2, 2, 0x7F000000)

    def test_short_buffer(self):
        with self.assertRaises(pixels.UnsupportedPixelFormat):
            pixels.to_gray(b"\x00" * 3, 2, 2, pixels.MONO8)


class TestDeadPixel(unittest.TestCase):
    def _flat(self, size=128, seed=3, hot=5, dead=5):
        rng = np.random.default_rng(seed)
        image = 100.0 + rng.normal(0, 1.0, (size, size))
        for _ in range(hot):
            image[int(rng.integers(2, size - 2)), int(rng.integers(2, size - 2))] = 255
        for _ in range(dead):
            image[int(rng.integers(2, size - 2)), int(rng.integers(2, size - 2))] = 0
        return np.clip(image, 0, 255).astype(np.uint8)

    def test_detect_known_defects(self):
        image = self._flat()
        mask = image_metrics.dead_pixel_mask(image)
        self.assertEqual(int(mask.sum()), 10)

    def test_rate(self):
        image = self._flat()
        result = image_metrics.dead_pixel_analysis([image] * 3)
        self.assertAlmostEqual(result.rate, 10 / (128 * 128), places=9)
        self.assertEqual(result.hot, 5)
        self.assertEqual(result.dead, 5)


class TestFft(unittest.TestCase):
    @staticmethod
    def _flat(size=256, sigma=3.0, banding=0.0, seed=11):
        rng = np.random.default_rng(seed)
        image = 128.0 + rng.normal(0, sigma, (size, size))
        if banding:
            rows = np.arange(size, dtype=np.float32)
            image += banding * np.sin(2 * np.pi * rows / 8.0)[:, None]
        return np.clip(image, 0, 255).astype(np.uint8)

    def test_clean_noise_is_near_one(self):
        result = image_metrics.fft_peak_ratio(self._flat(), roi=256)
        self.assertLess(result.ratio, 2.0)

    def test_periodic_stripes_detected(self):
        result = image_metrics.fft_peak_ratio(self._flat(banding=9.0), roi=256)
        self.assertGreater(result.ratio, 10.0)
        # 条纹周期 8 px => 频率 1/8 cycles/px
        self.assertAlmostEqual(abs(result.fy), 0.125, delta=0.01)

    def test_size_invariance(self):
        small = image_metrics.fft_peak_ratio(self._flat(size=192), roi=192)
        large = image_metrics.fft_peak_ratio(self._flat(size=384), roi=384)
        self.assertLess(small.ratio, 2.5)
        self.assertLess(large.ratio, 2.5)


class TestSummarize(unittest.TestCase):
    def test_gaps_graded(self):
        records = make_records(300)
        for index in range(100, 300):
            records[index].frame_num += 3
        result = CaptureResult(records=records)
        cfg = CaptureConfig(duration_s=5.0, quality_frames=0)
        report = summarize(result, cfg)
        item = {m.key: m for m in report.metrics}["frame_loss"]
        self.assertIs(item.grade, Grade.FAIL)
        self.assertEqual(report.overall, Grade.FAIL)
        self.assertIs({m.key: m for m in report.metrics}["dead_pixel"].grade, Grade.NA)

    def test_clean_simulated_all_pass(self):
        cfg = CaptureConfig(duration_s=6.0, quality_frames=4, sim_realtime=False)
        result = SimulatedSource(cfg, profile="clean", size=128).capture()
        report = summarize(result, cfg, sdk_version="模拟")
        self.assertIs(report.overall, Grade.PASS)

    def test_banding_simulated_fails(self):
        cfg = CaptureConfig(duration_s=4.0, quality_frames=3)
        result = SimulatedSource(cfg, profile="banding", size=256).capture()
        report = summarize(result, cfg)
        items = {m.key: m for m in report.metrics}
        self.assertIs(items["fft_ratio"].grade, Grade.FAIL)
        self.assertTrue(items["fft_ratio"].diagnostic)
        self.assertIs(report.image_quality_grade, Grade.FAIL)


class TestReport(unittest.TestCase):
    def _report(self):
        cfg = CaptureConfig(duration_s=6.0, quality_frames=3)
        result = SimulatedSource(cfg, profile="clean", size=128).capture()
        return summarize(result, cfg), cfg

    def test_console_contains_metrics(self):
        report, _ = self._report()
        text = report_mod.render_console(report, color=False)
        for name in thresholds.METRIC_NAMES.values():
            self.assertIn(name, text)
        self.assertIn("综合判定", text)

    def test_bundle_files(self):
        report, _ = self._report()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base_dir, "_test_output")
        paths = report_mod.write_bundle(report, out_dir)
        try:
            for path in paths.values():
                self.assertTrue(os.path.exists(path), path)
            with open(paths["json"], encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(len(data["metrics"]), 6)
            with open(paths["markdown"], encoding="utf-8") as handle:
                self.assertIn("综合判定", handle.read())
        finally:
            for path in paths.values():
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(out_dir)
            except OSError:
                pass

    def test_markdown_table(self):
        report, _ = self._report()
        text = report_mod.render_markdown(report)
        self.assertIn("| 指标 | 实测值 | 类别 | 判定 |", text)


if __name__ == "__main__":
    unittest.main()
