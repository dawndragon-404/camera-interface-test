"""平场有效性、条纹方向与质量帧落盘的单元测试（无需相机）。

运行方式（在项目根目录）::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from camera_test import image_metrics, metrics, report as report_mod  # noqa: E402
from camera_test.analyze import summarize  # noqa: E402
from camera_test.sdk import CaptureConfig  # noqa: E402
from camera_test.sources import CaptureResult  # noqa: E402
from camera_test.thresholds import Grade  # noqa: E402


def make_image(size: int, level: float = 128.0, noise: float = 2.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    image = level + rng.normal(0.0, noise, (size, size))
    return np.clip(image, 0, 255).astype(np.uint8)


def striped_image(size: int, amplitude: float = 40.0, period: float = 8.0, seed: int = 8):
    rng = np.random.default_rng(seed)
    rows = np.arange(size, dtype=np.float32)
    base = 128.0 + amplitude * np.sin(2.0 * np.pi * rows / period)
    image = np.tile(base[:, None], (1, size)) + rng.normal(0.0, 2.0, (size, size))
    return np.clip(image, 0, 255).astype(np.uint8)


def parity_image(size: int, amplitude: float = 40.0, seed: int = 9):
    """奇偶行交替（对应 fy = 0.5 的奈奎斯特条纹）。"""
    rng = np.random.default_rng(seed)
    rows = np.arange(size, dtype=np.float32)
    base = 128.0 + amplitude * ((rows % 2.0) * 2.0 - 1.0)
    image = np.tile(base[:, None], (1, size)) + rng.normal(0.0, 2.0, (size, size))
    return np.clip(image, 0, 255).astype(np.uint8)


def scene_image(size: int, seed: int = 10):
    """模拟“真实场景”：随机色块 + 边缘。"""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(20, 235, (size // 16, size // 16)).astype(np.float32)
    image = np.kron(blocks, np.ones((16, 16), dtype=np.float32))
    image = image + rng.normal(0.0, 2.0, (size, size))
    return np.clip(image, 0, 255).astype(np.uint8)


def records_with_images(images, fps=60.0, count=None):
    size = images[0].shape[0]
    count = count or max(240, len(images) * 30)
    records = []
    for index in range(count):
        records.append(
            metrics.FrameRecord(
                index=index,
                t_ns=int(index * 1e9 / fps),
                frame_num=index,
                lost_packet=0,
                n_bytes=size * size,
                width=size,
                height=size,
                pixel_type=0x01080001,
            )
        )
    return CaptureResult(records=records, images=list(images))


class TestFlatField(unittest.TestCase):
    def test_flat_noise_is_valid(self):
        report = image_metrics.flat_field_check([make_image(128)] * 3)
        self.assertEqual(report.verdict, "flat")
        self.assertTrue(report.ok_dead)
        self.assertTrue(report.ok_fft)

    def test_overexposed_is_invalid(self):
        image = np.full((128, 128), 255, dtype=np.uint8)
        report = image_metrics.flat_field_check([image])
        self.assertEqual(report.verdict, "overexposed")
        self.assertFalse(report.ok_dead)
        self.assertFalse(report.ok_fft)

    def test_dark_field(self):
        rng = np.random.default_rng(11)
        images = [np.clip(rng.normal(3, 1, (128, 128)), 0, 255).astype(np.uint8) for _ in range(3)]
        report = image_metrics.flat_field_check(images)
        self.assertEqual(report.verdict, "dark")
        self.assertFalse(report.ok_dead)
        self.assertTrue(report.ok_fft)

    def test_periodic_stripes_detected(self):
        report = image_metrics.flat_field_check([striped_image(256)] * 3)
        self.assertEqual(report.verdict, "row_banding")
        self.assertGreater(report.row_profile_ratio, 10.0)
        self.assertFalse(report.ok_dead)
        self.assertTrue(report.ok_fft)

    def test_parity_alternation_detected(self):
        report = image_metrics.flat_field_check([parity_image(256)] * 3)
        self.assertEqual(report.verdict, "row_banding")

    def test_scene_is_invalid(self):
        report = image_metrics.flat_field_check([scene_image(256)] * 3)
        self.assertEqual(report.verdict, "scene")
        self.assertFalse(report.ok_dead)
        self.assertFalse(report.ok_fft)


class TestFftAxis(unittest.TestCase):
    def test_row_axis_parity(self):
        result = image_metrics.fft_peak_ratio(parity_image(256), roi=256)
        self.assertEqual(result.axis, "row")
        self.assertAlmostEqual(abs(result.fy), 0.5, places=3)
        self.assertAlmostEqual(result.period_px, 2.0, places=3)

    def test_row_axis_periodic(self):
        result = image_metrics.fft_peak_ratio(striped_image(256, period=8.0), roi=256)
        self.assertEqual(result.axis, "row")
        self.assertAlmostEqual(result.period_px, 8.0, delta=0.1)

    def test_col_axis(self):
        image = striped_image(256, period=8.0).T.copy()
        result = image_metrics.fft_peak_ratio(image, roi=256)
        self.assertEqual(result.axis, "col")

    def test_scene_axis(self):
        result = image_metrics.fft_peak_ratio(scene_image(256), roi=256)
        self.assertEqual(result.axis, "scene")


class TestSummarizeInputValidity(unittest.TestCase):
    def test_scene_makes_image_metrics_unavailable(self):
        result = records_with_images([scene_image(256)] * 4)
        report = summarize(result, CaptureConfig(duration_s=4.0, quality_frames=4))
        grades = {item.key: item.grade for item in report.metrics}
        self.assertIs(grades["dead_pixel"], Grade.NA)
        self.assertIs(grades["fft_ratio"], Grade.NA)
        self.assertIs(report.overall, Grade.PASS)
        self.assertIs(report.image_quality_grade, Grade.NA)
        self.assertIn("非平场", {item.key: item.detail for item in report.metrics}["fft_ratio"])

    def test_banding_is_judged_and_flagged(self):
        result = records_with_images([striped_image(256)] * 4)
        report = summarize(result, CaptureConfig(duration_s=4.0, quality_frames=4))
        grades = {item.key: item.grade for item in report.metrics}
        self.assertIs(grades["fft_ratio"], Grade.FAIL)
        self.assertIs(grades["dead_pixel"], Grade.NA)
        self.assertIs(report.overall, Grade.PASS)
        self.assertIs(report.image_quality_grade, Grade.FAIL)
        detail = {item.key: item.detail for item in report.metrics}["fft_ratio"]
        self.assertIn("行方向条纹", detail)

    def test_clean_flat_keeps_pass(self):
        result = records_with_images([make_image(256)] * 4)
        report = summarize(result, CaptureConfig(duration_s=4.0, quality_frames=4))
        self.assertIs(report.overall, Grade.PASS)
        self.assertEqual(report.stats["flat_verdict"], "flat")


class TestSaveQualityImages(unittest.TestCase):
    def test_png_and_index(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base_dir, "_test_quality")
        images = [striped_image(64), parity_image(64)]
        try:
            saved = report_mod.save_quality_images(images, out_dir, tag="flat")
            self.assertEqual(len(saved), 2)
            for path in saved:
                self.assertTrue(os.path.exists(path))
                with open(path, "rb") as handle:
                    self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")
            self.assertTrue(os.path.exists(os.path.join(out_dir, "index.csv")))
            try:
                from PIL import Image

                with Image.open(saved[0]) as opened:
                    self.assertEqual(opened.size, (64, 64))
                    self.assertEqual(opened.mode, "L")
            except ImportError:
                pass
        finally:
            for name in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
                os.remove(os.path.join(out_dir, name))
            if os.path.isdir(out_dir):
                os.rmdir(out_dir)


if __name__ == "__main__":
    unittest.main()
