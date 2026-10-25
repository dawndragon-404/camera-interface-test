"""图像质量指标：坏点率与 FFT 峰值比。

两个指标的定义（报告中同样会打印，便于和实验室口径对齐）：

坏点率
    在“平场”图像（盖镜头 / 均匀面光）上，某像素与自身 3×3 邻域中值之差
    超过 ``max(6 * 局部噪声σ, max(12% * 局部中值, 8 DN))`` 判为坏点；
    局部噪声 σ 用 ``1.4826 * median(|I - median3x3(I)|)`` 估计。
    用相对对比度而不是固定 DN，是为了在换像素格式/位深/增益后口径一致；
    暗场下识别不出坏点（局部中值很低时 12% 很小），因此另用暗场专用判据。
    多帧时只统计“至少一半帧都被标记”的持久坏点。
    坏点率 = 持久坏点数 / 图像像素总数。

暗场热像素
    盖住镜头、尽量短曝光采若干帧，取多帧中值作为暗场参考；参考值高于
    ``中值 + max(6σ, 20 DN)`` 的像素判为热像素（暗电流亮点）。
    热像素率 = 热像素数 / 像素总数。该检查不参与 criterion 判定，
    仅作为补充检查输出。

FFT 峰值比
    取图像中心 ``roi``×``roi`` 区域，去均值、加 Hann 窗后做 2D FFT；
    排除 DC 及其邻域（半径 8 个频点）后，取频谱幅值最大值与中位数之比，
    再除以“纯白噪声下同尺寸频谱最大值的理论期望” ``sqrt(2 ln N) / 1.177``
    （N 为有效频点数），得到一个与图像尺寸无关、均匀噪声场约为 1 的比值。
    判据 3 / 5 来自 评判标准 v2.1-A：均匀噪声平场应 < 3；出现周期性条纹
    （工频频闪、滚动快门带状噪声）时该比值会迅速升高。该指标只作诊断，
    不参与接口综合判定。

另外附带一个“行均值谱峰比”作为条纹诊断参考，不参与判定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class DeadPixelResult:
    rate: float                 # 持久坏点率
    hot: int                    # 常亮（热）像素数
    dead: int                   # 常暗（死）像素数
    total_pixels: int
    frames: int                 # 参与统计的帧数
    per_frame_rates: List[float] = field(default_factory=list)


@dataclass
class FftResult:
    ratio: float                # 归一化峰值比（均匀噪声场 ≈ 1）
    fy: float                   # 归一化频率（cycles/pixel）
    fx: float
    size: int
    raw_ratio: float = 0.0      # 未归一化的 峰值/中位数
    bins: int = 0               # 参与统计的频点数
    axis: str = "scene"         # row（横带/行交替）、col（竖带/列交替）、scene（散点结构）
    axial_ratio: float = 0.0    # 轴向（fx≈0 或 fy≈0）上的最大归一化峰值比
    period_px: Optional[float] = None  # 峰值对应空间周期（像素）
    top_peaks: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class FlatFieldReport:
    """图像输入有效性判定（平场 / 场景 / 过曝 / 交替条纹）。"""

    verdict: str = "flat"       # flat / dark / scene / overexposed / row_banding / col_banding
    ok_dead: bool = True        # 是否可用于坏点率
    ok_fft: bool = True         # 是否可用于 FFT 峰值比
    saturation_ratio: float = 0.0   # >=250 DN 占比
    dark_ratio: float = 0.0         # <=5 DN 占比
    mean_level: float = 0.0
    sigma: float = 0.0
    structure_score: float = 0.0    # 3×3 局部极差 p99 / σ
    row_alt_score: float = 0.0      # |奇偶行均值差| / σ（参考）
    col_alt_score: float = 0.0      # |奇偶列均值差| / σ（参考）
    row_profile_ratio: float = 0.0  # 行均值谱峰比（全局行条纹判据）
    col_profile_ratio: float = 0.0  # 列均值谱峰比（全局列条纹判据）
    row_uniformity: float = 0.0     # 行内一致性（区分条纹与场景块状结构）
    col_uniformity: float = 0.0
    reason: str = ""

    @property
    def valid(self) -> bool:
        return self.ok_dead and self.ok_fft

    @property
    def label(self) -> str:
        return {
            "flat": "平场",
            "dark": "暗场",
            "scene": "场景结构（非平场）",
            "overexposed": "过曝",
            "row_banding": "全局行条纹",
            "col_banding": "全局列条纹",
        }.get(self.verdict, self.verdict)


@dataclass
class QualityReport:
    dead_pixel_rate: Optional[float] = None
    dead_pixel_hot: int = 0
    dead_pixel_dead: int = 0
    dead_pixel_frames: int = 0
    dead_pixel_total: int = 0
    fft_ratio: Optional[float] = None
    fft_raw_ratio: Optional[float] = None
    fft_frequency: str = ""
    fft_axis: str = ""
    fft_axial_ratio: Optional[float] = None
    fft_period_px: Optional[float] = None
    profile_ratio: Optional[float] = None
    # 图像输入有效性
    flat: Optional[FlatFieldReport] = None
    # 暗场热像素（补充检查）
    dark_hot_rate: Optional[float] = None
    dark_hot_count: int = 0
    dark_frames: int = 0
    dark_level: Optional[float] = None
    dark_sigma: Optional[float] = None
    dark_max: Optional[float] = None
    skipped: str = ""


@dataclass
class DarkFieldResult:
    rate: float
    count: int
    total_pixels: int
    frames: int
    level: float          # 暗场中位灰度
    sigma: float          # 暗场噪声 σ
    max_value: float      # 最亮像素值


# ---------------------------------------------------------------------------
# 坏点
# ---------------------------------------------------------------------------
def _local_median_and_center(block):
    """返回 3×3 邻域中值与中心像素（均为 ``(h-2, w-2)`` 网格）。"""
    windows = np.lib.stride_tricks.sliding_window_view(block, (3, 3))
    median = np.median(windows, axis=(-2, -1)).astype(np.float32)
    center = block[1:-1, 1:-1]
    return center, median


def dead_pixel_mask(
    gray,
    k: float = 6.0,
    rel_contrast: float = 0.12,
    abs_floor: float = 8.0,
    tile: int = 256,
):
    """返回与图像同尺寸的坏点布尔掩膜（边缘 1 像素不参与判定）。

    判据：``|I - median3x3| > max(k·σ, max(rel_contrast·median3x3, abs_floor))``。
    相对对比度让判据在换位深/增益后仍可比，绝对下限防止在暗背景下把噪声当坏点。
    """
    if np is None:
        raise RuntimeError("需要 numpy")
    img = np.asarray(gray)
    if img.ndim != 2 or img.shape[0] < 3 or img.shape[1] < 3:
        raise ValueError("图像尺寸过小，无法做坏点分析")
    height, width = img.shape
    mask = np.zeros((height, width), dtype=bool)
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1 = min(y0 + tile, height)
            x1 = min(x0 + tile, width)
            ys, xs = max(y0 - 1, 0), max(x0 - 1, 0)
            ye, xe = min(y1 + 1, height), min(x1 + 1, width)
            block = img[ys:ye, xs:xe].astype(np.float32)
            if block.shape[0] < 3 or block.shape[1] < 3:
                continue
            center, median = _local_median_and_center(block)
            deviation = np.abs(center - median)
            sigma = 1.4826 * float(np.median(deviation))
            threshold = np.maximum(
                k * sigma,
                np.maximum(rel_contrast * median, abs_floor),
            )
            local_mask = deviation > threshold
            # 中心网格在整图中的偏移：块起点 + 1（halo）
            oy = ys + 1
            ox = xs + 1
            mask[oy:oy + local_mask.shape[0], ox:ox + local_mask.shape[1]] = local_mask
    return mask


def _local_median_image(gray, tile: int = 256):
    """整图的 3×3 局部中值（边缘像素用自身值填充）。"""
    img = np.asarray(gray, dtype=np.float32)
    out = img.copy()
    height, width = img.shape
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1 = min(y0 + tile, height)
            x1 = min(x0 + tile, width)
            ys, xs = max(y0 - 1, 0), max(x0 - 1, 0)
            ye, xe = min(y1 + 1, height), min(x1 + 1, width)
            block = img[ys:ye, xs:xe]
            if block.shape[0] < 3 or block.shape[1] < 3:
                continue
            _, median = _local_median_and_center(block)
            oy, ox = ys + 1, xs + 1
            out[oy:oy + median.shape[0], ox:ox + median.shape[1]] = median
    return out


def dead_pixel_analysis(
    images: Sequence["np.ndarray"],
    k: float = 6.0,
    rel_contrast: float = 0.12,
    abs_floor: float = 8.0,
    tile: int = 256,
) -> DeadPixelResult:
    """多帧统计持久坏点。"""
    if np is None:
        raise RuntimeError("需要 numpy")
    if not images:
        raise ValueError("没有可用于坏点分析的图像")
    masks = []
    per_frame = []
    total = 0
    for image in images:
        mask = dead_pixel_mask(
            image, k=k, rel_contrast=rel_contrast, abs_floor=abs_floor, tile=tile
        )
        masks.append(mask)
        total = mask.size
        per_frame.append(float(mask.sum()) / float(mask.size))
    stack = np.stack(masks, axis=0)
    if len(masks) >= 2:
        needed = max(1, (len(masks) + 1) // 2)
        persistent = stack.sum(axis=0) >= needed
    else:
        persistent = masks[0]
    # 用第一帧区分“热像素”（比邻域亮）与“死像素”（比邻域暗）
    first = np.asarray(images[0], dtype=np.float32)
    median_image = _local_median_image(first, tile=tile)
    hot = int(np.count_nonzero(persistent & (first > median_image)))
    dead = int(persistent.sum()) - hot
    dead = max(dead, 0)
    return DeadPixelResult(
        rate=float(persistent.sum()) / float(persistent.size),
        hot=hot,
        dead=dead,
        total_pixels=int(persistent.size),
        frames=len(images),
        per_frame_rates=per_frame,
    )


# ---------------------------------------------------------------------------
# 图像输入有效性（平场判定 / 条纹方向）
# ---------------------------------------------------------------------------
def _axis_noise(reference) -> float:
    """用行/列相邻差分估计噪声 σ，避开条纹方向的影响。"""
    sigma_h = 1.4826 * float(np.median(np.abs(np.diff(reference, axis=1)))) / 1.41421356
    sigma_v = 1.4826 * float(np.median(np.abs(np.diff(reference, axis=0)))) / 1.41421356
    return max(min(sigma_h, sigma_v), 0.5)


def _profile_banding_metrics(profile, within_variance: float):
    """1D 均值剖面的谱峰比 + 行（列）内一致性。

    谱峰比：去趋势后只在“至少 4 个周期”的频率上找峰，约 3 表示噪声，
    ≥10 表示跨整幅的条纹。

    一致性：去趋势后均值剖面的方差 / 均值噪声底（行内方差 / 列数）。全局条纹时
    剖面起伏远高于平均带来的噪声底，一致性很高；场景纹理的行内方差很大，
    即使剖面起伏也不小，一致性仍然偏低。用它避免把“块状场景”误判成条纹。
    """
    values = np.asarray(profile, dtype=np.float32)
    if values.size < 32:
        return 0.0, 0.0
    kernel = 31
    pad = kernel // 2
    padded = np.pad(values, pad, mode="edge")
    cumsum = np.concatenate([[0.0], np.cumsum(padded, dtype=np.float64)])
    trend = (cumsum[kernel:] - cumsum[:-kernel]) / float(kernel)
    residual = values - trend
    residual = residual - residual.mean()
    noise_floor = within_variance / float(values.size)
    uniformity = float(np.var(residual)) / max(noise_floor, 1e-9)
    if float(np.max(np.abs(residual))) < 1e-6:
        return 1.0, uniformity
    spectrum = np.abs(np.fft.rfft(residual * np.hanning(values.size)))
    body = spectrum[1:]
    if body.size <= 4:
        return 0.0, uniformity
    baseline = float(np.median(body))
    if baseline <= 0:
        return 0.0, uniformity
    low_cut = min(max(4, body.size // 64), body.size - 1)
    return float(body[low_cut:].max() / baseline), uniformity


def _structure_score(reference, sigma: float, samples: int = 120000, seed: int = 0) -> float:
    """3×3 局部极差（max−min）的 p99 / σ，用来衡量“场景结构”强弱。

    比“局部中值偏差”更敏感：大片均匀区域之间的台阶边缘经过中值滤波后，
    偏差可能恰好为 0，但极差会非常大。
    """
    height, width = reference.shape
    count = int(min(samples, max(1, (height - 2) * (width - 2))))
    rng = np.random.default_rng(seed)
    ys = rng.integers(1, height - 1, count)
    xs = rng.integers(1, width - 1, count)
    values = np.empty((count, 9), dtype=np.float32)
    index = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            values[:, index] = reference[ys + dy, xs + dx]
            index += 1
    spread = values.max(axis=1) - values.min(axis=1)
    return float(np.percentile(spread, 99)) / sigma


def flat_field_check(
    images: Sequence,
    saturation_ratio: float = 0.02,
    structure_score: float = 8.0,
    banding_ratio: float = 10.0,
    banding_uniformity: float = 20.0,
) -> FlatFieldReport:
    """判断质量帧是否是可用的平场，并区分“场景 / 过曝 / 全局条纹”。

    * 过曝：>=250 DN 像素占比超阈值 → 坏点与 FFT 都不可用；
    * 暗场：<=5 DN 占比高 → 只可用于暗场热像素，坏点率需要平场；
    * 全局行/列条纹：均值谱峰比超阈值且行（列）内一致 → 条纹类缺陷，FFT 可用且应判不合格；
    * 场景结构：3×3 局部极差 p99 > 8σ → 非平场，坏点与 FFT 均不可用。
    """
    if np is None:
        return FlatFieldReport(
            verdict="scene", ok_dead=False, ok_fft=False, reason="未安装 numpy"
        )
    frames = [np.asarray(img, dtype=np.float32) for img in images]
    frames = [img for img in frames if img.ndim == 2 and min(img.shape) >= 8]
    if not frames:
        return FlatFieldReport(
            verdict="scene", ok_dead=False, ok_fft=False, reason="没有可用的灰度帧"
        )
    # 用单帧而不是多帧中值：条纹相位会随时间漂移，多帧中值会把弱条纹平均掉。
    # 条纹判定看的是行（列）均值剖面，已经对 512 列做了平均，单帧噪声足够。
    reference = np.asarray(frames[0], dtype=np.float32)
    level = float(np.median(reference))
    sigma = _axis_noise(reference)
    structure = _structure_score(reference, sigma)
    sat = float(np.mean(reference >= 250.0))
    dark = float(np.mean(reference <= 5.0))
    row_means = reference.mean(axis=1)
    col_means = reference.mean(axis=0)
    row_alt = (
        abs(float(row_means[0::2].mean() - row_means[1::2].mean())) / sigma
        if row_means.size >= 4
        else 0.0
    )
    col_alt = (
        abs(float(col_means[0::2].mean() - col_means[1::2].mean())) / sigma
        if col_means.size >= 4
        else 0.0
    )
    row_profile, row_uniform = _profile_banding_metrics(
        row_means, float(np.mean(np.var(reference, axis=1)))
    )
    col_profile, col_uniform = _profile_banding_metrics(
        col_means, float(np.mean(np.var(reference, axis=0)))
    )
    row_banding = row_profile > banding_ratio and row_uniform > banding_uniformity
    col_banding = col_profile > banding_ratio and col_uniform > banding_uniformity

    report = FlatFieldReport(
        saturation_ratio=sat,
        dark_ratio=dark,
        mean_level=level,
        sigma=sigma,
        structure_score=structure,
        row_alt_score=row_alt,
        col_alt_score=col_alt,
        row_profile_ratio=row_profile,
        col_profile_ratio=col_profile,
        row_uniformity=row_uniform,
        col_uniformity=col_uniform,
    )
    if sat > saturation_ratio:
        report.verdict = "overexposed"
        report.ok_dead = report.ok_fft = False
        report.reason = "过曝：%.1f%% 像素 ≥250 DN（均值 %.0f），坏点与 FFT 结果不可用，请降低曝光" % (
            sat * 100.0,
            level,
        )
    elif dark > 0.8:
        report.verdict = "dark"
        report.ok_dead, report.ok_fft = False, True
        report.reason = "暗场：%.0f%% 像素 ≤5 DN，可用于暗场热像素；坏点率需要平场（均匀面光）" % (
            dark * 100.0
        )
    elif row_banding:
        report.verdict = "row_banding"
        report.ok_dead, report.ok_fft = False, True
        parity = (
            "，奇偶行均值差 %.1fσ" % row_alt
            if row_alt >= 3.0
            else ""
        )
        report.reason = (
            "全局行条纹：行均值谱峰比 %.0f（噪声约 3），行内一致性 %.0f%s，"
            "疑似照明频闪、滚动快门或读出异常；若被摄物本身有横纹，请盖镜头复测"
            % (row_profile, row_uniform, parity)
        )
    elif col_banding:
        report.verdict = "col_banding"
        report.ok_dead, report.ok_fft = False, True
        parity = (
            "，奇偶列均值差 %.1fσ" % col_alt
            if col_alt >= 3.0
            else ""
        )
        report.reason = (
            "全局列条纹：列均值谱峰比 %.0f（噪声约 3），列内一致性 %.0f%s，"
            "疑似列 FPN 或读出异常；若被摄物本身有竖纹，请盖镜头复测"
            % (col_profile, col_uniform, parity)
        )
    elif structure > structure_score:
        report.verdict = "scene"
        report.ok_dead = report.ok_fft = False
        report.reason = (
            "非平场：3×3 局部极差 p99 = %.1fσ（场景纹理/边缘），坏点与 FFT 结果不可用，"
            "请盖镜头或换均匀面光重测" % structure
        )
    else:
        report.verdict = "flat"
        report.reason = "平场可用：均值 %.0f DN，噪声 σ %.2f DN，3×3 局部极差 p99 %.1fσ" % (
            level,
            sigma,
            structure,
        )
    return report


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------
def _box_sum(arr, radius: int):
    """保留工具函数：局部邻域和（需要做局部本底时可复用）。"""
    a = 2 * radius + 1
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)
    return ii[a:, a:] - ii[:-a, a:] - ii[a:, :-a] + ii[:-a, :-a]


def _center_crop(gray, size: int):
    height, width = gray.shape
    side = min(size, height, width)
    if side < 3:
        raise ValueError("图像尺寸过小，无法做频谱分析")
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return gray[y0:y0 + side, x0:x0 + side], side


def noise_max_gain(bins: int) -> float:
    """纯白噪声频谱中“最大值/中位数”的理论期望（Rayleigh 极值）。"""
    if bins <= 1:
        return 1.0
    return math.sqrt(2.0 * math.log(bins)) / 1.1774100225154747


def fft_peak_ratio(
    gray,
    roi: int = 512,
    dc_radius: int = 8,
    top_n: int = 3,
) -> FftResult:
    """计算 FFT 峰值比（定义见模块文档）。"""
    if np is None:
        raise RuntimeError("需要 numpy")
    crop, side = _center_crop(np.asarray(gray), roi)
    img = crop.astype(np.float32)
    img = img - float(img.mean())
    window = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(img * window)))

    center = side // 2
    rows = np.arange(side)[:, None]
    cols = np.arange(side)[None, :]
    valid = (np.abs(rows - center) > dc_radius) | (np.abs(cols - center) > dc_radius)
    body = spectrum[valid]
    bins = int(body.size)
    median = float(np.median(body))
    if median <= 0:
        return FftResult(ratio=float("nan"), fy=0.0, fx=0.0, size=side, raw_ratio=float("nan"))

    # 取得分最高的若干峰（排除 DC 邻域）
    coords = np.argwhere(valid)
    order = np.argsort(body)[::-1][:top_n]
    peaks: List[Tuple[float, float, float]] = []
    gain = noise_max_gain(bins)
    for index in order:
        y, x = coords[index]
        fy = (y - center) / float(side)
        fx = (x - center) / float(side)
        peaks.append((float(body[index] / median) / gain, float(fy), float(fx)))
    best_ratio, best_fy, best_fx = peaks[0]
    # 轴向判别：fx≈0 → 横带（行间变化）；fy≈0 → 竖带（列间变化）。
    # 只有“单轴明显占优”才归为条纹，两轴都强（网格状场景）判为 scene。
    tol = 2
    row_axis = valid & (np.abs(cols - center) <= tol)
    col_axis = valid & (np.abs(rows - center) <= tol)
    axial_row = float(spectrum[row_axis].max() / median / gain) if row_axis.any() else 0.0
    axial_col = float(spectrum[col_axis].max() / median / gain) if col_axis.any() else 0.0
    axial_ratio = max(axial_row, axial_col)
    if axial_row >= 0.8 * best_ratio and axial_row > 1.5 * axial_col:
        axis = "row"
    elif axial_col >= 0.8 * best_ratio and axial_col > 1.5 * axial_row:
        axis = "col"
    else:
        axis = "scene"
    peak_freq = max(abs(best_fy), abs(best_fx))
    period = (1.0 / peak_freq) if peak_freq > 0 else None
    return FftResult(
        ratio=best_ratio,
        fy=best_fy,
        fx=best_fx,
        size=side,
        raw_ratio=float(body.max() / median),
        bins=bins,
        axis=axis,
        axial_ratio=axial_ratio,
        period_px=period,
        top_peaks=peaks,
    )


def profile_peak_ratio(gray) -> float:
    """行均值谱峰比（条纹诊断参考，不参与判定）。"""
    if np is None:
        raise RuntimeError("需要 numpy")
    img = np.asarray(gray, dtype=np.float32)
    if img.ndim != 2 or img.shape[0] < 16:
        return float("nan")
    rows = img.mean(axis=1)
    # 移动平均去趋势（窗口 31），避免明暗渐变抬高低频
    kernel = 31
    pad = kernel // 2
    padded = np.pad(rows, pad, mode="edge")
    cumsum = np.concatenate([[0.0], np.cumsum(padded, dtype=np.float64)])
    trend = (cumsum[kernel:] - cumsum[:-kernel]) / float(kernel)
    residual = rows - trend
    residual = residual - residual.mean()
    if not np.any(np.abs(residual) > 1e-6):
        return 1.0
    window = np.hanning(rows.size)
    spectrum = np.abs(np.fft.rfft(residual * window))
    body = spectrum[1:]
    baseline = float(np.median(body))
    if baseline <= 0:
        return float("nan")
    return float(body.max() / baseline)


def dark_hot_pixel_analysis(
    images: Sequence["np.ndarray"],
    k: float = 6.0,
    abs_floor: float = 20.0,
) -> DarkFieldResult:
    """暗场热像素分析：取多帧中值后统计明显亮点。"""
    if np is None:
        raise RuntimeError("需要 numpy")
    if not images:
        raise ValueError("没有暗场图像")
    stack = np.stack([np.asarray(img, dtype=np.float32) for img in images], axis=0)
    reference = np.median(stack, axis=0)
    level = float(np.median(reference))
    sigma = 1.4826 * float(np.median(np.abs(reference - level)))
    threshold = level + max(k * sigma, abs_floor)
    hot = reference > threshold
    return DarkFieldResult(
        rate=float(hot.sum()) / float(hot.size),
        count=int(hot.sum()),
        total_pixels=int(hot.size),
        frames=len(images),
        level=level,
        sigma=sigma,
        max_value=float(reference.max()),
    )


def analyse_images(
    images: Sequence,
    roi: int = 512,
    dark_images: Optional[Sequence] = None,
    th=None,
) -> QualityReport:
    """汇总坏点、频谱、暗场热像素与输入有效性（平场判定）。"""
    if np is None:
        return QualityReport(skipped="未安装 numpy，跳过图像质量指标")
    if not images:
        return QualityReport(skipped="质量帧数为 0，跳过图像质量指标")
    valid = [np.asarray(img) for img in images]
    clean = []
    for img in valid:
        if img.ndim != 2:
            return QualityReport(skipped="图像不是单通道灰度，跳过图像质量指标")
        clean.append(img)
    shapes = {img.shape for img in clean}
    if len(shapes) > 1:
        return QualityReport(skipped="质量帧尺寸不一致，跳过图像质量指标")

    thresholds = th
    flat = flat_field_check(
        clean,
        saturation_ratio=getattr(thresholds, "flat_saturation_ratio", 0.02),
        structure_score=getattr(thresholds, "flat_structure_score", 8.0),
        banding_ratio=getattr(thresholds, "flat_banding_ratio", 10.0),
        banding_uniformity=getattr(thresholds, "flat_banding_uniformity", 20.0),
    )
    report = QualityReport(flat=flat)

    if flat.ok_dead:
        try:
            dead = dead_pixel_analysis(clean)
            report.dead_pixel_rate = dead.rate
            report.dead_pixel_hot = dead.hot
            report.dead_pixel_dead = dead.dead
            report.dead_pixel_frames = dead.frames
            report.dead_pixel_total = dead.total_pixels
        except Exception as exc:  # noqa: BLE001 - 质量指标失败不应中断测试
            report.skipped = "坏点率计算失败：%s" % exc

    if dark_images:
        try:
            dark = dark_hot_pixel_analysis(dark_images)
            report.dark_hot_rate = dark.rate
            report.dark_hot_count = dark.count
            report.dark_frames = dark.frames
            report.dark_level = dark.level
            report.dark_sigma = dark.sigma
            report.dark_max = dark.max_value
        except Exception as exc:  # noqa: BLE001
            report.skipped = (report.skipped + "；" if report.skipped else "") + (
                "暗场热像素计算失败：%s" % exc
            )

    ratios = []
    peaks = []
    for img in clean:
        try:
            result = fft_peak_ratio(img, roi=roi)
        except Exception:  # noqa: BLE001
            continue
        ratios.append(result.ratio)
        peaks.append(result)
    if ratios:
        report.fft_ratio = float(np.median(ratios))
        worst = max(peaks, key=lambda p: p.ratio)
        report.fft_raw_ratio = worst.raw_ratio
        report.fft_frequency = "%.4f,%.4f cycles/px" % (worst.fy, worst.fx)
        report.fft_axis = worst.axis
        report.fft_axial_ratio = worst.axial_ratio
        report.fft_period_px = worst.period_px
    profiles = []
    for img in clean:
        value = profile_peak_ratio(img)
        if not np.isnan(value):
            profiles.append(value)
    if profiles:
        report.profile_ratio = float(np.median(profiles))
    return report
