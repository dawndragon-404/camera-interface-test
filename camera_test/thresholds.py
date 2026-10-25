"""判定阈值与判定逻辑（A 方案）。

接口综合判定只统计“计入”项：

    指标            合格        警告            不合格
    帧号丢帧率      0          < 0.01%         >= 0.01%
    时间戳抖动比     < 0.05     0.05 ~ 0.10     > 0.10
    丢帧增长(30s)    0          1 ~ 5           > 5
    丢包增长(30s)    0          1 ~ 10          > 10
    坏点率          < 0.001     0.001 ~ 0.01    > 0.01   （默认诊断，enforce 时计入）
    FFT 峰值比      < 3        3 ~ 5           > 5      （只诊断，不计入）

区间边界按“合格区间的上界取开区间、警告区间闭区间”处理，
例如抖动比恰好 0.05 判为“警告”，恰好 0.10 仍判“警告”。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Grade(Enum):
    PASS = "合格"
    WARN = "警告"
    FAIL = "不合格"
    INCOMPLETE = "不完整"  # 有指标缺数据，不能给出完整结论
    NA = "不可用"          # 单项指标无法计算

    @property
    def order(self) -> int:
        return {
            Grade.PASS: 0,
            Grade.WARN: 1,
            Grade.INCOMPLETE: 2,
            Grade.FAIL: 3,
            Grade.NA: 4,
        }[self]


@dataclass
class Thresholds:
    """接口计入项、图像诊断项与补充检查阈值。"""

    # 帧号丢帧率：0 合格，< 0.01% 警告，>= 0.01% 不合格
    frame_loss_warn: float = 0.01 / 100.0
    # 时间戳抖动比：< 0.05 合格，0.05~0.10 警告，> 0.10 不合格
    jitter_pass: float = 0.05
    jitter_warn: float = 0.10
    # 丢帧增长(30s)：0 合格，1~5 警告，> 5 不合格
    frame_drop_warn: int = 5
    # 丢包增长(30s)：0 合格，1~10 警告，> 10 不合格
    packet_loss_warn: int = 10
    # 坏点率：< 0.001 合格，0.001~0.01 警告，> 0.01 不合格
    dead_pixel_pass: float = 0.001
    dead_pixel_warn: float = 0.01
    # FFT 峰值比：< 3 合格，3~5 警告，> 5 不合格
    fft_pass: float = 3.0
    fft_warn: float = 5.0

    # ---- 以下为“补充检查”阈值，不参与接口综合判定 ----
    # 实测帧率与目标帧率偏差
    fps_deviation_pass: float = 0.01
    fps_deviation_warn: float = 0.05
    # 最长帧间隔 / 中位帧间隔
    stall_ratio_pass: float = 1.5
    stall_ratio_warn: float = 2.5
    # 实测带宽 / 名义需求带宽
    bandwidth_ratio_pass: float = 0.95
    bandwidth_ratio_warn: float = 0.85
    # 暗场热像素率
    dark_hot_pass: float = 0.0001
    dark_hot_warn: float = 0.001
    # 采集期间进程内存增长（MB）
    rss_growth_warn_mb: float = 200.0
    # 图像输入有效性（平场判定）
    flat_saturation_ratio: float = 0.02   # >=250 DN 像素占比超过 2% 视为过曝
    flat_structure_score: float = 8.0     # p99 局部偏差 / σ 超过 8 视为非平场（场景结构）
    flat_banding_ratio: float = 10.0      # 行/列均值谱峰比（噪声约 3）超过 10 视为全局条纹
    flat_banding_uniformity: float = 20.0  # 条纹还需“行内一致”（剖面方差/行内方差）够高


@dataclass
class MetricResult:
    """单项指标的结果。"""

    key: str
    name: str
    value: Optional[float]
    display: str          # 用于报告显示的字符串
    grade: Grade
    rule: str             # 判定依据（人可读）
    detail: str = ""      # 附加说明 / 定位建议
    diagnostic: bool = False  # True 表示只诊断，不计入综合判定

    @property
    def ok(self) -> bool:
        return self.grade is Grade.PASS


def grade_frame_loss_rate(rate: Optional[float], th: Thresholds) -> Grade:
    if rate is None:
        return Grade.NA
    if rate <= 0:
        return Grade.PASS
    if rate < th.frame_loss_warn:
        return Grade.WARN
    return Grade.FAIL


def grade_jitter(ratio: Optional[float], th: Thresholds) -> Grade:
    if ratio is None:
        return Grade.NA
    if ratio < th.jitter_pass:
        return Grade.PASS
    if ratio <= th.jitter_warn:
        return Grade.WARN
    return Grade.FAIL


def grade_count_growth(count: Optional[int], warn_threshold: int) -> Grade:
    """丢帧/丢包增长：0 合格，1~N 警告，> N 不合格。"""
    if count is None:
        return Grade.NA
    if count <= 0:
        return Grade.PASS
    if count <= warn_threshold:
        return Grade.WARN
    return Grade.FAIL


def grade_dead_pixel_rate(rate: Optional[float], th: Thresholds) -> Grade:
    if rate is None:
        return Grade.NA
    if rate < th.dead_pixel_pass:
        return Grade.PASS
    if rate <= th.dead_pixel_warn:
        return Grade.WARN
    return Grade.FAIL


def grade_fft_ratio(ratio: Optional[float], th: Thresholds) -> Grade:
    if ratio is None:
        return Grade.NA
    if ratio < th.fft_pass:
        return Grade.PASS
    if ratio <= th.fft_warn:
        return Grade.WARN
    return Grade.FAIL


JUDGE_RULES = {
    "frame_loss": "0 / <0.01% / ≥0.01%",
    "jitter": "<0.05 / 0.05~0.10 / >0.10",
    "frame_drop_30s": "0 / 1~5 / >5",
    "packet_loss_30s": "0 / 1~10 / >10",
    "dead_pixel": "<0.001 / 0.001~0.01 / >0.01",
    "fft_ratio": "<3 / 3~5 / >5（诊断）",
}

METRIC_NAMES = {
    "frame_loss": "帧号丢帧率",
    "jitter": "时间戳抖动比",
    "frame_drop_30s": "丢帧增长(30s)",
    "packet_loss_30s": "丢包增长(30s)",
    "dead_pixel": "坏点率",
    "fft_ratio": "FFT 峰值比（诊断）",
}


def worst_grade(grades: list) -> Grade:
    """整体判定取各项中最差的一项（NA 不参与整体判定）。"""
    real = [g for g in grades if g is not Grade.NA]
    if not real:
        return Grade.NA
    return max(real, key=lambda g: g.order)


def overall_grade(grades: list) -> Grade:
    """综合判定：不合格 > 警告 > 数据不完整 > 合格。

    只要有一项因为缺数据（NA）算不出来，就不能声称“合格”，
    改用“不完整”提醒补测，避免“少算了指标却显示合格”。
    """
    real = [g for g in grades if g is not Grade.NA]
    if not real:
        return Grade.NA
    if any(g is Grade.FAIL for g in real):
        return Grade.FAIL
    if any(g is Grade.WARN for g in real):
        return Grade.WARN
    if any(g is Grade.NA for g in grades):
        return Grade.INCOMPLETE
    return Grade.PASS


# 补充检查项名称（不参与接口综合判定）
EXTRA_NAMES = {
    "bandwidth": "链路带宽",
    "integrity": "帧完整性",
    "fps": "帧率偏差",
    "stall": "最长帧间隔",
    "segment": "长时稳定性",
    "resource": "进程资源",
    "dark_hot": "暗场热像素",
}

EXTRA_ADVICE = {
    "bandwidth": (
        "实测带宽明显低于需求：确认插的是 USB3 口（不是 USB2）、换短屏蔽线，"
        "避免与其它 USB3 设备共用控制器；3072×2048 Mono8@59.6 约需 375 MB/s。"
    ),
    "integrity": (
        "出现帧长异常的帧，多半是传输丢包/半帧；先查线缆与供电，"
        "必要时降低分辨率或帧率复测。"
    ),
    "fps": "实测帧率与目标不符：检查 AcquisitionFrameRate 是否被曝光时间/带宽限制（曝光 >= 1/帧率 时必然掉帧）。",
    "stall": (
        "存在明显卡顿帧：关闭 CPU 节能、提高采集线程优先级，"
        "并把图像解码移出采集线程（本工具已用独立解码线程）。"
    ),
    "segment": "分段丢帧随时间恶化，重点查供电与温升；长测（>=180 s）更有说服力。",
    "resource": "采集期间内存增长较多，留意长时间运行的资源泄漏。",
    "dark_hot": "暗场仍有明显亮点：确认镜头已盖严，若确实存在热像素，需坏点校正或返厂。",
}



def advice(metric_key: str, grade: Grade) -> str:
    """针对不合格/警告项给出排查建议。"""
    if grade in (Grade.PASS, Grade.NA):
        return ""
    tips = {
        "frame_loss": (
            "检查 Host 侧缓冲区（已自动设 ImageNodeNum）、USB3 线缆长度/接口供电，"
            "降低分辨率或提高曝光前先确认带宽；NUC 上避免同时占用同一 USB3 控制器。"
        ),
        "jitter": (
            "时间戳抖动偏大通常来自主机调度：关闭 CPU 节能（performance governor）、"
            "提高采集线程优先级、避免在同一线程里做图像处理。"
        ),
        "frame_drop_30s": (
            "30 秒内出现帧丢失增长，优先排查 USB3 链路稳定性（线缆、供电、端口）"
            "与主机侧缓冲是否被占满。"
        ),
        "packet_loss_30s": (
            "出现丢包增长，检查 USB3 线缆、连接器是否插紧，供电是否充足；"
            "GigE 场景检查网卡巨帧/包大小与交换机。"
        ),
        "dead_pixel": (
            "坏点率偏高，确认镜头是否盖住/光照是否均匀；若确为传感器坏点，"
            "需启用坏点校正或返厂。"
        ),
        "fft_ratio": (
            "频谱诊断提示：单峰显著不等于相机不合格，先确认平场/光源无频闪；"
            "若峰值频率随光源变化，多为照明频闪；若频率固定且暗场仍在，查 FPN/读出。"
        ),
    }
    return tips.get(metric_key, "")
