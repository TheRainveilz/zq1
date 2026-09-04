# -*- coding: utf-8 -*-
"""
金属推力保持架垫片 —— 冲压翻边正/反面检测 (OpenCV-Python 后台算法)

实现思路
--------
1) 取图层抽象：LocalFolderSource(本地遍历调试) / HttpCameraSource(WTX-3000 HTTP)，
   由 SOURCE_MODE 一键切换，业务逻辑完全不变。
2) 工件定位(不用固定 ROI)：HoughCircles 粗找所有圆孔 -> 用孔心做最小二乘圆拟合
   (带迭代剔野点)得到"节圆"= 工件中心 + 节圆半径。工件任意旋转/偏移都自适应；
   即使外圆被相机视场切掉也能定位(现场样图正是这种半幅视野)。
   外圆/内孔 Hough 与工件掩膜质心作为二级/三级兜底。
3) 孔心精定位：以粗圆心为起点，沿 360° 射线找"孔内-台面"灰度 50% 跨越点，
   对跨越点做圆拟合(迭代剔野点)，得到亚像素级孔心与真实孔半径 r。
   同一工件所有孔径一致，故取全部孔 r 的中位数作为统一基准，抗单孔失配。
4) 特征A(翻边外圈)：孔 ROI 内多阈值(百分位序列)二值化 -> 轮廓计数，
   只保留"与孔同心"的轮廓(平均半径在环带内 + 径向标准差小 + 角度覆盖率够),
   再按平均半径聚类 -> 环数。正面=孔口环+翻边外环>=2，反面只有单圈冲裁轮廓=1。
   并行用"同心 Hough"在 [1.12,1.36]r 找翻边圆作为 OR 兜底(现场光照不均时更稳)。
5) 特征B(拐角小圆压痕)：4 个拐角相对"工件中心->孔心"径向方向固定角度分布
   (实测 -131°/-59.5°/+60°/+131°，随工件旋转自动跟随)，在每个拐角开微小 ROI，
   先用 HoughCircles 做形状级筛选(毛刺/铁屑/划痕非圆 -> 无响应),
   再对命中圆做多阈值分割并计算 circularity = 4*pi*area/perimeter**2，>0.75 记为有效压痕。
6) 判定：单孔 = 特征A AND 特征B；工件 = 孔1 OR 孔2 -> OK，全部无效 -> NG。
7) 异常保护：找不到工件/圆孔不足 -> 直接 NG。NG 图自动落盘，可选叠加调试图。
8) 预留 Modbus-TCP 对接 PLC 的注释占位(见文件末 ModbusReporter)。

用法
----
    python thrust_cage_flange_inspect.py                 # 按头部常量运行
    python thrust_cage_flange_inspect.py --dir  D:\\samples
    python thrust_cage_flange_inspect.py --mode http --ip 192.168.1.100
    python thrust_cage_flange_inspect.py --debug         # 输出叠加调试图
    python thrust_cage_flange_inspect.py --calib         # 拐角角度/尺寸标定(换型用)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# =====================================================================================
# ============================  参 数 区 (现场只改这里)  ================================
# =====================================================================================

# ---------- 1. 取图 / 运行模式 ----------
SOURCE_MODE = "local"                 # "local"=本地文件夹遍历调试; "http"=WTX-3000 网络取图
LOCAL_IMAGE_DIR = r"D:\新建文件夹\WTX3000-360C (DA7486717)"   # 样本目录(正/反面混放)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LOCAL_RECURSIVE = True                # 递归遍历子目录

CAMERA_IP = "192.168.1.100"           # WTX-3000 相机 IP
CAMERA_URL_TEMPLATE = "http://{camera_ip}/camera/currentImage"
HTTP_TIMEOUT_S = 2.0                  # 单帧取图超时(s)
HTTP_INTERVAL_S = 0.20                # 连续取图间隔(s)
HTTP_MAX_FRAMES = 0                   # 上线取 0 = 无限循环; 调试可设有限帧数
HTTP_RETRY = 3                        # 取图失败重试次数

# ---------- 2. 预处理 ----------
RESIZE_MAX_SIDE = 0                   # >0 按最长边缩放提速; 0=原图。所有阈值均为比例量,缩放不影响判定
MEDIAN_BLUR_K = 3                     # 中值滤波核(奇数,0=关)。压制铁屑/椒盐噪点
GAUSS_BLUR_K = 5                      # 高斯核(奇数,0=关)
CLAHE_CLIP = 2.0                      # 限制对比度自适应直方图均衡, 抗油污/光照不均
CLAHE_GRID = (8, 8)

# ---------- 3. 圆孔粗定位 (HoughCircles) ----------
HOLE_R_MIN_RATIO = 0.030              # 圆孔半径下限 / 图像宽度  (实测样图 ≈0.042)
HOLE_R_MAX_RATIO = 0.065              # 圆孔半径上限 / 图像宽度
HOLE_MIN_DIST_RATIO = 0.060           # 相邻孔心最小间距 / 图像宽度
HOLE_HOUGH_DP = 1.0
HOLE_HOUGH_P1 = 120                   # Canny 高阈值
HOLE_HOUGH_P2 = 55                    # 累加器阈值: 调小=多找孔(易误检), 调大=少找孔(易漏)
HOLE_HOUGH_P2_FALLBACK = 32           # 第一遍找到的孔不足时自动降阈值重找一次(0=关闭)
MIN_HOLE_COUNT = 2                    # 有效圆孔少于该数 -> 直接 NG (异常保护)
MAX_HOLE_CANDIDATES = 40              # Hough 候选上限, 防异常图卡死

# ---------- 4. 工件定位: 节圆(孔心圆)拟合 ----------
PITCH_FIT_MIN_HOLES = 3               # 少于该数无法拟合节圆 -> 走兜底定位
PITCH_FIT_ITERS = 6
PITCH_FIT_TOL_RATIO = 0.06            # 内点容差 / 节圆半径
PITCH_FIT_TOL_MIN_PX = 8.0            # 内点容差下限(px)
OUTER_R_MIN_RATIO = 0.18              # 兜底: 外圆/中心大孔 Hough 半径范围 / 图像宽度
OUTER_R_MAX_RATIO = 0.60
OUTER_HOUGH_P2 = 60

# ---------- 5. 孔心精定位 (径向 50% 灰度跨越 + 圆拟合) ----------
REFINE_ANGLE_STEP_DEG = 2.0           # 射线角度步长(度)
REFINE_RADIUS_STEP_PX = 0.5           # 射线径向步长(px)
REFINE_SCAN_BAND = (0.30, 1.70)       # 射线扫描范围 / 粗半径
REFINE_INNER_BAND = 0.55              # 孔内灰度取样: r < 该值*粗半径
REFINE_LAND_BAND = 1.45               # 台面灰度取样: r > 该值*粗半径
REFINE_MIN_CONTRAST = 12              # 孔内/台面灰度差 < 该值 视为不是孔 -> 丢弃
REFINE_EDGE_START = 0.45              # 跨越点搜索起始 / 粗半径
REFINE_MIN_EDGE_PTS = 40              # 有效跨越点下限
REFINE_ITERS = 4
REFINE_INLIER_RATIO = 0.10            # 圆拟合内点容差 / 拟合半径
USE_GLOBAL_HOLE_RADIUS = True         # True=用所有孔半径中位数做统一基准 r (同一工件孔径一致)
HOLE_R_DEV_MAX = 0.25                 # 单孔半径偏离中位数超过该比例 -> 该孔判为无效

# ---------- 6. 特征A: 翻边外圈 (孔 ROI 内轮廓计数) ----------
# 实测结论(见文末调试说明): "轮廓计数"在反面也可能计到 2 圈(单圈冲裁边的内外沿),
# 鉴别力弱于"同心 Hough 找翻边圆"。因此默认 contour_or_hough, 且 HOLE_LOGIC 必须保持 AND,
# 由特征B(拐角压痕)提供主要鉴别力。若现场出现反面误判 OK, 先改 contour_and_hough。
FEATURE_A_MODE = "contour_or_hough"   # "contour" / "hough" / "contour_or_hough" / "contour_and_hough"
RING_ROI_RATIO = 1.48                 # 孔 ROI 外扩倍数 (含翻边区域)
RING_MASK_RATIO = 1.42                # ROI 内圆形掩膜半径 / r, 屏蔽相邻孔与拐角压痕干扰
RING_BAND = (0.86, 1.36)              # 只接受平均半径落在该环带内的轮廓 / r
RING_THRESH_PCTS = (10, 20, 30, 40, 50, 60, 70, 80, 90)  # 多阈值(灰度百分位)扫描, 抗光照不均
RING_MIN_CONTOUR_PTS = 24             # 轮廓点数下限, 滤掉毛刺碎轮廓
RING_MAX_RADIAL_STD = 0.16            # 径向标准差/平均半径 上限 -> 只留"同心圆"形状
RING_MIN_ANGLE_COVER = 0.20           # 轮廓角度覆盖率下限(0~1), 滤掉短弧碎片
RING_CLUSTER_GAP = 0.08               # 环半径聚类间隔 / r (小于该间隔视为同一圈)
RING_CLUSTER_MIN_HITS = 2             # 一个半径簇至少被 N 个阈值命中才算真环
RING_COUNT_MIN = 2                    # 环数 >= 该值 判定"存在翻边外圈"(正面)
FLANGE_HOUGH_BAND = (1.12, 1.36)      # 同心 Hough 找翻边圆的半径范围 / r
FLANGE_HOUGH_P1 = 110
FLANGE_HOUGH_P2 = 20
FLANGE_CENTER_GATE = 0.14             # 翻边圆圆心允许偏离孔心 / r

# ---------- 7. 特征B: 4 个拐角小圆压痕 ----------
# (相对角度°, 距离/r)。角度以"工件中心 -> 孔心"的向外径向方向为 0°, 逆时针为正,
# 因此工件任意旋转时 4 个拐角自动跟随, 无需知道绝对角度。实测值见文件末调试说明。
CORNER_SPEC = ((-131.0, 1.58), (-59.5, 1.79), (60.0, 1.79), (131.0, 1.58))
CORNER_WIN_RATIO = 0.75               # 拐角微小 ROI 半宽 / r
MARK_R_RATIO_RANGE = (0.28, 0.55)     # 压痕半径 / r 允许范围 (实测中位数 ≈0.38)
MARK_HOUGH_P1 = 110
MARK_HOUGH_P2 = 16                    # 压痕 Hough 累加器阈值: 形状级预筛, 非圆毛刺无响应
MARK_CENTER_GATE = 0.22               # 压痕圆心允许偏离拐角 ROI 中心 / r (实测定位重复性 σ≈0.08)
MARK_MASK_RATIO = 1.25                # 圆度计算时的圆形裁剪掩膜半径 / 压痕半径
MARK_THRESH_PCTS = (15, 25, 35, 45, 55, 65, 75, 85)      # 多阈值扫描, 取最佳圆度
MARK_MIN_AREA_RATIO = 0.30            # 连通域面积下限 / (pi*压痕半径^2)
MARK_CIRCULARITY_MIN = 0.75           # circularity = 4*pi*area/perimeter**2 > 该值 判为有效压痕
MIN_VALID_MARKS = 2                   # 单孔有效压痕数 >= 该值 -> 特征B 通过(4 个拐角允许油污遮挡 2 个)

# ---------- 8. 判定逻辑 ----------
HOLE_CHECK_COUNT = 2                  # 参与判定的孔数(按质量排序取前 N); 0 = 全部孔
HOLE_LOGIC = "AND"                    # 单孔内 特征A 与 特征B 的组合: "AND"(双特征联合, 勿改) / "OR"
PART_LOGIC = "OR"                     # 孔之间: "OR" = 任一孔满足即 OK (按需求 5)

# ---------- 9. 调试 / 存图 ----------
PRINT_DEBUG = True                    # 打印每孔轮廓计数、有效压痕数、判定结果
SAVE_NG_IMAGE = True                  # NG 样本本地保存
SAVE_OK_IMAGE = False                 # OK 样本也保存(追溯用)
SAVE_OVERLAY = True                   # 保存时叠加检测结果(孔/ROI/拐角/环) 便于现场看图排查
NG_SAVE_DIR = r"D:\zq\result\NG"
OK_SAVE_DIR = r"D:\zq\result\OK"
JPEG_QUALITY = 92

# ---------- 10. Modbus-TCP 对接 PLC (占位, 默认关闭) ----------
ENABLE_MODBUS = False
PLC_IP = "192.168.1.10"
PLC_PORT = 502
PLC_UNIT_ID = 1
PLC_COIL_OK = 0                       # OK 线圈地址
PLC_COIL_NG = 1                       # NG 线圈地址
PLC_REG_RESULT = 100                  # 结果寄存器: 0=未检 1=OK 2=NG
PLC_REG_HEARTBEAT = 101               # 心跳寄存器

# =====================================================================================
# ==============================  以下为业务逻辑, 现场无需改动  ==========================
# =====================================================================================

NG_PART_NOT_FOUND = "NG_PART_NOT_FOUND"      # 找不到垫片
NG_HOLE_NOT_FOUND = "NG_HOLE_NOT_FOUND"      # 圆孔数量不足
NG_NO_FEATURE = "NG_NO_FEATURE"              # 两个孔都没有有效翻边/压痕特征
OK_PASS = "OK"


# ------------------------------------------------------------------ 基础工具
def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread 在 Windows 下不支持中文路径, 用 np.fromfile + imdecode 代替。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception as exc:                                  # noqa: BLE001
        print("[ERR ] 读图失败 %s : %s" % (path, exc))
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """支持中文路径的写图。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY] if ext.lower() in (".jpg", ".jpeg") else []
    try:
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            return False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        buf.tofile(path)
        return True
    except Exception as exc:                                  # noqa: BLE001
        print("[ERR ] 存图失败 %s : %s" % (path, exc))
        return False


def crop_pad(src: np.ndarray, cx: float, cy: float, half: int) -> Tuple[np.ndarray, bool]:
    """以 (cx,cy) 为中心裁 2*half 方形; 越界用边缘复制补齐, 第二返回值=是否越界。"""
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    h, w = src.shape[:2]
    pad = (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))
    sub = src[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if sub.size == 0:
        return np.zeros((2 * half, 2 * half), src.dtype), True
    if any(pad):
        sub = cv2.copyMakeBorder(sub, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_REPLICATE)
    return sub, any(pad)


def fit_circle_lsq(pts: np.ndarray) -> Tuple[float, float, float]:
    """Kasa 代数法最小二乘圆拟合: x^2+y^2 = 2ax + 2by + c。"""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    a = np.c_[2.0 * x, 2.0 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9)))
    return cx, cy, r


def fit_circle_robust(pts: np.ndarray, iters: int, tol_ratio: float,
                      tol_min: float, min_pts: int) -> Optional[Tuple[float, float, float, int]]:
    """迭代剔野点的圆拟合, 返回 (cx, cy, r, 内点数)。"""
    if len(pts) < min_pts:
        return None
    cur = pts.astype(np.float64)
    cx = cy = r = 0.0
    for _ in range(max(1, iters)):
        cx, cy, r = fit_circle_lsq(cur)
        dev = np.abs(np.hypot(cur[:, 0] - cx, cur[:, 1] - cy) - r)
        keep = dev < max(tol_ratio * r, tol_min)
        if keep.sum() < min_pts or keep.all():
            break
        cur = cur[keep]
    return cx, cy, r, int(len(cur))


def odd(v: float, lo: int = 3) -> int:
    """转成 >=lo 的奇数, 供形态学/滤波核使用。"""
    k = int(round(v))
    if k < lo:
        k = lo
    return k if k % 2 == 1 else k + 1


def rotate_unit(ux: float, uy: float, deg: float) -> Tuple[float, float]:
    """把单位向量 (ux,uy) 旋转 deg 度(图像坐标系, y 向下)。"""
    t = np.radians(deg)
    c, s = float(np.cos(t)), float(np.sin(t))
    return ux * c - uy * s, ux * s + uy * c


def angular_coverage(px: np.ndarray, py: np.ndarray, bins: int = 36) -> float:
    """轮廓点相对中心的角度覆盖率(0~1), 用于滤掉短弧碎片。"""
    if len(px) == 0:
        return 0.0
    ang = (np.degrees(np.arctan2(py, px)) + 360.0) % 360.0
    idx = np.unique((ang / (360.0 / bins)).astype(np.int32))
    return float(len(idx)) / float(bins)


def circularity(contour: np.ndarray) -> float:
    """需求指定的圆度: 4*pi*area/perimeter**2。"""
    area = float(cv2.contourArea(contour))
    per = float(cv2.arcLength(contour, True))
    if per <= 1e-6:
        return 0.0
    return 4.0 * float(np.pi) * area / (per * per)


def find_contours(binary: np.ndarray, mode: int, method: int = cv2.CHAIN_APPROX_SIMPLE) -> List[np.ndarray]:
    """兼容 OpenCV 3/4/5 的 findContours 返回值。"""
    res = cv2.findContours(binary, mode, method)
    return list(res[-2])


def radial_profile(src_f32: np.ndarray, cx: float, cy: float,
                   radii: np.ndarray, cos_t: np.ndarray, sin_t: np.ndarray) -> np.ndarray:
    """沿 360° 射线采样, 返回每个半径上"跨角度中位数"曲线(越界置 NaN)。
    取中位数而不是均值: 相邻孔/局部油污只占少数角度, 中位数天然抗干扰。"""
    xs = (cx + radii[:, None] * cos_t[None, :]).astype(np.float32)
    ys = (cy + radii[:, None] * sin_t[None, :]).astype(np.float32)
    h, w = src_f32.shape[:2]
    inside = (xs > 1.0) & (ys > 1.0) & (xs < w - 2.0) & (ys < h - 2.0)
    vals = cv2.remap(src_f32, np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    vals[~inside] = np.nan
    out = np.full(vals.shape[0], np.nan, np.float32)
    good = np.count_nonzero(~np.isnan(vals), axis=1) > 0
    if good.any():
        out[good] = np.nanmedian(vals[good], axis=1)
    return out


# ------------------------------------------------------------------ 取图层
class LocalFolderSource:
    """本地调试: 遍历文件夹里的样本图片。"""

    def __init__(self, folder: str, recursive: bool = True) -> None:
        self.folder = folder
        files: List[str] = []
        pattern = "**/*" if recursive else "*"
        for path in glob.glob(os.path.join(folder, pattern), recursive=recursive):
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS:
                files.append(path)
        self.files = sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def frames(self) -> Iterator[Tuple[str, Optional[np.ndarray]]]:
        for path in self.files:
            yield path, imread_unicode(path)


class HttpCameraSource:
    """上线取图: WTX-3000 HTTP 接口 http://{camera_ip}/camera/currentImage 。"""

    def __init__(self, ip: str, max_frames: int = 0, interval_s: float = 0.2) -> None:
        self.url = CAMERA_URL_TEMPLATE.format(camera_ip=ip)
        self.max_frames = max_frames
        self.interval_s = interval_s
        try:
            import requests                                   # 延迟导入, 本地调试无需安装
        except ImportError as exc:                            # pragma: no cover
            raise RuntimeError("HTTP 取图需要 requests 库: pip install requests") from exc
        self._requests = requests
        self._session = requests.Session()

    def _grab(self) -> Optional[np.ndarray]:
        for attempt in range(max(1, HTTP_RETRY)):
            try:
                resp = self._session.get(self.url, timeout=HTTP_TIMEOUT_S)
                resp.raise_for_status()
                buf = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
                print("[WARN] 第 %d 次取图: 数据无法解码" % (attempt + 1))
            except Exception as exc:                          # noqa: BLE001
                print("[WARN] 第 %d 次取图失败: %s" % (attempt + 1, exc))
            time.sleep(0.05)
        return None

    def frames(self) -> Iterator[Tuple[str, Optional[np.ndarray]]]:
        n = 0
        while self.max_frames <= 0 or n < self.max_frames:
            n += 1
            yield "HTTP#%06d" % n, self._grab()
            if self.interval_s > 0:
                time.sleep(self.interval_s)


def build_source(mode: str, folder: str, ip: str):
    """一键切换取图方式。"""
    if mode == "http":
        print("[INFO] 取图模式: HTTP  %s" % CAMERA_URL_TEMPLATE.format(camera_ip=ip))
        return HttpCameraSource(ip, HTTP_MAX_FRAMES, HTTP_INTERVAL_S)
    src = LocalFolderSource(folder, LOCAL_RECURSIVE)
    print("[INFO] 取图模式: LOCAL  %s  (%d 张)" % (folder, len(src)))
    return src


# ------------------------------------------------------------------ 数据结构
@dataclass
class HoleResult:
    """单个圆孔的检测结果。"""
    index: int
    cx: float
    cy: float
    r: float
    contrast: float = 0.0
    in_frame: bool = True
    raw_contour_count: int = 0          # ROI 内参与统计的原始轮廓数
    ring_count: int = 0                 # 同心环数(特征A 主判据)
    ring_radii: List[float] = field(default_factory=list)
    flange_hough_r: Optional[float] = None
    feature_a: bool = False
    corner_hits: List[dict] = field(default_factory=list)
    valid_marks: int = 0                # 有效小圆压痕数(特征B)
    corners_in_frame: int = 0
    feature_b: bool = False
    passed: bool = False


@dataclass
class InspectResult:
    """整幅图的检测结果。"""
    name: str
    verdict: str = OK_PASS
    reason: str = ""
    part_cx: float = 0.0
    part_cy: float = 0.0
    pitch_r: float = 0.0
    locate_method: str = ""
    hole_r: float = 0.0
    holes: List[HoleResult] = field(default_factory=list)
    checked: List[int] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def is_ok(self) -> bool:
        return self.verdict == OK_PASS


# ------------------------------------------------------------------ 预处理
def preprocess(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """返回 (缩放后的彩色图, 原始灰度, 增强灰度, 缩放系数)。"""
    scale = 1.0
    if RESIZE_MAX_SIDE > 0:
        long_side = max(bgr.shape[:2])
        if long_side > RESIZE_MAX_SIDE:
            scale = RESIZE_MAX_SIDE / float(long_side)
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    work = gray
    if MEDIAN_BLUR_K >= 3:
        work = cv2.medianBlur(work, odd(MEDIAN_BLUR_K))       # 先去铁屑/椒盐点
    work = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(work)
    if GAUSS_BLUR_K >= 3:
        work = cv2.GaussianBlur(work, (odd(GAUSS_BLUR_K), odd(GAUSS_BLUR_K)), 0)
    return bgr, gray, work, scale


# ------------------------------------------------------------------ 圆孔粗定位
def detect_hole_candidates(work: np.ndarray) -> np.ndarray:
    """HoughCircles 粗找圆孔, 返回 Nx3 的 (x, y, r)。

    第一遍用高累加器阈值(准), 数量不够时自动降阈值再找一遍(全)。
    后续还有精定位灰度对比度校验 + 孔径一致性校验兜底, 所以这里宁松勿紧。
    """
    w = work.shape[1]
    r_lo = max(3, int(HOLE_R_MIN_RATIO * w))
    r_hi = max(r_lo + 2, int(HOLE_R_MAX_RATIO * w))
    min_dist = max(8, int(HOLE_MIN_DIST_RATIO * w))
    thresholds = [HOLE_HOUGH_P2]
    if HOLE_HOUGH_P2_FALLBACK > 0 and HOLE_HOUGH_P2_FALLBACK < HOLE_HOUGH_P2:
        thresholds.append(HOLE_HOUGH_P2_FALLBACK)
    cand = np.zeros((0, 3), np.float64)
    for p2 in thresholds:
        circles = cv2.HoughCircles(work, cv2.HOUGH_GRADIENT, dp=HOLE_HOUGH_DP, minDist=min_dist,
                                   param1=HOLE_HOUGH_P1, param2=p2,
                                   minRadius=r_lo, maxRadius=r_hi)
        if circles is not None:
            cand = np.asarray(circles[0], dtype=np.float64)
        if len(cand) >= MIN_HOLE_COUNT:
            break
    if len(cand) > MAX_HOLE_CANDIDATES:                       # 半径由大到小截断, 防异常图卡死
        cand = cand[np.argsort(-cand[:, 2])][:MAX_HOLE_CANDIDATES]
    return cand


def locate_part(work: np.ndarray, cand: np.ndarray) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """自动定位工件(不用固定 ROI), 三级策略。返回 ((cx,cy,pitch_r), 方法名)。

    1) 节圆拟合: 孔心共圆 -> 工件中心/节圆半径, 对任意旋转偏移天然免疫,
       外圆被视场切掉也有效(现场半幅视野样图)。
    2) 大圆 Hough: 直接找垫片外圆或中心大孔。
    3) 工件掩膜质心: 最后兜底。
    """
    if len(cand) >= PITCH_FIT_MIN_HOLES:
        fit = fit_circle_robust(cand[:, :2], PITCH_FIT_ITERS, PITCH_FIT_TOL_RATIO,
                                PITCH_FIT_TOL_MIN_PX, PITCH_FIT_MIN_HOLES)
        if fit is not None:
            cx, cy, pr, n_in = fit
            if pr > 1.5 * float(np.median(cand[:, 2])) and n_in >= PITCH_FIT_MIN_HOLES:
                return (cx, cy, pr), "pitch_fit(n=%d)" % n_in

    w = work.shape[1]
    big = cv2.HoughCircles(work, cv2.HOUGH_GRADIENT, dp=1.0, minDist=int(0.30 * w),
                           param1=HOLE_HOUGH_P1, param2=OUTER_HOUGH_P2,
                           minRadius=int(OUTER_R_MIN_RATIO * w),
                           maxRadius=int(OUTER_R_MAX_RATIO * w))
    if big is not None:
        b = np.asarray(big[0], dtype=np.float64)
        bx, by, br = b[np.argmax(b[:, 2])]
        return (float(bx), float(by), float(br)), "boundary_circle"

    _, mask = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(0.02 * w), odd(0.02 * w)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts = find_contours(mask, cv2.RETR_EXTERNAL)
    if cnts:
        big_c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(big_c) > 0.02 * work.size:
            m = cv2.moments(big_c)
            if abs(m["m00"]) > 1e-6:
                (_, _), rr = cv2.minEnclosingCircle(big_c)
                return (m["m10"] / m["m00"], m["m01"] / m["m00"], float(rr)), "mask_centroid"
    return None, "none"


# ------------------------------------------------------------------ 孔心精定位
def refine_hole(gray: np.ndarray, cx0: float, cy0: float, r0: float,
                cos_t: np.ndarray, sin_t: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """沿 360° 射线找孔壁"灰度 50% 跨越点"再做圆拟合。
    返回 (cx, cy, r, 孔内/台面灰度差)。灰度差过小 -> 认为不是孔, 返回 None。
    该方法与孔内是亮(通孔透光)还是暗(暗场)无关, 自动判极性。
    """
    gray_f = gray.astype(np.float32)
    radii = np.arange(REFINE_SCAN_BAND[0] * r0, REFINE_SCAN_BAND[1] * r0,
                      max(0.2, REFINE_RADIUS_STEP_PX), dtype=np.float32)
    if len(radii) < 8:
        return None
    xs = (cx0 + radii[:, None] * cos_t[None, :]).astype(np.float32)
    ys = (cy0 + radii[:, None] * sin_t[None, :]).astype(np.float32)
    h, w = gray.shape[:2]
    inside = (xs > 1.0) & (ys > 1.0) & (xs < w - 2.0) & (ys < h - 2.0)
    vals = cv2.remap(gray_f, np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    vals[~inside] = np.nan

    med = np.full(len(radii), np.nan, np.float32)
    good = np.count_nonzero(~np.isnan(vals), axis=1) > 0
    if not good.any():
        return None
    med[good] = np.nanmedian(vals[good], axis=1)
    in_sel = radii < REFINE_INNER_BAND * r0
    la_sel = radii > REFINE_LAND_BAND * r0
    if not in_sel.any() or not la_sel.any():
        return None
    inner = float(np.nanmedian(med[in_sel]))
    land = float(np.nanmedian(med[la_sel]))
    if not np.isfinite(inner) or not np.isfinite(land):
        return None
    contrast = abs(inner - land)
    if contrast < REFINE_MIN_CONTRAST:                        # 无明显孔 -> 丢弃(抗油污误检)
        return None

    thr = 0.5 * (inner + land)
    sign = 1.0 if inner > land else -1.0
    start = np.searchsorted(radii, REFINE_EDGE_START * r0)
    pts: List[Tuple[float, float]] = []
    for j in range(vals.shape[1]):
        col = sign * (vals[:, j] - thr)
        col = np.nan_to_num(col, nan=-1.0)
        cross = np.where((col[start:-1] > 0.0) & (col[start + 1:] <= 0.0))[0]
        if cross.size:
            rr = float(radii[start + cross[0]])
            pts.append((cx0 + rr * float(cos_t[j]), cy0 + rr * float(sin_t[j])))
    if len(pts) < REFINE_MIN_EDGE_PTS:
        return None
    fit = fit_circle_robust(np.asarray(pts, np.float64), REFINE_ITERS,
                            REFINE_INLIER_RATIO, 3.0, max(12, REFINE_MIN_EDGE_PTS // 2))
    if fit is None:
        return None
    cx, cy, r, _ = fit
    if not (REFINE_SCAN_BAND[0] * r0 < r < REFINE_SCAN_BAND[1] * r0):
        return None
    return cx, cy, r, contrast


def local_enhance(sub: np.ndarray) -> np.ndarray:
    """小 ROI 局部增强: 局部 CLAHE + 高斯。翻边外圈在暗区/亮区对比度差异很大,
    局部均衡后同一套阈值才能通用(实测样图左侧暗、右侧过曝, 全局均衡不够)。"""
    out = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(sub)
    if GAUSS_BLUR_K >= 3:
        out = cv2.GaussianBlur(out, (odd(GAUSS_BLUR_K), odd(GAUSS_BLUR_K)), 0)
    return out


def local_smooth(sub: np.ndarray) -> np.ndarray:
    """小 ROI 只做平滑。压痕检测实测在原始灰度上最稳: 拐角窗口本来就小,
    再做 CLAHE 会把台面机加工纹理放大成假圆(实测有效命中率反而下降)。"""
    if GAUSS_BLUR_K >= 3:
        return cv2.GaussianBlur(sub, (odd(GAUSS_BLUR_K), odd(GAUSS_BLUR_K)), 0)
    return sub


# ------------------------------------------------------------------ 特征A: 翻边外圈
def feature_a_ring_contours(gray: np.ndarray, hx: float, hy: float, r: float) -> Tuple[int, int, List[float]]:
    """孔 ROI 内轮廓计数 -> 同心环数。

    抗干扰三重过滤:
      1) 圆形掩膜 (RING_MASK_RATIO) 屏蔽相邻孔与拐角压痕;
      2) 只保留"同心"轮廓: 平均半径在 RING_BAND 内、径向标准差比 < RING_MAX_RADIAL_STD、
         角度覆盖率 > RING_MIN_ANGLE_COVER (毛刺/铁屑/划痕碎轮廓全被剔除);
      3) 多阈值扫描后按半径聚类, 只有被 >=RING_CLUSTER_MIN_HITS 个阈值重复命中的簇才算真环
         (单圈冲裁轮廓的内外两条边距离很近, 会被聚成 1 圈, 不会误判成翻边)。
    返回 (原始轮廓数, 环数, 各环半径比)。
    """
    half = int(round(RING_ROI_RATIO * r))
    if half < 6:
        return 0, 0, []
    sub, _ = crop_pad(gray, hx, hy, half)
    sub = local_enhance(sub)
    mask = np.zeros(sub.shape[:2], np.uint8)
    cv2.circle(mask, (half, half), int(round(RING_MASK_RATIO * r)), 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lo, hi = RING_BAND[0] * r, RING_BAND[1] * r
    radii_hits: List[float] = []
    raw = 0
    for q in np.percentile(sub, RING_THRESH_PCTS):
        for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            _, bw = cv2.threshold(sub, float(q), 255, flag)
            bw = cv2.morphologyEx(cv2.bitwise_and(bw, mask), cv2.MORPH_OPEN, kernel)
            for cnt in find_contours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE):
                if len(cnt) < RING_MIN_CONTOUR_PTS:
                    continue
                raw += 1
                pts = cnt.reshape(-1, 2).astype(np.float32) - float(half)
                dist = np.hypot(pts[:, 0], pts[:, 1])
                mean_r = float(dist.mean())
                if mean_r < 1e-6 or not (lo <= mean_r <= hi):
                    continue
                if float(dist.std()) / mean_r > RING_MAX_RADIAL_STD:
                    continue
                if angular_coverage(pts[:, 0], pts[:, 1]) < RING_MIN_ANGLE_COVER:
                    continue
                radii_hits.append(mean_r)

    radii_hits.sort()
    clusters: List[List[float]] = []
    for v in radii_hits:
        if not clusters or v - clusters[-1][-1] > RING_CLUSTER_GAP * r:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    rings = [c for c in clusters if len(c) >= RING_CLUSTER_MIN_HITS]
    return raw, len(rings), [round(float(np.mean(c)) / r, 3) for c in rings]


def feature_a_flange_hough(gray: np.ndarray, hx: float, hy: float, r: float) -> Optional[float]:
    """同心 Hough: 在 [1.12,1.36]r 内找与孔同心的翻边圆, 作为轮廓计数的 OR 兜底。
    光照不均导致翻边外圈局部对比度低、轮廓断裂时, 形状级 Hough 仍能命中。"""
    half = int(round((FLANGE_HOUGH_BAND[1] + 0.12) * r))
    if half < 6:
        return None
    sub, _ = crop_pad(gray, hx, hy, half)
    sub = local_enhance(sub)
    circles = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.0,
                               minDist=max(6, int(0.20 * r)),
                               param1=FLANGE_HOUGH_P1, param2=FLANGE_HOUGH_P2,
                               minRadius=max(3, int(FLANGE_HOUGH_BAND[0] * r)),
                               maxRadius=max(5, int(FLANGE_HOUGH_BAND[1] * r)))
    if circles is None:
        return None
    best = None
    for (bx, by, br) in np.asarray(circles[0], dtype=np.float64):
        if np.hypot(bx - half, by - half) <= FLANGE_CENTER_GATE * r:
            if best is None or br > best:
                best = float(br)
    return None if best is None else best / r


# ------------------------------------------------------------------ 特征B: 拐角小圆压痕
def _best_mark_circularity(win: np.ndarray, bx: float, by: float, br: float) -> float:
    """对 Hough 命中的候选圆做多阈值分割, 取圆度最高的连通域。

    压痕与翻边外圈在图像上是紧邻的, 单一阈值会把两者粘成一团导致圆度骤降;
    这里用"圆形掩膜裁剪 + 多阈值(百分位)扫描"取最优解, 是本算法抗粘连的关键。
    """
    mask = np.zeros(win.shape[:2], np.uint8)
    cv2.circle(mask, (int(round(bx)), int(round(by))), max(2, int(round(MARK_MASK_RATIO * br))), 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    min_area = MARK_MIN_AREA_RATIO * float(np.pi) * br * br
    best = 0.0
    for q in np.percentile(win, MARK_THRESH_PCTS):
        for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            _, bw = cv2.threshold(win, float(q), 255, flag)
            bw = cv2.bitwise_and(bw, mask)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            for cnt in find_contours(bw, cv2.RETR_EXTERNAL):
                if cv2.contourArea(cnt) < min_area:
                    continue
                if cv2.pointPolygonTest(cnt, (float(bx), float(by)), False) < 0:
                    continue                                  # 必须包住 Hough 圆心
                best = max(best, circularity(cnt))
    return best


def feature_b_corner_marks(gray: np.ndarray, hx: float, hy: float, r: float,
                           part_cx: float, part_cy: float,
                           corner_spec: Sequence[Tuple[float, float]] = CORNER_SPEC
                           ) -> Tuple[int, int, List[dict]]:
    """4 个拐角微小 ROI 内找冲压小圆压痕。

    拐角位置由"工件中心 -> 孔心"的径向方向 + 固定相对角度确定, 工件任意旋转自动跟随。
    双重判据: (1) HoughCircles 形状级预筛 —— 毛刺/铁屑/划痕不是圆, 不产生响应;
              (2) circularity = 4*pi*area/perimeter**2 > MARK_CIRCULARITY_MIN。
    返回 (有效压痕数, 在视场内的拐角数, 每个拐角明细)。
    """
    ux, uy = hx - part_cx, hy - part_cy
    norm = float(np.hypot(ux, uy))
    if norm < 1e-6:
        return 0, 0, []
    ux, uy = ux / norm, uy / norm
    half = int(round(CORNER_WIN_RATIO * r))
    if half < 5:
        return 0, 0, []
    r_lo = max(3, int(round(MARK_R_RATIO_RANGE[0] * r)))
    r_hi = max(r_lo + 2, int(round(MARK_R_RATIO_RANGE[1] * r)))
    h, w = gray.shape[:2]
    valid = 0
    in_frame = 0
    details: List[dict] = []
    for ang, dist_ratio in corner_spec:
        vx, vy = rotate_unit(ux, uy, ang)
        px, py = hx + dist_ratio * r * vx, hy + dist_ratio * r * vy
        rec = {"angle": ang, "cx": px, "cy": py, "ok": False,
               "circ": 0.0, "r": 0.0, "in_frame": False}
        if not (half <= px < w - half and half <= py < h - half):
            details.append(rec)                               # 拐角越界(工件半幅进入视场) -> 不参与计数
            continue
        rec["in_frame"] = True
        in_frame += 1
        win, _ = crop_pad(gray, px, py, half)
        win = local_smooth(win)
        circles = cv2.HoughCircles(win, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(6, r_lo),
                                   param1=MARK_HOUGH_P1, param2=MARK_HOUGH_P2,
                                   minRadius=r_lo, maxRadius=r_hi)
        if circles is None:
            details.append(rec)
            continue
        cand = [c for c in np.asarray(circles[0], dtype=np.float64)
                if np.hypot(c[0] - half, c[1] - half) <= MARK_CENTER_GATE * r]
        if not cand:
            details.append(rec)
            continue
        bx, by, br = cand[0]
        circ = _best_mark_circularity(win, bx, by, br)
        rec["circ"] = round(circ, 3)
        rec["r"] = round(float(br) / r, 3)
        rec["mark_cx"] = px + (bx - half)
        rec["mark_cy"] = py + (by - half)
        rec["mark_r"] = float(br)
        if circ > MARK_CIRCULARITY_MIN:
            rec["ok"] = True
            valid += 1
        details.append(rec)
    return valid, in_frame, details


def count_corners_in_frame(shape: Tuple[int, int], hx: float, hy: float, r: float,
                           part_cx: float, part_cy: float) -> int:
    """预判 4 个拐角 ROI 有几个完整落在视场内 —— 用于挑选参与判定的孔,
    避免选到贴边、拐角被切掉的孔造成误 NG(现场半幅视野时很常见)。"""
    ux, uy = hx - part_cx, hy - part_cy
    norm = float(np.hypot(ux, uy))
    if norm < 1e-6:
        return 0
    ux, uy = ux / norm, uy / norm
    half = int(round(CORNER_WIN_RATIO * r))
    h, w = shape[:2]
    n = 0
    for ang, dist_ratio in CORNER_SPEC:
        vx, vy = rotate_unit(ux, uy, ang)
        px, py = hx + dist_ratio * r * vx, hy + dist_ratio * r * vy
        if half <= px < w - half and half <= py < h - half:
            n += 1
    return n


# ------------------------------------------------------------------ 主检测流程
def inspect(bgr: np.ndarray, name: str = "") -> Tuple[InspectResult, np.ndarray]:
    """单帧检测。返回 (结果, 预处理后的彩色图 —— 供画调试图用)。"""
    t0 = time.time()
    bgr, gray, work, _ = preprocess(bgr)
    res = InspectResult(name=name)

    # --- 1. 圆孔粗定位 + 工件定位 (异常保护) ---
    cand = detect_hole_candidates(work)
    part, method = locate_part(work, cand)
    res.locate_method = method
    if part is None:
        res.verdict = NG_PART_NOT_FOUND
        res.reason = "定位失败: 图中找不到垫片"
        res.elapsed_ms = (time.time() - t0) * 1e3
        return res, bgr
    res.part_cx, res.part_cy, res.pitch_r = part
    if len(cand) < MIN_HOLE_COUNT:
        res.verdict = NG_HOLE_NOT_FOUND
        res.reason = "圆孔候选不足: %d < %d" % (len(cand), MIN_HOLE_COUNT)
        res.elapsed_ms = (time.time() - t0) * 1e3
        return res, bgr

    # --- 2. 孔心精定位 ---
    ang = np.radians(np.arange(0.0, 360.0, REFINE_ANGLE_STEP_DEG))
    cos_t, sin_t = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    refined: List[Tuple[float, float, float, float]] = []
    for (x0, y0, r0) in cand:
        got = refine_hole(work, float(x0), float(y0), float(r0), cos_t, sin_t)
        if got is not None:
            refined.append(got)
    if len(refined) < MIN_HOLE_COUNT:
        res.verdict = NG_HOLE_NOT_FOUND
        res.reason = "精定位后有效圆孔不足: %d < %d" % (len(refined), MIN_HOLE_COUNT)
        res.elapsed_ms = (time.time() - t0) * 1e3
        return res, bgr

    # 同一工件孔径一致 -> 用中位数做统一基准, 单孔精定位失配不会带偏 ROI 尺寸
    r_med = float(np.median([q[2] for q in refined]))
    res.hole_r = r_med
    holes: List[HoleResult] = []
    h_img, w_img = gray.shape[:2]
    for i, (hx, hy, hr, contrast) in enumerate(refined):
        if abs(hr - r_med) / max(r_med, 1e-6) > HOLE_R_DEV_MAX:
            continue                                          # 半径异常 -> 不是同一种孔, 剔除
        r_use = r_med if USE_GLOBAL_HOLE_RADIUS else hr
        margin = RING_ROI_RATIO * r_use
        in_frame = (margin <= hx < w_img - margin) and (margin <= hy < h_img - margin)
        holes.append(HoleResult(index=i, cx=hx, cy=hy, r=r_use,
                                contrast=contrast, in_frame=in_frame))
    if len(holes) < MIN_HOLE_COUNT:
        res.verdict = NG_HOLE_NOT_FOUND
        res.reason = "孔径一致性筛选后不足: %d < %d" % (len(holes), MIN_HOLE_COUNT)
        res.elapsed_ms = (time.time() - t0) * 1e3
        return res, bgr

    # --- 3. 选参与判定的孔: 先要 ROI 完整在视场内, 再看 4 个拐角完整数, 最后按孔壁对比度 ---
    for hole in holes:
        hole.corners_in_frame = count_corners_in_frame(
            gray.shape, hole.cx, hole.cy, hole.r, res.part_cx, res.part_cy)
    holes.sort(key=lambda q: (not q.in_frame, -q.corners_in_frame, -q.contrast))
    n_check = len(holes) if HOLE_CHECK_COUNT <= 0 else min(HOLE_CHECK_COUNT, len(holes))
    res.holes = holes
    res.checked = [holes[i].index for i in range(n_check)]

    # --- 4. 双特征联合判断 ---
    for hole in holes[:n_check]:
        raw_cnt, ring_cnt, ring_radii = feature_a_ring_contours(gray, hole.cx, hole.cy, hole.r)
        hole.raw_contour_count = raw_cnt
        hole.ring_count = ring_cnt
        hole.ring_radii = ring_radii
        a_contour = ring_cnt >= RING_COUNT_MIN
        a_hough = False
        if FEATURE_A_MODE != "contour":
            hole.flange_hough_r = feature_a_flange_hough(gray, hole.cx, hole.cy, hole.r)
            a_hough = hole.flange_hough_r is not None
        if FEATURE_A_MODE == "contour":
            hole.feature_a = a_contour
        elif FEATURE_A_MODE == "hough":
            hole.feature_a = a_hough
        elif FEATURE_A_MODE == "contour_and_hough":
            hole.feature_a = a_contour and a_hough
        else:                                                 # contour_or_hough (默认)
            hole.feature_a = a_contour or a_hough

        marks, corners, details = feature_b_corner_marks(
            gray, hole.cx, hole.cy, hole.r, res.part_cx, res.part_cy)
        hole.valid_marks = marks
        hole.corners_in_frame = corners
        hole.corner_hits = details
        hole.feature_b = marks >= MIN_VALID_MARKS

        hole.passed = (hole.feature_a and hole.feature_b) if HOLE_LOGIC == "AND" \
            else (hole.feature_a or hole.feature_b)

    # --- 5. 工件级判定: 孔1 满足 OR 孔2 满足 -> OK ---
    checked_holes = holes[:n_check]
    if PART_LOGIC == "OR":
        part_ok = any(q.passed for q in checked_holes)
    else:
        part_ok = all(q.passed for q in checked_holes)
    if part_ok:
        res.verdict, res.reason = OK_PASS, "存在翻边外圈 + 冲压小圆压痕(正面)"
    else:
        res.verdict = NG_NO_FEATURE
        res.reason = "所有受检孔均无有效翻边/压痕特征(反面或漏冲)"
    res.elapsed_ms = (time.time() - t0) * 1e3
    return res, bgr


# ------------------------------------------------------------------ 调试输出
def print_result(res: InspectResult) -> None:
    """按需求 7 打印: 每孔轮廓计数、有效小圆痕迹数量、最终判定结果。"""
    print("=" * 108)
    print("[图像] %s" % res.name)
    if res.verdict in (NG_PART_NOT_FOUND, NG_HOLE_NOT_FOUND):
        print("  定位: %-22s  ->  %s : %s" % (res.locate_method, res.verdict, res.reason))
        print("  >>> 判定: NG   (%.1f ms)" % res.elapsed_ms)
        return
    print("  工件定位: %-22s 中心=(%.1f, %.1f)  节圆R=%.1f  孔径r=%.1f  有效孔=%d  受检孔=%s"
          % (res.locate_method, res.part_cx, res.part_cy, res.pitch_r,
             res.hole_r, len(res.holes), res.checked))
    print("  %-4s %-19s %-7s %-6s %-24s %-9s %-6s %-7s %-6s %s"
          % ("孔", "孔心(x,y)", "原始轮廓", "环数", "环半径比", "翻边Hough",
             "特征A", "有效压痕", "特征B", "单孔"))
    for hole in res.holes[:len(res.checked)]:
        flange = "-" if hole.flange_hough_r is None else "%.2f" % hole.flange_hough_r
        print("  #%-3d (%7.1f,%7.1f) %-7d %-6d %-24s %-9s %-6s %d/%-5d %-6s %s"
              % (hole.index, hole.cx, hole.cy, hole.raw_contour_count, hole.ring_count,
                 str(hole.ring_radii), flange, "PASS" if hole.feature_a else "FAIL",
                 hole.valid_marks, hole.corners_in_frame,
                 "PASS" if hole.feature_b else "FAIL", "OK" if hole.passed else "NG"))
        detail = "  ".join("%+6.1f°:%s(circ=%.2f,r=%.2f)"
                           % (d["angle"], "Y" if d["ok"] else ("-" if d["in_frame"] else "x"),
                              d["circ"], d["r"]) for d in hole.corner_hits)
        print("       拐角明细 %s" % detail)
    print("  >>> 判定: %s   %s   (%.1f ms)"
          % ("OK" if res.is_ok else "NG", res.reason, res.elapsed_ms))


def draw_overlay(bgr: np.ndarray, res: InspectResult) -> np.ndarray:
    """叠加检测结果, 现场看图排查用。"""
    vis = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    green, red, yellow, blue, cyan = (0, 220, 0), (0, 0, 255), (0, 220, 220), (255, 120, 0), (255, 255, 0)
    if res.pitch_r > 0:
        cv2.circle(vis, (int(res.part_cx), int(res.part_cy)), int(res.pitch_r), blue, 1, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(res.part_cx), int(res.part_cy)), blue, cv2.MARKER_CROSS, 26, 2)
    checked = set(res.checked)
    for hole in res.holes:
        col = green if hole.passed else (red if hole.index in checked else yellow)
        cv2.circle(vis, (int(hole.cx), int(hole.cy)), int(hole.r), col, 2, cv2.LINE_AA)
        if hole.index not in checked:
            continue
        cv2.circle(vis, (int(hole.cx), int(hole.cy)), int(RING_ROI_RATIO * hole.r), cyan, 1, cv2.LINE_AA)
        for rr in hole.ring_radii:                            # 命中的同心环
            cv2.circle(vis, (int(hole.cx), int(hole.cy)), int(rr * hole.r), (255, 0, 255), 1, cv2.LINE_AA)
        if hole.flange_hough_r:
            cv2.circle(vis, (int(hole.cx), int(hole.cy)),
                       int(hole.flange_hough_r * hole.r), (200, 200, 0), 1, cv2.LINE_AA)
        half = int(round(CORNER_WIN_RATIO * hole.r))
        for d in hole.corner_hits:
            c = green if d["ok"] else (red if d["in_frame"] else (128, 128, 128))
            p0 = (int(d["cx"]) - half, int(d["cy"]) - half)
            p1 = (int(d["cx"]) + half, int(d["cy"]) + half)
            cv2.rectangle(vis, p0, p1, c, 1)
            if d.get("mark_r"):
                cv2.circle(vis, (int(d["mark_cx"]), int(d["mark_cy"])), int(d["mark_r"]), c, 2, cv2.LINE_AA)
        cv2.putText(vis, "#%d ring=%d mark=%d" % (hole.index, hole.ring_count, hole.valid_marks),
                    (int(hole.cx) - half, int(hole.cy) - int(1.55 * hole.r)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    tag = "OK" if res.is_ok else "NG"
    cv2.rectangle(vis, (0, 0), (300, 44), (30, 30, 30), -1)
    cv2.putText(vis, "%s  %s" % (tag, res.verdict), (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, green if res.is_ok else red, 2, cv2.LINE_AA)
    return vis


def save_result_image(bgr: np.ndarray, res: InspectResult, overlay: bool) -> Optional[str]:
    """NG(可选 OK) 样本本地保存。文件名带时间戳 + 原图名 + NG 代码, 便于追溯。"""
    if res.is_ok and not SAVE_OK_IMAGE:
        return None
    if (not res.is_ok) and not SAVE_NG_IMAGE:
        return None
    folder = OK_SAVE_DIR if res.is_ok else NG_SAVE_DIR
    base = os.path.splitext(os.path.basename(res.name))[0] or "frame"
    base = "".join(ch if ch.isalnum() or ch in "-_#" else "_" for ch in base)[:60]
    fname = "%s_%s_%s.jpg" % (time.strftime("%Y%m%d_%H%M%S"), base, res.verdict)
    path = os.path.join(folder, fname)
    img = draw_overlay(bgr, res) if (overlay and SAVE_OVERLAY) else bgr
    return path if imwrite_unicode(path, img) else None


# ------------------------------------------------------------------ Modbus-TCP 占位
class ModbusReporter:
    """预留: 把判定结果写给 PLC (Modbus-TCP)。默认关闭, 不影响算法运行。

    上线时安装 pymodbus (pip install pymodbus) 并把下面注释解开即可:

        from pymodbus.client import ModbusTcpClient

        def connect(self):
            self.client = ModbusTcpClient(PLC_IP, port=PLC_PORT)
            return self.client.connect()

        def report(self, ok: bool):
            # 线圈: OK/NG 各一路, 便于 PLC 直接接分选气缸
            self.client.write_coil(PLC_COIL_OK, bool(ok),  slave=PLC_UNIT_ID)
            self.client.write_coil(PLC_COIL_NG, not ok,    slave=PLC_UNIT_ID)
            # 寄存器: 0=未检 1=OK 2=NG, 供 PLC 做流程判断/计数
            self.client.write_register(PLC_REG_RESULT, 1 if ok else 2, slave=PLC_UNIT_ID)

        def heartbeat(self, tick: int):
            self.client.write_register(PLC_REG_HEARTBEAT, tick & 0xFFFF, slave=PLC_UNIT_ID)

        def close(self):
            if self.client:
                self.client.close()

    注意现场时序: 建议 PLC 用"上升沿 + 应答清零"握手, 不要靠视觉端延时,
    否则节拍变化时会漏信号。
    """

    def __init__(self, enabled: bool = ENABLE_MODBUS) -> None:
        self.enabled = enabled
        self.client = None
        self.tick = 0
        if self.enabled:
            print("[INFO] Modbus 占位已启用, 但通信代码尚未接通 —— 请按 ModbusReporter 注释解开")

    def connect(self) -> bool:
        return False

    def report(self, ok: bool) -> None:                        # noqa: ARG002
        self.tick += 1

    def close(self) -> None:
        return None


# ------------------------------------------------------------------ 换型标定工具
def calibrate(bgr: np.ndarray, name: str = "") -> None:
    """换型/换相机后用来重新标定 CORNER_SPEC 与 MARK_R_RATIO_RANGE。

    做法: 全图 Hough 找小圆 -> 按"工件中心->孔心"径向方向换算相对角度/距离比 ->
    聚类统计。把打印出来的中位数直接填进头部 CORNER_SPEC 即可。
    """
    _, gray, work, _ = preprocess(bgr)
    cand = detect_hole_candidates(work)
    part, method = locate_part(work, cand)
    if part is None or len(cand) < MIN_HOLE_COUNT:
        print("[CALIB] %s 定位失败, 跳过" % name)
        return
    pcx, pcy, _ = part
    ang = np.radians(np.arange(0.0, 360.0, REFINE_ANGLE_STEP_DEG))
    cos_t, sin_t = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    holes = [q for q in (refine_hole(work, float(x), float(y), float(r), cos_t, sin_t)
                         for (x, y, r) in cand) if q is not None]
    if len(holes) < MIN_HOLE_COUNT:
        print("[CALIB] %s 精定位失败, 跳过" % name)
        return
    r_med = float(np.median([q[2] for q in holes]))
    small = cv2.HoughCircles(work, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(6, int(0.35 * r_med)),
                             param1=MARK_HOUGH_P1, param2=32,
                             minRadius=max(3, int(0.10 * r_med)),
                             maxRadius=max(5, int(0.60 * r_med)))
    if small is None:
        print("[CALIB] %s 未找到候选小圆" % name)
        return
    small = np.asarray(small[0], dtype=np.float64)
    rows: List[Tuple[float, float, float]] = []
    for (hx, hy, _hr, _c) in holes:
        ux, uy = hx - pcx, hy - pcy
        nn = float(np.hypot(ux, uy))
        if nn < 1e-6:
            continue
        ux, uy = ux / nn, uy / nn
        dist = np.hypot(small[:, 0] - hx, small[:, 1] - hy)
        for (sx, sy, sr) in small[(dist > 1.15 * r_med) & (dist < 2.10 * r_med)]:
            dx, dy = sx - hx, sy - hy
            rel = float(np.degrees(np.arctan2(ux * dy - uy * dx, ux * dx + uy * dy)))
            rows.append((rel, float(np.hypot(dx, dy)) / r_med, float(sr) / r_med))
    if not rows:
        print("[CALIB] %s 环带内无小圆" % name)
        return
    arr = np.asarray(rows)
    print("[CALIB] %s  定位=%s  孔数=%d  r=%.1f  候选压痕=%d"
          % (name, method, len(holes), r_med, len(arr)))
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    groups: List[List[np.ndarray]] = []
    for row in arr:
        if groups and row[0] - groups[-1][-1][0] <= 15.0:
            groups[-1].append(row)
        else:
            groups.append([row])
    spec = []
    for grp in groups:
        gg = np.asarray(grp)
        if len(gg) < 2:
            continue
        spec.append((round(float(np.median(gg[:, 0])), 1), round(float(np.median(gg[:, 1])), 2)))
        print("   簇 n=%2d  角度中位数 %+7.1f° (σ=%.1f)  距离比 %.2f (σ=%.02f)  半径比 %.2f"
              % (len(gg), np.median(gg[:, 0]), gg[:, 0].std(),
                 np.median(gg[:, 1]), gg[:, 1].std(), np.median(gg[:, 2])))
    print("   >>> 建议 CORNER_SPEC = %s" % (tuple(spec),))
    print("   >>> 建议 MARK_R_RATIO_RANGE = (%.2f, %.2f)"
          % (max(0.05, np.percentile(arr[:, 2], 5) * 0.9), np.percentile(arr[:, 2], 95) * 1.15))


# ------------------------------------------------------------------ 入口
def setup_console() -> None:
    """Windows 控制台默认 GBK, 中文调试信息会乱码 —— 强制切 UTF-8。"""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:                                         # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                     # noqa: BLE001
            pass


def guess_label(name: str) -> Optional[bool]:
    """本地调试时从文件名猜真值(含 正/OK/front -> True; 反/NG/back -> False),
    仅用于自检统计, 猜不出返回 None, 不影响判定。"""
    base = os.path.basename(name)
    low = base.lower()
    if "正" in base or any(k in low for k in ("_ok", "ok_", "front", "zheng")):
        return True
    if "反" in base or any(k in low for k in ("_ng", "ng_", "back", "fan")):
        return False
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="推力保持架垫片 冲压翻边正/反面检测")
    ap.add_argument("--mode", choices=("local", "http"), default=SOURCE_MODE, help="取图方式")
    ap.add_argument("--dir", default=LOCAL_IMAGE_DIR, help="本地样本目录")
    ap.add_argument("--ip", default=CAMERA_IP, help="WTX-3000 相机 IP")
    ap.add_argument("--debug", action="store_true", help="额外保存叠加调试图(OK 也存)")
    ap.add_argument("--calib", action="store_true", help="拐角角度/尺寸标定模式(换型用)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张(本地调试用)")
    ap.add_argument("--holes", type=int, default=None,
                    help="覆盖 HOLE_CHECK_COUNT: 参与判定的孔数, 0=全部孔(排查用)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = parse_args(argv)
    global SAVE_OK_IMAGE, HOLE_CHECK_COUNT
    if args.debug:
        SAVE_OK_IMAGE = True
    if args.holes is not None:
        HOLE_CHECK_COUNT = args.holes
    try:
        source = build_source(args.mode, args.dir, args.ip)
    except Exception as exc:                                  # noqa: BLE001
        print("[FATAL] 取图源初始化失败: %s" % exc)
        return 2
    if args.mode == "local" and len(getattr(source, "files", [])) == 0:
        print("[FATAL] 目录下没有图片: %s" % args.dir)
        return 2

    plc = ModbusReporter()
    n_ok = n_ng = n_bad = 0
    hit = miss = 0
    t_start = time.time()
    for idx, (name, bgr) in enumerate(source.frames()):
        if args.limit and idx >= args.limit:
            break
        if bgr is None:
            n_bad += 1
            print("[WARN] 帧无效, 按 NG 处理: %s" % name)
            plc.report(False)
            continue
        if args.calib:
            calibrate(bgr, name)
            continue
        res, proc = inspect(bgr, name)
        if PRINT_DEBUG:
            print_result(res)
        saved = save_result_image(proc, res, overlay=True)
        if saved:
            print("  存图: %s" % saved)
        plc.report(res.is_ok)
        n_ok += res.is_ok
        n_ng += (not res.is_ok)
        truth = guess_label(name)
        if truth is not None:
            hit += (truth == res.is_ok)
            miss += (truth != res.is_ok)
    plc.close()
    if not args.calib:
        total = n_ok + n_ng
        print("=" * 108)
        print("[汇总] 共 %d 帧: OK=%d  NG=%d  无效帧=%d  耗时 %.2f s"
              % (total, n_ok, n_ng, n_bad, time.time() - t_start))
        if hit + miss > 0:
            print("[自检] 文件名可推断真值 %d 张: 一致 %d, 不一致 %d (准确率 %.1f%%)"
                  % (hit + miss, hit, miss, 100.0 * hit / (hit + miss)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

