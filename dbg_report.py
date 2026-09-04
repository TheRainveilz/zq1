# -*- coding: utf-8 -*-
"""
dbg_report.py —— thrust_cage_flange_inspect 的外挂调试报告工具 (不修改原文件)

通过 import 复用主算法的函数与阈值, 产出三样东西:
  1) CSV  : 一行一个拐角, 供离线定 MARK_CIRCULARITY_MIN / MIN_VALID_MARKS
  2) 拼图 : 一行一个受检孔 = [上下文][特征A][4 x (窗口|诊断)], 固定单元格, PNG
  3) 分流 : 文件名带结论, 边界样本自动进 borderline/, 特征A 一票否决打 AVETO 标记
  4) 扫描 : --sweep 用一次检测的结果解析式推出整片阈值网格的正确率, 无需重跑

保真原则
--------
所有面板一律调用模块自己的 crop_pad / local_smooth / local_enhance, 不自己重写裁切与增强;
唯一复刻的是 _best_mark_circularity 的多阈值循环(为了拿到"赢的那个阈值/掩膜/轮廓"),
并对每个拐角断言复刻结果 == 模块记录的 circ, 一旦与模块实现分叉立即报错(--no-verify 可关)。

stage 四分类 (主算法把 circ=0.00 的三种失败合成了一种, 这里拆开):
  oof        拐角出画面, 不参与计数
  hough_miss Hough 一个圆都没找到      -> 调 MARK_HOUGH_P2 / MARK_R_RATIO_RANGE
  gate_miss  找到了圆但全在中心门外    -> 调 MARK_CENTER_GATE / --calib 重标 CORNER_SPEC
  seg_miss   圆找到了但分割无连通域    -> 调 MARK_MIN_AREA_RATIO / MARK_MASK_RATIO
  low_circ   分割到了但圆度不够        -> 调 MARK_CIRCULARITY_MIN
  ok         有效压痕

用法
----
    python dbg_report.py                        # 按默认样本目录跑, 默认全孔模式
    python dbg_report.py --dir "D:\\样本 目录" --limit 20      # 含空格的路径要加引号
    python dbg_report.py --sweep                # 末尾追加阈值网格表
    python dbg_report.py --no-sheet             # 只出 CSV (快)

换样本目录不用每次敲 --dir: 改本文件下方"路径"一节的 IMAGE_DIR / OUT_DIR 即可(主算法文件是
交付件, 不在那里改)。优先级 命令行 > 环境变量 DBG_DIR/DBG_OUT > 本文件 > 主模块常量。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thrust_cage_flange_inspect as T          # noqa: E402  主算法模块, 本文件只读不改

# ---------------------------------------------------------------- 契约检查
# 本工具刻意依赖主模块的内部实现(含带下划线的 _best_mark_circularity)。
# 主文件一旦重构, 宁可启动就炸, 也不要画出位置错误的框还看着挺像。
REQ_FUNCS = ("inspect", "crop_pad", "local_smooth", "local_enhance", "find_contours",
             "circularity", "_best_mark_circularity", "guess_label", "imwrite_unicode",
             "setup_console", "LocalFolderSource")
REQ_CONSTS = ("CORNER_SPEC", "CORNER_WIN_RATIO", "MARK_R_RATIO_RANGE", "MARK_HOUGH_P1",
              "MARK_HOUGH_P2", "MARK_CENTER_GATE", "MARK_MASK_RATIO", "MARK_THRESH_PCTS",
              "MARK_MIN_AREA_RATIO", "MARK_CIRCULARITY_MIN", "MIN_VALID_MARKS",
              "RING_ROI_RATIO", "RING_MASK_RATIO", "HOLE_CHECK_COUNT", "HOLE_LOGIC",
              "PART_LOGIC", "FEATURE_A_MODE", "OK_PASS", "NG_SAVE_DIR", "LOCAL_IMAGE_DIR",
              "IMAGE_EXTS")
REQ_CORNER_KEYS = ("angle", "cx", "cy", "ok", "circ", "r", "in_frame")

def check_contract() -> None:
    """启动即校验依赖的名字与结构都在, 缺一个就退出。"""
    missing = [n for n in REQ_FUNCS + REQ_CONSTS if not hasattr(T, n)]
    if missing:
        print("[FATAL] 主模块缺少本工具依赖的名字: %s" % ", ".join(missing))
        print("        thrust_cage_flange_inspect.py 可能已重构, 请同步更新 dbg_report.py")
        raise SystemExit(3)
    if len(T.CORNER_SPEC) != 4:
        print("[FATAL] CORNER_SPEC 长度 %d != 4, 拼图版式按 4 拐角写死" % len(T.CORNER_SPEC))
        raise SystemExit(3)


def check_corner_keys(rec: dict) -> None:
    lack = [k for k in REQ_CORNER_KEYS if k not in rec]
    if lack:
        print("[FATAL] corner_hits 缺字段: %s (主模块 feature_b_corner_marks 已改)" % lack)
        raise SystemExit(3)


# ---------------------------------------------------------------- 路径 (改这里)
# 主算法文件是交付件不改, 所以样本/输出目录的本地覆盖放在这里。
# 留空 = 沿用主模块的 LOCAL_IMAGE_DIR 与 NG_SAVE_DIR 的父目录。
# 优先级: 命令行 --dir/--out  >  环境变量 DBG_DIR/DBG_OUT  >  下面两行  >  主模块常量
IMAGE_DIR = r""        # 样本目录, 例: r"D:\新建文件夹\WTX3000-360C (DA7486717)"
OUT_DIR   = r""        # 输出根目录(CSV 与 sheet/ 都放这下面), 例: r"D:\zq\result"


def default_image_dir() -> str:
    return os.environ.get("DBG_DIR") or IMAGE_DIR or T.LOCAL_IMAGE_DIR


def default_out_dir() -> str:
    return (os.environ.get("DBG_OUT") or OUT_DIR
            or os.path.dirname(T.NG_SAVE_DIR) or "result")


# ---------------------------------------------------------------- 版式 / 配色
CELL = 200                    # 单元格边长(px)。拐角窗口 2*0.75*r≈76px -> 放大约 2.6 倍
CONTEXT_RATIO = 2.9           # 上下文裁切半宽 / r。拐角最远 1.79r + 方框半对角 1.06r = 2.85r
HEADER_H = 42                 # 顶部信息条高度
BORDERLINE_BAND = 0.07        # |maxcirc - MARK_CIRCULARITY_MIN| 落在该带内 -> 归为边界样本
SWEEP_TC = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)   # --sweep 的圆度阈值网格
SWEEP_TM = (1, 2, 3, 4)                                  # --sweep 的最少压痕数网格
SWEEP_R_UPPER = (0.55, 0.50, 0.47, 0.45, 0.42, 0.40)     # --sweep-radius 的半径上界网格

GREEN, RED, YELLOW, GRAY = (0, 220, 0), (0, 0, 255), (0, 220, 220), (140, 140, 140)
BLUE, CYAN, MAGENTA, OLIVE, ORANGE = (255, 120, 0), (255, 255, 0), (255, 0, 255), (200, 200, 0), (0, 150, 255)
STAGE_COLOR = {"ok": GREEN, "low_circ": ORANGE, "seg_miss": RED,
               "gate_miss": (255, 0, 200), "hough_miss": GRAY, "oof": (90, 90, 90)}


def ascii_safe(text: str) -> str:
    """cv2.putText 画不了中文, 图上文字统一降级为 ASCII(CSV 里仍保留原名)。"""
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(text))


def fit(img: np.ndarray, w: int = CELL, h: int = CELL) -> np.ndarray:
    """统一成 w*h 的 BGR 面板。放大用 NEAREST(看得见像素), 缩小用 AREA。"""
    if img is None or img.size == 0:
        return np.full((h, w, 3), 20, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    interp = cv2.INTER_NEAREST if img.shape[0] < h else cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interp)


def label(panel: np.ndarray, text: str, y: int = 13, color=(255, 255, 255), scale: float = 0.36) -> None:
    """左上角带底色的小字标注, 保证在亮/暗背景上都看得清。"""
    text = ascii_safe(text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(panel, (0, y - th - 4), (min(panel.shape[1], tw + 5), y + 4), (25, 25, 25), -1)
    cv2.putText(panel, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- 拐角复算 (与主模块同源)
def corner_window(gray: np.ndarray, hole, rec: dict) -> Tuple[np.ndarray, int]:
    """复现主模块 feature_b_corner_marks 里那一份拐角窗口。

    调的是模块自己的 crop_pad + local_smooth, 因此逐字节一致。
    注意必须是 local_smooth 而不是 local_enhance —— 模块注释写明拐角窗口做 CLAHE
    会把台面机加工纹理放大成假圆, 用错会导致"看到的掩膜不是检测器看到的掩膜"。
    """
    half = int(round(T.CORNER_WIN_RATIO * hole.r))
    win, _ = T.crop_pad(gray, rec["cx"], rec["cy"], half)
    return T.local_smooth(win), half


def hough_candidates(win: np.ndarray, r: float) -> np.ndarray:
    """按主模块同一套参数重跑压痕 Hough, 只用于拆开 hough_miss / gate_miss。
    Hough 是确定性的, 同输入同参数结果必然与检测时一致。"""
    r_lo = max(3, int(round(T.MARK_R_RATIO_RANGE[0] * r)))
    r_hi = max(r_lo + 2, int(round(T.MARK_R_RATIO_RANGE[1] * r)))
    circles = cv2.HoughCircles(win, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(6, r_lo),
                               param1=T.MARK_HOUGH_P1, param2=T.MARK_HOUGH_P2,
                               minRadius=r_lo, maxRadius=r_hi)
    if circles is None:
        return np.zeros((0, 3), np.float64)
    return np.asarray(circles[0], dtype=np.float64)


def sweep_circularity(win: np.ndarray, bx: float, by: float, br: float) -> dict:
    """复刻 T._best_mark_circularity 的多阈值循环, 额外带出赢的阈值/掩膜/轮廓与逐阈值曲线。

    这是本工具唯一重写的一段逻辑(模块只返回标量, 赢的掩膜在函数里就丢了)。
    调用方用 verify 断言 best == 模块记录的 circ, 把"实现分叉"从静默风险变成显式报错。
    """
    mask = np.zeros(win.shape[:2], np.uint8)
    cv2.circle(mask, (int(round(bx)), int(round(by))),
               max(2, int(round(T.MARK_MASK_RATIO * br))), 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    min_area = T.MARK_MIN_AREA_RATIO * float(np.pi) * br * br
    best, win_q, win_flag, win_bw, win_cnt = 0.0, None, None, None, None
    curve: List[float] = []
    n_pass = 0
    for q in np.percentile(win, T.MARK_THRESH_PCTS):
        per_q = 0.0
        for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            _, bw = cv2.threshold(win, float(q), 255, flag)
            bw = cv2.bitwise_and(bw, mask)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            local, local_cnt = 0.0, None
            for cnt in T.find_contours(bw, cv2.RETR_EXTERNAL):
                if cv2.contourArea(cnt) < min_area:
                    continue
                if cv2.pointPolygonTest(cnt, (float(bx), float(by)), False) < 0:
                    continue                                  # 必须包住 Hough 圆心
                v = T.circularity(cnt)
                if v > local:
                    local, local_cnt = v, cnt
            n_pass += int(local > T.MARK_CIRCULARITY_MIN)
            per_q = max(per_q, local)
            if local > best:
                best, win_q, win_flag, win_bw, win_cnt = local, float(q), flag, bw.copy(), local_cnt
        curve.append(per_q)
    return {"best": best, "q": win_q, "flag": win_flag, "bw": win_bw,
            "cnt": win_cnt, "curve": curve, "n_pass": n_pass,
            "n_combo": 2 * len(T.MARK_THRESH_PCTS)}


class VerifyError(RuntimeError):
    """复刻的分割循环与主模块结果不一致 —— 拼图会画错, 必须停下来。"""


def classify_corner(gray: np.ndarray, hole, rec: dict, verify: bool = True) -> dict:
    """把单个拐角判成 6 种 stage 之一, 并带回画面板需要的全部素材。"""
    out = {"stage": "oof", "win": None, "half": 0, "hough_n": "",
           "cands": None, "bx": None, "by": None, "br": None, "sweep": None}
    if not rec["in_frame"]:
        return out
    win, half = corner_window(gray, hole, rec)
    out["win"], out["half"] = win, half

    if "mark_r" in rec:
        # 反推检测时的 Hough 候选圆(模块记的是图像坐标, 这里换回窗口坐标)。
        # 必须过一遍 float32: HoughCircles 的圆心恒为 x.5, 而 (cx+d)-cx 的浮点余差约 1e-13,
        # 正好让 round() 的银行家舍入在 .5 处翻边(round(36.5)=36 但 round(36.5+1e-13)=37),
        # 掩膜中心偏 1 px 就可能选出不同的连通域。原值来自 float32, snap 回去即精确还原。
        bx = float(np.float32(float(rec["mark_cx"]) - float(rec["cx"]) + half))
        by = float(np.float32(float(rec["mark_cy"]) - float(rec["cy"]) + half))
        br = float(rec["mark_r"])
        sw = sweep_circularity(win, bx, by, br)
        out.update({"bx": bx, "by": by, "br": br, "sweep": sw})
        if verify and abs(round(sw["best"], 3) - float(rec["circ"])) > 1e-9:
            # 比"再调一次 _best_mark_circularity"更强: 同时校验了窗口复现与 bx/by/br 反推
            raise VerifyError(
                "拐角 %+.1f°: 复刻圆度 %.6f (记为 %.3f) != 模块记录 %.3f"
                % (rec["angle"], sw["best"], round(sw["best"], 3), rec["circ"]))
        if rec["ok"]:
            out["stage"] = "ok"
        else:
            out["stage"] = "seg_miss" if sw["best"] <= 0.0 else "low_circ"
        return out

    # 没有 mark_r 说明模块没拿到候选圆: 要么 Hough 无响应, 要么全被中心门挡掉
    cands = hough_candidates(win, hole.r)
    out["cands"], out["hough_n"] = cands, int(len(cands))
    out["stage"] = "hough_miss" if len(cands) == 0 else "gate_miss"
    return out


def checked_holes(res) -> List:
    """主模块把受检孔排在 res.holes 前面, 只有它们算过特征。"""
    n = len(res.checked)
    return [h for h in res.holes[:n] if h.corner_hits]


# ---------------------------------------------------------------- 面板
def panel_context(gray: np.ndarray, hole) -> np.ndarray:
    """上下文格: 孔 + ROI + 命中环 + 翻边圆 + 4 个拐角框, 用来核对框有没有落在压痕上。

    裁切半宽取 CONTEXT_RATIO=2.9r: 拐角中心最远 1.79r, 方框半对角 0.75*sqrt(2)=1.06r,
    合计 2.85r。早期调试图用 2.3r, 四个框全贴边被切掉。
    """
    half = int(round(CONTEXT_RATIO * hole.r))
    sub, _ = T.crop_pad(gray, hole.cx, hole.cy, half)
    x0, y0 = int(round(hole.cx)) - half, int(round(hole.cy)) - half
    vis = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
    s = CELL / float(max(1, vis.shape[1]))
    vis = cv2.resize(vis, (CELL, CELL), interpolation=cv2.INTER_LINEAR)

    def pt(x: float, y: float) -> Tuple[int, int]:
        return int(round((x - x0) * s)), int(round((y - y0) * s))

    def rad(v: float) -> int:
        return max(1, int(round(v * s)))

    col = GREEN if hole.passed else RED
    cv2.circle(vis, pt(hole.cx, hole.cy), rad(hole.r), col, 1, cv2.LINE_AA)
    cv2.circle(vis, pt(hole.cx, hole.cy), rad(T.RING_ROI_RATIO * hole.r), CYAN, 1, cv2.LINE_AA)
    for rr in hole.ring_radii:                                # 特征A 命中的同心环
        cv2.circle(vis, pt(hole.cx, hole.cy), rad(rr * hole.r), MAGENTA, 1, cv2.LINE_AA)
    if hole.flange_hough_r:
        cv2.circle(vis, pt(hole.cx, hole.cy), rad(hole.flange_hough_r * hole.r), OLIVE, 1, cv2.LINE_AA)
    ch = T.CORNER_WIN_RATIO * hole.r
    for rec in hole.corner_hits:
        c = GREEN if rec["ok"] else (RED if rec["in_frame"] else GRAY)
        cv2.rectangle(vis, pt(rec["cx"] - ch, rec["cy"] - ch), pt(rec["cx"] + ch, rec["cy"] + ch), c, 1)
        if "mark_r" in rec:
            cv2.circle(vis, pt(rec["mark_cx"], rec["mark_cy"]), rad(rec["mark_r"]), c, 1, cv2.LINE_AA)

    label(vis, "#%d A=%s B=%s -> %s" % (hole.index, "P" if hole.feature_a else "F",
                                        "P" if hole.feature_b else "F",
                                        "OK" if hole.passed else "NG"), color=col)
    label(vis, "r=%.1f mark=%d/%d ring=%d" % (hole.r, hole.valid_marks,
                                              hole.corners_in_frame, hole.ring_count), y=29)
    return vis


def panel_ring(gray: np.ndarray, hole) -> np.ndarray:
    """特征A 格: 孔 ROI(用模块自己的 local_enhance) + 命中环 + 掩膜边界 + 翻边 Hough 圆。

    特征A 只是 AND 里的一票否决(README 已知限制 3: 反面单圈冲裁边也会被计成 2 圈),
    所以这里只给一格看结果, 不铺全套多阈值画廊。
    """
    half = int(round(T.RING_ROI_RATIO * hole.r))
    sub, _ = T.crop_pad(gray, hole.cx, hole.cy, half)
    vis = cv2.cvtColor(T.local_enhance(sub), cv2.COLOR_GRAY2BGR)
    s = CELL / float(max(1, vis.shape[1]))
    vis = cv2.resize(vis, (CELL, CELL), interpolation=cv2.INTER_LINEAR)
    ctr = (int(round(half * s)), int(round(half * s)))
    cv2.circle(vis, ctr, max(1, int(round(hole.r * s))), GRAY, 1, cv2.LINE_AA)
    cv2.circle(vis, ctr, max(1, int(round(T.RING_MASK_RATIO * hole.r * s))), CYAN, 1, cv2.LINE_AA)
    for rr in hole.ring_radii:
        cv2.circle(vis, ctr, max(1, int(round(rr * hole.r * s))), MAGENTA, 1, cv2.LINE_AA)
    if hole.flange_hough_r:
        cv2.circle(vis, ctr, max(1, int(round(hole.flange_hough_r * hole.r * s))), OLIVE, 1, cv2.LINE_AA)
    flange = "-" if hole.flange_hough_r is None else "%.2f" % hole.flange_hough_r
    label(vis, "A=%s ring=%d fl=%s" % ("PASS" if hole.feature_a else "FAIL",
                                       hole.ring_count, flange),
          color=GREEN if hole.feature_a else RED)
    label(vis, ascii_safe(str([round(v, 2) for v in hole.ring_radii])), y=29)
    return vis


def panel_corner_win(rec: dict, ana: dict, hole) -> np.ndarray:
    """拐角窗口格: 原始窗口(检测器看到的那一份) + Hough 圆 + 赢的轮廓 + 中心门。"""
    if ana["win"] is None:
        vis = np.full((CELL, CELL, 3), 60, np.uint8)
        label(vis, "%+.1f x oof" % rec["angle"], color=GRAY)
        return vis
    win, half = ana["win"], ana["half"]
    s = CELL / float(win.shape[1])
    vis = fit(win)
    ctr = (int(round(half * s)), int(round(half * s)))
    cv2.circle(vis, ctr, max(1, int(round(T.MARK_CENTER_GATE * hole.r * s))), BLUE, 1, cv2.LINE_AA)
    col = STAGE_COLOR.get(ana["stage"], YELLOW)
    if ana["br"]:
        cv2.circle(vis, (int(round(ana["bx"] * s)), int(round(ana["by"] * s))),
                   max(1, int(round(ana["br"] * s))), col, 1, cv2.LINE_AA)
    sw = ana["sweep"]
    if sw and sw["cnt"] is not None:
        cv2.drawContours(vis, [(sw["cnt"].astype(np.float32) * s).astype(np.int32)], -1, col, 1)
    label(vis, "%+.1f circ=%.2f r=%.2f" % (rec["angle"], rec["circ"], rec["r"]), color=col)
    label(vis, "%s" % ana["stage"], y=29, color=col)
    return vis


def panel_corner_diag(rec: dict, ana: dict, hole) -> np.ndarray:
    """诊断格: 按 stage 显示最能指向参数的那张中间图。

      ok/low_circ/seg_miss -> 赢的那张二值掩膜 + 用的百分位/极性 + 通过组合数(阈值裕度)
      gate_miss            -> 被中心门挡掉的 Hough 圆 + 中心门圈
      hough_miss           -> Canny 边缘(HoughCircles 内部用的就是 p1 与 p1/2),
                              看得出是"根本没边"(照明/对比度)还是"有边但不成圆"(p2 偏高)
    """
    stage = ana["stage"]
    if ana["win"] is None:
        return np.full((CELL, CELL, 3), 40, np.uint8)
    win, half = ana["win"], ana["half"]
    s = CELL / float(win.shape[1])
    sw = ana["sweep"]

    if sw is not None and sw["bw"] is not None:
        vis = fit(sw["bw"])
        if sw["cnt"] is not None:
            cv2.drawContours(vis, [(sw["cnt"].astype(np.float32) * s).astype(np.int32)], -1, GREEN, 1)
        pol = "INV" if sw["flag"] == cv2.THRESH_BINARY_INV else "BIN"
        label(vis, "q=%.0f %s best=%.2f" % (sw["q"], pol, sw["best"]), color=STAGE_COLOR[stage])
        label(vis, "margin %d/%d combos" % (sw["n_pass"], sw["n_combo"]), y=29,
              color=GREEN if sw["n_pass"] >= 4 else ORANGE)
        return vis

    if stage == "gate_miss":
        vis = fit(win)
        ctr = (int(round(half * s)), int(round(half * s)))
        cv2.circle(vis, ctr, max(1, int(round(T.MARK_CENTER_GATE * hole.r * s))), BLUE, 1, cv2.LINE_AA)
        worst = 9e9
        for (bx, by, br) in ana["cands"]:
            cv2.circle(vis, (int(round(bx * s)), int(round(by * s))),
                       max(1, int(round(br * s))), ORANGE, 1, cv2.LINE_AA)
            worst = min(worst, float(np.hypot(bx - half, by - half)) / max(hole.r, 1e-6))
        label(vis, "gate_miss n=%d" % ana["hough_n"], color=STAGE_COLOR[stage])
        label(vis, "nearest %.2fr gate %.2fr" % (worst, T.MARK_CENTER_GATE), y=29)
        return vis

    vis = fit(cv2.Canny(win, max(1, T.MARK_HOUGH_P1 // 2), T.MARK_HOUGH_P1))
    label(vis, "hough_miss (canny)", color=STAGE_COLOR[stage])
    label(vis, "p1=%d p2=%d r=[%.2f,%.2f]r" % (T.MARK_HOUGH_P1, T.MARK_HOUGH_P2,
                                               T.MARK_R_RATIO_RANGE[0], T.MARK_R_RATIO_RANGE[1]), y=29)
    return vis


# ---------------------------------------------------------------- 拼图
N_COLS = 2 + 2 * 4            # 上下文 + 特征A + 4 拐角 x (窗口|诊断)
SHEET_W = N_COLS * CELL


def header_band(res, truth: Optional[bool]) -> np.ndarray:
    """顶部信息条。把当次生效的阈值印在图上 —— 隔一周回看才知道是哪个参数版本出的。"""
    band = np.full((HEADER_H, SHEET_W, 3), 30, np.uint8)
    tag = "OK" if res.is_ok else "NG"
    truth_s = {True: "front", False: "back", None: "?"}[truth]
    cv2.putText(band, ascii_safe("%s  %s   truth=%s   %s" % (tag, res.verdict, truth_s,
                                                             os.path.basename(res.name))),
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                GREEN if res.is_ok else RED, 1, cv2.LINE_AA)
    cv2.putText(band, ascii_safe(
        "locate=%s  pitchR=%.1f  r=%.1f  holes=%d checked=%d  %.0fms   |   "
        "circmin=%.2f minmarks=%d win=%.2fr gate=%.2fr markr=[%.2f,%.2f]r  "
        "A=%s logic=%s/%s  corner=%s"
        % (res.locate_method, res.pitch_r, res.hole_r, len(res.holes), len(res.checked),
           res.elapsed_ms, T.MARK_CIRCULARITY_MIN, T.MIN_VALID_MARKS, T.CORNER_WIN_RATIO,
           T.MARK_CENTER_GATE, T.MARK_R_RATIO_RANGE[0], T.MARK_R_RATIO_RANGE[1],
           T.FEATURE_A_MODE, T.HOLE_LOGIC, T.PART_LOGIC,
           ",".join("%+.1f/%.2f" % c for c in T.CORNER_SPEC))),
        (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    return band


def build_sheet(gray: np.ndarray, res, per_hole: List[Tuple[object, List[dict]]],
                truth: Optional[bool]) -> np.ndarray:
    """一行一个受检孔的固定版式拼图。行数随 --holes 变化, 列数恒定, 便于连翻对比。"""
    rows = [header_band(res, truth)]
    if not per_hole:
        blank = np.full((CELL, SHEET_W, 3), 45, np.uint8)
        label(blank, "no checked hole: %s  %s" % (res.verdict, res.reason), y=24, color=RED, scale=0.5)
        rows.append(blank)
        return np.vstack(rows)
    for hole, anas in per_hole:
        cells = [panel_context(gray, hole), panel_ring(gray, hole)]
        for rec, ana in zip(hole.corner_hits, anas):
            cells.append(panel_corner_win(rec, ana, hole))
            cells.append(panel_corner_diag(rec, ana, hole))
        while len(cells) < N_COLS:                             # CORNER_SPEC 不足 4 时补空格
            cells.append(np.full((CELL, CELL, 3), 40, np.uint8))
        row = np.hstack(cells[:N_COLS])
        cv2.line(row, (0, 0), (row.shape[1], 0), (70, 70, 70), 1)
        rows.append(row)
    return np.vstack(rows)


# ---------------------------------------------------------------- CSV
CSV_HEADER = ("image", "truth", "verdict", "reason", "elapsed_ms", "locate", "pitch_r", "hole_r",
              "n_holes", "n_checked",
              "hole_index", "hole_cx", "hole_cy", "contrast", "ring_count", "ring_radii",
              "flange_r", "feature_a",
              "corner_angle", "corner_cx", "corner_cy", "in_frame", "stage", "hough_n",
              "mark_r_ratio", "circ", "ok",
              "sweep_q", "sweep_pol", "sweep_pass", "sweep_combo", "circ_curve",
              "hole_valid_marks", "corners_in_frame", "feature_b", "hole_passed")


def analyse_image(gray: np.ndarray, res, truth: Optional[bool], verify: bool
                  ) -> Tuple[List[list], List[Tuple[object, List[dict]]]]:
    """跑完一张图的拐角复算, 同时产出 CSV 行与拼图素材(只算一遍)。"""
    head = [res.name, {True: "front", False: "back", None: ""}[truth], res.verdict, res.reason,
            round(res.elapsed_ms, 1), res.locate_method, round(res.pitch_r, 1),
            round(res.hole_r, 2), len(res.holes), len(res.checked)]
    rows: List[list] = []
    per_hole: List[Tuple[object, List[dict]]] = []
    holes = checked_holes(res)
    if not holes:                                             # 早退 NG(找不到工件/孔不足)也要占一行
        rows.append(head + [""] * (len(CSV_HEADER) - len(head)))
        return rows, per_hole
    for hole in holes:
        hmid = [hole.index, round(hole.cx, 2), round(hole.cy, 2), round(hole.contrast, 1),
                hole.ring_count, "|".join("%.3f" % v for v in hole.ring_radii),
                "" if hole.flange_hough_r is None else round(hole.flange_hough_r, 3),
                int(hole.feature_a)]
        tail = [hole.valid_marks, hole.corners_in_frame, int(hole.feature_b), int(hole.passed)]
        anas: List[dict] = []
        for rec in hole.corner_hits:
            check_corner_keys(rec)
            ana = classify_corner(gray, hole, rec, verify)
            anas.append(ana)
            sw = ana["sweep"]
            rows.append(head + hmid + [
                rec["angle"], round(rec["cx"], 2), round(rec["cy"], 2), int(rec["in_frame"]),
                ana["stage"], ana["hough_n"],
                rec["r"] if "mark_r" in rec else "", rec["circ"], int(rec["ok"]),
                "" if sw is None else round(sw["q"], 1) if sw["q"] is not None else "",
                "" if sw is None else ("INV" if sw["flag"] == cv2.THRESH_BINARY_INV else
                                       "BIN" if sw["flag"] is not None else ""),
                "" if sw is None else sw["n_pass"], "" if sw is None else sw["n_combo"],
                "" if sw is None else "|".join("%.3f" % v for v in sw["curve"]),
            ] + tail)
        per_hole.append((hole, anas))
    return rows, per_hole


def is_borderline(res, per_hole) -> bool:
    """边界样本 = 两个阈值任一小幅扰动就会翻转判定的图。

    直接复用 verdict_at 重算, 比"某个孔的压痕数正好等于下限"这类经验规则准得多 ——
    全孔模式下 10 个孔里总会有一个正好卡在下限, 那种规则会把几乎每张图都判成边界。
    """
    item = collect_sweep(res, [h for h, _ in per_hole], None)
    base = res.is_ok
    tc, tm = T.MARK_CIRCULARITY_MIN, T.MIN_VALID_MARKS
    probes = ((tc + BORDERLINE_BAND, tm), (tc - BORDERLINE_BAND, tm),
              (tc, tm + 1), (tc, max(1, tm - 1)))
    return any(verdict_at(item, q, m) != base for q, m in probes)


def sheet_name(res, per_hole, truth: Optional[bool]) -> Tuple[str, bool]:
    """文件名带结论 -> 资源管理器大图标视图就是界面; 同时判是否归入 borderline。"""
    circs = [rec["circ"] for hole, _ in per_hole for rec in hole.corner_hits if rec["in_frame"]]
    max_circ = max(circs) if circs else 0.0
    marks = sum(hole.valid_marks for hole, _ in per_hole)
    aveto = any(hole.feature_b and not hole.feature_a for hole, _ in per_hole)
    border = is_borderline(res, per_hole)
    wrong = truth is not None and truth != res.is_ok
    base = os.path.splitext(os.path.basename(res.name))[0] or "frame"
    base = "".join(ch if ch.isalnum() or ch in "-_#" else "_" for ch in base)[:48]
    name = "%s%s_m%d_c%.2f%s_%s_%s.png" % (
        "WRONG_" if wrong else "", "OK" if res.is_ok else "NG", marks, max_circ,
        "_AVETO" if aveto else "", base, time.strftime("%H%M%S"))
    return name, border


# ---------------------------------------------------------------- 阈值网格 (--sweep)
def collect_sweep(res, holes, truth: Optional[bool]) -> dict:
    """把一张图压成阈值重算所需的最小信息。circ 与 MARK_CIRCULARITY_MIN 无关,
    所以整片 (圆度阈值 x 最少压痕数) 网格都能从一次检测的结果解析式推出, 不必重跑。"""
    return {"truth": truth, "is_ok": res.is_ok, "verdict": res.verdict,
            "holes": [{"feature_a": bool(hole.feature_a),
                       "circs": [rec["circ"] for rec in hole.corner_hits if rec["in_frame"]]}
                      for hole in holes]}


def verdict_at(item: dict, tc: float, tm: int) -> bool:
    """在给定 (圆度阈值, 最少压痕数) 下重算工件判定, 逻辑与主模块一致。"""
    if not item["holes"]:
        return False                                          # 早退 NG: 根本没算到特征
    flags = []
    for h in item["holes"]:
        fb = sum(1 for c in h["circs"] if c > tc) >= tm
        flags.append((h["feature_a"] and fb) if T.HOLE_LOGIC == "AND" else (h["feature_a"] or fb))
    return any(flags) if T.PART_LOGIC == "OR" else all(flags)


def margin_at(item: dict, tc: float, tm: int) -> Optional[int]:
    """离翻判还差几个压痕 —— 成本不对称时真正要看的量, 单元格里的 OK/NG 只是符号。

    判 OK 的样本: 决定性的那个孔还能掉几个压痕才翻 NG (>=0)。
    判 NG 的样本: 最接近的那个孔还要再冒几个误检才翻 OK (>=1)。
    None = 该判定不由压痕数决定(特征A 一票否决 / 早退 NG), 加减压痕都翻不动。
    """
    holes = item["holes"]
    if not holes:
        return None                                           # 早退 NG, 与压痕数无关
    if T.HOLE_LOGIC == "AND":
        usable = [h for h in holes if h["feature_a"]]          # 特征A 否掉的孔, 补压痕也救不回
        if not usable or (T.PART_LOGIC == "AND" and len(usable) < len(holes)):
            return None
    else:
        if any(h["feature_a"] for h in holes):                 # OR: 特征A 单独定调
            return None
        usable = holes
    counts = [sum(1 for c in h["circs"] if c > tc) for h in usable]
    key = max(counts) if T.PART_LOGIC == "OR" else min(counts)  # OR 看最强孔, AND 看最弱孔
    return (key - tm) if verdict_at(item, tc, tm) else (tm - key)


def min_margin(items: List[dict], tc: float, tm: int) -> Optional[int]:
    """一组样本里离翻判最近的那个的裕度。None = 没有一个由压痕数决定。"""
    vals = [m for m in (margin_at(d, tc, tm) for d in items) if m is not None]
    return min(vals) if vals else None


def safe_combos(front: List[dict], back: List[dict]) -> List[tuple]:
    """零逃逸(反面无一判 OK)的 (tc, tm) 组合。

    成本不对称: 正面判 NG 是过杀(可接受), 反面判 OK 是逃逸(不可接受)。所以不是
    "灵敏度+特异性最大", 而是先把有逃逸的组合全部淘汰, 再在剩下的里挑正面通过率最高的。
    排序: 正面通过数 -> 反面裕度 -> 正面裕度, 都是越大越好。
    """
    out = []
    for tc in SWEEP_TC:
        for tm in SWEEP_TM:
            if any(verdict_at(d, tc, tm) for d in back):
                continue                                      # 有逃逸, 直接淘汰
            f_ok = sum(1 for d in front if verdict_at(d, tc, tm))
            bm, fm = min_margin(back, tc, tm), min_margin(front, tc, tm)
            out.append((f_ok, 99 if bm is None else bm, 99 if fm is None else fm, tc, tm, bm, fm))
    out.sort(reverse=True)
    return out


def fmt_margin(m: Optional[int]) -> str:
    return "-" if m is None else str(m)


def sweep_table(data: List[dict]) -> None:
    front = [d for d in data if d["truth"] is True]
    back = [d for d in data if d["truth"] is False]
    unk = [d for d in data if d["truth"] is None]
    print("=" * 108)
    print("[阈值网格] 行 = MARK_CIRCULARITY_MIN, 列 = MIN_VALID_MARKS")
    print("           单元 = 正面判 OK 数/正面总数  反面判 NG 数/反面总数"
          "   (样本: 正面 %d, 反面 %d, 未标注 %d)" % (len(front), len(back), len(unk)))
    print("  circmin |" + "".join("   tm=%-14d" % tm for tm in SWEEP_TM))
    for tc in SWEEP_TC:
        line = "   %.2f   |" % tc
        for tm in SWEEP_TM:
            f_ok = sum(1 for d in front if verdict_at(d, tc, tm))
            b_ng = sum(1 for d in back if not verdict_at(d, tc, tm))
            line += "  %3d/%-3d %3d/%-3d  " % (f_ok, len(front), b_ng, len(back))
        print(line)
    if unk:
        print("  未标注样本的 OK 率(仅供参考, 文件名里没有 正/反/front/back 无法判对错):")
        for tc in SWEEP_TC:
            print("   %.2f   |" % tc + "".join("  %3d/%-3d        "
                                               % (sum(1 for d in unk if verdict_at(d, tc, tm)), len(unk))
                                               for tm in SWEEP_TM))
    if back:
        margin_grid(front, back)
    recommend(front, back)


def margin_grid(front: List[dict], back: List[dict]) -> None:
    """裕度网格: 单元 = 反面离翻成 OK 还差几个压痕 / 正面离翻成 NG 还能掉几个。

    X = 该组合已有逃逸(反面判 OK), 直接淘汰; - = 判定不由压痕数决定(特征A 一票否决)。
    选阈值看的是这张表: 反面裕度 1 意味着任何一个新的误检就翻判, 不管网格上写着几比几。
    """
    print("-" * 108)
    print("[裕度网格] 单元 = 反面裕度/正面裕度 (最接近翻判的那张还差几个压痕; 越大越稳)")
    print("           判 NG 的样本 = 还要再冒几个才翻 OK; 判 OK 的 = 还能掉几个才翻 NG")
    print("           X = 已有反面逃逸(淘汰)   - = 判定与压痕数无关")
    print("  circmin |" + "".join("   tm=%-14d" % tm for tm in SWEEP_TM))
    for tc in SWEEP_TC:
        line = "   %.2f   |" % tc
        for tm in SWEEP_TM:
            if any(verdict_at(d, tc, tm) for d in back):
                line += "  %-16s" % "X"
            else:
                line += "  %-16s" % ("%s / %s" % (fmt_margin(min_margin(back, tc, tm)),
                                                  fmt_margin(min_margin(front, tc, tm))))
        print(line)


def recommend(front: List[dict], back: List[dict]) -> None:
    """按不对称成本给推荐值: 硬约束 反面 0 逃逸, 目标 正面通过率最高。"""
    print("-" * 108)
    if not front or not back:
        print("[推荐] 正面或反面样本缺一类, 无法给推荐值。")
        print("       文件名要带 正/反 (或 front/back), guess_label() 才推得出真值。")
        return
    safe = safe_combos(front, back)
    if not safe:
        print("[推荐] !! 整片网格没有一个组合能做到反面 0 逃逸 —— 靠调这两个阈值救不回来。")
        print("       说明反面误检的圆度落在正面分布之内。下一步只有: 收 MARK_R_RATIO_RANGE")
        print("       上界(见 --sweep-radius)、改判定聚合方式、或先把成像修好压低误检率。")
        return
    print("[推荐] 硬约束 反面 0 逃逸, 目标 正面通过率最高 (过杀可接受, 逃逸不可接受)")
    for f_ok, _, _, tc, tm, bm, fm in safe[:3]:
        print("       MARK_CIRCULARITY_MIN=%.2f  MIN_VALID_MARKS=%d   正面 %d/%d   "
              "反面裕度 %s  正面裕度 %s"
              % (tc, tm, f_ok, len(front), fmt_margin(bm), fmt_margin(fm)))
    top = safe[0]
    if top[0] < len(front):
        print("       注意: 最优组合仍会过杀 %d/%d 张正面。" % (len(front) - top[0], len(front)))
    if top[5] is not None and top[5] <= 1:
        print("       注意: 反面裕度只有 %d —— 一个新的误检压痕就翻判, 不能算 0 误判。" % top[5])
    if top[6] == 0:
        print("       注意: 上面第一档的正面裕度是 0 —— 正面正好卡在阈值上, 掉一个压痕就过杀。")
    # 均衡档只在"正面通过数已达最优"的组合里挑: 正面裕度的含义随判定翻转(判 NG 时它是
    # "还差几个才过"), 不先卡住正面通过数会把正面已经判 NG 的组合当成"两侧都很宽"。
    pool = [t for t in safe if t[0] == top[0]]
    bal = max(pool, key=lambda t: (min(t[1], t[2]), t[1]))     # 取小者最大, 平手看反面裕度
    if bal[3:5] != top[3:5]:
        print("       两侧裕度最均衡的一档(正面没那么贴阈值, 换批样本更抗得住):")
        print("       MARK_CIRCULARITY_MIN=%.2f  MIN_VALID_MARKS=%d   正面 %d/%d   "
              "反面裕度 %s  正面裕度 %s"
              % (bal[3], bal[4], bal[0], len(front), fmt_margin(bal[5]), fmt_margin(bal[6])))
    print("       n=%d 反面零逃逸只能声明逃逸率 < %.0f%% (95%% 置信, 1-0.05^(1/n))"
          % (len(back), 100.0 * (1.0 - 0.05 ** (1.0 / len(back)))))


# ---------------------------------------------------------------- 半径网格 (--sweep-radius)
def total_marks(items: List[dict], tc: float) -> int:
    return sum(sum(1 for c in h["circs"] if c > tc) for d in items for h in d["holes"])


def radius_scan(src, limit: int, uppers: Sequence[float],
                forced: Optional[bool] = None) -> List[Tuple[float, List[dict]]]:
    """按 MARK_R_RATIO_RANGE 上界逐个重跑全部样本。

    圆度阈值与压痕数能解析式扫(circ 与它们无关), 但半径范围是 HoughCircles 的
    minRadius/maxRadius —— 改它会改变 Hough 找到的圆本身(可能在同一个拐角改选一个更小的
    圆), 事后筛已记录的命中只能做减法, 推不出这种改选。所以这一维只能重跑。
    """
    lo = T.MARK_R_RATIO_RANGE[0]
    saved_range, saved_print = T.MARK_R_RATIO_RANGE, T.PRINT_DEBUG
    T.PRINT_DEBUG = False                                      # 6 个上界 x N 张, 逐帧表格太吵
    out: List[Tuple[float, List[dict]]] = []
    try:
        for up in uppers:
            if up <= lo:
                print("[WARN] 跳过上界 %.2f: 不大于下界 %.2f" % (up, lo))
                continue
            T.MARK_R_RATIO_RANGE = (lo, up)
            data: List[dict] = []
            for idx, (name, bgr) in enumerate(src.frames()):
                if limit and idx >= limit:
                    break
                if bgr is None:
                    continue
                res, _ = T.inspect(bgr, name)
                data.append(collect_sweep(res, checked_holes(res),
                                          forced if forced is not None else T.guess_label(name)))
            out.append((up, data))
            print("  上界 %.2f 完成 (%d 张)" % (up, len(data)))
    finally:
        T.MARK_R_RATIO_RANGE, T.PRINT_DEBUG = saved_range, saved_print
    return out


def radius_table(scan: List[Tuple[float, List[dict]]], csv_path: str) -> None:
    """半径上界表: 左半是代价(正面还剩多少压痕), 右半是收益(反面逃逸与裕度)。"""
    tc0, tm0 = T.MARK_CIRCULARITY_MIN, T.MIN_VALID_MARKS
    print("=" * 108)
    print("[半径网格] 行 = MARK_R_RATIO_RANGE 上界 (下界固定 %.2f), 每行重跑一次检测"
          % T.MARK_R_RATIO_RANGE[0])
    print("           现行 = 当前 circmin=%.2f/tm=%d 下的表现; 最佳 = 该上界下零逃逸约束的最优组合"
          % (tc0, tm0))
    print("   上界 | 正面压痕 反面压痕 | 现行:正面OK 反面逃逸 反面裕度 | 最佳零逃逸组合      正面OK 反面裕度")
    recs = []
    for up, data in scan:
        front = [d for d in data if d["truth"] is True]
        back = [d for d in data if d["truth"] is False]
        f_ok = sum(1 for d in front if verdict_at(d, tc0, tm0))
        esc = sum(1 for d in back if verdict_at(d, tc0, tm0))
        bm = min_margin(back, tc0, tm0) if back else None
        best = safe_combos(front, back) if (front and back) else []
        if best:
            b_fok, _, _, btc, btm, b_bm, _ = best[0]
            btxt = "circmin=%.2f tm=%d" % (btc, btm)
            bcol = "%3d/%-3d  %s" % (b_fok, len(front), fmt_margin(b_bm))
        else:
            btc = btm = b_fok = b_bm = None
            btxt, bcol = "(缺正面或反面样本)", "  -       -"
        print("   %.2f | %6d   %6d   | %6d/%-3d %6d    %6s   | %-20s %s"
              % (up, total_marks(front, tc0), total_marks(back, tc0),
                 f_ok, len(front), esc, fmt_margin(bm), btxt, bcol))
        recs.append([up, len(front), len(back), total_marks(front, tc0), total_marks(back, tc0),
                     f_ok, esc, fmt_margin(bm),
                     "" if btc is None else btc, "" if btm is None else btm,
                     "" if b_fok is None else b_fok, fmt_margin(b_bm)])
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(("r_upper", "n_front", "n_back", "front_marks", "back_marks",
                    "cur_front_ok", "cur_back_escape", "cur_back_margin",
                    "best_circmin", "best_minmarks", "best_front_ok", "best_back_margin"))
        w.writerows(recs)
    print("[输出] 半径网格 CSV  %s" % csv_path)
    if not any(d["truth"] is False for _, data in scan for d in data):
        print("       没有反面样本, 右半张表是空的。左半的\"正面压痕\"仍然有用: 它是收紧上界的代价,")
        print("       掉得快说明正面压痕本来就靠大半径命中撑着, 收紧会连正面一起杀。")


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="thrust_cage_flange_inspect 外挂调试报告 (不改原文件)")
    ap.add_argument("--dir", default=None,
                    help="样本目录, 默认 %s" % default_image_dir())
    ap.add_argument("--out", default=None,
                    help="输出根目录, 默认 %s" % default_out_dir())
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张")
    ap.add_argument("--holes", type=int, default=0,
                    help="参与判定的孔数, 0=全部孔(默认)。全孔约 40 个拐角样本/张, 2 孔只有 8 个")
    ap.add_argument("--no-sheet", action="store_true", help="只出 CSV, 不画拼图(快)")
    ap.add_argument("--sweep", action="store_true", help="末尾追加 (圆度阈值 x 最少压痕数) 网格表")
    ap.add_argument("--sweep-radius", dest="sweep_r", nargs="?", const="", default=None,
                    metavar="LIST",
                    help="末尾追加 MARK_R_RATIO_RANGE 上界扫描(每个值重跑一次检测)。"
                         "可给逗号分隔的值, 默认 %s"
                         % ",".join("%.2f" % v for v in SWEEP_R_UPPER))
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过复刻分割与模块记录值的等值断言(不建议)")
    ap.add_argument("--truth", choices=("front", "back"), default=None,
                    help="强制整批样本的真值, 用于按类分文件夹存放、文件名是相机时间戳的情况; "
                         "默认由 guess_label() 从文件名推断")
    ap.add_argument("--print", dest="show", action="store_true", help="同时打印主模块的逐帧表格")
    return ap


def summarise(n_ok: int, n_ng: int, n_bad: int, hit: int, miss: int,
              stages: Dict[str, int], n_ver: int, csv_path: str, sheet_dir: str,
              n_sheet: int, n_border: int, t0: float) -> None:
    print("=" * 108)
    print("[汇总] 共 %d 帧: OK=%d NG=%d 无效=%d   耗时 %.1f s"
          % (n_ok + n_ng, n_ok, n_ng, n_bad, time.time() - t0))
    if hit + miss:
        print("[自检] 文件名可推断真值 %d 张: 一致 %d, 不一致 %d (准确率 %.1f%%)"
              % (hit + miss, hit, miss, 100.0 * hit / (hit + miss)))
    total = sum(stages.values())
    if total:
        order = ("ok", "low_circ", "seg_miss", "gate_miss", "hough_miss", "oof")
        print("[拐角] 共 %d 个: %s" % (total, "  ".join(
            "%s=%d(%.0f%%)" % (k, stages[k], 100.0 * stages[k] / total)
            for k in order if stages.get(k))))
        in_f = total - stages.get("oof", 0)
        if in_f:
            print("       画面内 %d 个, 有效压痕率 %.1f%%   (失败最多的一档就是该调的参数, 见文件头注释)"
                  % (in_f, 100.0 * stages.get("ok", 0) / in_f))
    if n_ver:
        print("[保真] %d 个拐角的复刻分割与模块记录值逐一核对一致" % n_ver)
    print("[输出] CSV  %s" % csv_path)
    if n_sheet:
        print("       拼图 %s  (%d 张, 其中 %d 张进 borderline/)" % (sheet_dir, n_sheet, n_border))


def main(argv: Optional[Sequence[str]] = None) -> int:
    check_contract()
    T.setup_console()
    args = build_parser().parse_args(argv)
    T.HOLE_CHECK_COUNT = args.holes                            # 模块内是运行时读全局, 外部赋值即生效
    T.PRINT_DEBUG = bool(args.show)

    img_dir = args.dir or default_image_dir()
    if not os.path.isdir(img_dir):
        print("[FATAL] 样本目录不存在: %s" % img_dir)
        print('        用 --dir "路径" 指定(含空格或中文要加引号), 或改 dbg_report.py 顶部的 IMAGE_DIR。')
        return 2
    src = T.LocalFolderSource(img_dir, getattr(T, "LOCAL_RECURSIVE", True))
    if not len(src):
        print("[FATAL] 目录下没有图片: %s" % img_dir)
        print("        支持的扩展名: %s (含子目录)" % " ".join(sorted(T.IMAGE_EXTS)))
        return 2

    root = args.out or default_out_dir()
    sheet_dir = os.path.join(root, "sheet")
    border_dir = os.path.join(sheet_dir, "borderline")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(root, "dbg_%s.csv" % stamp)
    os.makedirs(root, exist_ok=True)

    print("[INFO] 样本 %d 张  目录 %s" % (len(src), img_dir))
    print("[INFO] 输出根目录 %s" % os.path.abspath(root))
    print("[INFO] 孔数模式 %s   圆度阈值 %.2f   最少压痕 %d   保真核对 %s"
          % ("全部孔" if args.holes <= 0 else "前 %d 孔" % args.holes,
             T.MARK_CIRCULARITY_MIN, T.MIN_VALID_MARKS, "关" if args.no_verify else "开"))
    forced = {"front": True, "back": False}.get(args.truth)
    if forced is not None:
        print("[INFO] 真值强制为 %s (--truth %s), 不看文件名"
              % ("正面" if forced else "反面", args.truth))

    n_ok = n_ng = n_bad = hit = miss = n_ver = n_sheet = n_border = 0
    stages: Dict[str, int] = {}
    sweep_data: List[dict] = []
    t0 = time.time()
    fh = open(csv_path, "w", newline="", encoding="utf-8-sig")   # BOM: 中文 Excel 直接双击能开
    writer = csv.writer(fh)
    writer.writerow(CSV_HEADER)
    try:
        for idx, (name, bgr) in enumerate(src.frames()):
            if args.limit and idx >= args.limit:
                break
            if bgr is None:
                n_bad += 1
                print("[WARN] 帧无效: %s" % name)
                continue
            res, proc = T.inspect(bgr, name)
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY) if proc.ndim == 3 else proc.copy()
            truth = forced if forced is not None else T.guess_label(name)
            try:
                rows, per_hole = analyse_image(gray, res, truth, not args.no_verify)
            except VerifyError as exc:
                print("[FATAL] 保真核对失败 %s: %s" % (name, exc))
                print("        本工具复刻的多阈值分割已与主模块实现分叉, 请同步 sweep_circularity()")
                return 4
            writer.writerows(rows)
            fh.flush()
            for row in rows:
                st = row[CSV_HEADER.index("stage")]
                if st:
                    stages[st] = stages.get(st, 0) + 1
                if row[CSV_HEADER.index("sweep_pass")] != "" and not args.no_verify:
                    n_ver += 1
            if args.sweep:
                sweep_data.append(collect_sweep(res, [h for h, _ in per_hole], truth))
            if not args.no_sheet:
                fname, border = sheet_name(res, per_hole, truth)
                path = os.path.join(border_dir if border else sheet_dir, fname)
                if T.imwrite_unicode(path, build_sheet(gray, res, per_hole, truth)):
                    n_sheet += 1
                    n_border += int(border)
            n_ok += res.is_ok
            n_ng += (not res.is_ok)
            if truth is not None:
                hit += (truth == res.is_ok)
                miss += (truth != res.is_ok)
            print("  %-52s %-20s marks=%-3d %6.1fms" % (
                ascii_safe(os.path.basename(name))[:52], res.verdict,
                sum(h.valid_marks for h, _ in per_hole), res.elapsed_ms))
    finally:
        fh.close()
    summarise(n_ok, n_ng, n_bad, hit, miss, stages, n_ver, csv_path, sheet_dir,
              n_sheet, n_border, t0)
    if args.sweep:
        sweep_table(sweep_data)
    if args.sweep_r is not None:
        uppers = SWEEP_R_UPPER
        if args.sweep_r.strip():
            try:
                uppers = tuple(sorted((float(v) for v in args.sweep_r.split(",") if v.strip()),
                                      reverse=True))
            except ValueError:
                print("[FATAL] --sweep-radius 只接受逗号分隔的数字, 例: 0.55,0.50,0.45")
                return 2
            if not uppers:
                print("[FATAL] --sweep-radius 的列表是空的")
                return 2
        n_img = min(len(src), args.limit) if args.limit else len(src)
        print("=" * 108)
        print("[半径扫描] %d 个上界 x %d 张, 每个上界重跑一次检测 (约 %.0f s)"
              % (len(uppers), n_img, len(uppers) * n_img * 0.27))
        radius_table(radius_scan(src, args.limit, uppers, forced),
                     os.path.join(root, "dbg_radius_%s.csv" % stamp))
    return 0


if __name__ == "__main__":
    sys.exit(main())

