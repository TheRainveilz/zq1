# dbg_report —— 外挂调试报告工具

`thrust_cage_flange_inspect.py` 的调试外挂：**不修改主算法文件**，通过 import 复用它的函数与
阈值，产出定阈值用的 CSV 与排查用的固定版式拼图。

主算法是交付件，不往里塞调试代码；工具本身可以随意改、随意扔，不牵连算法的回归。

## 设计取舍

| 取舍 | 原因 |
|---|---|
| 外挂而非改主文件 | 主算法已提交并写进 README，调试代码进去要跟着一起评审、一起回归 |
| 不做 GUI | 定阈值要的是"一屏看完 50 张"和可回溯的表，不是一次看一张的界面；PNG 落盘能贴进报告、能与上一版并排 diff |
| 阈值改模块全局变量 | 扫描不用编辑文件，git 上零噪声（模块内是运行时读全局，外部赋值即生效） |
| 只支持本地目录 | 生产路径（HTTP 无限循环）绝不挂调试，一帧一张 PNG 会写满硬盘 |

## 环境

与主算法相同，无额外依赖：

```bash
pip install opencv-python numpy
```

仓库里的 `.venv` **没有装 cv2**，用系统 Python 跑：`D:\Python\python.exe dbg_report.py`。

## 用法

```bash
python dbg_report.py                        # 全孔模式, 出 CSV + 拼图
python dbg_report.py --holes 2              # 按生产配置(HOLE_CHECK_COUNT=2)复现
python dbg_report.py --sweep                # 末尾追加阈值网格表
python dbg_report.py --no-sheet             # 只出 CSV(快)
python dbg_report.py --dir D:\反面样本 --limit 20
```

| 参数 | 说明 |
|---|---|
| `--dir DIR` | 样本目录，默认取主模块 `LOCAL_IMAGE_DIR` |
| `--out DIR` | 输出根目录，默认 `NG_SAVE_DIR` 的父目录（`result/`） |
| `--limit N` | 只处理前 N 张 |
| `--holes N` | 参与判定的孔数，`0` = 全部孔（默认）。全孔约 40 个拐角样本/张，2 孔只有 8 个 |
| `--no-sheet` | 只出 CSV，不画拼图 |
| `--sweep` | 追加 (圆度阈值 × 最少压痕数) 网格表 |
| `--no-verify` | 跳过保真断言（不建议） |
| `--print` | 同时打印主模块的逐帧表格 |

**定阈值时用默认的全孔模式。** 生产用 2 孔是为了节拍（154 ms vs 267 ms），离线统计不在乎这个，
而样本量差 5 倍。

## 输出

### CSV `result/dbg_<时间戳>.csv`

一行一个拐角，36 列，UTF-8 BOM（中文 Excel 双击可开）。字段分五组：

| 组 | 字段 |
|---|---|
| 图级 | `image truth verdict reason elapsed_ms locate pitch_r hole_r n_holes n_checked` |
| 孔级 | `hole_index hole_cx hole_cy contrast ring_count ring_radii flange_r feature_a` |
| 拐角级 | `corner_angle corner_cx corner_cy in_frame stage hough_n mark_r_ratio circ ok` |
| 分割细节 | `sweep_q sweep_pol sweep_pass sweep_combo circ_curve` |
| 孔级结论 | `hole_valid_marks corners_in_frame feature_b hole_passed` |

关键字段：

- `circ` —— 圆度。**与 `MARK_CIRCULARITY_MIN` 无关**，所以能事后重算任意阈值下的判定
- `sweep_pass` / `sweep_combo` —— 16 个（百分位 × 极性）组合里有几个过了圆度门，即**阈值裕度**。
  只有 1/16 说明该压痕靠某一个特定阈值勉强挤过去，不稳
- `circ_curve` —— 8 个百分位上的圆度（`|` 分隔），用来分辨"所有阈值都不圆"与"只在某一档够"
- `hough_n` —— 仅在失败拐角上填（成功的不重跑 Hough），留空表示未测而非 0 个候选
- `truth` —— 由主模块 `guess_label()` 从文件名推断（正/OK/front → front，反/NG/back → back），
  推不出留空

定阈值的做法：按 `truth` 分组画 `circ` 直方图，把 `MARK_CIRCULARITY_MIN` 放进正反两组分布之间的
空隙；按 `hole_valid_marks` 分组定 `MIN_VALID_MARKS`。

### 拼图 `result/sheet/*.png`

一行一个受检孔，固定 200 px 单元格，10 列共 2000 px 宽：

```
[上下文][特征A][角1 窗口|诊断][角2 窗口|诊断][角3 窗口|诊断][角4 窗口|诊断]
```

- **上下文**：裁切半宽 2.9 r。拐角中心最远 1.79 r，加方框半对角 0.75×√2 = 1.06 r，共 2.85 r；
  早期调试图用 2.3 r 会把四个框切掉。配色沿用主模块 `draw_overlay`：绿/红 = 孔通过/未通过，
  青 = 孔 ROI，品红 = 命中同心环，橄榄 = 翻边 Hough 圆，绿/红/灰方框 = 拐角窗口
- **特征A**：孔 ROI（过 `local_enhance`）+ 命中环 + 掩膜边界 + 翻边圆
- **拐角窗口**：检测器实际看到的那一份（过 `local_smooth`）+ Hough 圆 + 赢的轮廓 + 蓝色中心门
- **拐角诊断**：按 stage 给最能指向参数的中间图，见下节
- **顶部信息条**：判定、真值、定位方法、耗时，以及**当次生效的全部阈值** —— 隔一周回看才知道
  这张图是哪个参数版本出的

### 文件名与分流

```
result/sheet/OK_m30_c0.89_1_4256379_151228.png
result/sheet/borderline/NG_m1_c0.77_AVETO_xxx_151304.png
```

判定打头（名称排序自然分组），`m` = 压痕总数，`c` = 最大圆度。资源管理器切大图标视图就是界面。

- `WRONG_` 前缀 —— 判定与文件名推断的真值不一致
- `_AVETO` —— 特征B 全过但特征A 一票否决。`HOLE_LOGIC="AND"` 会让 A 否掉正面件，这类漏判最难查
- `borderline/` —— **圆度阈值 ±0.07 或最少压痕数 ±1，任一扰动就会翻转判定**的图。用
  `verdict_at()` 实算，比"某个孔压痕数正好等于下限"那种经验规则准得多：全孔模式下 10 个孔里
  总有一个卡在下限，经验规则会把几乎每张图都归为边界

真正需要细看的只有 `borderline/` 和 `WRONG_`，其余扫缩略图即可。

## stage 六分类

主算法把三种不同的失败都记成 `circ=0.00`，本工具拆开（对失败的拐角补跑一次同参数 Hough；
Hough 是确定性的，同输入结果必然与检测时一致）：

| stage | 含义 | 诊断格显示 | 该调的参数 |
|---|---|---|---|
| `ok` | 有效压痕 | 赢的二值掩膜 + 裕度 | — |
| `low_circ` | 分割到了但圆度不够 | 同上 | `MARK_CIRCULARITY_MIN` |
| `seg_miss` | 圆找到了但无连通域过筛 | 同上 | `MARK_MIN_AREA_RATIO`、`MARK_MASK_RATIO` |
| `gate_miss` | 找到圆但全在中心门外 | 被挡掉的 Hough 圆 + 中心门 | `MARK_CENTER_GATE`、`--calib` 重标 `CORNER_SPEC` |
| `hough_miss` | Hough 一个圆都没找到 | Canny 边缘 | `MARK_HOUGH_P2`、`MARK_R_RATIO_RANGE`；没边则是照明 |
| `oof` | 拐角出画面，不参与计数 | 灰底 | 取景 |

`hough_miss` 的诊断格画 Canny 边缘（`HoughCircles` 内部用的就是 `p1` 与 `p1/2`），
能分清"根本没边"（对比度/照明问题）与"有边但不成圆"（`p2` 偏高）。

## 保真机制

调试图与检测器不一致，是这类工具最容易犯也最难发现的错。两道保险：

1. **面板一律调模块自己的** `crop_pad` / `local_smooth` / `local_enhance`，不自己重写裁切与增强。
   拐角必须用 `local_smooth`（主算法注释写明拐角窗口做 CLAHE 会把台面机加工纹理放大成假圆），
   特征A 必须用 `local_enhance`
2. **唯一复刻的是** `_best_mark_circularity` 的多阈值循环（模块只返回标量，赢的掩膜在函数里就丢了），
   逐拐角断言复刻结果 `== rec["circ"]`，不一致立即退出（exit 4）。这比"再调一次该函数"更强：
   同时校验了窗口复现与 Hough 圆坐标的反推

第 2 条在首次运行就抓到一个真 bug，记录在此以免重踩：`HoughCircles` 返回的圆心恒为 `x.5`，
而从 `mark_cx - cx + half` 反推会带约 1e-13 的浮点余差；Python `round()` 是银行家舍入
（`round(36.5)=36` 但 `round(36.5+1e-13)=37`），掩膜中心偏 1 px 就会选出不同的连通域
（0.877 vs 0.875）。修法是把反推值 snap 回 float32 —— 原值本来就来自 float32，
半个 ulp 约 2e-6，远大于余差。

## 阈值网格 `--sweep`

`circ` 与 `MARK_CIRCULARITY_MIN` 无关，所以整片（圆度阈值 × 最少压痕数）网格都能从一次检测的
结果解析式推出，不必重跑检测。`verdict_at()` 按主模块同样的 `HOLE_LOGIC` / `PART_LOGIC` 重算
工件判定，早退 NG（找不到工件/圆孔不足）恒为 NG。

单元格 = 正面判 OK 数/正面总数、反面判 NG 数/反面总数，即灵敏度与特异性。两类样本都有时会给出
得分最高的三个组合；缺一类时明确提示无法给推荐值。

## 基准图实测（正面 1 张，全孔模式）

| 项目 | 结果 |
|---|---|
| 拐角总数 | 40（10 孔 × 4） |
| stage 分布 | `ok` 30、`hough_miss` 6、`oof` 4 |
| 画面内有效压痕率 | 83.3%（30/36） |
| 圆度分布 | 0.77 ~ 0.89（阈值 0.75，最弱一个余量仅 0.02） |
| 保真核对 | 30 个拐角逐一一致 |

失败模式只有 `hough_miss` 一种，`gate_miss` / `seg_miss` / `low_circ` 全为 0 —— 瓶颈不在圆度门
也不在中心门，而在压痕没形成足够边缘，指向照明而非算法。

按 Hough 命中半径分组，能看出一个特异性隐患：

| 压痕半径比 | 个数 | 圆度中位 | 阈值裕度中位 | 裕度 ≤ 1/16 |
|---|---|---|---|---|
| < 0.45 r | 19 | 0.873 | 4/16 | 0 个 |
| ≥ 0.45 r | 11 | 0.821 | 2/16 | 5 个 |

`MARK_R_RATIO_RANGE = (0.28, 0.55)` 跨了近 2 倍，Hough 会锁到比压痕更大的结构（翻边边缘之类）上，
这些命中的圆度更低、只在 1~2 个阈值组合上勉强过门。**这正是反面最可能产生误判 OK 的机制。**
反面样本到手后先看这批大半径命中的表现，很可能要把半径范围收到标定中位数（0.34）附近。

## 与主算法的耦合

工具刻意依赖主模块的内部实现，包括带下划线的 `_best_mark_circularity`。启动时做契约检查
（函数名、常量名、`CORNER_SPEC` 长度、`corner_hits` 键名），缺一个就退出 —— 宁可启动就炸，
也不要画出位置错误的框还看着挺像。

主算法重构后需要同步的地方：

| 主算法改动 | 需同步 |
|---|---|
| `feature_b_corner_marks` 改了 `corner_hits` 键名或坐标语义 | `classify_corner()` |
| `_best_mark_circularity` 改了分割逻辑 | `sweep_circularity()`（保真断言会先报错） |
| 新增阈值常量想印到拼图上 | `header_band()` |

## 已知限制

1. 上述实测数字全部来自 1 张正面样本，`--sweep` 网格现在只能显示 `0/0`。反面样本到手后跑一次
   即可读出推荐阈值
2. 只支持本地目录，不接 HTTP —— 有意为之，生产路径不挂调试
3. 全孔 267 ms/张、2 孔 154 ms/张，拼图另加约 0.2 s/张。批量跑几十张无所谓，别在产线节拍里开
4. 拼图文字全为 ASCII（`cv2.putText` 画不了中文），中文图名在图上显示为 `?`，CSV 里保留原名

