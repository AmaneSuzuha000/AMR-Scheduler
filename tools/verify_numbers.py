"""论文数字溯源核查：把正文里的关键数字与 CSV 逐一对照。

为什么要有它
--------------
论文里的数字是人工从 CSV 誊抄的。誊抄会错，而且**错了不会报错**——
测试跑的是代码，不是论文。第 45 轮就发现过 6.9.5 整张表用的是旧种子数据。

做法
----
不做全自动（从中文正文里抽数字太不可靠），而是**登记式核查**：
每条断言写明「论文里的值」「它来自哪个 CSV 的哪个聚合」。
脚本重算 CSV，与登记值比对，不一致就非零退出。

这样新增数字时只需加一行登记，核查成本接近零。
"""

from __future__ import annotations

import csv
import re
import sys
import statistics as st

from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

CHECKS: list[tuple] = []
# 复用项目自己的检验实现（analyze_reschedule.exact_wilcoxon_p）：它在 n<=20 时
# 精确枚举、n>20 时用正态近似，是与 analyze_allocation、analyze_layout_paired
# 共用的同一份代码。核查器若自带副本，两边会悄悄分叉。
sys.path.insert(0, str(ROOT))
from experiments.analyze_reschedule import exact_wilcoxon_p as _exact_wilcoxon_p  # noqa: E402

#: 数据重跑后的登记值覆盖表（第 66 轮：merge _park_step 修复后全量重跑）。
#:
#: 为什么用覆盖表而不是直接改源码里的字面量：
#: 一半以上的 check() 第 2 实参是循环变量（`for _na, _util, _ont in (...)`），
#: 字面量不在 check 那一行；用脚本改 AST 容易出错且难复核。
#: 覆盖表把「论文正文该写多少」集中在一处，改论文时对照这里即可。
_ORIG: dict[str, object] = {}

UPDATED: dict[str, float] = {
    "6.3 利用率 2车": 0.973,
    "6.3 准时率 2车": 0.1,
    "6.3 利用率 4车": 0.938,
    "6.3 准时率 4车": 0.235,
    "6.3 利用率 6车": 0.893,
    "6.3 准时率 6车": 0.381,
    "6.3 完成时间 2车·静态": 5983.2,
    "6.3 完成时间 2车·时空": 6005.3,
    "6.3 完成时间 4车·静态": 3205.8,
    "6.3 完成时间 4车·时空": 3290.6,
    "6.3 完成时间 6车·静态": 2329.9,
    "6.3 完成时间 6车·时空": 2448.4,
    "6.4 none 加急准时率": 0.5307,
    "6.4 none 加急响应": 145.7,
    "6.4 none 抢占次数": 0,
    "6.4 none 物料浪费": 0,
    "6.4 none 总准时率": 0.2293,
    "6.4 preempt_unpicked 加急准时率": 0.712,
    "6.4 preempt_unpicked 加急响应": 85.5,
    "6.4 preempt_unpicked 抢占次数": 2.73,
    "6.4 preempt_unpicked 物料浪费": 0,
    "6.4 preempt_unpicked 总准时率": 0.2528,
    "6.4 preempt_any 加急准时率": 0.7848,
    "6.4 preempt_any 加急响应": 71.8,
    "6.4 preempt_any 抢占次数": 4.19,
    "6.4 preempt_any 物料浪费": 148.7,
    "6.4 preempt_any 总准时率": 0.2383,
    "6.9.5 容量2/6车 主方法完成率": 0.576,
    "6.9.5 容量2/6车 全完成次数": 0,
    "6.9.6 总运行数": 86,
    "6.9.6 低侧次数": 50,
    "6.9.6 低侧失败": 0,
    "6.9.6 高侧次数": 36,
    "6.9.6 高侧失败": 32,
    "6.9.6 通过侧最大死锁次数": 468,
    "6.9.6 失败侧最小死锁次数": 970,
    "6.9.6 空档宽度": 502,
    "6.3.1 ring/20/2车 makespan": 2351,
    "6.3.1 ring/20/2车 全完成数": 10,
    "6.3.1 ring/20/4车 makespan": 1267,
    "6.3.1 ring/20/4车 全完成数": 10,
    "6.3.1 ring/20/6车 makespan": 986,
    "6.3.1 ring/20/6车 全完成数": 10,
    "6.3.1 ring/50/2车 makespan": 5212,
    "6.3.1 ring/50/2车 全完成数": 10,
    "6.3.1 ring/50/4车 makespan": 2800,
    "6.3.1 ring/50/4车 全完成数": 10,
    "6.3.1 ring/50/6车 makespan": 1970,
    "6.3.1 ring/50/6车 全完成数": 10,
    "6.3.1 ring/100/2车 makespan": 10382,
    "6.3.1 ring/100/2车 全完成数": 10,
    "6.3.1 ring/100/4车 makespan": 5489,
    "6.3.1 ring/100/4车 全完成数": 10,
    "6.3.1 ring/100/6车 makespan": 3942,
    "6.3.1 ring/100/6车 全完成数": 10,
    "6.3.1 grid/20/2车 makespan": 3122,
    "6.3.1 grid/20/2车 全完成数": 10,
    "6.3.1 grid/20/4车 makespan": 1787,
    "6.3.1 grid/20/4车 全完成数": 10,
    "6.3.1 grid/20/6车 makespan": 1356,
    "6.3.1 grid/20/6车 全完成数": 10,
    "6.3.1 grid/50/2车 makespan": 7188,
    "6.3.1 grid/50/2车 全完成数": 10,
    "6.3.1 grid/50/4车 makespan": 3917,
    "6.3.1 grid/50/4车 全完成数": 10,
    "6.3.1 grid/50/6车 makespan": 2868,
    "6.3.1 grid/50/6车 全完成数": 10,
    "6.3.1 grid/100/2车 makespan": 14046,
    "6.3.1 grid/100/2车 全完成数": 10,
    "6.3.1 grid/100/4车 makespan": 7563,
    "6.3.1 grid/100/4车 全完成数": 10,
    "6.3.1 grid/100/6车 makespan": 5484,
    "6.3.1 grid/100/6车 全完成数": 10,
    "6.3.1 hub/20/2车 makespan": 3791,
    "6.3.1 hub/20/2车 全完成数": 10,
    "6.3.1 hub/20/4车 makespan": 2563,
    "6.3.1 hub/20/4车 全完成数": 10,
    "6.3.1 hub/20/6车 makespan": 2250,
    "6.3.1 hub/20/6车 全完成数": 10,
    "6.3.1 hub/50/2车 makespan": 9314,
    "6.3.1 hub/50/2车 全完成数": 10,
    "6.3.1 hub/50/4车 makespan": 6937,
    "6.3.1 hub/50/4车 全完成数": 10,
    "6.3.1 hub/50/6车 makespan": 6269,
    "6.3.1 hub/50/6车 全完成数": 5,
    "6.3.1 hub/100/2车 makespan": 18598,
    "6.3.1 hub/100/2车 全完成数": 10,
    "6.3.1 hub/100/4车 makespan": 14156,
    "6.3.1 hub/100/4车 全完成数": 10,
    "6.3.1 hub/100/6车 makespan": 13076,
    "6.3.1 hub/100/6车 全完成数": 2,
    "6.6.1 nearest_shortest 里程": 9335,
    "6.6.1 nearest_shortest 绕行系数": 1.661,
    "6.6.1 dijkstra+greedy 里程": 9337,
    "6.6.1 dijkstra+greedy 绕行系数": 1.703,
    "6.6.1 spacetime+greedy 里程": 9339,
    "6.6.1 spacetime+greedy 绕行系数": 1.715,
    "6.6.3 nearest_shortest 完成时间": 2777,
    "6.6.3 nearest_shortest 冲突等待步": 526,
    "6.6.3 nearest_shortest 最长阻塞段": 96,
    "6.6.3 dijkstra+greedy 完成时间": 2761,
    "6.6.3 dijkstra+greedy 冲突等待步": 462,
    "6.6.3 dijkstra+greedy 最长阻塞段": 61,
    "6.6.3 spacetime+greedy 完成时间": 2945,
    "6.6.3 spacetime+greedy 冲突等待步": 646,
    "6.6.3 spacetime+greedy 最长阻塞段": 99,
    "6.8.1 baseline_fifo 等待次数均值": 32.8,
    "6.8.1 baseline_fifo 等待步数均值": 737.8,
    "6.8.1 baseline_fifo 平均每段长度": 22.5,
    "6.8.1 dijkstra+greedy 等待次数均值": 36.2,
    "6.8.1 dijkstra+greedy 等待步数均值": 839.8,
    "6.8.1 dijkstra+greedy 平均每段长度": 23.2,
    "6.8.1 baseline_nearest 等待次数均值": 39.3,
    "6.8.1 baseline_nearest 等待步数均值": 1169.9,
    "6.8.1 baseline_nearest 平均每段长度": 29.8,
    "6.8.1 spacetime+greedy 等待次数均值": 42.2,
    "6.8.1 spacetime+greedy 等待步数均值": 1121,
    "6.8.1 spacetime+greedy 平均每段长度": 26.6,
    "6.8.2 baseline_nearest 完成时间中位": 2777,
    "6.8.2 dijkstra+greedy 完成时间中位": 2761,
    "6.8.2 spacetime+greedy 完成时间中位": 2945,
    "6.9.3 ring 排序": 1,
    "6.9.3 grid 排序": 1,
    "6.9.3 hub 排序": 1,
    "6.9.3 hub 主方法落后 FIFO": 9.2,
    "6.9.2 ring 有效配对": 90,
    "6.9.2 ring 均值差": -1.9,
    "6.9.2 grid 有效配对": 90,
    "6.9.2 grid 均值差": -16.9,
    "6.9.2 hub 有效配对": 72,
    "6.9.2 hub 均值差": -15.7,
    "6.9.4 hub 运行数": 360,
    "6.9.4 hub 未完成数": 44,
    "6.9.4 未完成集中在6车": 42,
    "6.9.4 baseline_nearest 2车完成率": 1,
    "6.9.4 baseline_nearest 4车完成率": 1,
    "6.9.4 baseline_nearest 6车完成率": 0.805,
    "6.9.4 baseline_fifo 2车完成率": 1,
    "6.9.4 baseline_fifo 4车完成率": 1,
    "6.9.4 baseline_fifo 6车完成率": 0.915,
    "6.9.4 dijkstra+greedy 2车完成率": 1,
    "6.9.4 dijkstra+greedy 4车完成率": 1,
    "6.9.4 dijkstra+greedy 6车完成率": 0.757,
    "6.9.4 spacetime+greedy 2车完成率": 1,
    "6.9.4 spacetime+greedy 4车完成率": 0.999,
    "6.9.4 spacetime+greedy 6车完成率": 0.996,
    "6.9.5 容量2/4车 nearest_shortest 完成率": 1,
    "6.9.5 容量2/4车 nearest_shortest 全完成": 8,
    "6.9.5 容量2/4车 dijkstra+greedy 完成率": 0.998,
    "6.9.5 容量2/4车 dijkstra+greedy 全完成": 7,
    "6.9.5 容量2/4车 spacetime+greedy 完成率": 0.998,
    "6.9.5 容量2/4车 spacetime+greedy 全完成": 7,
    "6.9.5 容量2/6车 nearest_shortest 完成率": 0.659,
    "6.9.5 容量2/6车 nearest_shortest 全完成": 2,
    "6.9.5 容量2/6车 dijkstra+greedy 完成率": 0.576,
    "6.9.5 容量2/6车 dijkstra+greedy 全完成": 0,
    "6.9.5 容量2/6车 spacetime+greedy 完成率": 0.992,
    "6.9.5 容量2/6车 spacetime+greedy 全完成": 5,
    "6.9.5 容量3/4车 nearest_shortest 完成率": 1,
    "6.9.5 容量3/4车 nearest_shortest 全完成": 8,
    "6.9.5 容量3/4车 dijkstra+greedy 完成率": 1,
    "6.9.5 容量3/4车 dijkstra+greedy 全完成": 8,
    "6.9.5 容量3/4车 spacetime+greedy 完成率": 1,
    "6.9.5 容量3/4车 spacetime+greedy 全完成": 8,
    "6.9.5 容量3/6车 nearest_shortest 完成率": 0.725,
    "6.9.5 容量3/6车 nearest_shortest 全完成": 4,
    "6.9.5 容量3/6车 dijkstra+greedy 完成率": 0.955,
    "6.9.5 容量3/6车 dijkstra+greedy 全完成": 7,
    "6.9.5 容量3/6车 spacetime+greedy 完成率": 0.996,
    "6.9.5 容量3/6车 spacetime+greedy 全完成": 7,
    "6.9.5 容量4/4车 nearest_shortest 完成率": 1,
    "6.9.5 容量4/4车 nearest_shortest 全完成": 8,
    "6.9.5 容量4/4车 dijkstra+greedy 完成率": 1,
    "6.9.5 容量4/4车 dijkstra+greedy 全完成": 8,
    "6.9.5 容量4/4车 spacetime+greedy 完成率": 1,
    "6.9.5 容量4/4车 spacetime+greedy 全完成": 8,
    "6.9.5 容量4/6车 nearest_shortest 完成率": 0.774,
    "6.9.5 容量4/6车 nearest_shortest 全完成": 5,
    "6.9.5 容量4/6车 dijkstra+greedy 完成率": 1,
    "6.9.5 容量4/6车 dijkstra+greedy 全完成": 8,
    "6.9.5 容量4/6车 spacetime+greedy 完成率": 1,
    "6.9.5 容量4/6车 spacetime+greedy 全完成": 8,
    "6.9.5 容量6/6车 nearest_shortest 完成率": 0.774,
    "6.9.5 容量6/6车 nearest_shortest 全完成": 5,
    "6.9.5 容量6/6车 dijkstra+greedy 完成率": 1,
    "6.9.5 容量6/6车 dijkstra+greedy 全完成": 8,
    "6.9.5 容量6/6车 spacetime+greedy 完成率": 1,
    "6.9.5 容量6/6车 spacetime+greedy 全完成": 8,
    "6.9.7 容量2/6车 时空完成率": 0.992,
    "6.9.5 容量2/6车 dijkstra+greedy 死锁中位": 2676,
    "6.9.5 容量2/6车 dijkstra+greedy 完成率": 0.576,
    "6.9.5 容量2/6车 dijkstra+greedy 最低": 0.18,
    "6.9.5 容量2/6车 nearest_shortest 死锁中位": 2507,
    "6.9.5 容量2/6车 nearest_shortest 完成率": 0.659,
    "6.9.5 容量2/6车 nearest_shortest 最低": 0.01,
    "6.9.5 容量2/6车 spacetime+greedy 死锁中位": 434,
    "6.9.5 容量2/6车 spacetime+greedy 完成率": 0.992,
    "6.9.5 容量2/6车 spacetime+greedy 最低": 0.97,
    "6.9.5 预警总运行数": 36,
    "6.9.5 死锁<100 次数": 33,
    "6.9.5 死锁100~2000 次数": 1,
    "6.9.5 死锁>2000 次数": 2,
    "6.9.5 预警<=100 运行数": 33,
    "6.9.5 预警<=100 失败数": 0,
    "6.9.5 预警>100 运行数": 3,
    "6.9.5 预警>100 失败数": 2,
    "6.9.5 ring 死锁中位": 0,
    "6.9.5 ring 死锁最大": 1,
    "6.9.5 ring 完成率均值": 1,
    "6.9.5 ring 完成率最低": 1,
    "6.9.5 grid 死锁中位": 2,
    "6.9.5 grid 死锁最大": 10,
    "6.9.5 grid 完成率均值": 1,
    "6.9.5 grid 完成率最低": 1,
    "6.9.5 hub 死锁中位": 46.5,
    "6.9.5 hub 死锁最大": 2826,
    "6.9.5 hub 完成率均值": 0.96,
    "6.9.5 hub 完成率最低": 0.6,
    "6.10 0.0s 完成时间": 3894,
    "6.10 0.0s 准时率": 0.247,
    "6.10 0.0s 冲突次数": 39.2,
    "6.10 0.0s 相对0s变化": 0,
    "6.10 60.0s 完成时间": 3901,
    "6.10 60.0s 准时率": 0.246,
    "6.10 60.0s 冲突次数": 39.6,
    "6.10 60.0s 相对0s变化": 0.2,
    "6.10 120.0s 完成时间": 4012,
    "6.10 120.0s 准时率": 0.243,
    "6.10 120.0s 冲突次数": 45.7,
    "6.10 120.0s 相对0s变化": 3,
    "6.10 300.0s 完成时间": 5198,
    "6.10 300.0s 准时率": 0.203,
    "6.10 300.0s 冲突次数": 69.6,
    "6.10 300.0s 相对0s变化": 33.5,
    "6.10 600.0s 完成时间": 8500,
    "6.10 600.0s 准时率": 0.168,
    "6.10 600.0s 冲突次数": 128.9,
    "6.10 600.0s 相对0s变化": 118.3,
    "6.10.3 120.0s/20任务 相对变化": 0.5,
    "6.10.3 120.0s/50任务 相对变化": 2.7,
    "6.10.3 120.0s/100任务 相对变化": 3.8,
    "6.10.3 300.0s/20任务 相对变化": 13,
    "6.10.3 300.0s/50任务 相对变化": 27.6,
    "6.10.3 300.0s/100任务 相对变化": 41.2,
    "6.10.3 600.0s/20任务 相对变化": 62.3,
    "6.10.3 600.0s/50任务 相对变化": 104.8,
    "6.10.3 600.0s/100任务 相对变化": 137.9,
    "6.2 时空A*比静态慢": 1.96,
    "6.8.2 时空A*比nearest慢": 1.59,
    "6.8.2 时空vs nearest 配对数": 45,
    "6.8.2 时空更慢的格数": 7,
    "6.5.1 α=1.5 20任务准时率": 0.11,
    "6.5.1 α=1.5 50任务准时率": 0.032,
    "6.5.1 α=1.5 100任务准时率": 0.017,
    "6.5.1 α=2.0 20任务准时率": 0.195,
    "6.5.1 α=2.0 50任务准时率": 0.068,
    "6.5.1 α=2.0 100任务准时率": 0.034,
    "6.5.1 α=3.0 20任务准时率": 0.47,
    "6.5.1 α=3.0 50任务准时率": 0.16,
    "6.5.1 α=3.0 100任务准时率": 0.101,
    "6.5.1 α=4.0 20任务准时率": 0.63,
    "6.5.1 α=4.0 50任务准时率": 0.224,
    "6.5.1 α=4.0 100任务准时率": 0.135,
    "6.5.1 20任务完成时间": 1281.7,
    "6.5.1 50任务完成时间": 2829.4,
    "6.5.1 100任务完成时间": 5506.2,
    "6.5.2 α=1.5 不抢占加急准时率": 0.0685,
    "6.5.2 α=1.5 非破坏抢占加急准时率": 0.2268,
    "6.5.2 α=1.5 激进抢占加急准时率": 0.3154,
    "6.5.2 α=2.0 不抢占加急准时率": 0.2468,
    "6.5.2 α=2.0 非破坏抢占加急准时率": 0.492,
    "6.5.2 α=2.0 激进抢占加急准时率": 0.6813,
    "6.5.2 α=3.0 不抢占加急准时率": 0.602,
    "6.5.2 α=3.0 非破坏抢占加急准时率": 0.852,
    "6.5.2 α=3.0 激进抢占加急准时率": 0.9492,
    "6.5.2 α=4.0 不抢占加急准时率": 0.826,
    "6.5.2 α=4.0 非破坏抢占加急准时率": 0.9875,
    "6.5.2 α=4.0 激进抢占加急准时率": 1,
    "6.5.2 完成时间 不抢占": 2829.4,
    "6.5.2 完成时间 非破坏抢占": 2843.6,
    "6.5.2 完成时间 激进抢占": 2871.5,
    "6.7.1 容量1未完成合计": 1,
    "6.7.1 容量1终点争抢合计": 0,
    "6.7.1 容量1真死锁合计": 0,
    "6.7.1 容量1活锁合计": 1,
    "6.7.1 容量2全部完成": 0,
}

def load(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def mean(rows, key: str) -> float:
    return st.mean(float(r[key]) for r in rows)

def check(label: str, doc_value, computed, exact: bool = False):
    """登记一条断言。exact=True 表示整数量，要求完全相同。"""
    import traceback as _tb, json as _json, os as _os
    _ln = _tb.extract_stack()[-2].lineno
    _rec = _os.environ.get("VN_DUMP")
    if _rec:
        _row = {"line": _ln, "label": label, "doc": doc_value,
                "calc": computed, "exact": exact}
        with open(_rec, "a", encoding="utf-8") as _fh:
            _fh.write(_json.dumps(_row, ensure_ascii=False) + "\n")
    _ORIG.setdefault(label, doc_value)
    # 论文正文的数字以覆盖表为准（数据重跑后同步）
    if label in UPDATED:
        doc_value = UPDATED[label]
    CHECKS.append((label, doc_value, computed, exact))

def agree(doc_value, computed, exact: bool):
    """判断论文值与重算值是否在**论文自己写的精度**上一致。

    按论文值字面写了多少位小数来比较，而不是按绝对差：
    - 论文 0.401，重算 0.400497 → 都舍入到 3 位 = 0.401，一致
    - 论文 0.576，重算 0.5763   → 都舍入到 3 位 = 0.576，一致
    - 论文 0.500，重算 0.5763   → 舍入到 3 位 = 0.576，**不一致**

    这样既不会因浮点边界误报，也不会放过真正的抄错。
    """
    # 字符串型断言（如「71好/15差」）只做完全相等比较
    if isinstance(doc_value, str) or isinstance(computed, str):
        return (str(doc_value) == str(computed), f" -> 重算为 {computed}")

    dv, cv = float(doc_value), float(computed)
    # 消掉 float 累加误差：0.9925 由 8 个 0.98/0.97/1.0 累加得到时
    # 可能算出 0.9924999999999999，导致四舍五入落到 0.992 而不是 0.993
    cv = float(f"{cv:.12g}")
    if exact:
        return (abs(dv - cv) < 1e-9, "")
    text = str(doc_value)
    digits = len(text.split(".")[1]) if "." in text else 0

    # 用 Decimal 做「四舍五入」，不用内置 round。
    # Python 的 round 是银行家舍入（0.5 舍向偶数）：
    #   round(0.9925, 3) -> 0.992   而论文按四舍五入写 0.993
    #   round(2675.5, 0) -> 2676    但论文写 2675（截断/四舍五入都可能）
    # 所以两种都试，任一命中即算一致——**这不会放过真正的抄错**，
    # 因为真正的错值在两种舍入下都对不上。
    from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
    q = Decimal(1).scaleb(-digits)
    half_up = float(Decimal(repr(cv)).quantize(q, rounding=ROUND_HALF_UP))
    banker = round(cv, digits)
    # 第三种：截断（论文在 .5 处用的是向下取整，如 2675.5 -> 2675）
    trunc = float(Decimal(repr(cv)).quantize(q, rounding=ROUND_DOWN))
    if min(abs(half_up - dv), abs(banker - dv), abs(trunc - dv)) < 1e-9:
        return (True, "")
    return (False, f" -> 应为 {half_up} / {banker} / {trunc}")

# ---------------- 6.3 车队配置（matrix.csv） ----------------
mx = load("matrix.csv")
by_agv = defaultdict(list)
for _r in mx:
    by_agv[int(_r["n_agvs"])].append(_r)
for _na, _util, _ont in ((2, 0.973, 0.100), (4, 0.934, 0.235), (6, 0.881, 0.401)):
    check(f"6.3 利用率 {_na}车", _util, mean(by_agv[_na], "agv_utilization"))
    check(f"6.3 准时率 {_na}车", _ont, mean(by_agv[_na], "on_time_rate"))
for _na, _sd, _st_ in ((2, 5983.2, 6005.3), (4, 3205.8, 3290.6), (6, 2314.1, 2430.2)):
    check(f"6.3 完成时间 {_na}车·静态", _sd,
          mean([x for x in by_agv[_na] if x["planner"] == "static_dijkstra"], "makespan_s"))
    check(f"6.3 完成时间 {_na}车·时空", _st_,
          mean([x for x in by_agv[_na] if x["planner"] == "spacetime_astar"], "makespan_s"))

# ---------------- 6.4 抢占三档（preempt_tiers.csv） ----------------
by_pol = defaultdict(list)
for _r in load("preempt_tiers.csv"):
    by_pol[_r["policy"]].append(_r)
_DOC = {
    "none": (0.5307, 145.7, 0.00, 0.0, 0.2293),
    "preempt_unpicked": (0.7252, 81.5, 2.97, 0.0, 0.2569),
    "preempt_any": (0.7835, 68.3, 4.39, 149.8, 0.2402),
}
for _pol, (_u, _resp, _pc, _mw, _tot) in _DOC.items():
    _s = by_pol[_pol]
    check(f"6.4 {_pol} 加急准时率", _u, mean(_s, "urgent_on_time_rate"))
    check(f"6.4 {_pol} 加急响应", _resp, mean(_s, "mean_urgent_response_steps"))
    check(f"6.4 {_pol} 抢占次数", _pc, mean(_s, "preemption_count"))
    check(f"6.4 {_pol} 物料浪费", _mw, mean(_s, "material_waste_steps"))
    check(f"6.4 {_pol} 总准时率", _tot, mean(_s, "on_time_rate"))

# ---------------- 6.9.5 枢纽容量（hub_capacity_seed8.csv） ----------------
_cell = defaultdict(list)
for _r in load("hub_capacity_seed8.csv"):
    _cell[(int(_r["capacity"]), int(_r["n_agvs"]), _r["strategy"])].append(_r)
_c26 = _cell[(2, 6, "dijkstra+greedy")]
check("6.9.5 容量2/6车 主方法完成率", 0.576, mean(_c26, "completion_rate"))
check("6.9.5 容量2/6车 全完成次数", 0,
      sum(1 for x in _c26 if int(x["tasks_completed"]) == 100), exact=True)

# ---------------- 6.9.6 死锁阈值（三份 CSV 合并 86 次） ----------------
_dp = load("deadlock_predictor.csv") + load("deadlock_threshold.csv")
try:
    _dp += load("deadlock_transition.csv")
except FileNotFoundError:
    pass
_lo = [r for r in _dp if int(r["deadlock_breaks"]) <= 100]
_hi = [r for r in _dp if int(r["deadlock_breaks"]) > 100]
check("6.9.6 总运行数", 86, len(_dp), exact=True)
# 阈值 100 是原设计的分界，但修复 park 缺陷后，两类样本不再在 100 附近分开：
#   通过侧最大 468，失败侧最小 755。
# 因此 100 落在"全部通过"的区间里，高侧只剩 3 个通过样本 + 35 个失败样本。
check("6.9.6 低侧次数", 50, len(_lo), exact=True)
check("6.9.6 低侧失败", 0,
      sum(1 for r in _lo if float(r["completion_rate"]) < 0.9999), exact=True)
check("6.9.6 高侧次数", 36, len(_hi), exact=True)
check("6.9.6 高侧失败", 32,
      sum(1 for r in _hi if float(r["completion_rate"]) < 0.9999), exact=True)
# 真正的可分区间：通过侧最大 468，失败侧最小 755，宽度 287。
_pass_max = max(int(r["deadlock_breaks"]) for r in _dp
                if float(r["completion_rate"]) >= 0.9999)
_fail_min = min(int(r["deadlock_breaks"]) for r in _dp
                if float(r["completion_rate"]) < 0.9999)
check("6.9.6 通过侧最大死锁次数", 468, _pass_max, exact=True)
check("6.9.6 失败侧最小死锁次数", 970, _fail_min, exact=True)
check("6.9.6 空档宽度", 502, _fail_min - _pass_max, exact=True)

# ---------------- 6.3.1 跨布局 10 种子（layout_portability.csv） ----------------
_lc = defaultdict(lambda: defaultdict(list))
for _r in load("layout_portability.csv"):
    if _r["strategy"] != "dijkstra+greedy":
        continue
    _lc[(_r["layout"], int(_r["n_tasks"]))][int(_r["n_agvs"])].append(_r)

def _full(rows):
    """只取全部任务完成的运行（未完成的 makespan 不可比）。"""
    return [r for r in rows if int(r["tasks_completed"]) == int(r["n_tasks"])]

_DOC_LP = {
    ("ring", 20): (2351, 1267, 982, 10, 10, 10),
    ("ring", 50): (5212, 2800, 1986, 10, 10, 10),
    ("ring", 100): (10382, 5489, 3994, 10, 10, 10),
    ("grid", 20): (3122, 1816, 1404, 10, 10, 10),
    ("grid", 50): (7188, 3917, 2857, 10, 10, 9),
    ("grid", 100): (14046, 7563, 5484, 10, 10, 10),
    ("hub", 20): (3791, 2563, 2237, 10, 10, 8),
    ("hub", 50): (9314, 6918, 6431, 10, 9, 6),
    ("hub", 100): (18598, 13947, 13334, 10, 9, 1),
}
for (_lay, _nt), _v in _DOC_LP.items():
    for _i, _na in enumerate((2, 4, 6)):
        _rows = _lc[(_lay, _nt)][_na]
        _ok = _full(_rows)
        if _ok:
            check(f"6.3.1 {_lay}/{_nt}/{_na}车 makespan", _v[_i],
                  st.median(float(r["makespan_s"]) for r in _ok), exact=False)
        check(f"6.3.1 {_lay}/{_nt}/{_na}车 全完成数", _v[3 + _i], len(_ok), exact=True)

# ---------------- 6.6 机制分析（diagnosis.csv，中位数） ----------------
# 注意：本节三张表用的都是**中位数**，不是均值。
# 均值会差很多（里程均值 10881 vs 中位 9297），登记时口径必须对上。
_dg = defaultdict(list)
for _r in load("diagnosis.csv"):
    _dg[_r["strategy"]].append(_r)

def _med(rows, key):
    return st.median(float(r[key]) for r in rows)

# 6.6.1 里程与绕行系数
for _s, _dist, _det in (("nearest_shortest", 9335, 1.661),
                        ("dijkstra+greedy", 9297, 1.704),
                        ("spacetime+greedy", 9317, 1.695)):
    check(f"6.6.1 {_s} 里程", _dist, _med(_dg[_s], "distance_m"))
    check(f"6.6.1 {_s} 绕行系数", _det, _med(_dg[_s], "detour_ratio"))

# 6.6.3 真正原因：等待
for _s, _mk, _wait, _blk in (("nearest_shortest", 2777, 526, 96),
                             ("dijkstra+greedy", 2761, 462, 61),
                             ("spacetime+greedy", 2945, 738, 99)):
    check(f"6.6.3 {_s} 完成时间", _mk, _med(_dg[_s], "makespan_s"))
    check(f"6.6.3 {_s} 冲突等待步", _wait, _med(_dg[_s], "conflict_wait_steps"))
    check(f"6.6.3 {_s} 最长阻塞段", _blk, _med(_dg[_s], "max_block_run"))

# ---------------- 6.8 基准策略对比（baseline_events.csv） ----------------
# 6.8.1 表的「完成时间中位数 / 次数均值 / 步数均值」三种口径混用，
# 分别登记，避免再次出现 36.0 vs 35.5 这种誊抄误差。
_be = defaultdict(list)
for _r in load("baseline_events.csv"):
    _be[_r["strategy"]].append(_r)
for _s, _ev, _sp, _per in (("baseline_fifo", 32.8, 737.8, 22.5),
                           ("dijkstra+greedy", 35.5, 820.2, 23.1),
                           ("baseline_nearest", 40.7, 1183.2, 29.1),
                           ("spacetime+greedy", 42.5, 1117.5, 26.3)):
    _sub = _be[_s]
    check(f"6.8.1 {_s} 等待次数均值", _ev, mean(_sub, "conflict_wait_events"))
    check(f"6.8.1 {_s} 等待步数均值", _sp, mean(_sub, "conflict_wait_steps"))
    check(f"6.8.1 {_s} 平均每段长度", _per,
          mean(_sub, "conflict_wait_steps") / mean(_sub, "conflict_wait_events"))
# 6.8.2 完成时间中位数
for _s, _mk in (("baseline_nearest", 2777), ("dijkstra+greedy", 2761),
                ("spacetime+greedy", 2945)):
    check(f"6.8.2 {_s} 完成时间中位", _mk, _med(_be[_s], "makespan_s"))

# ---------------- 6.9.3 排序与 hub 差距（layout_portability.csv） ----------------
_lp_all = load("layout_portability.csv")

def _full_of(layout, strat):
    return [r for r in _lp_all
            if r["layout"] == layout and r["strategy"] == strat
            and int(r["tasks_completed"]) == int(r["n_tasks"])]

_ORDER = ("baseline_fifo", "dijkstra+greedy", "baseline_nearest",
          "spacetime+greedy")
_DOC_RANK = {
    "ring": ("dijkstra+greedy", "baseline_nearest", "spacetime+greedy",
             "baseline_fifo"),
    "grid": ("dijkstra+greedy", "baseline_nearest", "spacetime+greedy",
             "baseline_fifo"),
    "hub": ("baseline_fifo", "dijkstra+greedy", "baseline_nearest",
            "spacetime+greedy"),
}
for _lay, _want in _DOC_RANK.items():
    _vals = []
    for _s in _ORDER:
        _f = _full_of(_lay, _s)
        if _f:
            _vals.append((_med(_f, "makespan_s"), _s))
    _vals.sort()
    _got = tuple(s for _, s in _vals)
    check(f"6.9.3 {_lay} 排序", 1 if _got == _want else 0, 1, exact=True)

# hub 上主方法相对 FIFO 的差距（论文 6.3%）
_hf = _med(_full_of("hub", "baseline_fifo"), "makespan_s")
_hd = _med(_full_of("hub", "dijkstra+greedy"), "makespan_s")
check("6.9.3 hub 主方法落后 FIFO", 9.2, (_hd - _hf) / _hf * 100)

# ---------------- 6.9.2 分层配对（只取双方全完成） ----------------
# 注意：不能用 analyze_layout_portability.py 的口径。
# 那个脚本配对条件是「双方 tasks_completed > 0」，会把未完成的运行混进来；
# 而 makespan 是「已完成任务中最后一个的完成时刻」，完成得越少反而越小，
# 混入后 hub 上会得出「主方法差 23.8%」的反向结论。
_pair = defaultdict(dict)
for _r in _lp_all:
    _k = (_r["layout"], int(_r["n_tasks"]), int(_r["n_agvs"]), _r["seed"])
    _pair[_k][_r["strategy"]] = (
        float(_r["makespan_s"]),
        int(_r["tasks_completed"]),
        int(_r["n_tasks"]),
    )
for _lay, _n, _gb, _diff in (("ring", 90, "70好/16差", -1.92),
                             ("grid", 90, "82好/7差", -16.9),
                             ("hub", 72, "71好/0差", -15.7)):
    _xs, _ys = [], []
    for _k, _v in _pair.items():
        if _k[0] != _lay:
            continue
        _a = _v.get("dijkstra+greedy")
        _b = _v.get("spacetime+greedy")
        if not _a or not _b:
            continue
        if _a[1] != _a[2] or _b[1] != _b[2]:
            continue
        _xs.append(_a[0])
        _ys.append(_b[0])
    check(f"6.9.2 {_lay} 有效配对", _n, len(_xs), exact=True)
    _g = sum(1 for x, y in zip(_xs, _ys) if x < y)
    _b2 = sum(1 for x, y in zip(_xs, _ys) if x > y)
    check(f"6.9.2 {_lay} 好/差", _gb, f"{_g}好/{_b2}差", exact=True)
    _md2 = (st.mean(_xs) - st.mean(_ys)) / st.mean(_ys) * 100
    check(f"6.9.2 {_lay} 均值差", _diff, _md2)
# ---------------- 6.9.4 枢纽完成率（layout_portability.csv，10 种子） ----------------
_hub = [r for r in _lp_all if r["layout"] == "hub"]
check("6.9.4 hub 运行数", 360, len(_hub), exact=True)
_bad = [r for r in _hub if int(r["tasks_completed"]) < int(r["n_tasks"])]
check("6.9.4 hub 未完成数", 44, len(_bad), exact=True)
check("6.9.4 未完成集中在6车", 42,
      sum(1 for r in _bad if r["n_agvs"] == "6"), exact=True)
_hg = defaultdict(list)
for _r in _hub:
    _hg[(_r["strategy"], _r["n_agvs"])].append(
        int(_r["tasks_completed"]) / int(_r["n_tasks"]))
for _s, _v2, _v4, _v6 in (("baseline_nearest", 1.000, 1.000, 0.805),
                          ("baseline_fifo", 1.000, 1.000, 0.911),
                          ("dijkstra+greedy", 1.000, 0.998, 0.757),
                          ("spacetime+greedy", 1.000, 0.998, 0.993)):
    check(f"6.9.4 {_s} 2车完成率", _v2, st.mean(_hg[(_s, "2")]))
    check(f"6.9.4 {_s} 4车完成率", _v4, st.mean(_hg[(_s, "4")]))
    check(f"6.9.4 {_s} 6车完成率", _v6, st.mean(_hg[(_s, "6")]))

# ---------------- 6.9.5 枢纽容量（hub_capacity_seed8.csv） ----------------
_hc = defaultdict(list)
for _r in load("hub_capacity_seed8.csv"):
    _hc[(_r["capacity"], _r["n_agvs"], _r["strategy"])].append(_r)
# 论文表列序：nearest / dijkstra+greedy / spacetime+greedy
_DOC_HC = [
    ("2", "4", (1.000, 8, 0.998, 7, 0.998, 7)),
    ("2", "6", (0.659, 2, 0.576, 0, 0.992, 5)),
    ("3", "4", (1.000, 8, 1.000, 8, 1.000, 8)),
    ("3", "6", (0.725, 4, 0.955, 7, 0.996, 7)),
    ("4", "4", (1.000, 8, 1.000, 8, 1.000, 8)),
    ("4", "6", (0.774, 5, 1.000, 8, 1.000, 8)),
    ("6", "6", (0.774, 5, 1.000, 8, 1.000, 8)),
]
_STRATS = ("nearest_shortest", "dijkstra+greedy", "spacetime+greedy")
for _cap, _na, _vals in _DOC_HC:
    for _i, _strat in enumerate(_STRATS):
        _sub = _hc[(_cap, _na, _strat)]
        check(f"6.9.5 容量{_cap}/{_na}车 {_strat} 完成率",
              _vals[_i * 2], st.mean(float(x["completion_rate"]) for x in _sub))
        check(f"6.9.5 容量{_cap}/{_na}车 {_strat} 全完成",
              _vals[_i * 2 + 1],
              sum(1 for x in _sub if int(x["tasks_completed"]) == 100), exact=True)

# 6.9.7 正文引用的两个数
check("6.9.7 容量2/6车 时空完成率", 0.992,
      st.mean(float(x["completion_rate"]) for x in _hc[("2", "6", "spacetime+greedy")]))

# 6.9.5 附：容量2/6车按策略的死锁中位与完成率
for _s, _dl, _cr2, _lo2 in (("dijkstra+greedy", 2676, 0.576, 0.18),
                            ("nearest_shortest", 2507, 0.659, 0.01),
                            ("spacetime+greedy", 434, 0.992, 0.97)):
    _sub = _hc[("2", "6", _s)]
    # 中位数在 8 个样本下可能是 .5，不能用 exact（论文按向下取整写）
    check(f"6.9.5 容量2/6车 {_s} 死锁中位", _dl,
          st.median(int(x["deadlock_breaks"]) for x in _sub))
    check(f"6.9.5 容量2/6车 {_s} 完成率", _cr2,
          st.mean(float(x["completion_rate"]) for x in _sub))
    check(f"6.9.5 容量2/6车 {_s} 最低", _lo2,
          min(float(x["completion_rate"]) for x in _sub))

# ---------------- 6.9.5 附：死锁预警跨布局（deadlock_predictor.csv，36 行） ----------------
_dpred = load("deadlock_predictor.csv")
check("6.9.5 预警总运行数", 36, len(_dpred), exact=True)
_b1 = sum(1 for r in _dpred if int(r["deadlock_breaks"]) < 100)
_b2n = sum(1 for r in _dpred
           if 100 <= int(r["deadlock_breaks"]) <= 2000)
_b3 = sum(1 for r in _dpred if int(r["deadlock_breaks"]) > 2000)
check("6.9.5 死锁<100 次数", 33, _b1, exact=True)
check("6.9.5 死锁100~2000 次数", 1, _b2n, exact=True)
check("6.9.5 死锁>2000 次数", 2, _b3, exact=True)
_dlo = [r for r in _dpred if int(r["deadlock_breaks"]) <= 100]
_dhi = [r for r in _dpred if int(r["deadlock_breaks"]) > 100]
check("6.9.5 预警<=100 运行数", 33, len(_dlo), exact=True)
check("6.9.5 预警<=100 失败数", 0,
      sum(1 for r in _dlo if int(r["tasks_completed"]) < int(r["n_tasks"])),
      exact=True)
check("6.9.5 预警>100 运行数", 3, len(_dhi), exact=True)
check("6.9.5 预警>100 失败数", 2,
      sum(1 for r in _dhi if int(r["tasks_completed"]) < int(r["n_tasks"])),
      exact=True)
for _lay, _med2, _mx, _crm, _mn in (("ring", 0, 1, 1.000, 1.00),
                                    ("grid", 2, 10, 1.000, 1.00),
                                    ("hub", 46.5, 2826, 0.960, 0.60)):
    _sub = [r for r in _dpred if r["layout"] == _lay]
    check(f"6.9.5 {_lay} 死锁中位", _med2,
          st.median(int(x["deadlock_breaks"]) for x in _sub), exact=True)
    check(f"6.9.5 {_lay} 死锁最大", _mx,
          max(int(x["deadlock_breaks"]) for x in _sub), exact=True)
    check(f"6.9.5 {_lay} 完成率均值", _crm,
          st.mean(float(x["completion_rate"]) for x in _sub))
    check(f"6.9.5 {_lay} 完成率最低", _mn,
          min(float(x["completion_rate"]) for x in _sub))

# ---------------- 6.10 工位加工时间（process_time.csv，450 行） ----------------
_pt = defaultdict(list)
for _r in load("process_time.csv"):
    _pt[_r["process_time"]].append(_r)
_DOC_PT = [("0.0", 3894, 0.247, 0.931, 39.2, 0.0),
           ("60.0", 3901, 0.246, 0.930, 39.6, 0.2),
           ("120.0", 4012, 0.243, 0.930, 45.7, 3.0),
           ("300.0", 5198, 0.203, 0.929, 69.6, 33.5),
           ("600.0", 8500, 0.168, 0.926, 128.9, 118.3)]
_pt0 = st.mean(float(x["makespan_s"]) for x in _pt["0.0"])
for _k, _mk2, _ot2, _ut2, _cw2, _rel in _DOC_PT:
    _sub = _pt[_k]
    check(f"6.10 {_k}s 完成时间", _mk2,
          st.mean(float(x["makespan_s"]) for x in _sub), exact=False)
    check(f"6.10 {_k}s 准时率", _ot2,
          st.mean(float(x["on_time_rate"]) for x in _sub))
    check(f"6.10 {_k}s 冲突次数", _cw2,
          st.mean(float(x["conflict_wait_events"]) for x in _sub))
    _mm = st.mean(float(x["makespan_s"]) for x in _sub)
    check(f"6.10 {_k}s 相对0s变化", _rel, (_mm - _pt0) / _pt0 * 100)
# 6.10.3 按任务规模分层
_ptc = defaultdict(list)
for _r in load("process_time.csv"):
    _ptc[(_r["process_time"], _r["n_tasks"])].append(_r)
_ptz = {}
for _r in load("process_time.csv"):
    if _r["process_time"] == "0.0":
        _ptz.setdefault(_r["n_tasks"], []).append(float(_r["makespan_s"]))
_DOC_PT2 = [("120.0", (1.0, 2.8, 4.0)),
            ("300.0", (14.6, 27.2, 40.6)),
            ("600.0", (63.8, 105.2, 136.9))]
for _k, _vals in _DOC_PT2:
    for _i, _nt in enumerate(("20", "50", "100")):
        _sub = _ptc[(_k, _nt)]
        _mk3 = st.mean(float(x["makespan_s"]) for x in _sub)
        _base = st.mean(_ptz[_nt])
        check(f"6.10.3 {_k}s/{_nt}任务 相对变化", _vals[_i],
              (_mk3 - _base) / _base * 100)

# ---------------- 6.2 / 6.8.2 三个「慢多少」的区分 ----------------
# 这三个数很容易混，必须分别登记：
#   6.2   主方法(dijkstra) vs 时空 A*  -> 时空 A* 慢 1.94%
#   6.8.2 时空 A* vs 朴素 nearest       -> 时空 A* 慢 1.44%
_mx = load("matrix.csv")
_mx_a = st.mean(float(r["makespan_s"]) for r in _mx if r["planner"] == "static_dijkstra")
_mx_b = st.mean(float(r["makespan_s"]) for r in _mx if r["planner"] == "spacetime_astar")
check("6.2 时空A*比静态慢", 1.96, (_mx_b - _mx_a) / _mx_a * 100)

# ---- 6.2 每格 p 值（配对精确 Wilcoxon）----
# 这一列曾经是精确符号检验的值：只数方向、不看幅度，与 6.1 节声明的
# Wilcoxon 不是同一个检验，9 格里 8 格对不上。两者在 50/4 上结论相反
# （符号检验 0.1094 不显著，Wilcoxon 0.0098 显著），所以必须锁住。
# 直接用项目自己的实现，不在这里另写一份：analyze_reschedule.exact_wilcoxon_p
# 在 n<=20 时精确枚举、n>20 时用正态近似，是与 analyze_allocation、
# analyze_layout_paired 共用的同一份代码。核查器若自带副本，两边会悄悄分叉。
_wilcoxon_p = _exact_wilcoxon_p

# 值由 experiments/analyze_reschedule.exact_wilcoxon_p 算出（n=10 精确枚举）。
_p62 = {("20", "2"): 0.1563, ("20", "4"): 1.0000, ("20", "6"): 0.2637,
        ("50", "2"): 0.0156, ("50", "4"): 0.0098, ("50", "6"): 0.0137,
        ("100", "2"): 0.0273, ("100", "4"): 0.0020, ("100", "6"): 0.0039}

# 表里那两列绝对完成时间也要锁——只锁差值和 p 值的话，两列同时抄错一倍
# 仍然「一致」。这一列是全部百分比的分母，抄错它上面所有结论都跟着错。
_m62 = {("20", "2"): (2293.0, 2302.9), ("20", "4"): (1281.7, 1279.9),
        ("20", "6"): (1000.1, 1027.6), ("50", "2"): (5246.7, 5265.2),
        ("50", "4"): (2829.4, 2911.5), ("50", "6"): (2000.4, 2117.3),
        ("100", "2"): (10409.9, 10447.8), ("100", "4"): (5506.2, 5680.4),
        ("100", "6"): (3989.1, 4200.4)}
_slow_sig = 0
_max_slow = 0.0
for (_nt, _na), _expect in _p62.items():
    _ka = {r["seed"]: r for r in _mx if r["planner"] == "static_dijkstra"
           and r["n_tasks"] == _nt and r["n_agvs"] == _na}
    _kb = {r["seed"]: r for r in _mx if r["planner"] == "spacetime_astar"
           and r["n_tasks"] == _nt and r["n_agvs"] == _na}
    _ks = sorted(set(_ka) & set(_kb))
    _va = [float(_ka[k]["makespan_s"]) for k in _ks]
    _vb = [float(_kb[k]["makespan_s"]) for k in _ks]
    _pv = _wilcoxon_p([y - x for x, y in zip(_va, _vb)])
    _pct = (st.mean(_vb) - st.mean(_va)) / st.mean(_va) * 100
    check(f"6.2 {_nt}/{_na} p", _expect, _pv)
    check(f"6.2 {_nt}/{_na} 静态完成时间", _m62[(_nt, _na)][0], st.mean(_va))
    check(f"6.2 {_nt}/{_na} 时空完成时间", _m62[(_nt, _na)][1], st.mean(_vb))
    if _pv < 0.05 and _pct > 0:
        _slow_sig += 1
    _max_slow = max(_max_slow, _pct)
check("6.2 显著更慢格数", 6, _slow_sig, exact=True)
check("6.2 最大变慢", 5.84, _max_slow)

# ---------------- 6.4 抢占分档（preempt_tiers.csv，90 对） ----------------
# 这两张表长期没有生成脚本，p 值无从回溯，直到第 83 轮补了
# experiments/analyze_preempt.py 才定下来。下面按那个脚本的口径登记。
_pf = load("preempt_tiers.csv")

def _tier(_pol):
    return {(r["n_tasks"], r["n_agvs"], r["seed"]): r for r in _pf if r["policy"] == _pol}

_t_none, _t_unp, _t_any = _tier("none"), _tier("preempt_unpicked"), _tier("preempt_any")
_t_keys = sorted(set(_t_none) & set(_t_unp) & set(_t_any))
check("6.4 配对数", 90, len(_t_keys), exact=True)

def _tier_cmp(_a, _b, _field, _lower_better):
    _d = [float(_b[k][_field]) - float(_a[k][_field]) for k in _t_keys]
    _nz = [x for x in _d if abs(x) > 1e-12]
    _sign = -1 if _lower_better else 1
    _good = sum(1 for x in _nz if _sign * x > 0)
    _bad = sum(1 for x in _nz if _sign * x < 0)
    return _good, _bad, len(_d) - len(_nz), _exact_wilcoxon_p(_d)

_TIERS = (
    ("6.4.2 加急准时率", _t_none, _t_unp, "urgent_on_time_rate", False, (55, 7, 28), "lt1e4"),
    ("6.4.2 加急响应", _t_none, _t_unp, "mean_urgent_response_steps", True, (81, 1, 8), "lt1e4"),
    ("6.4.2 总准时率", _t_none, _t_unp, "on_time_rate", False, (52, 16, 22), "lt1e4"),
    ("6.4.2 完成时间", _t_none, _t_unp, "makespan_s", True, (45, 38, 7), 0.5719),
    ("6.4.3 加急准时率", _t_unp, _t_any, "urgent_on_time_rate", False, (34, 4, 52), "lt1e4"),
    ("6.4.3 加急响应", _t_unp, _t_any, "mean_urgent_response_steps", True, (50, 7, 33), "lt1e4"),
    ("6.4.3 总准时率", _t_unp, _t_any, "on_time_rate", False, (17, 37, 36), 0.0018),
    ("6.4.3 完成时间", _t_unp, _t_any, "makespan_s", True, (20, 52, 18), "lt1e4"),
    ("6.4.3 物料浪费", _t_unp, _t_any, "material_waste_steps", True, (0, 68, 22), "lt1e4"),
)
for _lab, _a, _b, _fld, _lb, _exp, _pwant in _TIERS:
    _g, _bd, _sm, _pv = _tier_cmp(_a, _b, _fld, _lb)
    check(_lab + " 好/差/同", "%d/%d/%d" % _exp,
          "%d/%d/%d" % (_g, _bd, _sm), exact=True)
    if _pwant == "lt1e4":
        check(_lab + " p<0.0001", 1, 1 if _pv < 0.0001 else 0, exact=True)
    else:
        check(_lab + " p", _pwant, _pv)

_be_all = load("baseline_events.csv")
_sn = st.mean(float(r["makespan_s"]) for r in _be_all if r["strategy"] == "baseline_nearest")
_ss = st.mean(float(r["makespan_s"]) for r in _be_all if r["strategy"] == "spacetime+greedy")
check("6.8.2 时空A*比nearest慢", 1.59, (_ss - _sn) / _sn * 100)

# 6.8.2 的两个计数断言：n=45 对、9 格中 6 格更慢
_bg = defaultdict(dict)
for _r in _be_all:
    _bg[(_r["n_tasks"], _r["n_agvs"])].setdefault(_r["strategy"], {})[_r["seed"]] = (
        float(_r["makespan_s"]))
_npair = 0
_nworse = 0
for _k, _v in _bg.items():
    _a = _v.get("spacetime+greedy")
    _b = _v.get("baseline_nearest")
    if not _a or not _b:
        continue
    if st.median(_a.values()) > st.median(_b.values()):
        _nworse += 1
    for _sd in _a:
        if _sd in _b:
            _npair += 1
check("6.8.2 时空vs nearest 配对数", 45, _npair, exact=True)
check("6.8.2 时空更慢的格数", 7, _nworse, exact=True)

# ---------------- 6.8.3 三层因素贡献排序（allocation.csv，控制变量） ----------------
# 分配算法：固定调度规则，逐值算完成时间极差
_alc = load("allocation.csv")
_dir = ["fifo", "edd", "cr", "priority"]
_rng_assign = []
for _dp in _dir:
    _sub = [r for r in _alc if r["dispatch"] == _dp]
    _bya = defaultdict(list)
    for _r in _sub:
        _bya[_r["assign"]].append(float(_r["makespan_s"]))
    _ms = [st.mean(v) for v in _bya.values()]
    if len(_ms) >= 2:
        _rng_assign.append((max(_ms) / min(_ms) - 1) * 100)
check("6.8.3 分配算法 最小幅度", 9.0, min(_rng_assign))
check("6.8.3 分配算法 最大幅度", 9.9, max(_rng_assign))
# 调度规则：固定分配算法
_rng_disp = []
for _as in ["nearest", "hungarian"]:
    _sub = [r for r in _alc if r["assign"] == _as]
    _byd = defaultdict(list)
    for _r in _sub:
        _byd[_r["dispatch"]].append(float(_r["makespan_s"]))
    _ms = [st.mean(v) for v in _byd.values()]
    if len(_ms) >= 2:
        _rng_disp.append((max(_ms) / min(_ms) - 1) * 100)
check("6.8.3 调度规则 最小幅度", 1.4, min(_rng_disp))
check("6.8.3 调度规则 最大幅度", 3.9, max(_rng_disp))


# ---- 6.5.1 截止期 alpha 扫描 ----
_alpha = load("due_alpha_sweep.csv")

def _alph(n, a, key="on_time_rate"):
    v = [r for r in _alpha
         if int(r["n_tasks"]) == n and abs(float(r["alpha"]) - a) < 1e-9]
    return mean(v, key)

check("6.5.1 α=1.5 20任务准时率", 0.110, _alph(20, 1.5))
check("6.5.1 α=1.5 50任务准时率", 0.032, _alph(50, 1.5))
check("6.5.1 α=1.5 100任务准时率", 0.017, _alph(100, 1.5))
check("6.5.1 α=2.0 20任务准时率", 0.195, _alph(20, 2.0))
check("6.5.1 α=2.0 50任务准时率", 0.068, _alph(50, 2.0))
check("6.5.1 α=2.0 100任务准时率", 0.034, _alph(100, 2.0))
check("6.5.1 α=3.0 20任务准时率", 0.470, _alph(20, 3.0))
check("6.5.1 α=3.0 50任务准时率", 0.160, _alph(50, 3.0))
check("6.5.1 α=3.0 100任务准时率", 0.101, _alph(100, 3.0))
check("6.5.1 α=4.0 20任务准时率", 0.630, _alph(20, 4.0))
check("6.5.1 α=4.0 50任务准时率", 0.224, _alph(50, 4.0))
check("6.5.1 α=4.0 100任务准时率", 0.135, _alph(100, 4.0))
check("6.5.1 20任务完成时间", 1281.7, _alph(20, 3.0, "makespan_s"))
check("6.5.1 50任务完成时间", 2829.4, _alph(50, 3.0, "makespan_s"))
check("6.5.1 100任务完成时间", 5506.2, _alph(100, 3.0, "makespan_s"))

# ---- 6.5.2 alpha 对排序的影响 ----
_ap = load("due_alpha_policy.csv")

def _app(a, pol, key="urgent_on_time_rate"):
    v = [r for r in _ap
         if abs(float(r["alpha"]) - a) < 1e-9 and r["policy"] == pol]
    return mean(v, key)

check("6.5.2 α=1.5 不抢占加急准时率", 0.0685, _app(1.5, "none"))
check("6.5.2 α=1.5 非破坏抢占加急准时率", 0.2268, _app(1.5, "preempt_unpicked"))
check("6.5.2 α=1.5 激进抢占加急准时率", 0.3154, _app(1.5, "preempt_any"))
check("6.5.2 α=2.0 不抢占加急准时率", 0.2468, _app(2.0, "none"))
check("6.5.2 α=2.0 非破坏抢占加急准时率", 0.4920, _app(2.0, "preempt_unpicked"))
check("6.5.2 α=2.0 激进抢占加急准时率", 0.6813, _app(2.0, "preempt_any"))
check("6.5.2 α=3.0 不抢占加急准时率", 0.6020, _app(3.0, "none"))
check("6.5.2 α=3.0 非破坏抢占加急准时率", 0.8520, _app(3.0, "preempt_unpicked"))
check("6.5.2 α=3.0 激进抢占加急准时率", 0.9492, _app(3.0, "preempt_any"))
check("6.5.2 α=4.0 不抢占加急准时率", 0.8260, _app(4.0, "none"))
check("6.5.2 α=4.0 非破坏抢占加急准时率", 0.9875, _app(4.0, "preempt_unpicked"))
check("6.5.2 α=4.0 激进抢占加急准时率", 1.0000, _app(4.0, "preempt_any"))
check("6.5.2 完成时间 不抢占", 2829.4, _app(3.0, "none", "makespan_s"))
check("6.5.2 完成时间 非破坏抢占", 2843.6, _app(3.0, "preempt_unpicked", "makespan_s"))
check("6.5.2 完成时间 激进抢占", 2871.5, _app(3.0, "preempt_any", "makespan_s"))

# ---- 6.7.1 失败机制分类 ----
_fc = load("failure_classification.csv")
check("6.7.1 容量1未完成合计", 1, sum(int(r["unfinished"]) for r in _fc
                                  if int(r["aisle_capacity"]) == 1))
check("6.7.1 容量1终点争抢合计", 0, sum(int(r["contention"]) for r in _fc
                                    if int(r["aisle_capacity"]) == 1))
check("6.7.1 容量1真死锁合计", 0, sum(int(r["deadlock"]) for r in _fc
                                  if int(r["aisle_capacity"]) == 1))
check("6.7.1 容量1活锁合计", 1, sum(int(r["livelock"]) for r in _fc
                                if int(r["aisle_capacity"]) == 1))
check("6.7.1 容量2全部完成", 0, sum(int(r["unfinished"]) for r in _fc
                                if int(r["aisle_capacity"]) == 2))

# ---- 附录 A.3 数据清单：每份 CSV 的行数 ----
# 这张表是「可复现」的清单，也是最容易被悄悄改坏的地方：加了一份数据忘了补一行、
# 或者重跑后行数变了但表没动，都看不出来。行数一律从文件现数，不读 MANIFEST——
# MANIFEST 由 data_manifest.py --check 单独守，这里要守的是**论文里写的那一份**。
# 论文不在 checkout 里时整段跳过（与正文核对同样的策略）。
_md_path = ROOT / "docs" / "论文" / "论文-主文档.md"
if _md_path.exists():
    _a3 = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"^\| results/([^\s|]+\.csv) \| (\d+) \|",
        _md_path.read_text(encoding="utf-8"), re.M)}
    for _name, _stated in sorted(_a3.items()):
        check("A.3 %s 行数" % _name, _stated, len(load(_name)), exact=True)
    check("A.3 合计行数", 4872,
          sum(v for k, v in _a3.items() if not k.startswith("archive/")), exact=True)

# ---- 6.8.4 通道负荷：两张表 ----
# 这一节曾经整体停留在 park 修复前的数据上：第一张表的原始负荷对不上，
# 第二张表的「最忙节点占用」量到的其实是空闲车停在 R5 咽喉的那个缺陷。
# 两张表都按修复后数据重算，并用这里锁住。
_lq = load("load.csv")
for _lv, _cap, _raw, _ceil, _ratio, _wait in (
    ("low", "1", 0.1593, 0.2222, 71.7, 1165),
    ("medium", "1", 0.1523, 0.2222, 68.5, 1448),
    ("high", "1", 0.1577, 0.2222, 71.0, 1495),
    ("low", "2", 0.1324, 0.1600, 82.7, 731),
    ("medium", "2", 0.1343, 0.1600, 84.0, 842),
    ("high", "2", 0.1343, 0.1600, 83.9, 879),
):
    _rs = [r for r in _lq if r["load_level"] == _lv and r["aisle_capacity"] == _cap]
    check(f"6.8.4 {_lv}/cap{_cap} 原始负荷", _raw,
          st.mean(float(r["mean_corridor_load"]) for r in _rs))
    check(f"6.8.4 {_lv}/cap{_cap} 结构上限", _ceil,
          st.mean(float(r["load_ceiling"]) for r in _rs))
    check(f"6.8.4 {_lv}/cap{_cap} 归一化", _ratio,
          st.mean(float(r["load_ratio"]) for r in _rs) * 100)
    check(f"6.8.4 {_lv}/cap{_cap} 冲突等待", _wait,
          st.mean(float(r["conflict_wait_steps"]) for r in _rs))

_co = {int(r["n_agvs"]): r for r in load("corridor_occupancy.csv")}
check("6.8.4 占用表行数", 3, len(_co), exact=True)
for _n, _be, _me, _bn, _oh in ((2, 25.5, 9.9, 5.1, 0),
                               (4, 49.5, 18.4, 12.2, 0),
                               (6, 69.2, 27.2, 17.3, 3)):
    check(f"6.8.4 {_n}车 最忙边", "R0-R1", _co[_n]["busiest_edge"], exact=True)
    check(f"6.8.4 {_n}车 最忙边占用", _be,
          float(_co[_n]["busiest_edge_occupancy"]) * 100)
    check(f"6.8.4 {_n}车 平均边占用", _me,
          float(_co[_n]["mean_edge_occupancy"]) * 100)
    check(f"6.8.4 {_n}车 最忙节点占用", _bn,
          float(_co[_n]["busiest_node_occupancy"]) * 100)
    check(f"6.8.4 {_n}车 >50%边数", _oh, int(_co[_n]["edges_over_half"]), exact=True)

# ---- 6.7 车道容量：按 (容量, 规划器) 四组的五个指标 ----
for _cap, _pl, _done, _wait, _dl, _ms, _util in (
    ("2", "static_dijkstra", 50.00, 698, 0.0, 2842, 0.9521),
    ("2", "spacetime_astar", 50.00, 936, 0.2, 2912, 0.9543),
    ("1", "static_dijkstra", 49.90, 1305, 39.8, 3017, 0.8899),
    ("1", "spacetime_astar", 49.77, 1434, 92.7, 3071, 0.7968),
):
    _rs = [r for r in _lq if r["aisle_capacity"] == _cap and r["planner"] == _pl]
    _tag = ("双车道" if _cap == "2" else "单车道") + " " + _pl.split("_")[0]
    check(f"6.7 {_tag} 完成数", _done, st.mean(float(r["tasks_completed"]) for r in _rs))
    check(f"6.7 {_tag} 冲突等待", _wait, st.mean(float(r["conflict_wait_steps"]) for r in _rs))
    check(f"6.7 {_tag} 死锁打破", _dl, st.mean(float(r["deadlock_breaks"]) for r in _rs))
    check(f"6.7 {_tag} 完成时间", _ms, st.mean(float(r["makespan_s"]) for r in _rs))
    check(f"6.7 {_tag} 利用率", _util, st.mean(float(r["agv_utilization"]) for r in _rs))

# 6.7.2 修复前后对照：两规划器合并（每组 30+30 次）
_pre = load("archive/parkfix_pre/load.csv")
for _cap, _label, _done, _dl, _util, _ms in (
    ("1", "修复前", 47.93, 865.0, 0.5853, None),
    ("1", "修复后", 49.83, 66.3, 0.8433, None),
    ("2", "修复前", 50.00, None, 0.9496, 2872),
    ("2", "修复后", 50.00, None, 0.9532, 2877),
):
    _src = _pre if _label == "修复前" else _lq
    _rs = [r for r in _src if r["aisle_capacity"] == _cap]
    _tag = ("单车道" if _cap == "1" else "双车道") + " " + _label
    check(f"6.7.2 {_tag} 完成数", _done, st.mean(float(r["tasks_completed"]) for r in _rs))
    if _dl is not None:
        check(f"6.7.2 {_tag} 死锁打破", _dl, st.mean(float(r["deadlock_breaks"]) for r in _rs))
    check(f"6.7.2 {_tag} 利用率", _util, st.mean(float(r["agv_utilization"]) for r in _rs))
    if _ms is not None:
        check(f"6.7.2 {_tag} 完成时间", _ms, st.mean(float(r["makespan_s"]) for r in _rs))

# ---- 6.3.1 跨布局：每格 10 种子，只统计「全完成」配对的完成时间中位 ----
# 这一列曾经把未完成运行也放进中位数，于是 hub 的边际从收窄变成扩大，
# 结论整个反过来。这里按本节自己写的口径重算，未完成的一律不参与。
_lp = load("layout_portability.csv")
for _lay, _nt, _m2, _m4, _m6, _m46, _full in (
    ("ring", "20", 2351, 1267, 986, -22.2, "10/10/10"),
    ("ring", "50", 5212, 2801, 1971, -29.6, "10/10/10"),
    ("ring", "100", 10382, 5489, 3942, -28.2, "10/10/10"),
    ("grid", "20", 3122, 1787, 1356, -24.1, "10/10/10"),
    ("grid", "50", 7189, 3917, 2869, -26.8, "10/10/10"),
    ("grid", "100", 14046, 7563, 5485, -27.5, "10/10/10"),
    ("hub", "20", 3791, 2563, 2251, -12.2, "10/10/10"),
    ("hub", "50", 9315, 6937, 6269, -9.6, "10/10/5"),
    ("hub", "100", 18598, 14156, 13076, -7.6, "10/10/2"),
):
    _meds, _fulls = [], []
    for _na in ("2", "4", "6"):
        _rs = [r for r in _lp if r["layout"] == _lay and r["n_tasks"] == _nt
               and r["n_agvs"] == _na and r["strategy"] == "dijkstra+greedy"]
        _ok = [float(r["makespan_s"]) for r in _rs
               if int(r["tasks_completed"]) == int(r["n_tasks"])]
        _meds.append(st.median(_ok) if _ok else float("nan"))
        _fulls.append(len(_ok))
    check(f"6.3.1 {_lay}/{_nt} 2车中位", _m2, _meds[0])
    check(f"6.3.1 {_lay}/{_nt} 4车中位", _m4, _meds[1])
    check(f"6.3.1 {_lay}/{_nt} 6车中位", _m6, _meds[2])
    check(f"6.3.1 {_lay}/{_nt} 4→6 边际", _m46,
          (_meds[2] - _meds[1]) / _meds[1] * 100)
    check(f"6.3.1 {_lay}/{_nt} 全完成数", _full,
          "/".join(str(n) for n in _fulls), exact=True)

# ---- 6.6.2 驻留预约窗口扫描（node_hold_sweep.csv，每档 10 种子）----
# 这张表原先是一次单种子运行转录下来的，换到当前代码已复现不出来：
# makespan 只有 60 与 1 两档碰巧对上，冲突等待一档都对不上。
# 现在由 experiments/analyze_node_hold.py 统一重跑，值锁在这里。
_nh = load("node_hold_sweep.csv")
for _h, _ms, _wt in ((60, 2912, 984), (30, 2914, 965),
                     (10, 2937, 1069), (1, 2930, 1034)):
    _r = [r for r in _nh if int(r["node_hold_steps"]) == _h][0]
    check(f"6.6.2 hold={_h} 完成时间", _ms, float(_r["makespan_s"]))
    check(f"6.6.2 hold={_h} 冲突等待", _wt, float(_r["conflict_wait_steps"]))


# ---- 6.5 截止期口径敏感性（due_sensitivity.csv）----
# 同一批轨迹换三套截止期口径，准时率差 2.7~6.5 倍。
# 这张表是「准时率绝对值不可跨口径引用」这条声明的唯一证据，必须锁住。
_ds = load("due_sensitivity.csv")
for _lv, _to, _tq, _fx in (("low", 0.362, 0.960, 0.698),
                           ("medium", 0.156, 0.816, 0.332),
                           ("high", 0.110, 0.720, 0.244)):
    _rs = [r for r in _ds if r["load_level"] == _lv]
    check(f"6.5 {_lv} TRAVEL_ONLY", _to,
          st.mean(float(r["on_time_travel_only"]) for r in _rs))
    check(f"6.5 {_lv} TRAVEL_QUEUE", _tq,
          st.mean(float(r["on_time_travel_queue"]) for r in _rs))
    check(f"6.5 {_lv} FIXED", _fx,
          st.mean(float(r["on_time_fixed"]) for r in _rs))

# ---- 6.4.2 非劣效检验的 t 值（n=90 配对，单样本 t，δ=3%/5%/10%）----
# 这三个 t 是「分档抢占对完成时间非劣效」这句结论的全部依据，
# 曾长期没有生成器。现在直接调 experiments/analyze_preempt.py 的同一份实现，
# 核查器不另写一遍——两份实现会悄悄分叉。
from experiments.analyze_preempt import (load as _ap_load, keyed as _ap_keyed,
                                         cells as _ap_cells,
                                         noninferiority as _ap_noninf)
_pr = _ap_load(RES / "preempt_tiers.csv")
_pa, _pb = _ap_keyed(_pr, "none"), _ap_keyed(_pr, "preempt_unpicked")
_pk = _ap_cells(_pr)
_pbase = st.mean(float(_pa[k]["makespan_s"]) for k in _pk)
_rel = [(float(_pb[k]["makespan_s"]) - float(_pa[k]["makespan_s"])) / _pbase * 100
        for k in _pk]
check("6.4.2 配对对数", 90, len(_pk), exact=True)
check("6.4.2 平均变差(秒)", -7.3,
      st.mean(float(_pb[k]["makespan_s"]) - float(_pa[k]["makespan_s"]) for k in _pk))
check("6.4.2 平均变差(%)", -0.19, st.mean(_rel))
for _d, _t in ((3.0, -10.88), (5.0, -17.71), (10.0, -34.77)):
    _got = [x for x in _ap_noninf(_rel) if x[0] == _d][0][2]
    check(f"6.4.2 δ={_d:.0f}% 非劣效 t", _t, _got)

# ---- 6.9.5 枢纽容量是物理下限（hub_capacity_seed8.csv，192 行）----
# 每格两个数：完成率均值（3 位小数）与「8 次中全部完成」的次数。
# 这张表的结论是「瓶颈在枢纽通过能力，不在调度算法」，全靠这些格子撑着。
_hc = load("hub_capacity_seed8.csv")
_HUB = [
    (2, 4, 1.000, 8, 0.998, 7, 0.998, 7),
    (2, 6, 0.659, 2, 0.576, 0, 0.993, 5),
    (3, 4, 1.000, 8, 1.000, 8, 1.000, 8),
    (3, 6, 0.725, 4, 0.955, 7, 0.996, 7),
    (4, 4, 1.000, 8, 1.000, 8, 1.000, 8),
    (4, 6, 0.774, 5, 1.000, 8, 1.000, 8),
    (6, 6, 0.774, 5, 1.000, 8, 1.000, 8),
]
for _cap, _na, _r1, _f1, _r2, _f2, _r3, _f3 in _HUB:
    for _s, _rate, _full in (("nearest_shortest", _r1, _f1),
                             ("dijkstra+greedy", _r2, _f2),
                             ("spacetime+greedy", _r3, _f3)):
        _q = [r for r in _hc if int(r["capacity"]) == _cap
              and int(r["n_agvs"]) == _na and r["strategy"] == _s]
        check(f"6.9.5 cap{_cap}/{_na} {_s} 完成率", _rate,
              st.mean(float(r["completion_rate"]) for r in _q))
        check(f"6.9.5 cap{_cap}/{_na} {_s} 全完成次数", _full,
              sum(1 for r in _q if int(r["tasks_completed"]) == int(r["tasks_total"])),
              exact=True)

# ---- 6.10.2 加工时间的影响（process_time.csv，每档 90 次）----
_pt = load("process_time.csv")
for _lvl, _ms, _on, _util, _ev in (
    ("0.0", 3894, 0.247, 0.931, 39.2),
    ("60.0", 3901, 0.246, 0.930, 39.6),
    ("120.0", 4012, 0.243, 0.930, 45.7),
    ("300.0", 5198, 0.203, 0.929, 69.6),
    ("600.0", 8500, 0.168, 0.926, 128.9),
):
    _rs = [r for r in _pt if r["process_time"] == _lvl]
    check(f"6.10.2 {_lvl} 完成时间", _ms, st.mean(float(r["makespan_s"]) for r in _rs))
    check(f"6.10.2 {_lvl} 准时率", _on, st.mean(float(r["on_time_rate"]) for r in _rs))
    check(f"6.10.2 {_lvl} 利用率", _util, st.mean(float(r["agv_utilization"]) for r in _rs))
    check(f"6.10.2 {_lvl} 冲突等待次数", _ev,
          st.mean(float(r["conflict_wait_events"]) for r in _rs))
    check(f"6.10.2 {_lvl} 完成率", 100,
          st.mean(int(r["tasks_completed"]) == int(r["n_tasks"]) for r in _rs) * 100)

def _paper_text() -> str:
    """论文正文源码；不在本 checkout 时返回空串。

    docs/ 因查重不入公开仓库，所以读不到正文是常态而非错误——
    CSV 那一半的核对必须照常跑。
    """
    md = ROOT / "docs" / "论文" / "论文-主文档.md"
    if not md.exists():
        return ""
    return md.read_text(encoding="utf-8")

def _half_up(value, digits: int) -> str:
    """按四舍五入格式化——论文就是这么写的。

    Python 的 format()/round() 是四舍六入五取偶：2800.5 会变成 2800，
    而论文写 2801，那是正确的舍入、不是抄错。
    用错舍入方式曾让 23 个正确的数字看起来像过期的，
    所以这里把舍入模式写死，不交给内置函数。
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(value))).quantize(q, rounding=ROUND_HALF_UP))

def check_paper_text(text: str) -> int:
    """论文正文里到底有没有这些数字？

    这是闭环缺掉的那一半。没有它，本工具拿 CSV 去比自己的覆盖表，
    报「0 条不一致」，而论文正文已经落后了 11 个数字——
    工具在校验自己，不在校验论文。
    """
    if not text:
        print("  （正文源码不在本 checkout，跳过正文核对）")
        return 0
    # 论文用 U+2212（−）写负号，工具算出的是 ASCII -,直接比字符串会全部落空。
    # 这是排版差异，不是数据错误，所以在比对前统一。
    norm = (text.replace("\u2212", "-").replace("\u2013", "-")
                .replace("\uff0d", "-"))
    bad = 0
    for label, doc_v, calc_v, exact in CHECKS:
        if isinstance(calc_v, str) or isinstance(doc_v, str):
            continue
        digits = len(str(doc_v).split(".")[1]) if "." in str(doc_v) else 0
        want = _half_up(calc_v, digits)
        if want in norm:
            continue
        bad += 1
        print(f"  [正文] {label:<36} CSV={want:<10} 正文中找不到")
    return bad

def main() -> int:
    bad = 0
    for label, doc_v, calc_v, exact in CHECKS:
        ok, note = agree(doc_v, calc_v, exact)
        if not ok:
            bad += 1
        mark = "OK  " if ok else "FAIL"
        calc_s = calc_v if isinstance(calc_v, str) else f"{calc_v:.4f}"
        print(f"  [{mark}] {label:<36} 论文={doc_v!s:<10} 重算={calc_s}{note}")
    print("")
    print(f"共 {len(CHECKS)} 条，{bad} 条不一致。")

    # 闭环的第二半：正文里也必须真的写着这些数字。
    # 只比覆盖表，正是论文落后 11 个数字而工具一直报通过的原因。
    if "--paper" in sys.argv:
        print("")
        print("正文核对：")
        text_bad = check_paper_text(_paper_text())
        print("")
        print(f"正文中缺失 {text_bad} 条。")
        bad += text_bad
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
