"""配对差值的 95% 置信区间与效应量。

为什么需要它
------------
论文里到处是「提升了 6.7 个百分点」「慢了 1.44%」这类点估计，
但没有一个带区间。点估计不说精度，读者无法判断这个差是稳定还是偶然。

统计审计（第 62 轮）发现整个项目零个置信区间，这是硬缺口。

方法
----
配对差 d_i = a_i - b_i，按 (任务数, 车数, 种子) 配对。
均值差的 CI 用 t 分布：mean ± t_(0.975,n-1) · sd/√n。
n=90 时 t≈1.987，与正态的 1.96 几乎一致，但小样本时必须用 t。

同时报 Cohen's dz = mean/sd —— 配对设计的标准化效应量。
为什么不用 Cohen's d（合并标准差）：配对设计里组间方差含个体差异，
dz 才是正确的分母。经验档位 0.2/0.5/0.8 对应小/中/大。
"""
from __future__ import annotations

import csv
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_reschedule import exact_wilcoxon_p  # noqa: E402


def t_critical(n: int, conf: float = 0.95) -> float:
    """t 分布双侧临界值。

    小样本必须用 t 而不是 1.96。这里用查表 + 插值，
    避免为了一个数引入 scipy 依赖（本项目当前没有 scipy）。
    """
    # df -> t_(0.975, df)
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
    }
    df = n - 1
    if conf != 0.95:
        raise ValueError("只实现了 95% 临界值")
    if df < 1:
        return float("nan")
    if df in table:
        return table[df]
    keys = sorted(table)
    if df < keys[0]:
        return table[keys[0]]
    if df > keys[-1]:
        return 1.960  # 大样本趋近正态
    # 线性插值
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return table[lo]
    w = (df - lo) / (hi - lo)
    return table[lo] * (1 - w) + table[hi] * w


def mean_ci(diffs: list[float], conf: float = 0.95) -> dict:
    """配对差的均值、95% CI、效应量。"""
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "dz": float("nan"), "p": float("nan")}
    m = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    tc = t_critical(n, conf)
    dz = m / sd if sd > 0 else float("inf")
    p = exact_wilcoxon_p(diffs)
    return {"n": n, "mean": m, "lo": m - tc * se, "hi": m + tc * se,
            "sd": sd, "dz": dz, "p": p}


def load_pairs(path: str, key: str, group_col: str,
               a: str, b: str) -> list[float]:
    """从长表 CSV 读出 a-b 的配对差。

    配对键 = (任务数, 车数, 种子)，即同一算例下两种策略的结果相减。
    统计单位是**一次运行**，不是单个任务——这点必须在论文里写清楚。
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    tbl: dict[str, dict] = {}
    for r in rows:
        pk = (r["n_tasks"], r["n_agvs"], r["seed"])
        tbl.setdefault(r[group_col], {})[pk] = float(r[key])
    if a not in tbl or b not in tbl:
        raise KeyError("%s 或 %s 不在 %s 里" % (a, b, group_col))
    common = sorted(set(tbl[a]) & set(tbl[b]))
    return [tbl[a][k] - tbl[b][k] for k in common]


def fmt(d: dict, scale: float = 1.0, unit: str = "") -> str:
    return ("n=%d  均值 %+.4f%s  [95%% CI %+.4f, %+.4f]  dz=%.2f  p=%.2g"
            % (d["n"], d["mean"] * scale, unit, d["lo"] * scale,
               d["hi"] * scale, d["dz"], d["p"]))


def main() -> int:
    print("=" * 78)
    print("配对差值 95% 置信区间")
    print("统计单位：一次运行（同一任务数/车数/种子下配对）")
    print("=" * 78)

    print("")
    print("[1] 抢占三档 × 完成时间（秒）")
    base_p = "none"
    for pol in ("preempt_unpicked", "preempt_any"):
        d = mean_ci(load_pairs("results/preempt_tiers.csv", "makespan_s",
                              "policy", pol, base_p))
        print("  %-18s %s" % (pol, fmt(d)))

    print("")
    print("[2] 抢占三档 × 加急准时率（百分点）")
    for pol in ("preempt_unpicked", "preempt_any"):
        d = mean_ci(load_pairs("results/preempt_tiers.csv",
                              "urgent_on_time_rate", "policy", pol, base_p))
        print("  %-18s %s" % (pol, fmt(d, 100, " pp")))

    print("")
    print("[3] 规划器对照 × 完成时间")
    # matrix.csv 里两个规划器各占一行，同一 (任务数,车数,种子) 有两行。
    # 直接按 (任务数,车数,种子) 建字典会互相覆盖 —— 这正是 load_pairs 的
    # 隐藏前提：它假设每个配对键在组内唯一。对 matrix.csv 必须显式 pivot。
    rows = list(csv.DictReader(open("results/matrix.csv", encoding="utf-8")))
    piv: dict = {}
    for r in rows:
        k = (r["n_tasks"], r["n_agvs"], r["seed"])
        piv.setdefault(k, {})[r["planner"]] = float(r["makespan_s"])
    diffs = [v["spacetime_astar"] - v["static_dijkstra"]
             for v in piv.values()
             if "spacetime_astar" in v and "static_dijkstra" in v]
    d = mean_ci(diffs)
    rel = 100 * d["mean"] / (d["mean"] / (piv and 1) if False else 1)
    print("  时空A* - 静态Dijkstra  %s" % fmt(d))
    # 相对差要除以静态基线的均值，不能除以差值
    base_mean = st.mean([v["static_dijkstra"] for v in piv.values()
                         if "static_dijkstra" in v])
    print("  相对差：%+.2f%%  （静态基线均值 %.1f s）"
          % (100 * d["mean"] / base_mean, base_mean))
    return 0


if __name__ == "__main__":
    sys.exit(main())
