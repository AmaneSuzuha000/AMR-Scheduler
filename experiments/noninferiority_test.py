"""非劣效检验：PREEMPT_UNPICKED 相对无抢占，完成时间是否"实质上没有变差"。

为什么需要它
------------
论文原来写「完成时间无损失」。但配对 Wilcoxon 得到 p=0.7590，
这只说明**未检测到显著差异**。不显著 != 等效：样本量不足时，
一个真实的 15% 退化同样会得到不显著的 p 值。

要正当地声称「没有实质性变差」，必须做**非劣效检验**：
   H0: mu_unpicked - mu_none >= delta   （变差了至少 delta）
   H1: mu_unpicked - mu_none <  delta   （变差小于 delta）
其中 delta 是**事前约定**的可接受损失界限。

delta 怎么定
-------------
不能事后按数据挑一个让检验通过的数——那是 p-hacking。
本文按业务可接受度事前设定：完成时间退化 **5%** 视为可接受上限，
依据是 6.3 节实测「2→4 车」带来的收益是 45.8%、「4→6 车」是 27.0%，
相较之下 5% 属于同一量级里的噪声水平，不构成工程意义上的损失。
同时报告 3% 与 10% 两档作敏感性分析，避免结论依赖单一边界。
"""
from __future__ import annotations

import csv
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from analyze_reschedule import exact_wilcoxon_p  # noqa: E402


def paired_diff(a: dict, b: dict, key: str) -> list[float]:
    """按 (n_tasks, n_agvs, seed) 配对，返回 a-b。"""
    ks = sorted(set(a) & set(b))
    return [a[k][key] - b[k][key] for k in ks]


def tost_p(diffs: list[float], delta: float) -> tuple[float, float, float]:
    """单侧非劣效检验。返回 (均值差, 标准化统计量, 单侧 p)。

    用配对差的单样本 t 检验近似（n=90，正态近似合理）。
    H0: mean(diff) >= delta  vs  H1: mean(diff) < delta
    """
    n = len(diffs)
    m = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("inf")
    if se == 0:
        return m, float("inf") if m < delta else float("-inf"), 0.0 if m < delta else 1.0
    t = (m - delta) / se
    # 单侧 p：P(T <= t)，用正态近似
    p = 0.5 * (1 + math.erf(t / math.sqrt(2)))
    return m, t, p


def main() -> int:
    rows = list(csv.DictReader(open("results/preempt_tiers.csv", encoding="utf-8")))
    by_policy: dict[str, dict] = {}
    for r in rows:
        key = (r["n_tasks"], r["n_agvs"], r["seed"])
        by_policy.setdefault(r["policy"], {})[key] = {
            "makespan_s": float(r["makespan_s"]) if r["makespan_s"] else float("nan"),
            "urgent_on_time_rate": float(r["urgent_on_time_rate"]),
            "material_waste_steps": float(r["material_waste_steps"]),
        }

    print("策略:", list(by_policy.keys()))
    base = by_policy["none"]
    tgt = by_policy["preempt_unpicked"]

    diffs = paired_diff(tgt, base, "makespan_s")
    base_vals = [base[k]["makespan_s"] for k in sorted(set(base) & set(tgt))]
    mean_base = st.mean(base_vals)
    print("")
    print("配对 n = %d" % len(diffs))
    print("基线均值 %.1f s，主方法均值 %.1f s，均值差 %+.1f s（%+.2f%%）"
          % (mean_base, mean_base + st.mean(diffs), st.mean(diffs),
             st.mean(diffs) / mean_base * 100))
    print("")

    # 等价性/差异性：原 Wilcoxon
    w = exact_wilcoxon_p(diffs)
    n_better = sum(1 for d in diffs if d < 0)
    n_worse = sum(1 for d in diffs if d > 0)
    n_same = sum(1 for d in diffs if d == 0)
    print("原 Wilcoxon（差异性检验）: p = %.4f   好/差/相同 = %d/%d/%d"
          % (w, n_better, n_worse, n_same))
    print("  => 未检测到显著差异（不等于等效）")
    print("")

    print("非劣效检验（H0: 变差 >= delta）")
    print("%-8s %12s %10s %10s %s" % ("delta", "实际变差", "t 统计量", "单侧 p", "判定"))
    for pct in (3, 5, 10):
        delta = mean_base * pct / 100.0
        m, t, p = tost_p(diffs, delta)
        ok = "非劣效成立" if p < 0.05 else "不能判定"
        print("%-8s %12.2f %10.2f %10.4f %s"
              % ("%d%%" % pct, m, t, p, ok))

    print("")
    print("其它指标（unpicked vs none，均值）")
    for key in ("urgent_on_time_rate", "material_waste_steps"):
        a = st.mean(v[key] for v in tgt.values())
        b = st.mean(v[key] for v in base.values())
        print("  %-24s %.4f -> %.4f" % (key, b, a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())