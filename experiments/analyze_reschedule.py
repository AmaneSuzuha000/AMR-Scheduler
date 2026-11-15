"""加急重调度对比分析。"""

import argparse
import csv
import math
import sys
from collections import defaultdict


def exact_wilcoxon_p(diffs):
    """Wilcoxon 符号秩检验（双侧）。

    n <= 20 用精确枚举（2^n），n > 20 用带连续性校正的正态近似。
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0

    # 秩（处理并列，取平均秩）
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank_of = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank_of[order[k]] = avg
        i = j + 1

    w_plus = sum(rank_of[i] for i in range(n) if d[i] > 0)
    total = n * (n + 1) / 2.0
    w = min(w_plus, total - w_plus)

    if n <= 20:
        obs = round(w)
        count = 0
        for mask in range(1 << n):
            s = 0.0
            for i in range(n):
                if mask >> i & 1:
                    s += rank_of[i]
            if min(s, total - s) <= obs + 1e-9:
                count += 1
        return count / (1 << n)

    # 正态近似（含并列校正）
    mu = n * (n + 1) / 4.0
    # 并列校正项
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        tt = j - i + 1
        tie_term += tt ** 3 - tt
        i = j + 1
    sigma = ((n * (n + 1) * (2 * n + 1)) / 24.0 - tie_term / 48.0) ** 0.5
    if sigma <= 0:
        return 1.0
    z = (w - mu + 0.5) / sigma
    # 双侧 p = 2 * Phi(z)
    p = 2.0 * 0.5 * math.erfc(-z / math.sqrt(2.0))
    return min(1.0, p)


def mean(xs):
    xs = [x for x in xs if x == x]  # 去 NaN
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/reschedule.csv")
    args = p.parse_args(argv)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    for r in rows:
        for k in ("makespan_s", "on_time_rate", "urgent_on_time_rate",
                  "mean_urgent_response_steps", "agv_utilization",
                  "conflict_wait_steps", "preemption_count"):
            r[k] = float(r[k])
        r["n_tasks"] = int(r["n_tasks"])
        r["n_agvs"] = int(r["n_agvs"])
        r["seed"] = int(r["seed"])

    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["n_tasks"], r["n_agvs"])][r["policy"]].append(r)

    print("=" * 108)
    print("%5s %4s | %12s %12s %9s | %10s %10s | %8s %8s" %
          ("任务", "AGV", "加急准时(基线)", "加急准时(抢占)", "变化", "加急响应(基)", "加急响应(抢)", "抢占次数", "完成时间差"))
    print("-" * 108)
    for (nt, na) in sorted(cells):
        b = cells[(nt, na)]["none"]
        t = cells[(nt, na)]["preempt_unpicked"]
        bm = {k: mean([x[k] for x in b]) for k in
              ("urgent_on_time_rate", "mean_urgent_response_steps", "makespan_s", "on_time_rate")}
        tm = {k: mean([x[k] for x in t]) for k in bm}
        pmp = mean([x["preemption_count"] for x in t])
        print("%5d %4d | %12.3f %12.3f %+8.1f%% | %10.1f %10.1f | %8.2f %+8.2f%%" %
              (nt, na, bm["urgent_on_time_rate"], tm["urgent_on_time_rate"],
               (tm["urgent_on_time_rate"] - bm["urgent_on_time_rate"]) * 100,
               bm["mean_urgent_response_steps"], tm["mean_urgent_response_steps"],
               pmp,
               (tm["makespan_s"] - bm["makespan_s"]) / bm["makespan_s"] * 100))
    print("=" * 108)
    print("")
    print("配对显著性（同 seed 配对，精确 Wilcoxon 符号秩检验）：")
    # (字段, 中文名, 是否"越低越好")
    keys = [("urgent_on_time_rate", "加急准时率", False),
            ("mean_urgent_response_steps", "加急响应步数", True),
            ("makespan_s", "配送完成时间", True),
            ("on_time_rate", "总准时率", False)]
    print("  （改善 = 相对基线变好；越低越好的指标已按方向折算）")
    for k, label, lower_better in keys:
        diffs = []
        for (nt, na) in sorted(cells):
            b = {r["seed"]: r for r in cells[(nt, na)]["none"]}
            t = {r["seed"]: r for r in cells[(nt, na)]["preempt_unpicked"]}
            for s in b:
                if s in t:
                    diffs.append(t[s][k] - b[s][k])
        if lower_better:
            pos = sum(1 for d in diffs if d < 0)   # 变小 = 改善
            neg = sum(1 for d in diffs if d > 0)
        else:
            pos = sum(1 for d in diffs if d > 0)
            neg = sum(1 for d in diffs if d < 0)
        pv = exact_wilcoxon_p(diffs)
        print("  %-14s 改善/恶化 = %2d/%2d  (共%d对)  中位差=%+9.3f  p=%.4f%s"
              % (label, pos, neg, len(diffs), sorted(diffs)[len(diffs) // 2], pv,
                 "  *显著" if pv < 0.05 else ""))
    print("")
    print("整体（跨所有单元格）：")
    for pol in ("none", "preempt_unpicked"):
        g = [r for r in rows if r["policy"] == pol]
        print("  %-17s 加急准时率=%.3f  加急响应=%.1f步  总准时率=%.3f  完成时间=%.1f  抢占=%.2f" %
              (pol, mean([x["urgent_on_time_rate"] for x in g]),
               mean([x["mean_urgent_response_steps"] for x in g]),
               mean([x["on_time_rate"] for x in g]),
               mean([x["makespan_s"] for x in g]),
               mean([x["preemption_count"] for x in g])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
