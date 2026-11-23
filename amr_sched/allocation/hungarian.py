"""匈牙利算法：一步内的最优二分匹配（最小总代价）。

为什么需要它
------------
贪心是"全局最便宜的一对先配"，但局部最优不等于全局最优：
先配掉的那一对可能占用了别人更需要的那台车。
匈牙利算法在**一个决策步内**求总代价最小的完整匹配，作为"上界参照"。

复杂度 O(n^3)；本系统的空闲车与待分配任务一般不超过几十个，够用。
"""

from __future__ import annotations

INF = float("inf")


def hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """求解方阵的最小代价完美匹配。

    返回 ``[(row, col), ...]``。cost 必须方阵且元素有限（用大数代替无穷）。
    使用经典的 O(n^3) Jonker-Volgenant 风格实现（带势函数）。
    """
    n = len(cost)
    if n == 0:
        return []
    if any(len(r) != n for r in cost):
        raise ValueError("cost 必须是方阵")

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)      # p[j] = 第 j 列匹配的行（1-based）
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    out: list[tuple[int, int]] = []
    for j in range(1, n + 1):
        if p[j] != 0:
            out.append((p[j] - 1, j - 1))
    return out
