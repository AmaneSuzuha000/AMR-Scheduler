"""Self-contained MAPF / Lifelong-MAPD grid benchmarks reproducing CPS-A* Section 5.1 (Maps A/B/C)
plus MovingAI `.map` / `.scen` loader support.

Strictly zero dependency on FMS kitting DAGs, workstations, crews, tools, or RL checkpoints.
Every cell `(x, y)` maps directly to a capacity-1 node `"x_y"` with 4-connected unit-step
(`length=1.0, speed_limit=1.0, capacity=1`) bidirectional edges matching Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

from amr_sched.network import Edge, Network, Node, NodeKind


@dataclass(frozen=True)
class GridMapSpec:
    """Grid MAPF/MAPD benchmark layout + designated pickup/dropoff candidate zones."""

    name: str
    width: int
    height: int
    obstacles: frozenset[tuple[int, int]]
    traversable: tuple[tuple[int, int], ...]
    pickup_cells: tuple[tuple[int, int], ...]
    dropoff_cells: tuple[tuple[int, int], ...]
    network: Network

    @staticmethod
    def cell_id(x: int, y: int) -> str:
        return f"C_{x}_{y}"

    @staticmethod
    def parse_cell_id(nid: str) -> tuple[int, int]:
        _, xs, ys = nid.split("_", 2)
        return int(xs), int(ys)


def _grid_to_network(width: int, height: int, obstacles: set[tuple[int, int]]) -> tuple[Network, tuple[tuple[int, int], ...]]:
    net = Network()
    traversable: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                continue
            traversable.append((x, y))
            net.add_node(Node(GridMapSpec.cell_id(x, y), float(x), float(y), kind=NodeKind.AISLE, capacity=1))
    for x, y in traversable:
        u = GridMapSpec.cell_id(x, y)
        for dx, dy in ((1, 0), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
                v = GridMapSpec.cell_id(nx, ny)
                net.add_bidirectional(u, v, length=1.0, speed_limit=1.0, capacity=1)
    return net, tuple(traversable)


def _make_grid_spec(
    name: str,
    width: int,
    height: int,
    obstacles: Iterable[tuple[int, int]],
    pickup_cells: list[tuple[int, int]],
    dropoff_cells: list[tuple[int, int]],
) -> GridMapSpec:
    obs_set = {(x, y) for x, y in obstacles if 0 <= x < width and 0 <= y < height}
    net, trav = _grid_to_network(width, height, obs_set)
    trav_set = set(trav)
    p_clean = tuple(sorted(c for c in set(pickup_cells) if c in trav_set))
    d_clean = tuple(sorted(c for c in set(dropoff_cells) if c in trav_set))
    if not p_clean:
        p_clean = trav
    if not d_clean:
        d_clean = trav
    return GridMapSpec(
        name=name,
        width=width,
        height=height,
        obstacles=frozenset(obs_set),
        traversable=trav,
        pickup_cells=p_clean,
        dropoff_cells=d_clean,
        network=net,
    )


def build_map_a() -> GridMapSpec:
    """Map A (16 x 11): vertical rack aisles + central horizontal cross corridor + narrow right bottleneck."""
    w, h = 16, 11
    obs: set[tuple[int, int]] = set()
    for wall_x in (1, 5, 8, 11):
        for y in (1, 2, 3, 4, 6, 7, 8, 9):
            obs.add((wall_x, y))
    # Right-hand narrow single-lane corridor along x=14 flanked by vertical obstacles at x=13 and x=15
    for y in range(2, 9):
        obs.add((13, y))
        obs.add((15, y))

    pickups = [
        (x, y)
        for x in (2, 3, 4, 6, 7, 9, 10)
        for y in (2, 3, 7, 8)
        if (x, y) not in obs
    ]
    # Concentrated boundary outbound docks creating high traffic through shared entrances & bottleneck
    dropoffs = [(x, y) for x in (2, 6, 9, 12, 14) for y in (0, 10) if (x, y) not in obs]
    return _make_grid_spec("Map_A", w, h, obs, pickups, dropoffs)


def build_map_b() -> GridMapSpec:
    """Map B (20 x 20): dense discrete-point storage array with flexible aisles and perimeter docks."""
    w, h = 20, 20
    obs: set[tuple[int, int]] = set()
    for x in (2, 5, 8, 11, 14, 17):
        for y in (2, 5, 8, 11, 14, 17):
            obs.add((x, y))
    pickups = [
        (x, y)
        for x in range(3, 17)
        for y in range(3, 17)
        if (x, y) not in obs and (x % 3 == 0 or y % 3 == 0)
    ]
    dropoffs = [
        (x, 0) for x in range(4, 16, 3)
    ] + [
        (x, 19) for x in range(4, 16, 3)
    ] + [
        (0, y) for y in range(4, 16, 3)
    ] + [
        (19, y) for y in range(4, 16, 3)
    ]
    return _make_grid_spec("Map_B", w, h, obs, pickups, dropoffs)


def build_map_c() -> GridMapSpec:
    """Map C (20 x 20): concentric ring-structured layout with 4 narrow shared entrances."""
    w, h = 20, 20
    obs: set[tuple[int, int]] = set()
    # Outer ring at margin 2..17 (2-thick walls? 1-thick clear ring walls with central gates at x=9..10 / y=9..10)
    for k in range(2, 18):
        for x, y in ((k, 2), (k, 17), (2, k), (17, k)):
            if x in (9, 10) or y in (9, 10):
                continue
            obs.add((x, y))
    # Inner ring at margin 5..14 with gates at x=9..10 / y=9..10
    for k in range(5, 15):
        for x, y in ((k, 5), (k, 14), (5, k), (14, k)):
            if x in (9, 10) or y in (9, 10):
                continue
            obs.add((x, y))
    # Central core hub block leaving 1-cell ring corridor at 7..12
    for x in range(8, 12):
        for y in range(8, 12):
            obs.add((x, y))

    pickups = [
        (x, y)
        for x in (3, 4, 6, 7, 12, 13, 15, 16)
        for y in (3, 4, 6, 7, 12, 13, 15, 16)
        if (x, y) not in obs
    ]
    dropoffs = [
        (x, 0) for x in (2, 7, 12, 17)
    ] + [
        (x, 19) for x in (2, 7, 12, 17)
    ] + [
        (0, y) for y in (5, 14)
    ] + [
        (19, y) for y in (5, 14)
    ]
    return _make_grid_spec("Map_C", w, h, obs, pickups, dropoffs)


def build_map_rooms_32x32() -> GridMapSpec:
    """Large 32 x 32 multi-room MovingAI-style benchmark with narrow single-cell doorways."""
    w, h = 32, 32
    obs: set[tuple[int, int]] = set()
    for wall_pos in (7, 15, 23):
        for k in range(w):
            # Leave 2-cell spaced narrow 1-cell doorways per quadrant
            if k not in (3, 11, 19, 27):
                obs.add((wall_pos, k))
                obs.add((k, wall_pos))
    # Interior pillars for realistic detour structure
    for qx in (3, 11, 19, 27):
        for qy in (3, 11, 19, 27):
            if (qx + 1, qy + 1) not in obs:
                obs.add((qx + 1, qy + 1))

    pickups = [
        (x, y)
        for x in range(1, w - 1, 2)
        for y in range(1, h - 1, 2)
        if (x, y) not in obs and x not in (7, 15, 23) and y not in (7, 15, 23)
    ]
    dropoffs = [
        (x, 0) for x in range(2, w - 1, 4) if (x, 0) not in obs
    ] + [
        (x, h - 1) for x in range(2, w - 1, 4) if (x, h - 1) not in obs
    ] + [
        (0, y) for y in range(2, h - 1, 4) if (0, y) not in obs
    ] + [
        (w - 1, y) for y in range(2, h - 1, 4) if (w - 1, y) not in obs
    ]
    return _make_grid_spec("Rooms_32x32", w, h, obs, pickups, dropoffs)


def build_map_warehouse_40x40() -> GridMapSpec:
    """Large 40 x 40 robotic fulfillment warehouse with deep rack aisles & perimeter stations."""
    w, h = 40, 40
    obs: set[tuple[int, int]] = set()
    # Vertical rack blocks: 2-wide racks every 5 cols across 4 horizontal cross-aisles
    for rx in range(4, 36, 5):
        for dx in (0, 1):
            for y in range(3, 37):
                if y in (10, 19, 20, 29):
                    continue
                obs.add((rx + dx, y))

    pickups = [
        (x, y)
        for x in range(3, 37)
        for y in range(3, 37)
        if (x, y) not in obs and (x % 5 in (2, 3)) and y not in (10, 19, 20, 29)
    ]
    dropoffs = [
        (x, 0) for x in range(2, 38, 3) if (x, 0) not in obs
    ] + [
        (x, 39) for x in range(2, 38, 3) if (x, 39) not in obs
    ] + [
        (0, y) for y in range(4, 36, 4) if (0, y) not in obs
    ] + [
        (39, y) for y in range(4, 36, 4) if (39, y) not in obs
    ]
    return _make_grid_spec("Warehouse_40x40", w, h, obs, pickups, dropoffs)


def save_movingai_map_scen(
    grid: GridMapSpec,
    requests: list[tuple[str, str, str]],
    map_path: str | Path,
    scen_path: str | Path,
) -> None:
    """Export `GridMapSpec` + requests in standard MovingAI `.map` / `.scen` benchmark format."""
    mp = Path(map_path)
    sp = Path(scen_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    sp.parent.mkdir(parents=True, exist_ok=True)
    rows = ["type octile", f"height {grid.height}", f"width {grid.width}", "map"]
    for y in range(grid.height):
        rows.append("".join("@" if (x, y) in grid.obstacles else "." for x in range(grid.width)))
    mp.write_text("\n".join(rows) + "\n", encoding="utf-8")

    srows = ["version 1"]
    for i, (_, st, gl) in enumerate(requests):
        sx, sy = GridMapSpec.parse_cell_id(st)
        gx, gy = GridMapSpec.parse_cell_id(gl)
        dist = abs(gx - sx) + abs(gy - sy)
        srows.append(f"{i // 10}\t{mp.name}\t{grid.width}\t{grid.height}\t{sx}\t{sy}\t{gx}\t{gy}\t{dist:.4f}")
    sp.write_text("\n".join(srows) + "\n", encoding="utf-8")


def load_movingai_map(path: str | Path, *, name: str | None = None) -> GridMapSpec:
    """Load a standard MovingAI `.map` benchmark file into `GridMapSpec`."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    h = 0
    w = 0
    map_idx = -1
    for i, raw in enumerate(lines):
        low = raw.strip().lower()
        if low.startswith("height"):
            h = int(low.split()[1])
        elif low.startswith("width"):
            w = int(low.split()[1])
        elif low == "map":
            map_idx = i + 1
            break
    if map_idx < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid MovingAI .map header in {path}")
    obs: set[tuple[int, int]] = set()
    for y in range(h):
        row = lines[map_idx + y].strip()
        for x, ch in enumerate(row[:w]):
            if ch != ".":
                obs.add((x, y))
    net, trav = _grid_to_network(w, h, obs)
    return GridMapSpec(
        name=name or Path(path).stem,
        width=w,
        height=h,
        obstacles=frozenset(obs),
        traversable=trav,
        pickup_cells=trav,
        dropoff_cells=trav,
        network=net,
    )


def load_movingai_scen(path: str | Path, grid: GridMapSpec, *, limit: int | None = None) -> list[tuple[str, str]]:
    """Parse MovingAI `.scen` file as `(start_node_id, goal_node_id)` pairs."""
    pairs: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        st = raw.strip()
        if not st or st.lower().startswith("version"):
            continue
        parts = st.split()
        if len(parts) < 8:
            continue
        sx, sy, gx, gy = int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
        if (sx, sy) in grid.obstacles or (gx, gy) in grid.obstacles:
            continue
        if (sx, sy) == (gx, gy):
            continue
        pairs.append((GridMapSpec.cell_id(sx, sy), GridMapSpec.cell_id(gx, gy)))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def generate_oneshot_requests(
    grid: GridMapSpec,
    n_agvs: int,
    *,
    seed: int,
    concentrated_goals: bool = True,
) -> list[tuple[str, str, str]]:
    """Generate distinct `(agv_id, start_node, goal_node)` requests for one-shot MAPF."""
    rng = random.Random(int(seed))
    trav = [GridMapSpec.cell_id(x, y) for x, y in grid.traversable]
    if n_agvs > len(trav):
        raise ValueError(f"n_agvs={n_agvs} exceeds traversable cells={len(trav)} on {grid.name}")
    starts = rng.sample(trav, n_agvs)

    goal_pool = (
        [GridMapSpec.cell_id(x, y) for x, y in grid.dropoff_cells]
        if concentrated_goals and len(grid.dropoff_cells) >= n_agvs
        else list(trav)
    )
    rng.shuffle(goal_pool)
    used_goals: set[str] = set()
    reqs: list[tuple[str, str, str]] = []
    fallback_pool = list(trav)
    rng.shuffle(fallback_pool)
    for idx, st in enumerate(starts):
        cand = next((g for g in goal_pool if g != st and g not in used_goals), None)
        if cand is None:
            cand = next(g for g in fallback_pool if g != st and g not in used_goals)
        used_goals.add(cand)
        reqs.append((f"AGV_{idx:03d}", st, cand))
    return reqs


def generate_fixed_bg_and_test_requests(
    grid: GridMapSpec,
    n_bg: int,
    n_test: int,
    *,
    seed: int,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Generate disjoint fixed-background + inserted test requests (reproducing CPS-A* Table 5)."""
    rng = random.Random(int(seed))
    trav = [GridMapSpec.cell_id(x, y) for x, y in grid.traversable]
    total = n_bg + n_test
    if 2 * total > len(trav):
        # Allow goal sharing across non-overlapping starts if crowded, while keeping starts distinct
        starts = rng.sample(trav, total)
        goals = [rng.choice([g for g in trav if g != s]) for s in starts]
    else:
        picked = rng.sample(trav, 2 * total)
        starts = picked[:total]
        goals = picked[total:]

    bg = [(f"BG_{i:03d}", starts[i], goals[i]) for i in range(n_bg)]
    tst = [(f"TST_{i:03d}", starts[n_bg + i], goals[n_bg + i]) for i in range(n_test)]
    return bg, tst


def generate_lifelong_requests(
    grid: GridMapSpec,
    n_agvs: int,
    n_requests: int,
    *,
    seed: int,
    concentrated_dropoff: bool = True,
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Generate `(initial_agv_nodes, [(req_id, pickup, dropoff), ...])` for Lifelong MAPD."""
    rng = random.Random(int(seed))
    trav = [GridMapSpec.cell_id(x, y) for x, y in grid.traversable]
    init_cells = rng.sample(trav, min(n_agvs, len(trav)))
    init_pos = {f"AGV_{i:03d}": init_cells[i] for i in range(n_agvs)}

    p_pool = [GridMapSpec.cell_id(x, y) for x, y in grid.pickup_cells] or trav
    d_pool = (
        [GridMapSpec.cell_id(x, y) for x, y in grid.dropoff_cells]
        if concentrated_dropoff
        else trav
    ) or trav

    reqs: list[tuple[str, str, str]] = []
    for k in range(n_requests):
        pu = rng.choice(p_pool)
        do = rng.choice(d_pool)
        while do == pu and len(d_pool) > 1:
            do = rng.choice(d_pool)
        reqs.append((f"REQ_{k:04d}", pu, do))
    return init_pos, reqs


def build_official_trc_map(which: str) -> GridMapSpec:
    """Build exact obstacle & outbound/workstation geometry from `pengguo-hub/CPSAStar-TRC` `map_config.py`."""
    code = which.strip().upper()
    if code == "A":
        w, h = 16, 11
        obs = {(x, y) for x in range(1, 14, 2) for y in (*range(1, 5), *range(6, 10))}
        outbound = [(15, y) for y in range(3, 8)]
        net, trav = _grid_to_network(w, h, obs)
        p_cells = tuple((x, y) for x, y in trav if (x, y) not in outbound)
        d_cells = tuple(c for c in outbound if c in trav)
        return GridMapSpec("Official_Map_A", w, h, frozenset(obs), trav, p_cells, d_cells, net)
    if code == "B":
        w, h = 20, 20
        obs = {(x, y) for y in (2, 5, 8, 11, 14, 17) for x in (2, 5, 8, 11, 14, 17)}
        outbound = [(0, 5), (0, 14), (5, 0), (5, 19), (14, 0), (14, 19), (19, 5), (19, 14)]
        net, trav = _grid_to_network(w, h, obs)
        p_cells = tuple((x, y) for x, y in trav if (x, y) not in outbound)
        d_cells = tuple(c for c in outbound if c in trav)
        return GridMapSpec("Official_Map_B", w, h, frozenset(obs), trav, p_cells, d_cells, net)
    if code == "C":
        w, h = 20, 20
        obs = set()
        for x0, y0, x1, y1 in ((2, 2, 17, 17), (5, 5, 14, 14)):
            mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
            ent_x, ent_y = {mid_x, mid_x + 1}, {mid_y, mid_y + 1}
            for x in range(x0, x1 + 1):
                if x not in ent_x:
                    obs.add((x, y0))
                    obs.add((x, y1))
            for y in range(y0 + 1, y1):
                if y not in ent_y:
                    obs.add((x0, y))
                    obs.add((x1, y))
        inner = (
            [(x, 8) for x in range(8, 12)]
            + [(x, 11) for x in range(8, 12)]
            + [(8, y) for y in range(9, 11)]
            + [(11, y) for y in range(9, 11)]
        )
        net, trav = _grid_to_network(w, h, obs)
        p_cells = tuple((x, y) for x, y in trav if (x, y) not in inner)
        d_cells = tuple(c for c in inner if c in trav)
        return GridMapSpec("Official_Map_C", w, h, frozenset(obs), trav, p_cells, d_cells, net)
    raise ValueError(f"Unknown official TRC map code: {which!r}")
