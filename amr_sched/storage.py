"""Data storage and persistence -- the storage module of R3, the system design.

Why SQLite
------------
* It ships with Python (``sqlite3``), so there is no dependency at all, which matches the
  specification's Python technology stack requirement.
* It is a single file, so it can be submitted alongside the thesis and a reviewer can query
  it directly to reproduce results.
* It supports indexes and aggregate queries, which suits questions like "where was each
  vehicle at step 37" far better than CSV does.

Design trade-off
----------------
*Simulation results are written out both ways*, to CSV and to SQLite:

* CSV, for a human to look at and for the ``analyze_*.py`` scripts to read directly.
* SQLite, for querying trajectories and states by condition, which is what the visualisation
  and the later web layer need.

Both come from the same ``SimulationResult``, so the two can never disagree.

Schema
-------
=========  ================================================
table      contents
=========  ================================================
runs       summary metrics for one simulation run (one row per run)
tasks      every task in that run, with assign/pickup/complete times
agvs       per-AGV statistics for that run (busy/idle/blocked steps)
steps      step-by-step snapshots (optional: each vehicle's position and state per step)
=========  ================================================

Usage::

    from amr_sched.storage import ResultStore
    store = ResultStore("results/runs.db")
    store.save_run(result, scenario=cfg, label="matrix/spacetime/50/4/seed0")
    store.query("SELECT label, makespan_s FROM runs ORDER BY makespan_s LIMIT 5")
    store.close()
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from typing import Any, Iterable, Optional, Sequence

from amr_sched.entities import AgvState, TaskStatus
from amr_sched.simulation import SimulationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    solver          TEXT,
    allocator       TEXT,
    reschedule      TEXT,
    n_tasks         INTEGER,
    n_agvs          INTEGER,
    seed            INTEGER,
    tasks_completed INTEGER,
    makespan_s      REAL,
    on_time_rate    REAL,
    urgent_on_time_rate REAL,
    agv_utilization REAL,
    conflict_wait_steps INTEGER,
    preemption_count INTEGER,
    material_waste_steps REAL,
    deadlock_breaks INTEGER,
    collision_events INTEGER,
    hard_failures   INTEGER,
    distance_m      REAL,
    config_json     TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    run_id       INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    task_id      TEXT    NOT NULL,
    pickup       TEXT    NOT NULL,
    dropoff      TEXT    NOT NULL,
    priority     INTEGER NOT NULL,
    release_step INTEGER NOT NULL,
    due_step     INTEGER,
    status       TEXT    NOT NULL,
    assigned_agv TEXT,
    assigned_step INTEGER,
    pickup_step  INTEGER,
    complete_step INTEGER,
    flow_steps   INTEGER,
    PRIMARY KEY (run_id, task_id)
);

CREATE TABLE IF NOT EXISTS agvs (
    run_id       INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    agv_id       TEXT    NOT NULL,
    start_node   TEXT,
    end_node     TEXT,
    busy_steps   INTEGER NOT NULL DEFAULT 0,
    idle_steps   INTEGER NOT NULL DEFAULT 0,
    blocked_steps INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, agv_id)
);

CREATE TABLE IF NOT EXISTS steps (
    run_id   INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step     INTEGER NOT NULL,
    agv_id   TEXT    NOT NULL,
    node     TEXT,
    state    TEXT    NOT NULL,
    task_id  TEXT,
    PRIMARY KEY (run_id, step, agv_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step);
CREATE INDEX IF NOT EXISTS idx_runs_label ON runs(label);
"""


class ResultStore:
    """SQLite persistence for simulation results.

    Every write happens in **a single transaction**: either the whole run lands in the
    database,
    or nothing is left behind, so a partial write can never contaminate later statistics.
    """

    def __init__(self, path: str = "results/runs.db") -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------- writes ----------

    def save_run(self, result: SimulationResult,
                 label: str = "",
                 config: Any = None,
                 record_steps: bool = False) -> int:
        """Write one run to the database and return its run_id.

        With ``record_steps=True`` the per-step snapshots are stored as well. At 100 tasks and
        6 vehicles that is about 4000 steps
        x 6 vehicles = 24 thousand rows, which is acceptable; but batch experiments leave it
        off by default to save space.
        """
        s = result.summary()
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO runs (label, solver, allocator, reschedule,"
                " n_tasks, n_agvs, seed, tasks_completed, makespan_s,"
                " on_time_rate, urgent_on_time_rate, agv_utilization,"
                " conflict_wait_steps, preemption_count,"
                " material_waste_steps, deadlock_breaks, collision_events,"
                " hard_failures, distance_m, config_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (label,
                 type(getattr(result, "planner", None)).__name__,
                 getattr(result, "allocator_name", None),
                 getattr(result, "reschedule_policy", None),
                 len(result.tasks), len(result.agvs),
                 getattr(config, "seed", None),
                 int(s["tasks_completed"]),
                 float(s["makespan_seconds"]),
                 float(s["on_time_rate"]),
                 float(s["urgent_on_time_rate"]),
                 float(s["agv_utilization"]),
                 int(s["conflict_wait_steps"]),
                 int(s["preemption_count"]),
                 float(s.get("material_waste_steps", 0.0)),
                 int(s["deadlock_breaks"]),
                 int(s["collision_events"]),
                 int(s["hard_failures"]),
                 float(getattr(result, "total_distance", 0.0)),
                 self._dump_config(config)))
            run_id = int(cur.lastrowid)

            cur.executemany(
                "INSERT INTO tasks (run_id, task_id, pickup, dropoff,"
                 " priority, release_step, due_step, status, assigned_agv,"
                 " assigned_step, pickup_step, complete_step, flow_steps)"
                 " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, t.task_id, t.pickup, t.dropoff,
                  int(getattr(t.priority, "value", t.priority)),
                  int(t.release_step), t.due_step,
                  t.status.value if isinstance(t.status, TaskStatus) else str(t.status),
                  t.assigned_agv, t.assigned_step, t.pickup_step,
                  t.complete_step,
                  (t.complete_step - t.release_step)
                  if t.complete_step is not None else None)
                 for t in result.tasks])

            cur.executemany(
                "INSERT INTO agvs (run_id, agv_id, start_node, end_node,"
                 " busy_steps, idle_steps, blocked_steps)"
                 " VALUES (?,?,?,?,?,?,?)",
                [(run_id, a.agv_id, a.start_node, a.cur_node,
                  int(a.busy_steps), int(a.idle_steps), int(a.blocked_steps))
                 for a in result.agvs])

            if record_steps and getattr(result, "trajectory", None):
                cur.executemany(
                    "INSERT INTO steps (run_id, step, agv_id, node, state, task_id)"
                    " VALUES (?,?,?,?,?,?)",
                    self._step_rows(run_id, result))

            self._conn.commit()
            return run_id
        except Exception:
            self._conn.rollback()
            raise

    @staticmethod
    def _step_rows(run_id: int, result: SimulationResult) -> Iterable[tuple]:
        for rec in result.trajectory:
            for agv_id, state in rec.agv_states.items():
                node = rec.agv_positions.get(agv_id)
                task = rec.agv_tasks.get(agv_id)
                st = state.value if isinstance(state, AgvState) else str(state)
                yield (run_id, int(rec.step), agv_id, node, st, task)

    @staticmethod
    def _dump_config(config: Any) -> Optional[str]:
        if config is None:
            return None
        # A plain dict has no __dict__ and is not a dataclass, so it needs handling on its
        # own;
        # otherwise passing in an already-serialised dict crashes at write time.
        if isinstance(config, dict):
            d = dict(config)
        else:
            try:
                d = asdict(config)
            except TypeError:
                d = {k: v for k, v in vars(config).items()
                     if not k.startswith("_")}
        return json.dumps(d, ensure_ascii=False, default=str)

    # ---------- queries ----------

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a read-only query and return the rows."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        self._conn.row_factory = None
        return rows

    def run_count(self) -> int:
        return int(self.query("SELECT COUNT(*) AS n FROM runs")[0]["n"]);

    def label_count(self, label: str) -> int:
        return int(self.query(
            "SELECT COUNT(*) AS n FROM runs WHERE label = ?", (label,))[0]["n"]);

    def agv_track(self, run_id: int, agv_id: str) -> list[sqlite3.Row]:
        """The step-by-step trajectory of one vehicle within one run."""
        return self.query(
            "SELECT step, node, state, task_id FROM steps"
            " WHERE run_id = ? AND agv_id = ? ORDER BY step",
            (run_id, agv_id));

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()