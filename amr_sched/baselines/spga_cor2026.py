"""External Benchmark: Similarity-based Parallel Genetic Algorithm (Jia Li et al., COR 2026).

faithful adaptation of:
  Jia Li et al., "Dynamic flexible job-shop scheduling with AGV failures:
  Rescheduling based on a similarity-based parallel genetic algorithm",
  Computers & Operations Research 193 (2026) 107524.

Implements:
  - 3-layer chromosome encoding (OS: topological op order, MS: bay/crew selection, AS: AGV assignment)
  - Greedy insertion decoding under Assumption 10 (no path conflicts, static shortest-path transit)
  - Eq. (12) standardized cosine-similarity mate selection (GAS) avoiding premature convergence
  - Synchronized Precedence Operation Crossover (POX) across OS/MS/AS + layer mutation
  - Individual life-cycle L-value renewal (`U[1, max(1, round(alpha * max_iters))]`, GAL)

Supports two evaluation/deployment protocols on our shop-floor runtime:
  1. `mode="relaxed"`: Optimizes under Jia Li's Assumption-10 relaxed model (static shortest-path
     travel, zero reservation/congestion blocking), then submits the resulting assignment/bays/crews
     through `NativeTransactionOwner` for full closed-loop trajectory audit.
  2. `mode="g1_unified"`: Respects live 5-tier CVCR commitments and scores elite chromosomes on
     our spacetime-coupled `CVCREvaluator` (`B2`) within an explicit real-time `budget_ms` window,
     emitting full `CandidateProposal` fields (`assignment`, `bays`, `crews`, `task_sequences`,
     `op_priorities`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import time
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot, OperationSnapshot, TaskSnapshot, op_predecessors
from amr_sched.network import Network


@dataclass(frozen=True)
class SPGAConfig:
    """Configuration for Jia Li et al. (COR 2026) SPGA solver."""

    pop_size: int = 80
    max_iters: int = 40
    alpha: float = 0.04
    p_cross: float = 0.9
    p_mut: float = 0.1
    b2_elite_evals: int = 6
    mode: str = "relaxed"  # "relaxed" | "g1_unified"
    budget_ms: float = 100.0
    seed: int = 0


@dataclass
class SPGADecodedPlan:
    """Decoded 3-layer chromosome solution and surrogate/physical objective."""

    assign: dict[str, str]
    bays: dict[str, str]
    crews: dict[str, int]
    op_prio: dict[str, int]
    task_orders: dict[str, list[str]]
    makespan: int
    tardiness: int
    objective: float


class SPGAProblem:
    """Normalized FJSP-AGV view extracted from a CVCR NativeSnapshot."""

    def __init__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        alpha_makespan: float = 3.0,
    ) -> None:
        self.snapshot = snapshot
        self.network = network
        self.alpha_makespan = float(alpha_makespan)
        self.clf = classify_commitments(snapshot)
        self.ops: list[OperationSnapshot] = sorted(snapshot.operations, key=lambda o: o.op_id)
        self.op_index: dict[str, int] = {op.op_id: i for i, op in enumerate(self.ops)}
        self.tasks: list[TaskSnapshot] = sorted(snapshot.tasks, key=lambda t: t.task_id)
        self.tasks_by_id: dict[str, TaskSnapshot] = {t.task_id: t for t in self.tasks}

        disturbed = set(self.clf.disturbed_robots)
        self.healthy_robots: list[str] = sorted(
            r.robot_id for r in snapshot.robots if r.robot_id not in disturbed and r.state != "faulted"
        )
        if not self.healthy_robots:
            self.healthy_robots = sorted(r.robot_id for r in snapshot.robots if r.state != "faulted")
        if not self.healthy_robots:
            self.healthy_robots = sorted(r.robot_id for r in snapshot.robots)
        self.robots_by_id = {r.robot_id: r for r in snapshot.robots}

        self.surr = _LazyKittingSurrogate(
            network, snapshot, {op.op_id: max(snapshot.tick, op.due_step - op.proc_steps) for op in self.ops},
            alpha_makespan=alpha_makespan,
        )

        # Build legal bay/crew pairs per op honoring irrevocability when kit task picked/completed or started
        self.eligible_ms: list[list[tuple[str, int]]] = []
        for op in self.ops:
            kit_picked = any(
                self.tasks_by_id[kt].status.lower() in ("picked", "completed")
                for kt in op.kit_task_ids
                if kt in self.tasks_by_id
            )
            if op.started or kit_picked:
                self.eligible_ms.append([(op.bay_id, op.crew_id)])
            else:
                bays = tuple(op.eligible_bays) if op.eligible_bays else (op.bay_id,)
                crews = tuple(op.eligible_crews) if op.eligible_crews else (op.crew_id,)
                pairs = [(b, int(c)) for b in bays for c in crews]
                if (op.bay_id, int(op.crew_id)) not in pairs:
                    pairs.insert(0, (op.bay_id, int(op.crew_id)))
                self.eligible_ms.append(pairs)

        # Build legal AGV choices per task respecting CVCR commitments
        self.movable_task_indices: list[int] = []
        self.task_agv_choices: list[list[str]] = []
        for idx, t in enumerate(self.tasks):
            status_l = t.status.lower()
            is_irrev = (t.task_id in self.clf.frozen_tasks) or status_l in ("picked", "completed")
            if is_irrev and t.assigned_agv is not None:
                self.task_agv_choices.append([t.assigned_agv])
            else:
                self.task_agv_choices.append(self.healthy_robots if self.healthy_robots else [t.assigned_agv or "agv_0"])
                if len(self.task_agv_choices[-1]) > 1:
                    self.movable_task_indices.append(idx)

        # Build operation precedence graph
        self.preds: list[list[int]] = [[] for _ in self.ops]
        self.succs: list[list[int]] = [[] for _ in self.ops]
        for j, op in enumerate(self.ops):
            for pid in op_predecessors(op):
                if pid in self.op_index:
                    p = self.op_index[pid]
                    self.preds[j].append(p)
                    self.succs[p].append(j)


class SPGASolver:
    """Jia Li et al. (COR 2026) SPGA over integrated FJSP-AGV chromosomes."""

    def __init__(self, problem: SPGAProblem, cfg: SPGAConfig | None = None) -> None:
        self.prob = problem
        self.cfg = cfg or SPGAConfig()
        self.rng = random.Random(int(self.cfg.seed) + int(problem.snapshot.tick) * 131 + 17)

    def _topo_repair(self, seq: list[int]) -> list[int]:
        """Repair an operation sequence into a valid topological ordering via Kahn selection."""
        n = len(self.prob.ops)
        pos = {op_idx: i for i, op_idx in enumerate(seq)}
        indeg = [len(self.prob.preds[i]) for i in range(n)]
        ready = [i for i in range(n) if indeg[i] == 0]
        out: list[int] = []
        while ready:
            ready.sort(key=lambda idx: pos.get(idx, idx))
            cur = ready.pop(0)
            out.append(cur)
            for nxt in self.prob.succs[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        return out

    def _random_chromosome(self) -> tuple[list[int], list[int], list[int]]:
        n_ops = len(self.prob.ops)
        n_tasks = len(self.prob.tasks)
        indeg = [len(self.prob.preds[i]) for i in range(n_ops)]
        ready = [i for i in range(n_ops) if indeg[i] == 0]
        os_seq: list[int] = []
        while ready:
            pick_i = self.rng.randrange(len(ready))
            cur = ready.pop(pick_i)
            os_seq.append(cur)
            for nxt in self.prob.succs[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        ms = [self.rng.randrange(len(self.prob.eligible_ms[i])) for i in range(n_ops)]
        as_ = [self.rng.randrange(len(self.prob.task_agv_choices[j])) for j in range(n_tasks)]
        return os_seq, ms, as_

    def _seed_chromosome(self) -> tuple[list[int], list[int], list[int]]:
        """Seed chromosome matching current shop-floor incumbent state."""
        order_ops = sorted(
            range(len(self.prob.ops)),
            key=lambda i: (self.prob.ops[i].due_step, i),
        )
        os_seq = self._topo_repair(order_ops)
        ms = []
        for i, op in enumerate(self.prob.ops):
            pair = (op.bay_id, int(op.crew_id))
            choices = self.prob.eligible_ms[i]
            ms.append(choices.index(pair) if pair in choices else 0)
        as_ = []
        for j, t in enumerate(self.prob.tasks):
            choices = self.prob.task_agv_choices[j]
            if t.assigned_agv in choices:
                as_.append(choices.index(t.assigned_agv))
            else:
                # Greedy nearest healthy robot seed
                pick = min(
                    range(len(choices)),
                    key=lambda c_idx: self.prob.surr.tt(
                        self.prob.robots_by_id[choices[c_idx]].cur_node, t.pickup
                    ),
                )
                as_.append(pick)
        return os_seq, ms, as_

    def decode_relaxed(self, chrom: tuple[list[int], list[int], list[int]]) -> SPGADecodedPlan:
        """Decode (OS, MS, AS) via Assumption-10 greedy insertion (no path collision/delay)."""
        os_seq, ms, as_ = chrom
        snap = self.prob.snapshot
        t_now = int(snap.tick)

        bays: dict[str, str] = {}
        crews: dict[str, int] = {}
        op_prio: dict[str, int] = {}
        for pos, op_i in enumerate(os_seq):
            op = self.prob.ops[op_i]
            bay, crew = self.prob.eligible_ms[op_i][ms[op_i]]
            bays[op.op_id] = bay
            crews[op.op_id] = int(crew)
            op_prio[op.op_id] = t_now + pos * 2

        assign: dict[str, str] = {}
        for j, t in enumerate(self.prob.tasks):
            assign[t.task_id] = self.prob.task_agv_choices[j][as_[j]]

        # Group tasks per robot and order them according to op precedence order
        op_rank = {op.op_id: pos for pos, op_i in enumerate(os_seq) for op in (self.prob.ops[op_i],)}
        rob_node: dict[str, str] = {}
        rob_time: dict[str, float] = {}
        for rid in set(assign.values()):
            rob = self.prob.robots_by_id.get(rid)
            rob_node[rid] = rob.cur_node if rob is not None else self.prob.tasks[0].pickup
            rob_time[rid] = float(max(t_now, rob.faulted_until if rob is not None else t_now))

        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in assign.values()}
        for t in self.prob.tasks:
            rid = assign[t.task_id]
            by_robot.setdefault(rid, []).append(t)

        task_orders: dict[str, list[str]] = {}
        task_unload: dict[str, int] = {}
        for rid, tlist in by_robot.items():
            # Active picked/executing tasks MUST go first physically
            tlist.sort(
                key=lambda t: (
                    0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    op_rank.get(t.kit_op_id or "", 10000),
                    t.release_step,
                    t.task_id,
                )
            )
            task_orders[rid] = [t.task_id for t in tlist if t.status.lower() != "completed"]
            cur_n = rob_node.get(rid, tlist[0].pickup)
            cur_tm = rob_time.get(rid, float(t_now))
            for t in tlist:
                st_l = t.status.lower()
                if st_l == "completed":
                    task_unload[t.task_id] = int(t.complete_step or t_now)
                    continue
                target_drop = bays.get(t.kit_op_id or "", t.dropoff)
                if st_l in ("picked", "to_dropoff", "unloading"):
                    d_tt = self.prob.surr.tt(cur_n, target_drop)
                    cur_tm = max(cur_tm, float(t_now)) + d_tt + 5.0
                else:
                    e_tt = self.prob.surr.tt(cur_n, t.pickup)
                    arr_p = max(float(t.release_step), cur_tm + e_tt)
                    d_tt = self.prob.surr.tt(t.pickup, target_drop)
                    cur_tm = arr_p + 5.0 + d_tt + 5.0
                cur_n = target_drop
                task_unload[t.task_id] = int(round(cur_tm))

        # Greedy interval insertion on bays, crews, and fixed tools in OS order
        bay_intervals: dict[str, list[tuple[int, int]]] = {}
        crew_intervals: dict[int, list[tuple[int, int]]] = {}
        tool_intervals: dict[int, list[tuple[int, int]]] = {}
        op_end: dict[str, int] = {}

        def _earliest_slot(intervals: list[tuple[int, int]], ready: int, dur: int) -> int:
            s = ready
            for a, b in intervals:
                if s + dur <= a:
                    return s
                if s < b:
                    s = b
            return s

        def _insert_slot(intervals: list[tuple[int, int]], start: int, end: int) -> None:
            intervals.append((start, end))
            intervals.sort()

        for op_i in os_seq:
            op = self.prob.ops[op_i]
            if op.started and op.end_step is not None:
                op_end[op.op_id] = int(op.end_step)
                continue
            kit_ready = max(
                [t_now] + [task_unload.get(kt, t_now) for kt in op.kit_task_ids]
            )
            preds = op_predecessors(op)
            pred_ready = max([t_now] + [op_end.get(p, t_now) for p in preds])
            rdy = max(kit_ready, pred_ready)
            bay = bays[op.op_id]
            crew = crews[op.op_id]
            tool = int(op.tool_id)
            bi = bay_intervals.setdefault(bay, [])
            ci = crew_intervals.setdefault(crew, [])
            ti = tool_intervals.setdefault(tool, [])
            st = rdy
            for _ in range(16):
                sb = _earliest_slot(bi, st, op.proc_steps)
                sc = _earliest_slot(ci, st, op.proc_steps)
                stt = _earliest_slot(ti, st, op.proc_steps)
                nxt = max(sb, sc, stt)
                if nxt == st:
                    break
                st = nxt
            en = st + int(op.proc_steps)
            _insert_slot(bi, st, en)
            _insert_slot(ci, st, en)
            _insert_slot(ti, st, en)
            op_end[op.op_id] = en

        makespan = max(op_end.values(), default=t_now)
        tard = sum(
            (12 if op.is_urgent else 4) * max(0, op_end.get(op.op_id, makespan) - op.due_step)
            for op in self.prob.ops
        )
        obj = float(self.prob.alpha_makespan * makespan + tard)
        return SPGADecodedPlan(
            assign=assign,
            bays=bays,
            crews=crews,
            op_prio=op_prio,
            task_orders=task_orders,
            makespan=int(makespan),
            tardiness=int(tard),
            objective=obj,
        )

    @staticmethod
    def _to_vec(chrom: tuple[list[int], list[int], list[int]]) -> np.ndarray:
        os_seq, ms, as_ = chrom
        return np.asarray(os_seq + ms + as_, dtype=np.float64)

    @staticmethod
    def _cosine_sim_matrix(pop_vecs: np.ndarray) -> np.ndarray:
        """Eq. (12): z-score standardized cosine similarity matrix across chromosomes."""
        if len(pop_vecs) <= 1:
            return np.ones((len(pop_vecs), len(pop_vecs)), dtype=np.float64)
        mu = pop_vecs.mean(axis=0, keepdims=True)
        sd = pop_vecs.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z = (pop_vecs - mu) / sd
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        Zn = Z / norms
        return Zn @ Zn.T

    def _pox_crossover(
        self,
        p1: tuple[list[int], list[int], list[int]],
        p2: tuple[list[int], list[int], list[int]],
    ) -> tuple[list[int], list[int], list[int]]:
        """POX crossover on OS synchronized with corresponding MS & AS alleles."""
        n_ops = len(self.prob.ops)
        if n_ops <= 1:
            return (list(p1[0]), list(p1[1]), list(p1[2]))
        os1, ms1, as1 = p1
        os2, ms2, as2 = p2
        subset_k = max(1, self.rng.randint(1, max(1, n_ops - 1)))
        keep_ops = set(self.rng.sample(range(n_ops), subset_k))

        child_os = [-1] * n_ops
        for idx, op_i in enumerate(os1):
            if op_i in keep_ops:
                child_os[idx] = op_i
        fill_iter = (op_i for op_i in os2 if op_i not in keep_ops)
        for idx in range(n_ops):
            if child_os[idx] == -1:
                child_os[idx] = next(fill_iter)
        child_os = self._topo_repair(child_os)

        child_ms = [ms1[i] if i in keep_ops else ms2[i] for i in range(n_ops)]
        child_as = list(as1)
        for j, t in enumerate(self.prob.tasks):
            op_idx = self.prob.op_index.get(t.kit_op_id or "", -1)
            if op_idx not in keep_ops:
                child_as[j] = as2[j]
        return child_os, child_ms, child_as

    def _mutate(
        self, chrom: tuple[list[int], list[int], list[int]]
    ) -> tuple[list[int], list[int], list[int]]:
        os_seq, ms, as_ = list(chrom[0]), list(chrom[1]), list(chrom[2])
        if len(os_seq) >= 2:
            i, j = self.rng.sample(range(len(os_seq)), 2)
            os_seq[i], os_seq[j] = os_seq[j], os_seq[i]
            os_seq = self._topo_repair(os_seq)
        flex_ops = [i for i, c in enumerate(self.prob.eligible_ms) if len(c) > 1]
        if flex_ops:
            oi = self.rng.choice(flex_ops)
            ms[oi] = self.rng.randrange(len(self.prob.eligible_ms[oi]))
        if self.prob.movable_task_indices:
            tj = self.rng.choice(self.prob.movable_task_indices)
            as_[tj] = self.rng.randrange(len(self.prob.task_agv_choices[tj]))
        return os_seq, ms, as_

    def solve(
        self,
        evaluator: CVCREvaluator | None = None,
    ) -> tuple[SPGADecodedPlan, EvalResult | None, int]:
        """Run SPGA under relaxed or G1-unified mode; returns best decoded plan + optional B2 eval."""
        t0 = time.perf_counter()
        deadline = t0 + max(0.010, float(self.cfg.budget_ms) / 1000.0)
        is_unified = self.cfg.mode == "g1_unified" and evaluator is not None
        pop_n = min(self.cfg.pop_size, 48 if is_unified else self.cfg.pop_size)
        pop_n = max(12, pop_n)
        max_l = max(1, int(round(float(self.cfg.alpha) * float(self.cfg.max_iters))))

        pop: list[tuple[list[int], list[int], list[int]]] = [self._seed_chromosome()]
        while len(pop) < pop_n:
            pop.append(self._random_chromosome())
        lives = [self.rng.randint(1, max_l) for _ in pop]
        plans = [self.decode_relaxed(ch) for ch in pop]

        eval_count = 0
        best_idx = min(range(len(plans)), key=lambda i: plans[i].objective)
        best_plan = plans[best_idx]
        best_eval: EvalResult | None = None

        if is_unified and evaluator is not None:
            best_eval = evaluator.evaluate(
                best_plan.assign,
                op_priority=best_plan.op_prio,
                op_bay_overrides=best_plan.bays,
                op_crew_overrides=best_plan.crews,
                task_order_override=best_plan.task_orders or None,
            )
            eval_count += 1
            if best_eval.valid:
                evaluator.set_incumbent(
                    best_plan.assign,
                    best_eval,
                    op_priority=best_plan.op_prio,
                    op_bay_overrides=best_plan.bays,
                    task_order_override=best_plan.task_orders or None,
                )

        # Leave ~25% of budget for B2 verification in unified mode
        ga_cutoff = (
            t0 + (float(self.cfg.budget_ms) * 0.55) / 1000.0
            if is_unified
            else deadline - 0.002
        )

        for _it in range(self.cfg.max_iters):
            if time.perf_counter() >= ga_cutoff:
                break
            pop_vecs = np.stack([self._to_vec(ch) for ch in pop], axis=0)
            sim_mat = self._cosine_sim_matrix(pop_vecs)

            offspring: list[tuple[list[int], list[int], list[int]]] = []
            half = max(4, pop_n // 2)
            # Pair low-similarity elites cleanly (Eq. 12)
            fit_rank = sorted(range(len(pop)), key=lambda i: plans[i].objective)
            for p_i in fit_rank[:half]:
                if time.perf_counter() >= ga_cutoff:
                    break
                sims = sim_mat[p_i]
                # Choose mate with lowest standardized similarity (different individual)
                order_mates = np.argsort(sims)
                mate_j = int(order_mates[0]) if int(order_mates[0]) != p_i else int(order_mates[min(1, len(pop) - 1)])
                c1 = pop[p_i]
                if self.rng.random() < self.cfg.p_cross:
                    c1 = self._pox_crossover(pop[p_i], pop[mate_j])
                if self.rng.random() < self.cfg.p_mut:
                    c1 = self._mutate(c1)
                offspring.append(c1)

            off_plans = [self.decode_relaxed(ch) for ch in offspring]

            # Update life-cycle L-values (GAL) & deduplicate
            next_pop: list[tuple[list[int], list[int], list[int]]] = []
            next_plans: list[SPGADecodedPlan] = []
            next_lives: list[int] = []
            seen_keys: set[tuple[int, ...]] = set()

            combined = [
                (pop[i], plans[i], lives[i] - 1) for i in range(len(pop))
            ] + [
                (offspring[k], off_plans[k], self.rng.randint(1, max_l))
                for k in range(len(offspring))
            ]
            combined.sort(key=lambda item: item[1].objective)
            for idx_c, (ch, pl, lv) in enumerate(combined):
                # Always keep global elite (idx_c == 0) alive; cull expired L <= 0 otherwise
                if idx_c > 0 and lv <= 0:
                    continue
                key = tuple(ch[0]) + tuple(ch[1]) + tuple(ch[2])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                next_pop.append(ch)
                next_plans.append(pl)
                next_lives.append(max(1, lv))
                if len(next_pop) >= pop_n:
                    break

            while len(next_pop) < pop_n:
                rch = self._random_chromosome()
                next_pop.append(rch)
                next_plans.append(self.decode_relaxed(rch))
                next_lives.append(self.rng.randint(1, max_l))

            pop, plans, lives = next_pop, next_plans, next_lives
            if plans[0].objective < best_plan.objective:
                best_plan = plans[0]

        if is_unified and evaluator is not None:
            # Verify top diverse elites on true spacetime-coupled B2 evaluator inside budget
            fit_elites = sorted(range(len(plans)), key=lambda i: plans[i].objective)
            tested = 0
            seen_assign_keys: set[tuple[tuple[str, str], ...]] = set()
            for idx_e in fit_elites:
                if tested >= int(self.cfg.b2_elite_evals) or time.perf_counter() + 0.005 >= deadline:
                    break
                pl = plans[idx_e]
                akey = tuple(sorted(pl.assign.items()))
                if akey in seen_assign_keys and tested > 0:
                    continue
                seen_assign_keys.add(akey)
                ev = evaluator.evaluate(
                    pl.assign,
                    op_priority=pl.op_prio,
                    op_bay_overrides=pl.bays,
                    op_crew_overrides=pl.crews,
                    task_order_override=pl.task_orders or None,
                )
                eval_count += 1
                tested += 1
                if ev.valid and (best_eval is None or not best_eval.valid or ev.objective < best_eval.objective):
                    best_eval = ev
                    best_plan = pl
                    evaluator.set_incumbent(
                        pl.assign,
                        ev,
                        op_priority=pl.op_prio,
                        op_bay_overrides=pl.bays,
                        task_order_override=pl.task_orders or None,
                    )
        elif evaluator is not None and best_eval is None:
            # Measure physical B2 objective of relaxed winner ONLY for reporting, without search optimization
            best_eval = evaluator.evaluate(
                best_plan.assign,
                op_priority=best_plan.op_prio,
                op_bay_overrides=best_plan.bays,
                op_crew_overrides=best_plan.crews,
                task_order_override=best_plan.task_orders or None,
            )

        return best_plan, best_eval, eval_count


def run_spga_policy_proposal(
    snapshot: NativeSnapshot,
    network: Network,
    cfg: SPGAConfig | None = None,
    use_rust: bool = True,
) -> CandidateProposal:
    """Wrap Jia Li (COR 2026) SPGA solver into a legal CVCR CandidateProposal."""
    t0 = time.perf_counter()
    use_cfg = cfg or SPGAConfig()
    prob = SPGAProblem(snapshot, network)
    evaluator = CVCREvaluator(network, snapshot, use_rust=use_rust)
    solver = SPGASolver(prob, use_cfg)
    best_plan, best_eval, eval_count = solver.solve(evaluator=evaluator)

    clf = prob.clf
    snap_by_id = {t.task_id: t for t in snapshot.tasks}
    commit_assign: dict[str, str] = {}
    affected_tasks: set[str] = set()
    for tid, new_agv in best_plan.assign.items():
        t_orig = snap_by_id.get(tid)
        if t_orig is None:
            continue
        orig_agv = t_orig.assigned_agv
        is_dist = orig_agv is not None and orig_agv in clf.disturbed_robots
        is_releasable = tid in clf.releasable_tasks
        is_urgent_unassigned = orig_agv is None and getattr(t_orig, "is_urgent", False)
        if orig_agv is not None or is_urgent_unassigned:
            commit_assign[tid] = orig_agv if (orig_agv is not None and not is_releasable and not is_dist) else new_agv
        if orig_agv != commit_assign.get(tid) or is_urgent_unassigned:
            affected_tasks.add(tid)

    # Filter routes cleanly and keep sequences valid
    proposed_seq: dict[str, list[str]] = {}
    for rid, seq in best_plan.task_orders.items():
        clean_q = [tid for tid in seq if commit_assign.get(tid) == rid]
        if clean_q:
            proposed_seq[rid] = clean_q

    routes = best_eval.robot_paths if (best_eval is not None and best_eval.valid) else {}
    pred_obj = float(best_eval.objective) if best_eval is not None else float(best_plan.objective)
    pred_cmax = (
        max(best_eval.op_end.values())
        if (best_eval is not None and best_eval.op_end)
        else int(best_plan.makespan)
    )
    snap_digest = snapshot.compute_digest()
    cand_digest = hashlib.sha256(
        json.dumps(
            {
                "policy": f"SPGA_{use_cfg.mode.upper()}",
                "assign": sorted(commit_assign.items()),
                "bays": sorted(best_plan.bays.items()),
                "crews": sorted(best_plan.crews.items()),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    affected_robots = sorted({r for r in commit_assign.values()} | set(clf.disturbed_robots))
    affected_ops = sorted(
        op.op_id
        for op in snapshot.operations
        if op.bay_id != best_plan.bays.get(op.op_id, op.bay_id)
        or op.crew_id != best_plan.crews.get(op.op_id, op.crew_id)
    )
    return CandidateProposal(
        schema="amr-lgcvcr.proposal.v1",
        policy_id=f"SPGA_COR2026_{use_cfg.mode.upper()}",
        snapshot_digest=snap_digest,
        affected_robots=affected_robots,
        affected_tasks=sorted(affected_tasks),
        affected_operations=affected_ops,
        proposed_assignment=commit_assign,
        proposed_routes=routes,
        proposed_bays=best_plan.bays,
        proposed_crews=best_plan.crews,
        predicted_obj=pred_obj,
        predicted_makespan=int(pred_cmax),
        candidate_digest=cand_digest,
        proposed_task_sequences=proposed_seq or None,
        proposed_op_priorities=best_plan.op_prio,
        telemetry={
            "eval_count": int(eval_count),
            "relaxed_obj": round(float(best_plan.objective), 4),
            "relaxed_makespan": int(best_plan.makespan),
            "requested_budget_ms": round(float(use_cfg.budget_ms), 3),
            "elapsed_ms": round(elapsed_ms, 3),
            "budget_ms": float(use_cfg.budget_ms),
            "deadline_met": elapsed_ms <= float(use_cfg.budget_ms),
            "timed_out_gracefully": True,
            "mode": use_cfg.mode,
        },
    )
