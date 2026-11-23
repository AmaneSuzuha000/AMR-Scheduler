"""Unified policy registry decoupling scheduling algorithms from environments & bridges.

Provides a single canonical registry and factory API for all 4 internal CVCR
policies and all 9 literature SOTA baselines so the exact same policy identifier
can run seamlessly in BOTH:
  - Physical Gazebo + ROS 2 + Nav2 closed-loop execution (`PhysicalCVCRCoordinator`)
  - Discrete Python + Rust closed-loop benchmarking (`AMRReschedulingEnv`)

All heavy/optional dependencies (e.g., `torch`, `gymnasium`) are lazily
imported only when constructing policies or environments that require them,
allowing ROS 2 / Gazebo environments without `torch` to execute pure-NumPy,
heuristic, evolutionary, and bandit policies natively.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from amr_sched.baselines.numpy_backends import file_sha256, resolve_numpy_checkpoint_path
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.network import Network

if TYPE_CHECKING:
    from amr_sched.rl import AMRReschedulingEnv

ExternalProposer = Callable[..., CandidateProposal]

_V16_CKPT = resolve_default_checkpoint("v16_frontier.npz")
_V17_CKPT = resolve_default_checkpoint("v17_window_protection.npz")
_V19_BC_A123_CKPT = resolve_default_checkpoint("v19/v19_hetero_bc_a123.pt")
_V19_BC_CKPT = resolve_default_checkpoint("v19/v19_hetero_bc.pt")
_V19_RL_CKPT = resolve_default_checkpoint("v19/v19_hetero_rl.pt")
_V19_STAGED_RL_CKPT = resolve_default_checkpoint("v19/v19_staged_rl.pt")
_V19_NO_PROD_COMM_CKPT = resolve_default_checkpoint("v19/v19_no_prod_comm.pt")
_EJOR26_CKPT = resolve_default_checkpoint("baselines_2026/ejor26_wu_gru_attn.pt")
_IJCAI26_CKPT = resolve_default_checkpoint("baselines_2026/ijcai26_choi_pgmpo.pt")
_TII26_CKPT = resolve_default_checkpoint("baselines_2026/tii26_ma_e2g_moe.pt")
_ICRA26_CKPT = resolve_default_checkpoint("baselines_2026/icra26_zhang_kitting_hgnn.pt")
_IJPR25_DFJSPT_CKPT = resolve_default_checkpoint("baselines_2026/ijpr25_dfjspt_hmarl.pt")

_COST_COLUMNS: tuple[str, ...] = (
    "cmax",
    "task_tardiness_cost",
    "op_tardiness_cost",
    "unfinished_task_penalty",
    "unfinished_op_penalty",
    "churn_cost",
    "compute_cost",
)


@dataclass(frozen=True)
class PolicySpec:
    """Metadata describing one registered scheduling policy."""

    name: str
    family: str
    description: str
    uses_external_proposer: bool
    checkpoint_path: Optional[Path] = None


POLICY_SPECS: dict[str, PolicySpec] = {
    "V18_EA_JOINT": PolicySpec(
        name="V18_EA_JOINT",
        family="ours",
        description="V18 Frontier Synthesis (V17 window protection + pure-NumPy PGMPO shortlist)",
        uses_external_proposer=False,
        checkpoint_path=_V17_CKPT,
    ),
    "V17_EA_JOINT": PolicySpec(
        name="V17_EA_JOINT",
        family="ours",
        description="V17 Dual-Op Window Protection & Execution-Aware Joint Routing",
        uses_external_proposer=False,
        checkpoint_path=_V17_CKPT,
    ),
    "V16_EA_JOINT": PolicySpec(
        name="V16_EA_JOINT",
        family="ours",
        description="V16 Unified Execution-Aware Frontier",
        uses_external_proposer=False,
        checkpoint_path=_V16_CKPT,
    ),
    "rule_v10": PolicySpec(
        name="rule_v10",
        family="heuristic",
        description="Churn-Calibrated V10 Multi-Expert Rule Portfolio",
        uses_external_proposer=True,
    ),
    "IJCAI26_PGMPO": PolicySpec(
        name="IJCAI26_PGMPO",
        family="literature_2026",
        description="Choi et al. (IJCAI-26) Bit-Conditioned Dual-Attention PGMPO",
        uses_external_proposer=True,
        checkpoint_path=_IJCAI26_CKPT,
    ),
    "TII26_E2G_MoE": PolicySpec(
        name="TII26_E2G_MoE",
        family="literature_2026",
        description="Ma & Zhou (IEEE TII 2026) Dual-Graph Hierarchical MoE",
        uses_external_proposer=True,
        checkpoint_path=_TII26_CKPT,
    ),
    "EJOR26_Wu_GRU_Attn": PolicySpec(
        name="EJOR26_Wu_GRU_Attn",
        family="literature_2026",
        description="Wu et al. (EJOR 2026) GRU Pointer-Attention + Semi-Circle Proximity",
        uses_external_proposer=True,
        checkpoint_path=_EJOR26_CKPT,
    ),
    "ICRA26_Kitting_HGNN": PolicySpec(
        name="ICRA26_Kitting_HGNN",
        family="literature_2026",
        description="Zhang et al. (ICRA 2026) FJSP-LB-MK Kitting Heterogeneous GNN",
        uses_external_proposer=True,
        checkpoint_path=_ICRA26_CKPT,
    ),
    "JAIR26_DyMAB": PolicySpec(
        name="JAIR26_DyMAB",
        family="literature_2026",
        description="Sow et al. (JAIR 2026) Sliding-Window alpha-UCB Dynamic LNS",
        uses_external_proposer=True,
    ),
    "MIT26_TAPFPC_LNS": PolicySpec(
        name="MIT26_TAPFPC_LNS",
        family="literature_2026",
        description="Parimi & Williams (MIT CSAIL 2026) Precedence-Closure LNS",
        uses_external_proposer=True,
    ),
    "RAL26_DetourDelta": PolicySpec(
        name="RAL26_DetourDelta",
        family="literature_2026",
        description="Hirayama et al. (IEEE RA-L 2026) Detour-Delta Online MAPD",
        uses_external_proposer=True,
    ),
    "SPGA_AMR": PolicySpec(
        name="SPGA_AMR",
        family="literature_2026",
        description="Jia Li et al. (COR 2026) Similarity-based Parallel Genetic Algorithm [Paper-faithful reproduction]",
        uses_external_proposer=True,
    ),
    "IJPR25_DFJSPT_HMARL": PolicySpec(
        name="IJPR25_DFJSPT_HMARL",
        family="literature_2025",
        description="Wang et al. (IJPR 2025) 3-Level Hierarchical Multi-Agent DRL for DFJSP-T [Paper-faithful reproduction]",
        uses_external_proposer=True,
        checkpoint_path=_IJPR25_DFJSPT_CKPT,
    ),
    "BALANS_IJCAI25": PolicySpec(
        name="BALANS_IJCAI25",
        family="literature_2025",
        description="Cai et al. (IJCAI 2025) Plain Online Multi-Radius Bandit ALNS",
        uses_external_proposer=True,
    ),
    "V19_HETERO_BC_A123": PolicySpec(
        name="V19_HETERO_BC_A123",
        family="v19_neural_joint",
        description="V19 Heterogeneous-Graph Joint Constructor BC (Stage A123)",
        uses_external_proposer=True,
        checkpoint_path=_V19_BC_A123_CKPT,
    ),
    "V19_HETERO_BC": PolicySpec(
        name="V19_HETERO_BC",
        family="v19_neural_joint",
        description="V19 Heterogeneous-Graph Joint Constructor BC (Multi-Expert B2 Behavior Cloning)",
        uses_external_proposer=True,
        checkpoint_path=_V19_BC_CKPT,
    ),
    "V19_HETERO_RL": PolicySpec(
        name="V19_HETERO_RL",
        family="v19_neural_joint",
        description="V19 Heterogeneous-Graph Critical-Chain Joint Constructor + KL-Anchored SMDP PPO",
        uses_external_proposer=True,
        checkpoint_path=_V19_RL_CKPT,
    ),
    "V19_STAGED_RL": PolicySpec(
        name="V19_STAGED_RL",
        family="v19_neural_joint",
        description="V19 Staged Production-Then-Transport Ablation + KL-Anchored SMDP PPO",
        uses_external_proposer=True,
        checkpoint_path=_V19_STAGED_RL_CKPT,
    ),
    "V19_NO_PROD_COMM": PolicySpec(
        name="V19_NO_PROD_COMM",
        family="v19_neural_joint",
        description="V19 No-Production-Communication Ablation + KL-Anchored SMDP PPO",
        uses_external_proposer=True,
        checkpoint_path=_V19_NO_PROD_COMM_CKPT,
    ),
    "V19_ADAPTIVE_RL": PolicySpec(
        name="V19_ADAPTIVE_RL",
        family="v19_neural_joint",
        description="CoRe-Sched: Commitment-aware Rescheduling (AHG-RL + CVCR-IR Seq-Diff + Atomic Transaction Commit)",
        uses_external_proposer=True,
        checkpoint_path=_V19_RL_CKPT,
    ),
}

SUPPORTED_POLICIES: tuple[str, ...] = tuple(POLICY_SPECS.keys())

DEFAULT_12_POLICIES: tuple[str, ...] = tuple(
    p for p in SUPPORTED_POLICIES if p not in ("BALANS_IJCAI25", "IJPR25_DFJSPT_HMARL") and not p.startswith("V19_")
)

_POLICY_ALIASES: dict[str, str] = {
    "v18": "V18_EA_JOINT",
    "v18_ea_joint": "V18_EA_JOINT",
    "v17": "V17_EA_JOINT",
    "v17_ea_joint": "V17_EA_JOINT",
    "v16": "V16_EA_JOINT",
    "v16_ea_joint": "V16_EA_JOINT",
    "rule_v10": "rule_v10",
    "v10_rule": "rule_v10",
    "v10": "rule_v10",
    "ijcai26_pgmpo": "IJCAI26_PGMPO",
    "pgmpo": "IJCAI26_PGMPO",
    "pgmpo_amr_aligned": "IJCAI26_PGMPO",
    "pgmpo_aligned": "IJCAI26_PGMPO",
    "tii26_e2g_moe": "TII26_E2G_MoE",
    "tii26": "TII26_E2G_MoE",
    "ejor26_wu_gru_attn": "EJOR26_Wu_GRU_Attn",
    "ejor26": "EJOR26_Wu_GRU_Attn",
    "icra26_kitting_hgnn": "ICRA26_Kitting_HGNN",
    "icra26": "ICRA26_Kitting_HGNN",
    "jair26_dymab": "JAIR26_DyMAB",
    "jair26": "JAIR26_DyMAB",
    "mit26_tapfpc_lns": "MIT26_TAPFPC_LNS",
    "mit26": "MIT26_TAPFPC_LNS",
    "ral26_detourdelta": "RAL26_DetourDelta",
    "ral26": "RAL26_DetourDelta",
    "spga_amr": "SPGA_AMR",
    "spga": "SPGA_AMR",
    "spga_cor2026": "SPGA_AMR",
    "ijpr25_dfjspt_hmarl": "IJPR25_DFJSPT_HMARL",
    "ijpr25_dfjspt": "IJPR25_DFJSPT_HMARL",
    "dfjspt_hmarl": "IJPR25_DFJSPT_HMARL",
    "dfjspt": "IJPR25_DFJSPT_HMARL",
    "balans_ijcai25": "BALANS_IJCAI25",
    "ijcai25_balans_plain": "BALANS_IJCAI25",
    "balans": "BALANS_IJCAI25",
    "v19_hetero_bc_a123": "V19_HETERO_BC_A123",
    "v19_bc_a123": "V19_HETERO_BC_A123",
    "v19_hetero_bc": "V19_HETERO_BC",
    "v19_hetero_bc_a1234": "V19_HETERO_BC",
    "v19_bc": "V19_HETERO_BC",
    "v19": "V19_HETERO_RL",
    "v19_rl": "V19_HETERO_RL",
    "v19_hetero_rl": "V19_HETERO_RL",
    "v19_staged_rl": "V19_STAGED_RL",
    "v19_staged": "V19_STAGED_RL",
    "v19_no_prod_comm": "V19_NO_PROD_COMM",
    "v19_adaptive_rl": "V19_ADAPTIVE_RL",
    "v19_adaptive": "V19_ADAPTIVE_RL",
    "v19_adaptive_bandit": "V19_ADAPTIVE_RL",
    "ahg_rl": "V19_ADAPTIVE_RL",
    "ahg-rl": "V19_ADAPTIVE_RL",
    "hg_acdrl": "V19_ADAPTIVE_RL",
    "core_sched": "V19_ADAPTIVE_RL",
    "core-sched": "V19_ADAPTIVE_RL",
    "coresched": "V19_ADAPTIVE_RL",
}


def resolve_policy_name(name: str) -> str:
    """Resolve user/CLI policy identifier or alias into its canonical registry name."""
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("Policy name must be non-empty.")
    if raw in POLICY_SPECS:
        return raw
    canon = _POLICY_ALIASES.get(raw.lower())
    if canon is None:
        supported = ", ".join(SUPPORTED_POLICIES)
        raise ValueError(f"Unsupported policy {name!r}. Supported policies: {supported}")
    return canon


def is_registered_policy(name: str) -> bool:
    """Return True if `name` matches any registered canonical policy or alias."""
    raw = str(name or "").strip()
    return raw in POLICY_SPECS or raw.lower() in _POLICY_ALIASES


def is_external_policy(name: str) -> bool:
    """Return True when a registered policy executes via an external CandidateProposal callable."""
    canon = resolve_policy_name(name)
    return bool(POLICY_SPECS[canon].uses_external_proposer)


class RuleV10PhysicalProposer:
    """Causal V10 portfolio rule adapter for physical bridges running outside RL envs."""

    def __init__(self, *, use_rust: bool = True) -> None:
        self.use_rust = bool(use_rust)

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        clf = classify_commitments(snapshot)
        active = [
            t for t in snapshot.tasks
            if int(t.release_step) <= int(snapshot.tick) and str(t.status) not in ("completed", "failed")
        ]
        is_fault = bool(clf.disturbed_robots)
        is_urg = any(bool(t.is_urgent) and str(t.status) in ("pending", "assigned") for t in active)
        has_prec = any(bool(op_predecessors(op)) for op in snapshot.operations)
        has_win = any(bool(getattr(op, "resource_windows", ())) for op in snapshot.operations)
        util = len(active) / float(max(1, len(snapshot.robots)))

        scope = "V10_E6_BALANCE_KM_LRHO"
        eff_budget = float(budget_ms)
        if is_fault or is_urg:
            if has_win and is_fault:
                scope = "V10_E1_FAST_RESCUE"
                eff_budget = min(eff_budget, 20.0) if eff_budget > 20.0 else eff_budget
            elif is_urg:
                scope = "V10_E2_CRIT_PROD_CI"
            else:
                scope = "V10_E3_JOINT_PC_HYS"
        elif has_prec and not has_win:
            scope = "V10_E7_COND_CPSAT"
        elif util >= 2.0:
            scope = "V10_E5_AGGRESSIVE_GUIDE"

        prop = LGCVCRPolicy(scope, network, use_rust=self.use_rust).propose(
            snapshot,
            budget_ms=max(12.0, eff_budget),
        )
        prop.policy_id = "rule_v10"
        return prop


def build_external_proposer(
    policy_name: str,
    *,
    seed: int = 0,
    use_rust: bool = True,
    backend: str = "auto",
    allow_untrained: bool = False,
) -> Optional[ExternalProposer]:
    """Instantiate a stateless/stateful `CandidateProposal` proposer for a policy name."""
    canon = resolve_policy_name(policy_name)
    if canon in ("V18_EA_JOINT", "V17_EA_JOINT", "V16_EA_JOINT"):
        return None
    if canon == "rule_v10":
        return RuleV10PhysicalProposer(use_rust=use_rust)

    spec = POLICY_SPECS[canon]
    if spec.checkpoint_path is not None and not allow_untrained:
        allow_np_alias = (
            (canon in ("IJCAI26_PGMPO", "EJOR26_Wu_GRU_Attn", "IJPR25_DFJSPT_HMARL") or canon.startswith("V19_"))
            and backend in ("auto", "numpy")
        )
        chk = (
            resolve_numpy_checkpoint_path(spec.checkpoint_path, prefer_npz=True)
            if allow_np_alias
            else spec.checkpoint_path
        )
        if not chk.is_file():
            raise FileNotFoundError(
                f"Required checkpoint for {canon} not found: {spec.checkpoint_path}"
            )

    if canon == "RAL26_DetourDelta":
        from amr_sched.baselines.top_venue_baselines import DetourDeltaRAL26Proposer
        return DetourDeltaRAL26Proposer(use_rust=use_rust)
    if canon == "BALANS_IJCAI25":
        from amr_sched.baselines.top_venue_baselines import BalansIJCAI25PlainProposer
        return BalansIJCAI25PlainProposer(seed=seed, use_rust=use_rust)
    if canon == "IJPR25_DFJSPT_HMARL":
        from amr_sched.baselines.dfjspt_hmarl_ijpr25 import DFJSPTHMARLProposer

        proposer = DFJSPTHMARLProposer(
            ckpt_path=_IJPR25_DFJSPT_CKPT,
            policy_id=canon,
            backend=backend,
            allow_untrained=allow_untrained,
            use_rust=use_rust,
            seed=seed,
        )
        proposer._get_net()
        return proposer
    if canon == "SPGA_AMR":
        from amr_sched.baselines.spga_cor2026 import SPGAConfig, run_spga_policy_proposal

        def _spga_proposer(
            snapshot: NativeSnapshot,
            network: Network,
            *,
            budget_ms: float = 35.0,
        ) -> CandidateProposal:
            cfg = SPGAConfig(
                pop_size=40,
                max_iters=30,
                mode="g1_unified",
                budget_ms=float(budget_ms),
                seed=int(seed) + int(snapshot.tick),
            )
            return run_spga_policy_proposal(snapshot, network, cfg, use_rust=use_rust)

        return _spga_proposer

    from amr_sched.baselines.baselines_2026 import (
        ICRA26KittingHGNNProposer,
        IJCAI26PGMPOProposer,
        ParimiMIT26TAPFPCLNSProposer,
        SowJAIR26DyMABProposer,
        TII26E2GMoEProposer,
        WuEJOR26Proposer,
    )

    if canon == "IJCAI26_PGMPO":
        proposer = IJCAI26PGMPOProposer(
            ckpt_path=_IJCAI26_CKPT, use_rust=use_rust, seed=seed, backend=backend, allow_untrained=allow_untrained
        )
        proposer._get_net()
        return proposer
    if canon == "TII26_E2G_MoE":
        proposer = TII26E2GMoEProposer(
            ckpt_path=_TII26_CKPT, use_rust=use_rust, seed=seed, allow_untrained=allow_untrained
        )
        proposer._get_net()
        return proposer
    if canon == "EJOR26_Wu_GRU_Attn":
        proposer = WuEJOR26Proposer(
            ckpt_path=_EJOR26_CKPT, use_rust=use_rust, seed=seed, backend=backend, allow_untrained=allow_untrained
        )
        proposer._get_net()
        return proposer
    if canon == "ICRA26_Kitting_HGNN":
        proposer = ICRA26KittingHGNNProposer(
            ckpt_path=_ICRA26_CKPT, use_rust=use_rust, seed=seed, allow_untrained=allow_untrained
        )
        proposer._get_net()
        return proposer
    if canon == "JAIR26_DyMAB":
        return SowJAIR26DyMABProposer(use_rust=use_rust, seed=seed)
    if canon == "MIT26_TAPFPC_LNS":
        return ParimiMIT26TAPFPCLNSProposer(use_rust=use_rust, seed=seed)
    if canon in (
        "V19_HETERO_BC_A123",
        "V19_HETERO_BC",
        "V19_HETERO_RL",
        "V19_STAGED_RL",
        "V19_NO_PROD_COMM",
    ):
        from amr_sched.rl.v19_hetero_constructor import V19HeteroConstructorProposer

        ablation_map = {
            "V19_STAGED_RL": "staged",
            "V19_NO_PROD_COMM": "no_prod_comm",
        }
        proposer = V19HeteroConstructorProposer(
            ckpt_path=spec.checkpoint_path,
            policy_id=canon,
            ablation_mode=ablation_map.get(canon, "full"),
            backend=backend,
            allow_untrained=allow_untrained,
            use_rust=use_rust,
            seed=seed,
        )
        proposer._get_net()
        return proposer
    if canon == "V19_ADAPTIVE_RL":
        from amr_sched.rl.v19_hetero_constructor import V19AdaptiveCouplingProposer

        proposer = V19AdaptiveCouplingProposer(
            full_ckpt_path=_V19_RL_CKPT,
            staged_ckpt_path=_V19_STAGED_RL_CKPT,
            policy_id=canon,
            mode="bandit",
            backend=backend,
            allow_untrained=allow_untrained,
            use_rust=use_rust,
            seed=seed,
            evaluator_mode="B2_SEEDED",
        )
        proposer._ensure_nets()
        return proposer
    raise ValueError(f"Unhandled policy in build_external_proposer: {canon}")


def load_internal_cvcr_policy(
    policy_name: str,
    *,
    max_rust_verifications: int = 4,
    candidate_mode: str = "full",
    hysteresis_mode: str = "default",
    allow_missing_pgmpo: bool = False,
    allow_v17_fallback: bool = False,
) -> Any:
    """Load the cached NumPy/LR policy object strictly from packaged checkpoints for V16, V17, or V18."""
    canon = resolve_policy_name(policy_name)
    if canon == "V18_EA_JOINT":
        from amr_sched.cvcr.v18_frontier_synthesis import load_v18_synthesis_policy

        pol = load_v18_synthesis_policy(
            path=_V17_CKPT,
            max_rust_verifications=max_rust_verifications,
            candidate_mode=candidate_mode,
            hysteresis_mode=hysteresis_mode,
            allow_missing_pgmpo=allow_missing_pgmpo,
            allow_v17_fallback=allow_v17_fallback,
        )
        if not pol.pgmpo.ready and not allow_missing_pgmpo:
            raise RuntimeError("V18_EA_JOINT requires the trained PGMPO checkpoint")
        v17_sha = getattr(pol, "v17_checkpoint_sha256", None) or (file_sha256(_V17_CKPT) if _V17_CKPT.is_file() else "none")
        pg_sha = getattr(pol.pgmpo, "checkpoint_sha256", None) or "none"
        pol._checkpoint_sha256 = f"v17:{v17_sha[:16]}+pgmpo:{pg_sha[:16]}"
        return pol
    if canon == "V17_EA_JOINT":
        from amr_sched.cvcr.v17_window_protection import load_v17_window_policy

        pol = load_v17_window_policy(
            _V17_CKPT,
            router_mode="learned",
            ranker_mode="pairwise",
            enable_restraint=True,
            allow_fallback=allow_v17_fallback,
        )
        pol._checkpoint_sha256 = file_sha256(_V17_CKPT) if _V17_CKPT.is_file() else None
        return pol
    if canon == "V16_EA_JOINT":
        from amr_sched.cvcr.v16_frontier import V16FrontierPolicy

        if not _V16_CKPT.is_file():
            raise FileNotFoundError(f"Required V16 checkpoint not found: {_V16_CKPT}")
        pol = V16FrontierPolicy.from_npz(_V16_CKPT)
        pol._checkpoint_sha256 = file_sha256(_V16_CKPT)
        return pol
    return None


def create_discrete_policy_env(
    policy_name: str,
    *,
    topology: str = "workshop",
    seed: int = 0,
    max_steps: int = 500,
    use_rust: bool = True,
    p2_cfg: Optional[P2PerturbationConfig] = None,
    execution_aware: bool = True,
    p4_choke_robust: bool = True,
    loaded_repair_threshold: int = 20,
    epoch_interval: int = 16,
    backend: str = "auto",
    allow_untrained: bool = False,
    v18_candidate_mode: str = "full",
    v18_hysteresis_mode: str = "default",
    allow_missing_pgmpo: bool = False,
) -> tuple[str, Any, "AMRReschedulingEnv"]:
    """Construct paired `(kind, controller, AMRReschedulingEnv)` for any registered policy."""
    from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy

    canon = resolve_policy_name(policy_name)
    common_kwargs: dict[str, Any] = {
        "topology": topology,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "use_rust": bool(use_rust),
        "allow_kitting_shortcut": False,
        "pure_physical_reward": True,
        "real_compute_window_coupling": True,
        "loaded_repair_mode": "haven_v12",
        "loaded_repair_threshold": int(loaded_repair_threshold),
        "record_trajectory": True,
        "execution_perturbation": p2_cfg,
        "execution_aware": bool(execution_aware),
        "p4_choke_robust": bool(p4_choke_robust),
    }

    if canon == "V18_EA_JOINT":
        from amr_sched.cvcr.v18_frontier_synthesis import V18AdaptiveController

        pol = load_internal_cvcr_policy(
            canon,
            candidate_mode=v18_candidate_mode,
            hysteresis_mode=v18_hysteresis_mode,
            allow_missing_pgmpo=allow_missing_pgmpo,
        )
        ctrl = V18AdaptiveController(
            epoch_interval=epoch_interval,
            stall_threshold=loaded_repair_threshold,
            policy=pol,
            profile="quality",
        )
        ctrl.reset()
        env = AMRReschedulingEnv(
            **common_kwargs,
            expert_v12_portfolio=True,
            v13_kitting_mode="V18_EA_JOINT",
            v15_policy=pol,
        )
        env._policy_checkpoint_sha256 = getattr(pol, "_checkpoint_sha256", None)
        env._policy_inference_backend = "numpy"
        return "v18", ctrl, env

    if canon == "V17_EA_JOINT":
        from amr_sched.cvcr.v17_window_protection import V17AdaptiveController

        pol = load_internal_cvcr_policy(canon)
        ctrl = V17AdaptiveController(
            epoch_interval=epoch_interval,
            stall_threshold=loaded_repair_threshold,
            policy=pol,
            profile="quality",
        )
        ctrl.reset()
        env = AMRReschedulingEnv(
            **common_kwargs,
            expert_v12_portfolio=True,
            v13_kitting_mode="V17_EA_JOINT",
            v15_policy=pol,
        )
        env._policy_checkpoint_sha256 = getattr(pol, "_checkpoint_sha256", None)
        env._policy_inference_backend = "numpy"
        return "v17", ctrl, env

    if canon == "V16_EA_JOINT":
        from amr_sched.cvcr.v16_frontier import V16AdaptiveController

        pol = load_internal_cvcr_policy(canon)
        ctrl = V16AdaptiveController(
            epoch_interval=epoch_interval,
            stall_threshold=loaded_repair_threshold,
            policy=pol,
            profile="quality",
        )
        ctrl.reset()
        env = AMRReschedulingEnv(
            **common_kwargs,
            expert_v12_portfolio=True,
            v13_kitting_mode="V16_EA_JOINT",
            v15_policy=pol,
        )
        env._policy_checkpoint_sha256 = getattr(pol, "_checkpoint_sha256", None)
        env._policy_inference_backend = "numpy"
        return "v16", ctrl, env

    ctrl = RuleV10PortfolioPolicy(epoch_interval=epoch_interval)
    ctrl.reset()
    ext_proposer = None if canon == "rule_v10" else build_external_proposer(
        canon,
        seed=int(seed),
        use_rust=bool(use_rust),
        backend=backend,
        allow_untrained=allow_untrained,
    )
    env = AMRReschedulingEnv(
        **common_kwargs,
        expert_v10_portfolio=True,
        external_proposer=ext_proposer,
    )
    env._policy_checkpoint_sha256 = getattr(ext_proposer, "_checkpoint_sha256", None)
    env._policy_inference_backend = (
        ext_proposer._effective_backend()
        if hasattr(ext_proposer, "_effective_backend")
        else ("torch" if ext_proposer is not None and canon in ("TII26_E2G_MoE", "ICRA26_Kitting_HGNN") else "algorithmic")
    )
    return "rule_v10", ctrl, env


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Return nearest-rank percentile from a pre-sorted non-empty float list."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return float(sorted_vals[idx])


def _scenario_digest(env: "AMRReschedulingEnv", *, p2_cfg: Optional[P2PerturbationConfig] = None) -> str:
    """Compute canonical SHA-256 fingerprint over scenario network, tasks, AGVs, ops, disruptions & obstacles."""
    assert env.sim is not None and env.tracker is not None
    ep = getattr(env, "execution_perturbation", None)
    eff_p2 = p2_cfg if p2_cfg is not None else (getattr(ep, "config", None) if ep is not None else None)
    obstacles = [asdict(item) for item in ep.obstacles] if ep is not None and hasattr(ep, "obstacles") else []
    payload = {
        "nodes": sorted((nid, n.x, n.y, str(n.kind)) for nid, n in env.sim.network.nodes.items()),
        "edges": sorted((e.src, e.dst, e.length, e.speed_limit) for e in env.sim.network.edges.values()),
        "agvs": sorted((a.agv_id, a.start_node, a.max_speed) for a in env.sim.agvs),
        "tasks": sorted(
            (
                t.task_id,
                t.pickup,
                t.dropoff,
                t.release_step,
                t.due_step,
                str(getattr(t, "material_type", "")),
                bool(getattr(t, "priority", None) == 1 or str(getattr(t, "priority", "")) == "URGENT"),
            )
            for t in env.sim.tasks
        ),
        "ops": sorted(
            (
                op.op_id,
                op.bay_id,
                op.crew_id,
                op.proc_steps,
                op.due_step,
                tuple(op_predecessors(op)),
                tuple(op.kit_task_ids),
                tuple(op.eligible_bays),
                tuple(op.eligible_crews),
                tuple(getattr(op, "resource_windows", ())),
            )
            for op in env.tracker.operations
        ),
        "disruptions": dict(getattr(env, "disruptions", {}) or {}),
        "p2_cfg": asdict(eff_p2) if eff_p2 is not None else None,
        "scheduled_obstacles": obstacles,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_discrete_episode(
    policy_name: str,
    scenario_tuple: tuple[Any, ...],
    *,
    seed: int = 0,
    max_steps: int = 500,
    use_rust: bool = True,
    p2_cfg: Optional[P2PerturbationConfig] = None,
    execution_aware: bool = True,
    p4_choke_robust: bool = True,
    clock_mode: str = "wall",
    virtual_poll_ms: Optional[float] = None,
    episode_key: str = "",
    backend: str = "auto",
    allow_untrained: bool = False,
    v18_candidate_mode: str = "full",
    v18_hysteresis_mode: str = "default",
    allow_missing_pgmpo: bool = False,
) -> dict[str, Any]:
    """Execute one discrete closed-loop episode and return audited metrics + 7 cost terms."""
    from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state

    if clock_mode not in ("wall", "virtual"):
        raise ValueError(f"Unsupported clock_mode {clock_mode!r}; expected 'wall' or 'virtual'.")
    if clock_mode == "wall" and virtual_poll_ms is not None:
        raise ValueError("virtual_poll_ms cannot be combined with clock_mode='wall'.")

    import copy

    scenario_copy = copy.deepcopy(scenario_tuple)
    p2_copy = copy.deepcopy(p2_cfg)

    canon = resolve_policy_name(policy_name)
    if clock_mode == "virtual":
        poll_val = 0.002 if virtual_poll_ms is None else float(virtual_poll_ms)
        scope = virtual_clock_scope(poll_ms=poll_val)
    else:
        scope = nullcontext()

    wall_t0 = time.perf_counter()
    with scope:
        kind, ctrl, env = create_discrete_policy_env(
            canon,
            seed=seed,
            max_steps=max_steps,
            use_rust=use_rust,
            p2_cfg=p2_copy,
            execution_aware=execution_aware,
            p4_choke_robust=p4_choke_robust,
            backend=backend,
            allow_untrained=allow_untrained,
            v18_candidate_mode=v18_candidate_mode,
            v18_hysteresis_mode=v18_hysteresis_mode,
            allow_missing_pgmpo=allow_missing_pgmpo,
        )
        proposer_stats: Counter[str] = Counter()
        observed_ckpt_sha = getattr(env, "_policy_checkpoint_sha256", None)
        observed_backend = getattr(env, "_policy_inference_backend", "algorithmic")
        if env.external_proposer is not None:
            raw_ext = env.external_proposer

            def _tracked_proposer(*args: Any, **kwargs: Any) -> CandidateProposal:
                nonlocal observed_ckpt_sha, observed_backend
                prop = raw_ext(*args, **kwargs)
                if prop is not None:
                    telem = prop.telemetry or {}
                    proposer_stats["keep_fallbacks"] += int(bool(telem.get("fallback_keep", False)))
                    proposer_stats["deadline_overruns"] += int(not bool(telem.get("deadline_met", True)))
                    proposer_stats["neural_candidates_evaluated"] += int(
                        telem.get("neural_candidates_evaluated", 0)
                    )
                    if telem.get("checkpoint_sha256"):
                        observed_ckpt_sha = str(telem["checkpoint_sha256"])
                    if telem.get("inference_backend"):
                        observed_backend = str(telem["inference_backend"])
                return prop

            env.external_proposer = _tracked_proposer

        obs, info = env.reset(seed=int(seed), options={"scenario_tuple": scenario_copy})
        assert env.sim is not None and env.tracker is not None
        sc_sha = _scenario_digest(env, p2_cfg=p2_copy)
        obstacles = (
            [asdict(item) for item in env.execution_perturbation.obstacles]
            if (p2_copy is not None and env.execution_perturbation is not None)
            else []
        )
        tm = False
        tr = False
        while not (tm or tr):
            if kind == "rule_v10":
                feats = compute_expert_v9_state(
                    env.sim,
                    env.tracker,
                    last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks,
                    decision_logs=env.decision_logs,
                )
                step_info = dict(
                    info,
                    action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick),
                )
            elif kind == "v18":
                feats = ctrl.observe(env, obs)
                step_info = info
            else:
                feats = env.get_expert_v12_obs(obs)
                step_info = info
            act = ctrl.select_action(feats, step_info, sim=env.sim)
            obs, _, tm, tr, info = env.step(act)

    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(
        env.sim,
        env.tracker.operations,
        require_trajectory=True,
    )
    decision_ms = [
        float(row["t_propose_ms"]) + float(row["t_validate_commit_ms"])
        for row in env.decision_logs
    ]
    decision_sorted = sorted(decision_ms)
    funnel_totals: dict[str, Any] = {}
    if kind == "v18" and hasattr(ctrl, "policy") and getattr(ctrl.policy, "funnel_history", None):
        f_hist = ctrl.policy.funnel_history
        agg_by_fam: dict[str, dict[str, int]] = {}
        neural_wins = 0
        both_valid = 0
        for item in f_hist:
            for fam, bkt in item.get("by_family", {}).items():
                dst = agg_by_fam.setdefault(fam, {k: 0 for k in bkt})
                for k, v in bkt.items():
                    dst[k] = dst.get(k, 0) + int(v)
            bn = item.get("best_valid_neural_b2_obj")
            bh = item.get("best_valid_heuristic_b2_obj")
            if bn is not None and bh is not None:
                both_valid += 1
                if bn + 1e-6 < bh:
                    neural_wins += 1
        funnel_totals = {
            "calls": len(f_hist),
            "by_family": agg_by_fam,
            "head_to_head_valid_snapshots": both_valid,
            "neural_beat_heuristic_snapshots": neural_wins,
        }
    rec: dict[str, Any] = {
        "policy": canon,
        "requested_policy": str(policy_name),
        "family": POLICY_SPECS[canon].family,
        "episode_key": episode_key,
        "seed": int(seed),
        "clock_mode": clock_mode,
        "checkpoint_sha256": observed_ckpt_sha,
        "inference_backend": observed_backend,
        "scenario_sha": sc_sha,
        "scenario_digest_includes_disruptions": True,
        "audit_passed": bool(aud.pass_audit),
        "audit_issues": list(aud.issues),
        "ticks": int(aud.total_steps),
        "terminated": bool(tm),
        "truncated": bool(tr),
        "net_obj": round(float(m["net_obj"]), 4),
        "total_obj": round(float(m["total_obj"]), 4),
        "cmax": round(float(m["cmax"]), 4),
        "task_tardiness_cost": round(float(m.get("task_tardiness_cost", 0.0)), 4),
        "op_tardiness_cost": round(float(m.get("op_tardiness_cost", 0.0)), 4),
        "unfinished_task_penalty": round(float(m.get("unfinished_task_penalty", 0.0)), 4),
        "unfinished_op_penalty": round(float(m.get("unfinished_op_penalty", 0.0)), 4),
        "churn_cost": round(float(m.get("churn_cost", 15.0 * float(m.get("total_churn", 0)))), 4),
        "compute_cost": round(float(m.get("compute_cost", 0.05 * float(m.get("total_compute_ms", 0.0)))), 4),
        "tardiness": round(float(m["total_tardiness"]), 4),
        "churn": int(m["total_churn"]),
        "replans": int(m["total_replans"]),
        "task_comp": round(float(m["task_completion_rate"]), 6),
        "prod_comp": round(float(m["production_completion_rate"]), 6),
        "compute_ms": round(float(m["total_compute_ms"]), 3),
        "wall_ms": round(wall_ms, 2),
        "decision_count": len(decision_ms),
        "mean_decision_ms": round(sum(decision_ms) / len(decision_ms), 3) if decision_ms else 0.0,
        "decision_ms_p95": round(_percentile(decision_sorted, 0.95), 3),
        "decision_ms_p99": round(_percentile(decision_sorted, 0.99), 3),
        "transaction_rejections": sum(not row["receipt_accepted"] for row in env.decision_logs),
        "expired_windows": int(env.total_window_expirations),
        "starvation_steps": int(env.tracker.starvation_steps),
        "proposer_stats": dict(proposer_stats),
        "v18_funnel": funnel_totals,
        "scheduled_obstacles": obstacles,
    }
    rec["cost_sum_err"] = round(
        abs(sum(float(rec[k]) for k in _COST_COLUMNS) - float(rec["net_obj"])),
        6,
    )
    return rec


def summarize_physical_run(coord: Any, ops: list[Any], *, budget_ms: Optional[float] = None) -> dict[str, Any]:
    """Summarize causal coordination latency, churn, makespan, and tardiness from PhysicalCVCRCoordinator."""
    triggered = [h for h in getattr(coord, "history", ()) if h.triggered]
    propose_ms = sorted(float(h.propose_ms) for h in triggered)
    e2e_ms = sorted(float(getattr(h, "e2e_ms", 0.0)) for h in triggered if float(getattr(h, "e2e_ms", 0.0)) > 0.0)
    eff_budget = float(budget_ms if budget_ms is not None else getattr(coord, "budget_ms", 20.0))
    overruns = sum(1 for v in propose_ms if v > eff_budget)
    eval_objs = [float(h.eval_objective) for h in triggered if h.eval_objective is not None]
    tasks = list(getattr(coord.simulator, "tasks", ()))
    task_tard = sum(
        max(0.0, float(t.complete_step) - float(t.due_step))
        for t in tasks
        if t.complete_step is not None and t.due_step is not None
    )
    op_tard = sum(
        max(0.0, float(o.end_step) - float(o.due_step))
        for o in ops
        if o.end_step is not None and o.due_step is not None
    )
    makespan_steps = max(
        (int(o.end_step) for o in ops if o.end_step is not None),
        default=int(getattr(coord.simulator, "_step_cursor", 0)),
    )
    return {
        "policy": str(getattr(coord, "policy_method", "")),
        "requested_policy": str(getattr(coord, "requested_policy_method", getattr(coord, "policy_method", ""))),
        "triggered_count": len(triggered),
        "accepted_count": sum(1 for h in triggered if h.receipt_accepted),
        "propose_ms_mean": round(sum(propose_ms) / len(propose_ms), 3) if propose_ms else 0.0,
        "propose_ms_p95": round(_percentile(propose_ms, 0.95), 3) if propose_ms else 0.0,
        "propose_ms_p99": round(_percentile(propose_ms, 0.99), 3) if propose_ms else 0.0,
        "propose_ms_max": round(propose_ms[-1], 3) if propose_ms else 0.0,
        "deadline_overrun_rate": round(overruns / len(propose_ms), 6) if propose_ms else 0.0,
        "e2e_feedback_to_commit_ms": {
            "count": len(e2e_ms),
            "mean": round(sum(e2e_ms) / len(e2e_ms), 3) if e2e_ms else 0.0,
            "p95": round(_percentile(e2e_ms, 0.95), 3) if e2e_ms else 0.0,
            "p99": round(_percentile(e2e_ms, 0.99), 3) if e2e_ms else 0.0,
            "max": round(e2e_ms[-1], 3) if e2e_ms else 0.0,
        },
        "eval_objective_mean": round(sum(eval_objs) / len(eval_objs), 4) if eval_objs else None,
        "eval_objective_last": round(eval_objs[-1], 4) if eval_objs else None,
        "makespan_steps": int(makespan_steps),
        "task_tardiness": round(float(task_tard), 4),
        "op_tardiness": round(float(op_tard), 4),
        "reassigned_events": sum(len(h.reassigned_tasks) for h in triggered),
        "detour_events": sum(len(h.replanned_detours) for h in getattr(coord, "history", ())),
    }


__all__ = [
    "DEFAULT_12_POLICIES",
    "POLICY_SPECS",
    "PolicySpec",
    "RuleV10PhysicalProposer",
    "SUPPORTED_POLICIES",
    "build_external_proposer",
    "create_discrete_policy_env",
    "is_external_policy",
    "is_registered_policy",
    "load_internal_cvcr_policy",
    "resolve_policy_name",
    "run_discrete_episode",
    "summarize_physical_run",
]
