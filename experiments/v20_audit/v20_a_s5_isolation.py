"""V20-A Task A1: Single-factor isolation of S5@35ms 4-Topo timeout & regression."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_stage1_quarantine,
    build_val72_specs,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.proposer_variants import UnifiedE0E9PrototypeProposer


def run_s5_isolation() -> dict[str, Any]:
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs_4t = [sp for sp in build_val72_specs() if sp["suite_group"] == "4topo"]
    assert_stage1_quarantine([sp["seed"] for sp in specs_4t], label="V20A_S5_Isolation")

    variants: list[tuple[str, dict[str, Any]]] = [
        ("A0_Tier1_Only", dict(enable_e4=False, enable_e5=False, enable_e6=False, enable_e7=False, enable_e9=False, commit_unwin_seq=True)),
        ("A1_Plus_E4_PGMPO", dict(enable_e4=True, enable_e5=False, enable_e6=False, enable_e7=False, enable_e9=False, commit_unwin_seq=True)),
        ("A2_Plus_E5_MIT", dict(enable_e4=False, enable_e5=True, enable_e6=False, enable_e7=False, enable_e9=False, commit_unwin_seq=True)),
        ("A3_Plus_E6_Polish", dict(enable_e4=False, enable_e5=False, enable_e6=True, enable_e7=False, enable_e9=False, commit_unwin_seq=True)),
        ("A4_Plus_E9_CPSAT", dict(enable_e4=False, enable_e5=False, enable_e6=False, enable_e7=False, enable_e9=True, commit_unwin_seq=True)),
        ("A5_Plus_E7_Cone", dict(enable_e4=False, enable_e5=False, enable_e6=False, enable_e7=True, enable_e9=False, commit_unwin_seq=True)),
        ("A6_Pruned8_NoE5_WithUnwinSeq", dict(enable_e4=True, enable_e5=False, enable_e6=True, enable_e7=True, enable_e9=True, commit_unwin_seq=True)),
        ("A7_Pruned8_NoE5_NoUnwinSeq", dict(enable_e4=True, enable_e5=False, enable_e6=True, enable_e7=True, enable_e9=True, commit_unwin_seq=False)),
    ]

    report: dict[str, Any] = {}
    for label, kw in variants:
        prop = UnifiedE0E9PrototypeProposer(hard_budget_ms=35.0, seed=0, **kw)
        rows = []
        failed_keys = []
        for sp in specs_4t:
            r, _ = run_wrapped_episode(
                variant_id=label,
                spec=sp,
                base_env_policy="V19_ADAPTIVE_RL",
                external_proposer=prop,
                hard_budget_ms=35.0,
            )
            rows.append(r)
            if not r["comp100"] or r["net_obj"] > 5000.0:
                failed_keys.append((r["episode_key"], r["net_obj"], r["comp100"], r["ticks"]))
        ov = summarize_episode_rows(rows)
        ov["outliers"] = failed_keys
        report[label] = ov
        print(
            f"  [{label}] 4Topo net={ov['mean_net_obj']} (med={ov['median_net_obj']}), "
            f"comp={ov['comp100_count']}/48, p95={ov['pooled_p95_ms']}ms, outliers={failed_keys}",
            flush=True,
        )

    out_p = OUT_AUDIT_DIR / "v20_a_s5_isolation.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_s5_isolation()
