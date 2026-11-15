"""Replay archived simulator behavior for the published research baselines."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ARCHIVE = Path(__file__).resolve().parents[1] / "experiments/reference/amr-f62c064.zip"
SOURCE_COMMIT = "f62c0641bd0d56c86ddcf21d06e725c75d5b6a0f"
ARCHIVE_SHA256 = "a9f249557b742f339331ccd781d8ed66d37fcc6be14c88d6d20020c158a0f164"

_RUNNER = """
import json
from pathlib import Path
import sys

archive = sys.argv[1]
sys.path.insert(0, archive)
from amr_sched import simulation
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.allocation.baseline import NearestShortestAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario

if not Path(simulation.__file__).is_relative_to(archive):
    raise RuntimeError("historical replay did not load the frozen simulator")

planners = {"static_dijkstra": StaticDijkstraPlanner, "spacetime_astar": SpaceTimeAStarPlanner}
allocators = {"greedy": GreedyAllocator, "nearest": NearestShortestAllocator}
summaries = []
for case in json.load(sys.stdin):
    scenario = ScenarioConfig(**case["scenario"])
    net, tasks, agvs = generate_scenario(scenario)
    planner = planners[case["planner"]](scenario.max_speed, scenario.dt)
    allocator = allocators[case["allocator"]]()
    config = simulation.SimulationConfig(**case["config"])
    summaries.append(simulation.Simulator(net, tasks, agvs, planner, allocator, config).run().summary())
print(json.dumps(summaries))
"""


def replay_legacy_cases(cases: list[dict]) -> list[dict]:
    """Run cases in an isolated interpreter using the byte-checked source archive."""
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"Historical source archive mismatch for {SOURCE_COMMIT}: {digest}")
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", _RUNNER, str(ARCHIVE)],
        input=json.dumps(cases), encoding="utf-8", capture_output=True, timeout=300, check=True,
    )
    return json.loads(completed.stdout)