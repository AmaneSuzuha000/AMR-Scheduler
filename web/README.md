# Multi-AGV scheduling console

## What this is

An **interactively running** scheduler UI, not a result player.

The difference: after you press *Run*, the server advances the simulation with
`Simulator.step_once()`, one batch of steps at a time, and pushes the current
state to the front end. So *Pause* really stops the simulation, *insert urgent*
really changes the task set being executed, and *switch planner* really changes
the algorithm used for the next path segment.

## Launch

```bash
python web/server.py
```

Then open <http://127.0.0.1:8077>. Override the port with the `AMR_WEB_PORT`
environment variable.

### Checking that it actually drives itself

```bash
python -m web.server --self-check --port 8099
```

This boots a real uvicorn server on a real socket, POSTs one small job, and waits for
the **background task** to finish it — nobody calls `_drive_sync`. It prints
`SELF-CHECK PASS` and exits 0 on success.

The pytest suite cannot cover that path: FastAPI's `TestClient` never runs the
`asyncio.create_task` that `start_run` spawns, so those tests drive the run by hand.
If the background driver broke, every one of them would still pass and the console
would sit at step 0 in front of the user. This mode is the one that would catch it.

Expected output:

```
SELF-CHECK run_id=20261105T110400-eeb21a
SELF-CHECK state=finished completed=4 elapsed=0.0s frames=4
SELF-CHECK PASS
```

## What you can do

| Control | Description |
|---|---|
| Layout | Ring (main study instance, 17 nodes / 36 edges) / grid / hub |
| Fleet size | 1–12 vehicles |
| Task count | 1–300 |
| Planner | Static Dijkstra / space-time A* |
| Allocator | Greedy (primary) / nearest-first / FIFO |
| Reschedule policy | No preemption / non-destructive / aggressive |
| Speed | Steps per second; 0 means run as fast as possible |
| Run / pause / stop | Pausing really freezes the step counter |
| **Insert urgent task** | Join the schedule at any time during a run |
| Vehicle state | Position, state, current task, remaining route per vehicle |
| Export and review | Every run is written to `results/web_runs/<run_id>/` |

## Output files

At the end of a run (or on stop) the server writes:

```
results/web_runs/<run_id>/
  config.json    configuration (reproducible)
  summary.json   all metrics
  frames.json    last 2000 trajectory frames
  tasks.csv      final state of every task
  agvs.csv       per-vehicle statistics
```

Past runs are listed by `GET /api/runs` and remain loadable after a service
restart clears in-memory state.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/meta` | Available layouts / planners / allocators / policies |
| POST | `/api/run` | Create and start a run |
| POST | `/api/pause?run_id=` | Pause / resume |
| POST | `/api/speed?run_id=` | Change advance rate |
| POST | `/api/urgent?run_id=` | Insert an urgent task mid-run |
| POST | `/api/stop?run_id=` | Stop |
| GET | `/api/runs` | Run history |
| GET | `/api/runs/{id}` | Config, summary, and frames of one run |
| GET | `/api/runs/{id}/export?kind=tasks` | Export CSV |
| WS | `/ws` | Live frame stream |

## Design constraints

1. **The scheduling core is not rewritten here.** `web/server.py` only drives,
   transports, and persists; all scheduling logic stays in `amr_sched/`.
2. **Live, not pre-computed.** Advancement goes through `Simulator.step_once()`
   rather than computing everything up front and playing it back.
3. **Bit-exact equivalence.** `reset()` plus repeated `step_once()` produces
   exactly the same result as `run()`; `Tests/test_simulator_stepping.py` locks
   this with SHA256 digests.

## Tests

```bash
python -m pytest Tests/test_web_console.py -v           # 7 endpoint-level tests
python -m pytest Tests/test_simulator_stepping.py -v    # 3 equivalence tests
```
