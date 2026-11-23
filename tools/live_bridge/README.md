# live_bridge — chassis-side bridge

Streams the vehicle pose to the RAWSim-O window in real time and receives A→B
routes back from it. Uses only the Python standard library; Python 3.8+ works.
When measuring latency on Windows use Python 3.13 or newer — earlier versions
have only 15.6 ms resolution in `time.time()`, which makes the pose timestamp
`t` inaccurate. The send loop is scheduled against absolute deadlines; measured
20.00 Hz on a local machine.

## Three modes

| Mode | Purpose | Launch |
|---|---|---|
| `circle` | Drive in a circle, test position display | `python live_bridge.py --mode circle --center 1.5 1.5 --radius 0.9` |
| `sim` | Simulated chassis following a route from the window, tests A→B | `python live_bridge.py --mode sim --start 0.825 1.275 0` |
| `replay` | Replay poses from an old JSONL log | `python live_bridge.py --mode replay --log run.jsonl` |

Listens on `127.0.0.1:9300` by default and sends 20 poses per second
(`--rate`). When the script runs on the vehicle computer, use
`--host 0.0.0.0` so a laptop can connect — the protocol has no authentication,
so only do this on a trusted network.

## Tests

```bash
python -m unittest -v test_live_bridge
```

## Connecting a real chassis

Subclass `Vehicle` in `live_bridge.py` and implement three methods:

- `step(dt)` — return the chassis pose `(x, y, yaw)` in the chassis's own frame,
  in meters and radians.
- `nav(goal, path, max_speed)` — drive along the point list `[(x, y), ...]` and
  return `True`; call `self._report(goal, 'succeeded' | 'failed', reason)` on
  arrival or failure.
- `cancel(goal)` — stop and call `self._report(goal, 'canceled')`.

No coordinate conversion is needed: the window uses a calibration file to map
chassis coordinates into map coordinates, and converts back before sending a route.

## Native fleet execution configuration

Scene road geometry now follows the directed `Edge.centerline` through scheduler MOVE `path_xy`, native sampling and explicit stopped-action `resume_path_xy`. Recovery retains future corners and joins the measured stop to its road projection within the existing arrival tolerance. The native adapter latches sensor departures beyond the unchanged .35m road bound; the independent physics guard still enforces velocity/lease/stop limits. Empty edge geometry retains the legacy straight-edge interpretation. Checkpoint definitions include geometry and implementation hashes, so old snapshots require their original frozen runtime rather than migration into changed geometry. Physical acceptance for bent roads is tracked separately from code regression.

The sensor-based fleet owner in `fleet_scheduler.py` uses one target pose per node. Build an explicit physical copy with `fleet_config.point_execution_network(algorithm_network)` before constructing that owner. This preserves node IDs, geometry, road directions, lengths and speeds, while setting node and edge capacities to one. The source scene and algorithm network remain unchanged; abstract multiple slots require separate physical slot geometry before native execution can support them. The owner rejects capacities above one before creating output or adapters.

The complete synthetic workshop evidence and active limitations are tracked in `out/archive/workshop_navigation/README.md`. This configuration check alone does not establish physical navigation acceptance.
