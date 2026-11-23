# Gazebo execution: what is verified, what is not

This file records the boundary between what has been measured on this install and what
has not. It exists because the boundary moved several times, each time because a number
that looked measured turned out to be a single unrepeated sample.

## Verified: the whole chain, in one command

```
python tools/live_bridge/verify_scheduler_to_gazebo.py
```

Run on the Gazebo host, this exports routes from `amr_sched.network`, checks each waypoint
equals its node's position under the documented scale, drives every route, requires all of
them to complete, and writes `results/gazebo_routes.csv`.

That CSV is the source of every accuracy number below, and it is covered by
`tools/data_manifest.py` alongside the other 22 result files. Numbers typed into prose go
stale without anyone noticing; a checksummed row does not. To reproduce:

    python tools/live_bridge/verify_scheduler_to_gazebo.py
    python tools/data_manifest.py --check

Latest recorded run (`results/gazebo_routes.csv`):

| route | result | mean error | max error |
| --- | --- | --- | --- |
| D -> W3 | 5/5 | 0.168 m | 0.416 m |
| D -> S | 5/5 | 0.163 m | 0.411 m |
| W1 -> W5 | 6/6 | 0.122 m | 0.433 m |
| R0 -> B1 | 4/4 | 0.151 m | 0.409 m |

The route file used to be typed by hand. The coordinates happened to match what the network
implies to 0.0000 m, but nothing enforced it, so a divergence between the vehicle's path and
the scheduler's path would have been invisible while every individual test still passed.
`gz_export_routes.py` now derives the file from the network; `Tests/test_gz_route_export.py`
pins the scale, the origin, and the fact that moving a node moves its waypoint.

## Verified: individual stages

**The pose channel works.** `/get_entity_state` and `/set_entity_state` are provided by
`libgazebo_ros_state.so`. That plugin was **not present** in the launch's world file and
had to be added to `empty.world`; without it both services are absent and the only channel
is `gz model -m <name> -i`, which answers in ~0.41 s and then stops answering entirely after
a few minutes while gzserver keeps simulating. With the plugin, ground truth reads take
about 0.2 ms and teleporting between routes works.

The reader/writer for those services is `tools/live_bridge/gz_pose_node.py`; the route
exporter is `gz_export_routes.py`; the route runner is `gz_follow_routes.py`. Measurements
quoted below come from `gz_actuator_calib.py`, `gz_repeat_probe.py`, `gz_rtf_probe.py`,
`gz_envelope_probe.py` and `gz_loop_rate.py`, all in the same directory.

**Translation is repeatable and accurate.** Three identical runs of a straight-line
command, 2.5 s each, measured against the service:

| vx commanded | distance, three runs | spread | ratio |
| --- | --- | --- | --- |
| 0.10 m/s | 0.236 / 0.249 / 0.251 m | 0.015 m | 0.94-1.00 |
| 0.20 m/s | 0.499 / 0.489 / 0.486 m | 0.013 m | 0.98-1.00 |

**The control loop is fast enough.** 19.7 Hz with a 50 ms sleep; a pose read costs 0.2 ms
median. At 0.30 m/s that is a decision every 0.02 m of travel, so loop rate is not a
limiting factor here.

**The simulator runs at real time.** Real-time factor 0.983, measured from `/clock` with a
BEST_EFFORT subscription (the default RELIABLE subscription is rejected by QoS
compatibility and silently receives nothing).

**All four nominal routes complete, reproducibly.** Three consecutive independent runs of
the full route set, each driving all 20 waypoints (this was before the results file existed,
so the figures are transcribed from the runs rather than read back from it):

| run | D->W3 | D->S | W1->W5 | R0->B1 |
| --- | --- | --- | --- | --- |
| 1 | 5/5, mean 0.139 | 5/5, 0.135 | 6/6, 0.126 | 4/4, 0.139 |
| 2 | 5/5, mean 0.141 | 5/5, 0.118 | 6/6, 0.140 | 4/4, 0.149 |
| 3 | 5/5, mean 0.190 | 5/5, 0.148 | 6/6, 0.123 | 4/4, 0.147 |

Mean waypoint error is 0.12-0.19 m. Most waypoints are much better than that -- 0.012 to
0.13 m -- and the mean is pulled up by the final waypoint of each route, where the vehicle
stops instead of driving through (0.40-0.45 m). That asymmetry is in the data, not averaged
away.

**Rotation repeats in part of the command space.** A 20-cell grid of (vx, wz), two runs per
cell, mapped by `gz_envelope_probe.py`. 14 cells repeated, 6 did not, and the 6 cluster
where one wheel is reversing or nearly stopped: (0.10, 0.80), (0.20, 0.40), (0.25, 0.80),
(0.30, 0.80). Inside the stable cells the achieved yaw rate is close to constant:

| wz commanded | wz achieved | ratio |
| --- | --- | --- |
| 0.40 | 0.15 | 0.38 |
| 0.80 | 0.34 | 0.43 |
| 1.20 | 0.51 | 0.43 |

The response saturates near 0.51 rad/s whatever is asked for. `gz_follow_routes.py` uses
0.43 and stays inside the envelope (vx 0.15-0.25, wz at most 1.20) rather than tuning until
a route happens to pass.

**A measurement bug was hiding the real accuracy.** The runner originally counted a
waypoint as reached the moment the vehicle entered its 0.45 m tolerance circle. Every
reported error then came out at 0.42-0.45 m -- the tolerance reflected back as though it
were a result, on every waypoint, which looks precise and measures nothing. A closest-
approach rule fixed it: a waypoint is done when the distance starts growing again, and the
recorded number is the closest the vehicle actually got. The same routes, unchanged, then
reported 0.12-0.19 m. The routes had always been driven this well; the measurement had not
been able to see it.

## Not verified, and deliberately not claimed

**Per-waypoint accuracy is not a precision result.** It is measured against a 0.45 m
tolerance on a 1.6 m grid, at 0.15-0.25 m/s, over three runs. It says the vehicle arrives
where the route says, not how accurately it could be made to with better hardware.

**Earlier ratios in this document were wrong.** Versions of `GAZEBO.md` and of the follower
carried yaw attenuation ratios of 0.16, 0.31, 0.62 and 0.70, each from a single run. All
four were noise. They are removed rather than revised, because the measurement behind them
cannot support any value. The ratios in the table above come from the two-run grid and
changed when the envelope was respected, which is the difference between the two.
**`/odom` is not ground truth and is not used as such.** Measured against a bounded
command it reported 0.879 m of travel where the vehicle could have moved at most 0.45 m.
Wheel slip can only make the reported distance larger than the true one, never
smaller -- but 1.95x is beyond what slip explains, so `/odom` here disagrees with the
commanded motion in a way that cannot be corrected by scaling. A test enforces that
`/odom` does not appear in executable path.

## What would change this

The rotation response would need characterisation with many repeats per command, and a
stability check that the plugin is not saturating. With repeatable rotation we could state a
turn rate with a confidence interval instead of the nominal value used in
`gz_follow_routes.py`. Until then the honest summary is: translation measured, rotation not.
