# AMR Gazebo execution guard

The guard runs inside the physics process and forwards Nav2 velocity commands
to the existing native differential-drive plugin. It contains no route planner,
task allocation or reservation logic. It is locked at startup.

Build on Ubuntu Humble with the existing SCOUT description workspace sourced:

```bash
colcon build --base-paths tools/ros2/amr_gazebo_guard --packages-select amr_gazebo_guard
source install/setup.bash
ros2 launch amr_gazebo_guard scout_execution.launch.py world:=/absolute/path/to/world.world
```

The launch keeps the original collision/drive model and changes the wheel
friction basis to ODE's automatic tangent selection (`fdir1=0 0 0`). It remaps
the wheel controller input to `/guarded_cmd_vel`. Nav2 retains its standard
`/cmd_vel` output; only this plugin publishes `/guarded_cmd_vel`.

`/motion_guard/authorize` requires the current guard instance ID, owner run ID,
route ID and a strictly increasing revision. Arming requires a fresh physical
stop: planar speed <0.01 m/s and |yaw rate| <0.02 rad/s for 0.3 s in both wall
and simulation time. A restart/world reset changes the guard ID. A different
owner is rejected; process restart reconciliation is a separate requirement.

After arming, `/motion_guard/lease` must repeat the exact action identity with
increasing sequence numbers and fresh simulation timestamps. Loss of lease
for 0.5 s or velocity commands for 0.25 s latches zero output. The first command
has a 1 s startup allowance, still requiring a lease within 0.5 s. Expiry is
checked before accepting late input as well as during each physics update.
Nonfinite, nonplanar or out-of-range commands latch zero. New data alone does
not clear a latch. Explicit stop also requires the matching action identity.

The 0.25 s command watchdog is not relaxed for a controller that goes quiet
mid-route. It is suspended only while the last accepted command was exactly
zero and the body is no longer meaningfully moving (planar speed <0.05 m/s,
|yaw rate| <0.1 rad/s) for 0.05 s. That suspension is itself bounded to 1.5 s,
after which the ordinary silence fault applies. The reason is that a zero
target cannot move the vehicle, so silence after a stop command is not an
uncontrolled-motion hazard, while a dead controller that was commanding motion
still latches after 0.25 s exactly as before. Nav2 reaches its goal, publishes
one final zero and stops publishing before the action result is delivered to
the owner; without this bounded allowance that ordinary completion handoff can
outlast 0.25 s and be reported as a fault against a *successful* goal.

This gate forwards **unstamped** native Nav2 Twist messages. The execution
adapter must confirm the old action terminated before authorizing a new one;
the guard cannot infer a route ID from a Twist. Status, actuation and authority
must be verified together. The scaled fixture and uncalibrated tire parameters
remain integration evidence, not a full-scale clearance or real-robot test.
