#!/usr/bin/env python3
"""Map out the (vx, wz) response surface in planar mode.

Known so far:
  pure angular velocity (0, w)  -> no response at all
  pure linear velocity (v, 0)   -> moves forward normally
  (0.15, 0.5)                   -> it does turn, but only 54% of the expected angle

The question to answer: which combination reliably produces a turn?
That decides how the controller in the bridge should be written.
"""
import subprocess, re, math, time

def sh(cmd, timeout=25):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None

def gt():
    r = sh("gz model -m scout_mini -i 2>/dev/null")
    t = r.stdout if r else ""
    m = re.search(r'position \{\s*x: (-?[0-9.eE+-]+)\s*y: (-?[0-9.eE+-]+)', t)
    q = re.search(r'orientation \{\s*x: (-?[0-9.eE+-]+)\s*y: (-?[0-9.eE+-]+)'
                  r'\s*z: (-?[0-9.eE+-]+)\s*w: (-?[0-9.eE+-]+)', t)
    if not (m and q):
        return None
    x, y = float(m.group(1)), float(m.group(2))
    qx, qy, qz, qw = (float(q.group(i)) for i in range(1, 5))
    return (x, y, math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))

def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a <= -math.pi: a += 2*math.pi
    return a

def settle():
    sh('ros2 topic pub --times 5 --rate 20 /cmd_vel geometry_msgs/msg/Twist '
       '"{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1', timeout=15)
    sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
    time.sleep(1.2)

def drive(vx, wz, sec=3.0):
    n = max(1, int(sec*20))
    sh('ros2 topic pub --times %d --rate 20 /cmd_vel geometry_msgs/msg/Twist '
       '"{linear: {x: %.3f}, angular: {z: %.3f}}" >/dev/null 2>&1' % (n, vx, wz),
       timeout=int(sec)+25)
    sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
    settle()

sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
time.sleep(1)
settle()

grid = []
for vx in (0.0, 0.05, 0.10, 0.20):
    for wz in (0.0, 0.3, 0.6, 1.0):
        grid.append((vx, wz))

print("%6s %6s | %9s | %10s | %10s | %s" % ("vx", "wz", "displacement(m)", "yaw change", "expected yaw", "turns"))
print("-" * 74)
ok_turn = []
for vx, wz in grid:
    settle()
    b = gt()
    if b is None:
        print("%6.2f %6.2f | pose unavailable" % (vx, wz)); continue
    drive(vx, wz, 3.0)
    a = gt()
    if a is None:
        print("%6.2f %6.2f | pose unavailable" % (vx, wz)); continue
    d = math.hypot(a[0]-b[0], a[1]-b[1])
    dy = wrap(a[2]-b[2])
    exp = wz*3.0
    mark = ""
    if abs(wz) > 1e-6:
        if abs(dy) > 0.3:
            mark = "YES"
            if abs(dy) > 0.5*abs(exp):
                ok_turn.append((vx, wz, dy, exp))
        else:
            mark = "no"
    print("%6.2f %6.2f | %9.4f | %+10.4f | %+10.4f | %s" % (vx, wz, d, dy, exp, mark))

print("")
print("=== combinations that turn reliably (|yaw| > half the expectation) ===")
for vx, wz, dy, exp in ok_turn:
    print("  vx=%.2f wz=%.2f -> actual %+.3f / expected %+.3f (%.0f%%)" % (vx, wz, dy, exp, 100*dy/exp))
if not ok_turn:
    print("  none")
