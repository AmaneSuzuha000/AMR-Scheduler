#!/usr/bin/env python3
"""Observe wheel angular velocity and true pose together under pure angular velocity.

Only one thing matters here: when given (v=0, w=+0.8),
  the left wheels should spin at -2.08 rad/s and the right wheels at +2.08 rad/s (opposite signs).
If all four wheels share the same sign, the differential mapping is not in effect.
"""
import subprocess, re, math, time

def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None

def gt():
    r = sh("gz model -m scout_mini -i 2>/dev/null")
    t = r.stdout if r else ""
    m = re.search(r'position \{\s*x: (-?[0-9.eE+-]+)\s*y: (-?[0-9.eE+-]+)\s*z: (-?[0-9.eE+-]+)', t)
    if not m:
        return None
    return tuple(float(m.group(i)) for i in (1,2,3))

def js():
    """Read the velocity field of /joint_states (once)."""
    r = sh("timeout 4 ros2 topic echo /joint_states --once 2>/dev/null", timeout=10)
    t = r.stdout if r else ""
    names = re.findall(r'^- (\w+wheel)$', t, re.M)
    mv = re.search(r'velocity:\s*\n((?:- [^\n]+\n)+)', t)
    if not mv:
        return None, None
    vals = [float(x) for x in re.findall(r'- (-?[0-9.eE+-]+)', mv.group(1))]
    return names, vals

print("=== before publishing ===")
sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
time.sleep(1)
p0 = gt()
print("  true pose:", p0)
print("  joint_states:", js())

print("")
print("=== publish (v=0, w=+0.8) continuously for 3 s, sampling wheel speeds meanwhile ===")
# publish in the background with setsid and an explicit pid so it can be killed precisely
pub = subprocess.Popen(
    'ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist '
    '"{linear: {x: 0.0}, angular: {z: 0.8}}"',
    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    preexec_fn=__import__('os').setsid)

for k in range(3):
    time.sleep(1.0)
    names, vals = js()
    print("  sample #%d:" % (k+1), dict(zip(names, vals)) if names else "unreadable")

pub.terminate()
sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
sh('ros2 topic pub --times 5 --rate 20 /cmd_vel geometry_msgs/msg/Twist '
   '"{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1', timeout=20)
sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
time.sleep(1.5)

print("")
print("=== after publishing ===")
p1 = gt()
print("  true pose:", p1)
if p0 and p1:
    print("  displacement %.4f m" % math.hypot(p1[0]-p0[0], p1[1]-p0[1]))

print("")
print("=== theoretical values (wheel_separation=0.416503, wheel_radius=0.08) ===")
v = 0.8 * 0.416503 / 2.0
print("  v_left  = -%.4f m/s -> %.4f rad/s" % (v, -v/0.08))
print("  v_right = +%.4f m/s -> %.4f rad/s" % (v, +v/0.08))
