#!/usr/bin/env python3
"""修复 live_bridge 的 GazeboVehicle 控制器：去掉「原地转」分支。

问题
----
原实现是 turn-then-drive：

    if abs(err) > self.heading_tol:
        self._publish_cmd(0.0, wz)      # 纯角速度
    else:
        self._publish_cmd(vz, wz)

而 Gazebo 的 planar_move 插件对纯角速度**完全不响应**（实测 yaw 变化
≤0.047 rad，期望最高 3.0 rad）；但只要 vx>0，转角立刻与 wz 成正比。

于是朝向误差一旦超过 heading_tol，车就永远转不过来，卡死到超时。

修复
----
1. 永不发 vx=0 的转向指令：转向时给一个很小的前进速度。
2. planar 转向约有 50% 的损失，用 gain 补偿。
3. 朝向误差大时减速但不归零。

本脚本只改 _control_once 这一个方法，其余代码原样保留。
"""
import re, sys, shutil, time

F = "/home/suzuha/AMR/tools/live_bridge/live_bridge.py"

src = open(F, encoding='utf-8').read()

# ---------- 定位 GazeboVehicle._control_once 的转向段 ----------
old = '''        x, y, yaw, tx, ty, speed = active
        dist = math.hypot(tx - x, ty - y)
        err = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
        if abs(err) > self.heading_tol:
            wz = max(-self.turn_rate, min(self.turn_rate, 2.0 * err))
            self._publish_cmd(0.0, wz)
        else:
            vz = min(speed, max(0.05, dist))
            wz = max(-self.turn_rate, min(self.turn_rate, 1.5 * err))
            self._publish_cmd(vz, wz)'''

new = '''        x, y, yaw, tx, ty, speed = active
        dist = math.hypot(tx - x, ty - y)
        err = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
        # --- GZ-2 fix ---
        # planar_move 插件对纯角速度完全不响应，因此绝不允许 vx 归零。
        # 转角实测约为指令的 50%，用 TURN_GAIN 补偿。
        TURN_GAIN = 2.0
        CRAWL = 0.05          # 转向时的最小前进速度 m/s
        wz = max(-self.turn_rate, min(self.turn_rate, TURN_GAIN * 2.0 * err))
        if abs(err) > self.heading_tol:
            # 大误差：慢速前进 + 大角速度，边挪边转
            vz = CRAWL
        else:
            # 小误差：正常速度，按误差比例减速
            slow = max(0.35, 1.0 - abs(err) / 0.6)
            vz = max(CRAWL, min(speed, dist, speed * slow))
        self._publish_cmd(vz, wz)'''

if new in src:
    print("已修过，跳过")
    sys.exit(0)
if old not in src:
    print("!! 未命中原 _control_once 转向段")
    idx = src.find('def _control_once')
    print("   _control_once 附近内容：")
    print(src[idx:idx+1400] if idx > 0 else "   找不到 _control_once")
    sys.exit(1)

backup = F + ".bak_ctl_" + time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(F, backup)
print("备份 ->", backup)

open(F, 'w', encoding='utf-8').write(src.replace(old, new, 1))
print("已替换 _control_once 的转向逻辑")

chk = open(F, encoding='utf-8').read()
print("含 CRAWL:", "CRAWL" in chk)
print("含 TURN_GAIN:", "TURN_GAIN" in chk)
print("仍有 publish_cmd(0.0, wz):", "publish_cmd(0.0, wz)" in chk)
