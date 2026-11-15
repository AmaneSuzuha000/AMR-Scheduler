#!/usr/bin/env python3
"""修复 SCOUT Mini 的轮子 collision 几何。

问题
----
URDF 里轮子的 collision 是：
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.08" length="0.084"/></geometry>
    </collision>

URDF 的 <cylinder> 默认轴线沿 **Z 轴**。而轮子的视觉/运动轴线应当是水平的
（沿关节局部 Y 轴，因为 joint origin 有 rpy="-1.57 0 0"）。

于是碰撞体是一个「立着的圆盘」，轴线竖直：
  - 它能靠接触摩擦推着车走直线；
  - 但绕竖直轴的力臂为零，差速无法产生偏航力矩
    -> 纯角速度指令下四个轮子符号正确、幅值正确，车却不转。

修复
----
把 collision 的圆柱旋转 90 度绕 X 轴，使其轴线水平，
与真实轮子（以及 joint 的旋转轴）一致：

    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      ...
    </collision>

只改这一处，不动别的。
"""
import re, sys, shutil, time

X = "/home/suzuha/scout_gazebo_ws/src/scout_gazebo_description/urdf/scout_mini.urdf.xacro"

src = open(X, encoding='utf-8').read()

# 定位 wheel 宏里的 collision 块
old = '''      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="0.084"/>
        </geometry>
      </collision>'''

new = '''      <collision>
        <origin xyz="0 0 0" rpy="1.5708 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="0.084"/>
        </geometry>
      </collision>'''

if old not in src:
    print("!! 未命中原始 collision 块，未修改")
    print("   实际内容（前 60 行里含 collision 的部分）：")
    for i, line in enumerate(src.splitlines(), 1):
        if 'collision' in line or 'cylinder' in line:
            print("   %3d: %s" % (i, line))
    sys.exit(1)

if new in src:
    print("已经修过了（rpy=1.5708 已存在），跳过")
    sys.exit(0)

backup = X + ".bak_geo_" + time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(X, backup)
print("备份 ->", backup)

open(X, 'w', encoding='utf-8').write(src.replace(old, new, 1))
print("已修改 collision 的 rpy 为 1.5708 0 0")

# 验证
chk = open(X, encoding='utf-8').read()
m = re.search(r'<collision>\s*<origin[^>]*rpy="([^"]+)"', chk)
print("当前 collision rpy =", m.group(1) if m else "读不到")
