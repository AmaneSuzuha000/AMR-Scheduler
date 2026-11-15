"""Corroborate stopped callback events with independently retained tick samples."""
import math


def corroborate_stopped(actual,observed):
    # Callback events can fall between 20Hz recording ticks (native publisher
    # is 40Hz). Sequence/force fields then legitimately differ. Require a
    # nearby native stamp, <=2mm spatial agreement, and both velocities stopped.
    assert abs(actual['time_s']-observed['time_s'])<=.05
    assert math.dist(actual['xyz'],observed['xyz'])<=.002
    assert max(abs(v) for v in actual['velocity'])<.035
    assert max(abs(v) for v in observed['velocity'])<.035
