"""Decode measured native model states; no command or scheduler decisions."""
import math


def decode(message, robots=('scout_a','scout_b')):
    assert message.header.frame_id=='wheelset_world'
    assert list(message.joint_names)==list(robots)
    assert len(message.transforms)==len(message.twist)==len(robots)
    stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
    assert math.isfinite(stamp) and stamp>=0
    result={}
    for name,transform,twist in zip(message.joint_names,message.transforms,message.twist):
        p=transform.translation; q=transform.rotation
        assert abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1)<1e-5
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        observation=dict(pose=[p.x,p.y,yaw],z=p.z,linear_speed=math.hypot(twist.linear.x,twist.linear.y),
            angular_speed=abs(twist.angular.z),sim_time_s=stamp)
        assert all(math.isfinite(v) for v in [*observation['pose'],observation['z'],observation['linear_speed'],observation['angular_speed']])
        result[name]=observation
    return result
