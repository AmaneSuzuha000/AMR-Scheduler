"""Inspect the running scene through ROS services and /clock, without moving any equipment."""
import hashlib
import json
import os
from pathlib import Path
import time
import sys
os.environ["ROS_DOMAIN_ID"]="73"
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetModelList
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.wheelset_scene.evidence import require_running_build
from tools.wheelset_scene.build import paint_strips
fingerprint=require_running_build()
scene_path=ROOT/"scenes/wheelset_workshop_video_v1.json"
scene=json.loads(scene_path.read_text(encoding="utf-8"))
rclpy.init();node=Node("wheelset_runtime_audit")
client=node.create_client(GetModelList,"/get_model_list")
if not client.wait_for_service(timeout_sec=30):raise RuntimeError("GetModelList not available")
future=client.call_async(GetModelList.Request());rclpy.spin_until_future_complete(node,future,timeout_sec=20)
if not future.done() or not future.result().success:raise RuntimeError("Model query failed")
names=set(future.result().model_names)
expected={i["id"] for i in scene["instances"]}|{i["id"] for i in scene["boxes"]}|{"review_camera"}
expected_without_paint=len(expected)
expected|={i['id'] for i in paint_strips(scene)}
missing=sorted(expected-names)
samples=[]
def clock_cb(m):samples.append((time.monotonic(),m.clock.sec+m.clock.nanosec*1e-9))
sub=node.create_subscription(Clock,"/clock",clock_cb,qos_profile_sensor_data)
end=time.monotonic()+10
while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.2)
factor=(samples[-1][1]-samples[0][1])/(samples[-1][0]-samples[0][0]) if len(samples)>1 else None
server_count=0
for process in Path('/proc').glob('[0-9]*'):
    try:
        command=(process/'cmdline').read_bytes().split(b'\0')[0].decode()
        if command and Path(command).name=='gzserver':server_count+=1
    except (OSError,UnicodeDecodeError):pass
report={"passed":not missing and factor is not None and factor>0,"build_fingerprint":fingerprint,"scene_sha256":hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "runtime_model_count":len(names),"expected_model_count":len(expected),
        "expected_model_count_excluding_paint":expected_without_paint,"missing_models":missing,
        "clock_sample_count":len(samples),"observed_realtime_factor":factor,
        "observed_gzserver_process_count":server_count,
        "conditions":f"Observed {server_count} Gazebo server process(es); this scene uses software rendering. Diagnostic only, not a benchmark.",
        "physics_validation":False,"source_camera_registration_validated":False}
out=ROOT/"simulation/wheelset_workshop/runtime/runtime_audit.json";out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
node.destroy_node();rclpy.shutdown()
if not report["passed"]:raise SystemExit(1)
