"""Capture actual Gazebo camera frames using ROS image messages and state services."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import zlib

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument("--domain",type=int,default=73)
parser.add_argument("--cameras",default="all");parser.add_argument("--timeout",type=float,default=90)
parser.add_argument("--resume",action="store_true",help="Reuse only current-build, hash-checked captures at the requested pose.")
args=parser.parse_args()
os.environ["ROS_DOMAIN_ID"]=str(args.domain)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetEntityState,GetEntityState
import sys
sys.path.insert(0,str(ROOT))
from tools.wheelset_scene.build import camera_pose
from tools.wheelset_scene.evidence import require_running_build
fingerprint=require_running_build()


def write_png(message,path):
    w,h=message.width,message.height;data=bytes(message.data)
    encoding=message.encoding.lower();channels={"rgb8":3,"bgr8":3,"rgba8":4,"bgra8":4}.get(encoding)
    if channels is None:raise RuntimeError("Unsupported camera encoding "+encoding)
    rows=[]
    for y in range(h):
        row=data[y*message.step:y*message.step+w*channels]
        if encoding!="rgb8":
            buf=bytearray(w*3)
            if encoding.startswith("bgr"):
                for x in range(w):buf[x*3:x*3+3]=row[x*channels:x*channels+3][::-1]
            else:
                for x in range(w):buf[x*3:x*3+3]=row[x*channels:x*channels+3]
            row=bytes(buf)
        rows.append(b"\0"+row)
    def chunk(kind,body):return struct.pack(">I",len(body))+kind+body+struct.pack(">I",zlib.crc32(kind+body)&0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+
                     chunk(b"IDAT",zlib.compress(b"".join(rows),6))+chunk(b"IEND",b""))


rclpy.init();node=Node("wheelset_visual_capture")
latest={"message":None,"count":0}
def on_image(message):latest.update(message=message,count=latest["count"]+1)
image_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
subscription=None
setter=node.create_client(SetEntityState,"/wheelset_review/set_entity_state")
getter=node.create_client(GetEntityState,"/wheelset_review/get_entity_state")
if not setter.wait_for_service(timeout_sec=args.timeout):raise RuntimeError("Gazebo state service unavailable")
if not getter.wait_for_service(timeout_sec=args.timeout):raise RuntimeError("Gazebo get state service unavailable")
scene=json.loads((ROOT/"scenes/wheelset_workshop_video_v1.json").read_text(encoding="utf-8"))


def registration_status(camera):
    registration=camera.get('registration','')
    if registration=='manual_sparse_landmark_fit':
        return 'MANUAL_SPARSE_LANDMARK_FIT_NOT_FULL_IMAGE_VALIDATED'
    if registration.startswith('same_video_'):
        return 'SAME_VIDEO_JOINT_GEOMETRY_ESTIMATE_NOT_FULL_IMAGE_VALIDATED'
    return 'INITIAL_ESTIMATE_NOT_CALIBRATED_TO_VIDEO'


out=ROOT/"visual_comparison/wheelset_workshop_v1/gazebo";out.mkdir(parents=True,exist_ok=True)
cameras=scene["cameras"]
if args.cameras!="all":cameras=[c for c in cameras if c["id"] in args.cameras.split(",")]
report=[]
report_path=out/'capture_report.json'
if args.resume and report_path.exists():
    for row in json.loads(report_path.read_text(encoding='utf-8')):
        camera=next((c for c in scene['cameras'] if c['id']==row['camera']),None)
        path=out/row['file']
        if (camera and row.get('build_fingerprint')==fingerprint and path.is_file()
                and row.get('camera_registration')==registration_status(camera)
                and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
                and math.dist(row['requested_pose'],camera_pose(camera,scene['global_scene_scale']))<1e-6):
            report.append(row)
for camera in cameras:
    if any(r['camera']==camera['id'] for r in report):continue
    pose=camera_pose(camera,scene["global_scene_scale"])
    request=SetEntityState.Request();request.state.name="review_camera";request.state.reference_frame="world"
    request.state.pose.position.x,request.state.pose.position.y,request.state.pose.position.z=pose[:3]
    pitch,yaw=pose[4],pose[5];q=request.state.pose.orientation
    q.x=-math.sin(pitch/2)*math.sin(yaw/2);q.y=math.sin(pitch/2)*math.cos(yaw/2)
    q.z=math.cos(pitch/2)*math.sin(yaw/2);q.w=math.cos(pitch/2)*math.cos(yaw/2)
    future=setter.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=20)
    if not future.done() or not future.result().success:raise RuntimeError("Camera pose command failed")
    # A fresh volatile reader prevents a previous view's large-image backlog from being reused.
    if subscription is not None:node.destroy_subscription(subscription)
    latest.update(message=None,count=0)
    subscription=node.create_subscription(Image,"/wheelset_review/reference/image_raw",on_image,image_qos)
    start=latest["count"];deadline=time.monotonic()+args.timeout
    while latest["count"]<start+4 and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.5)
    if latest["count"]<start+4:raise RuntimeError("Camera frames timed out")
    query=GetEntityState.Request();query.name="review_camera";query.reference_frame="world"
    future=getter.call_async(query);rclpy.spin_until_future_complete(node,future,timeout_sec=20)
    if not future.done() or not future.result().success:raise RuntimeError("Camera pose read failed")
    actual=future.result().state.pose.position
    residual=math.dist(pose[:3],[actual.x,actual.y,actual.z])
    if residual>.001:raise RuntimeError("Camera did not reach requested pose")
    target=out/(camera["id"]+".png");write_png(latest["message"],target)
    row={"camera":camera["id"],"reference_second":camera["reference_second"],"file":target.name,
         "width":latest["message"].width,"height":latest["message"].height,"pose_residual_m":residual,
         "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"source":"Gazebo ROS camera image",
         "camera_registration":registration_status(camera),"requested_pose":pose,"build_fingerprint":fingerprint,
         "scene_sha256":hashlib.sha256((ROOT/"scenes/wheelset_workshop_video_v1.json").read_bytes()).hexdigest(),
         "image_stamp":{"sec":latest["message"].header.stamp.sec,"nanosec":latest["message"].header.stamp.nanosec}}
    row['image_transport']='RELIABLE_DEPTH_1_NEW_READER_PER_VIEW'
    report.append(row)
    report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(row),flush=True)
report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
node.destroy_node();rclpy.shutdown()
