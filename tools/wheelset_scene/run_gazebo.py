"""Start this scene on its own Gazebo master and ROS domain; no existing process is stopped."""
import argparse
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import time
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.wheelset_scene.evidence import build_fingerprint
parser=argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=11359)
parser.add_argument("--domain",type=int,default=73)
parser.add_argument("--gui",action="store_true")
parser.add_argument("--camera",default="overview")
parser.add_argument("--stop",action="store_true",help="Stop only the processes recorded for this scene.")
args=parser.parse_args()
if os.name=="nt":raise SystemExit("Run on the Ubuntu Gazebo host.")
runtime=ROOT/"simulation/wheelset_workshop/runtime";runtime.mkdir(parents=True,exist_ok=True)
if args.stop:
    path=runtime/"session.json"
    if not path.exists():raise SystemExit("No scene session recorded.")
    record=json.loads(path.read_text())
    stopped=[];identities={}
    for key,exe in [("gui_pid","gzclient"),("server_pid","gzserver")]:
        pid=record.get(key)
        if not pid or not Path(f"/proc/{pid}").exists():continue
        command=Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        environment=Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if Path(command[0].decode()).name!=exe:raise SystemExit("PID identity mismatch; leaving it alone.")
        if ("GAZEBO_MASTER_URI="+record["master_uri"]).encode() not in environment:
            raise SystemExit("Process master URI mismatch; leaving it alone.")
        if exe=="gzserver" and str(ROOT).encode() not in b" ".join(command):
            raise SystemExit("World path mismatch; leaving it alone.")
        identities[pid]=Path(f"/proc/{pid}/stat").read_text().split()[21]
        os.kill(pid,signal.SIGINT);stopped.append(pid)
    deadline=time.monotonic()+8
    while time.monotonic()<deadline and any(Path(f"/proc/{pid}").exists() for pid in stopped):
        time.sleep(.2)
    forced=[]
    for pid in stopped:
        stat=Path(f"/proc/{pid}/stat")
        if stat.exists():
            fields=stat.read_text().split()
            if fields[21]==identities[pid] and fields[2]!="Z":
                os.kill(pid,signal.SIGKILL);forced.append(pid)
    print(json.dumps({"stopped_owned_pids":stopped,"forced_owned_pids":forced}));raise SystemExit(0)
sock=socket.socket();busy=sock.connect_ex(("127.0.0.1",args.port))==0;sock.close()
if busy:raise SystemExit(f"Port {args.port} already in use; leaving that instance unchanged.")
loaded=subprocess.run(["bash","-c","source /opt/ros/humble/setup.bash && env -0"],capture_output=True,check=True)
env=dict(os.environ)
env.update(dict(item.decode().split("=",1) for item in loaded.stdout.split(b"\0") if b"=" in item))
env.update({"DISPLAY":":0","ROS_DOMAIN_ID":str(args.domain),"GAZEBO_MASTER_URI":f"http://127.0.0.1:{args.port}",
            "GAZEBO_MODEL_PATH":str(ROOT/"assets/wheelset_workshop/models"),"GAZEBO_MODEL_DATABASE_URI":"",
            "LIBGL_ALWAYS_SOFTWARE":"1","OMP_NUM_THREADS":"4"})
world=ROOT/"simulation/wheelset_workshop/worlds"/("camera_"+args.camera+".world")
if not world.exists():raise SystemExit(f"Unknown camera world: {world}")
command=["gzserver","--verbose","-s","libgazebo_ros_init.so","-s","libgazebo_ros_factory.so",str(world)]
log=(runtime/"gzserver.log").open("w")
server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(1)
if server.poll() is not None:raise SystemExit("Gazebo server exited; inspect runtime/gzserver.log")
record={"server_pid":server.pid,"master_uri":env["GAZEBO_MASTER_URI"],"ros_domain_id":args.domain,"build_fingerprint":build_fingerprint(),
        "world":str(world),"root":str(ROOT),"started_unix":time.time(),"gui_pid":None}
if args.gui:
    gui_log=(runtime/"gzclient.log").open("w")
    client=subprocess.Popen(["gzclient","--verbose"],env=env,stdout=gui_log,stderr=subprocess.STDOUT,start_new_session=True)
    record["gui_pid"]=client.pid
(runtime/"session.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record))
