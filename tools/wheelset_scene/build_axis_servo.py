"""Build the isolated native effort plugin and bind binary to source hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .build import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'out/wheelset_axis_servo_build')
    args=parser.parse_args()
    if os.name=='nt':raise SystemExit('Build on Ubuntu after sourcing ROS2 Humble.')
    source=ROOT/'tools/ros2/wheelset_axis_servo';build=args.output.resolve()
    build.mkdir(parents=True,exist_ok=True)
    commands=[['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'],
        ['cmake','--build',str(build),'-j2']]
    with (build/'build.log').open('w') as log:
        for command in commands:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    library=build/'libwheelset_axis_servo.so'
    result=dict(passed=True,commands=commands,compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
        source_sha256={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()},
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        auxiliary_libraries={'libwheelset_fleet_truth.so':hashlib.sha256((build/'libwheelset_fleet_truth.so').read_bytes()).hexdigest()})
    (build/'build_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
