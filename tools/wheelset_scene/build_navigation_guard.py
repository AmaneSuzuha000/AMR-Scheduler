"""Build an owned, source-snapshotted guard/Nav2 overlay without other workspaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .build import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if os.name == 'nt':
        raise SystemExit('Run on Ubuntu after sourcing ROS2 Humble.')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out/'src/amr_gazebo_guard'
    shutil.copytree(ROOT/'tools/ros2/amr_gazebo_guard', source,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='1', MAKEFLAGS='-j1')
    commands = [['colcon', '--log-base', str(out/'logs'), 'build', '--base-paths', str(source),
        '--build-base', str(out/'build'), '--install-base', str(out/'install'),
        '--executor', 'sequential', '--cmake-args', '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON'],
        ['ctest', '--test-dir', str(out/'build/amr_gazebo_guard'), '--output-on-failure', '-R', 'execution_twist_bounds']]
    with (out/'build.log').open('x') as log:
        for command in commands:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    prefix = out/'install/amr_gazebo_guard'
    libraries = {name: hashlib.sha256((prefix/'lib'/name).read_bytes()).hexdigest()
        for name in ['libamr_motion_guard.so', 'libamr_bounded_pursuit.so', 'libamr_continuous_goal.so']}
    manifest = dict(passed=True, commands=commands, install_prefix=str(prefix),
        source_sha256={p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()}, library_sha256=libraries,
        compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
        claim_boundary='Native guard and Nav2 plugin compilation plus existing twist-bound unit test. '
            'No workshop vehicle motion or localization runtime acceptance.')
    (out/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
