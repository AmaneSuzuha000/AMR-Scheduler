#!/usr/bin/env bash
# Build only inside the extracted tree; run only on an explicitly selected port/domain.
set -eo pipefail
wheelset_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$wheelset_root"
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash tools/wheelset_scene/reproduce_workshop.sh prepare out/NEW_BUILD'
  echo '       bash tools/wheelset_scene/reproduce_workshop.sh run out/BUILD delivery|faults out/NEW_RUN PORT DOMAIN'
  exit 2
fi
wheelset_action="$1"
wheelset_build="$(realpath -m -- "$2")"
[[ "$wheelset_build" == "$wheelset_root/out/"* && "$wheelset_build" != "$wheelset_root/out" ]] || { echo 'Build must be a directory under this extracted tree/out.'; exit 2; }
[[ "$(dpkg --print-architecture)" == amd64 ]] || { echo 'Ubuntu amd64 required.'; exit 2; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || { echo 'Ubuntu 22.04 required.'; exit 2; }
# A prior workspace in the caller must not supply hidden packages or binaries.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
unset ROS_PACKAGE_PATH GAZEBO_PLUGIN_PATH GAZEBO_MODEL_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
python3 -m tools.wheelset_scene.portable_workshop verify
if [[ "$wheelset_action" == prepare && $# == 2 ]]; then
  [[ ! -e "$wheelset_build" ]] || { echo 'Existing build preserved; select a new directory.'; exit 2; }
  mkdir -p -- "$wheelset_build"
  python3 -m tools.wheelset_scene.portable_workshop dependencies --output "$wheelset_build/dependencies"
  source "$wheelset_build/dependencies/setup.bash"
  CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 colcon --log-base "$wheelset_build/scout/log" build \
    --base-paths "$wheelset_root/tools/ros2/wheelset_dependencies/scout_gazebo_description" \
    --build-base "$wheelset_build/scout/build" --install-base "$wheelset_build/scout/install" \
    --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release
  source "$wheelset_build/scout/install/local_setup.bash"
  python3 -m tools.wheelset_scene.build_axis_servo --output "$wheelset_build/axis"
  python3 -m tools.wheelset_scene.build_navigation_guard --output "$wheelset_build/guard"
  source "$wheelset_build/guard/install/local_setup.bash"
  python3 -m tools.wheelset_scene.workshop_navigation --output "$wheelset_build/normal_bundle" \
    --whole-line-axis-build "$wheelset_build/axis" --without-review-camera --physics-island-threads 4
  python3 -m tools.wheelset_scene.workshop_navigation --output "$wheelset_build/fault_bundle" \
    --whole-line-axis-build "$wheelset_build/axis" --fault-relay --without-review-camera --physics-island-threads 4
  python3 -c 'import json,sys; from pathlib import Path; from tools.wheelset_scene.portable_workshop import sha256; p=Path(sys.argv[1]); (p/"prepared.json").write_text(json.dumps(dict(prepared=True, source_manifest_sha256=sha256("source_bundle_manifest.json"),runtime_acceptance=False),indent=2)+"\n")' "$wheelset_build"
elif [[ "$wheelset_action" == run && $# == 6 ]]; then
  [[ -f "$wheelset_build/prepared.json" ]] || { echo 'Complete prepare first.'; exit 2; }
  python3 -c 'import json,sys; from pathlib import Path; from tools.wheelset_scene.portable_workshop import sha256; p=Path(sys.argv[1]); r=json.loads((p/"prepared.json").read_text()); assert r["prepared"] and r["source_manifest_sha256"]==sha256("source_bundle_manifest.json")' "$wheelset_build"
  source "$wheelset_build/dependencies/setup.bash"
  source "$wheelset_build/scout/install/local_setup.bash"
  source "$wheelset_build/guard/install/local_setup.bash"
  case "$3" in
    delivery) wheelset_bundle="$wheelset_build/normal_bundle" ;;
    faults) wheelset_bundle="$wheelset_build/fault_bundle" ;;
    *) echo 'Use delivery or faults.'; exit 2 ;;
  esac
  python3 -u -m tools.wheelset_scene.run_navigation_readiness --bundle "$wheelset_bundle" \
    --guard-build "$wheelset_build/guard" --whole-line --mode "$3" --output "$4" --port "$5" --domain "$6"
else
  echo 'Invalid action or arguments.'; exit 2
fi
