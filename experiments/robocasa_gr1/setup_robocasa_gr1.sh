#!/usr/bin/env bash
# Sets up the RoboCasa-GR1 tabletop evaluation (starVLA's Robocasa_tabletop protocol) in third_party/:
#   robosuite v1.5.1, robocasa-gr1-tabletop-tasks and starVLA at pinned revisions, and a separate Python 3.10
#   environment for the simulator (third_party/robocasa_env) with the task assets.
# The policy server runs in the LoopedWAM environment (pip install -e ".[robocasa_gr1]").
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TP=$ROOT/third_party
PY=${PYTHON:-python3.10}
mkdir -p "$TP"

clone() {  # url directory revision
  [ -d "$TP/$2/.git" ] || git clone "$1" "$TP/$2"
  git -C "$TP/$2" fetch --tags origin
  git -C "$TP/$2" checkout "$3"
}
clone https://github.com/ARISE-Initiative/robosuite.git robosuite v1.5.1
clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git robocasa-gr1-tabletop-tasks 4840e671596f93ca03651524b9f72ffb1aadfeff
clone https://github.com/starVLA/starVLA.git starVLA 13d8ad43eb4128554c57227507055ecf9c3a62a2

E=$TP/robocasa_env
[ -x "$E/bin/python" ] || "$PY" -m venv "$E"
"$E/bin/pip" install -U pip "setuptools<81" wheel
"$E/bin/pip" install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
"$E/bin/pip" install -e "$TP/robosuite"
"$E/bin/pip" install mink==0.0.5 robosuite_models
"$E/bin/pip" install -e "$TP/robocasa-gr1-tabletop-tasks"
# robocasa pins numpy 1.26.4; opencv 4.10 is the last headless wheel built for numpy 1.
"$E/bin/pip" uninstall -y opencv-python opencv-python-headless || true
"$E/bin/pip" install numpy==1.26.4 mujoco==3.2.6 gymnasium==1.2.2 websockets==16.0 msgpack av matplotlib mediapy \
    opencv-python-headless==4.10.0.84 pydantic tyro typing_extensions websocket-client pyarrow==17.0.0
(cd "$TP/robocasa-gr1-tabletop-tasks" && "$E/bin/python" robocasa/scripts/download_tabletop_assets.py -y)
echo "RoboCasa-GR1 simulator ready: $E/bin/python"
