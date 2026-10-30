#!/usr/bin/env bash
# Genesis（sim_server.py）と ROS2 のノード 3 つ・rosbag2 を起動し、試行表の 1 試行を実時間で走らせる。
# 使い方: ros2/run_trial.sh <exp_name> <table> <trial_id> <run_name> [video]
#   例: ros2/run_trial.sh go2-degr-h10-s1 val 300 val300-s1
# 出力: logs/ros2/<run_name>/（result.json・rosbag2・各ノードのログ）。video を付けると video/ros2_<run_name>.mp4 も録る。
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXP=$1 TABLE=$2 TRIAL=$3 NAME=$4 VIDEO=${5:-}
OUT="logs/ros2/$NAME"
TOPICS=(/odom /leg_tracking_error /sim/fallen /sim/ground_truth /cmd_vel /mission/mode /mission/status
        /health/fault /health/status)
rm -rf "${ROOT:?}/$OUT"
mkdir -p "$ROOT/$OUT"

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
set +u
# shellcheck disable=SC1091
source /opt/ros/lyrical/setup.bash
set -u
ROSPY="$ROOT/.venv-ros/bin/python"

pids=()
# 非対話のシェルから & で起動したプロセスは SIGINT を無視するので、SIGTERM で止める（rosbag2 は SIGTERM でも bag を閉じる）
cleanup() { kill -TERM "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT
cd "$ROOT/ros2"
"$ROSPY" genesis_bridge.py >"$ROOT/$OUT/bridge.log" 2>&1 & pids+=($!)
"$ROSPY" health_monitor.py --ros-args -p exp_name:="$EXP" >"$ROOT/$OUT/health_monitor.log" 2>&1 & pids+=($!)
"$ROSPY" mission_node.py >"$ROOT/$OUT/mission_node.log" 2>&1 & pids+=($!)
ros2 bag record -o "$ROOT/$OUT/bag" --topics "${TOPICS[@]}" >"$ROOT/$OUT/rosbag.log" 2>&1 & pids+=($!)
sleep 3

# Genesis は別の Python（3.12）で動かすので、ROS2 が足した環境変数を外す
cd "$ROOT/genesis_env"
video_args=()
if [[ -n "$VIDEO" ]]; then video_args=(--video "video/ros2_$NAME.mp4" --label "Genesis + ROS 2（$EXP / $TABLE #$TRIAL）"); fi
env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH PYOPENGL_PLATFORM=egl \
  "$ROOT/.venv/bin/python" sim_server.py -e "$EXP" --table "$TABLE" --trial-id "$TRIAL" --out "$OUT/result.json" \
  "${video_args[@]}" >"$ROOT/$OUT/sim.log" 2>&1
tail -1 "$ROOT/$OUT/sim.log"
