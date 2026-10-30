#!/usr/bin/env bash
# 結合試験: 提案版の 3 シードで、val の試行（正常・s=0.6・s=0.45、4 脚）を ROS2 経由で 20 回走らせる。
# 同じ試行の評価B（検知器あり、バッチ実行）の結果と比べるのは summarize_runs.py。
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNS=(
  "1 0" "1 1" "1 50" "1 100" "1 150" "1 200" "1 250" "1 300" "1 350" "1 400"
  "2 0" "2 150" "2 300" "2 350" "2 400"
  "3 1" "3 100" "3 250" "3 300" "3 400"
)
for run in "${RUNS[@]}"; do
  read -r seed trial <<<"$run"
  "$ROOT/ros2/run_trial.sh" "go2-degr-h10-s$seed" val "$trial" "batch/val$trial-s$seed"
done
