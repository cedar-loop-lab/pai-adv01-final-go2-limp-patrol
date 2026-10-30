"""結合試験（logs/ros2/batch/*/result.json）の結果を、同じ試行の評価B（検知器あり、バッチ実行）と並べる。

結果の分類は評価B と同じ（成功 = 劣化から 40 s 以内に帰還完了し、それまで転倒しない）。正常条件は歩き始めからの完了で見る。
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEADLINE_S = 40.0


def outcome(r: dict) -> str:
    onset = r["onset_t"] if r["onset_t"] is not None else math.nan
    if r["fall_t"] is not None and (math.isnan(onset) or r["fall_t"] - onset <= DEADLINE_S):
        return "fell"
    if r["done_t"] is not None and (math.isnan(onset) or r["done_t"] - onset <= DEADLINE_S):
        return "success"
    return "timeout"


def since(value, onset) -> str:
    return "   -  " if value is None or onset is None else f"{value - onset:6.2f}"


def main() -> None:
    rows, agree = [], 0
    batch_cache: dict[int, dict] = {}
    for path in sorted((ROOT / "logs" / "ros2" / "batch").glob("*/result.json")):
        r = json.loads(path.read_text())
        seed = int(r["exp"].rsplit("-s", 1)[1])
        if seed not in batch_cache:
            data = json.loads((ROOT / "logs" / "eval_b" / "val" / f"proposed-detector-s{seed}.json").read_text())
            batch_cache[seed] = {t["id"]: t for t in data["trials"]}
        b = batch_cache[seed][r["trial_id"]]
        ros = outcome(r)
        same = ros == ("fell" if b["outcome"].startswith("fell") else "success" if b["outcome"] == "success" else "timeout")
        agree += same
        rows.append(f"seed {seed} #{r['trial_id']:<3} s={r['trial_s']!s:<4} {r['trial_leg']!s:<4} | ROS2: {ros:<7} "
                    f"notify {since(r['notify_t'], r['onset_t'])} done {since(r['done_t'], r['onset_t'])} "
                    f"overruns {r['overruns']:>3} | batch: {b['outcome']:<12} notify {b['notify_after_onset_s']} "
                    f"done {b['done_after_onset_s']}")
    print("\n".join(rows))
    print(f"outcome agrees with the batch evaluation: {agree}/{len(rows)}")


if __name__ == "__main__":
    main()
