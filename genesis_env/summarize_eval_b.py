"""評価B の結果（logs/eval_b/<table>/proposed-<通知>-s<seed>.json）を、通知の条件 × 強さで、学習シード別に並べる。

成功率（主指標）、転倒率、失敗の内訳、通知した割合（検知率）と誤報、劣化から帰還完了までの時間（成功した試行の中央値）。
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict

from eval_common import REPO_ROOT

CONDITIONS = ("none", "detector", "oracle")


def load(table: str) -> dict[str, dict[int, list[dict]]]:
    runs: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for path in sorted((REPO_ROOT / "logs" / "eval_b" / table).glob("proposed-*.json")):
        data = json.loads(path.read_text())
        runs[data["notify"]][data["train_seed"]] = data["trials"]
    return runs


def line(trials: list[dict]) -> str:
    n = len(trials)
    outcomes = Counter(t["outcome"] for t in trials)
    success = outcomes["success"] / n
    fall = sum(v for k, v in outcomes.items() if k.startswith("fell")) / n
    notified = sum(t["notified"] for t in trials) / n
    false_alarm = sum(t["false_alarm"] for t in trials)
    done = [t["done_after_onset_s"] for t in trials if t["outcome"] == "success"]
    done_median = statistics.median(done) if done else float("nan")
    failures = {k: v for k, v in outcomes.items() if k != "success"}
    return (f"success {success:.3f} fall {fall:.3f} notified {notified:.2f} false alarm {false_alarm} "
            f"done median {done_median:5.1f} s | {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", choices=["val", "final_b"])
    args = parser.parse_args()
    runs = load(args.table)
    table = json.loads((REPO_ROOT / "genesis_env" / "trial_tables" / f"{args.table}.json").read_text())
    for s in [None] + table["severities"]:
        print(f"== s={s}")
        for cond in CONDITIONS:
            if cond not in runs:
                continue
            by_seed = runs[cond]
            pooled = [t for trials in by_seed.values() for t in trials if t["s"] == s]
            print(f"  {cond:<8} all : {line(pooled)}")
            for seed, trials in sorted(by_seed.items()):
                print(f"  {cond:<8} s{seed}  : {line([t for t in trials if t['s'] == s])}")


if __name__ == "__main__":
    main()
