"""評価A の結果（logs/eval_a/<table>/*.json）を、方式 × 倍率 s で、学習シード別・脚別に並べる。

--select-slow を付けると、検証用の表で減速率 k を決める（8章の規則: 転倒率が最小の k、
最小値から 0.02 以内が複数なら進んだ距離の平均が大きいほう）。
"""

import argparse
import json
from collections import defaultdict

from eval_common import LEG_NAMES, REPO_ROOT

SELECT_MARGIN = 0.02


def load(table: str) -> dict[str, dict[int, list[dict]]]:
    """方式のタグ -> 学習シード -> 試行の記録。"""
    runs: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for path in sorted((REPO_ROOT / "logs" / "eval_a" / table).glob("*.json")):
        data = json.loads(path.read_text())
        tag = path.stem.rsplit("-s", 1)[0]
        runs[tag][data["train_seed"]] = data["trials"]
    return runs


def stats(trials: list[dict]) -> tuple[float, float, int]:
    n = len(trials)
    fall = sum(t["outcome"] == "fell" for t in trials) / n
    progress = sum(t["progress_m"] for t in trials) / n
    return fall, progress, n


def print_table(runs: dict, severities: list) -> None:
    for tag, by_seed in runs.items():
        print(f"== {tag}")
        for s in severities:
            pooled = [t for trials in by_seed.values() for t in trials if t["s"] == s]
            fall, progress, n = stats(pooled)
            per_seed = " ".join(f"s{seed}:{stats([t for t in tr if t['s'] == s])[0]:.2f}" for seed, tr in sorted(by_seed.items()))
            per_leg = " ".join(
                f"{leg}:{stats([t for t in pooled if t['leg'] == leg])[0]:.2f}" for leg in LEG_NAMES if s is not None
            )
            invalid = sum(t["outcome"] in ("fell_before_onset", "no_onset") or t["left_patrol"] for t in pooled)
            print(f"  s={s}: n={n} fall {fall:.3f} progress {progress:5.1f} m | seed {per_seed} | leg {per_leg}"
                  + (f" | INVALID {invalid}" if invalid else ""))
        worst = [
            (stats([t for t in tr if t["s"] == s and t["leg"] == leg])[0], seed, s, leg)
            for seed, tr in by_seed.items() for s in severities if s is not None for leg in LEG_NAMES
        ]
        fall, seed, s, leg = max(worst)
        print(f"  worst cell: seed {seed}, s={s}, {leg}: fall {fall:.2f}")


def select_slow(runs: dict) -> None:
    candidates = []
    for tag, by_seed in runs.items():
        if not tag.startswith("nominal_slow"):
            continue
        pooled = [t for trials in by_seed.values() for t in trials if t["s"] is not None]
        fall, progress, _ = stats(pooled)
        candidates.append((float(tag.removeprefix("nominal_slow")), fall, progress))
        print(f"k={candidates[-1][0]}: fall {fall:.3f}, progress {progress:.1f} m (seeds {sorted(by_seed)})")
    best_fall = min(c[1] for c in candidates)
    near = [c for c in candidates if c[1] <= best_fall + SELECT_MARGIN]
    chosen = max(near, key=lambda c: c[2])
    print(f"selected k = {chosen[0]} (min fall {best_fall:.3f}, candidates within {SELECT_MARGIN}: {[c[0] for c in near]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", choices=["val", "final"])
    parser.add_argument("--select-slow", action="store_true")
    args = parser.parse_args()
    runs = load(args.table)
    table = json.loads((REPO_ROOT / "genesis_env" / "trial_tables" / f"{args.table}.json").read_text())
    print_table(runs, [None] + table["severities"])
    if args.select_slow:
        select_slow(runs)


if __name__ == "__main__":
    main()
