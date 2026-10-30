"""評価の試行表を事前に作る。全方式・全学習シードが同じ表を使い、同じ試行どうしを比べる。

1試行 = 不調の強さ s（正常条件は None）・脚・不調の発生地点（コースに沿った距離）・リセット時の関節角度の乱れ。
発生地点は1周 16 m のコースの 25〜75%（4〜12 m）から一様に選ぶ。関節角度の乱れは既定の姿勢に
U[-0.1, 0.1] rad を関節ごとに足し、リセット直後（最初の制御ステップの前）に1回だけ入れる。
初期の位置と向きは固定（基地、+x 向き）。
val は開発・検証用（減速率と検知の閾値を決める）、final は評価A の最終テスト用、final_b は評価B の最終テスト用で、乱数を分ける。
"""

import argparse
import json

import numpy as np

from eval_common import LEG_NAMES, REPO_ROOT

TABLES = {
    "val": {"seed": 1001, "severities": [0.6, 0.45], "per_leg": 50, "normal": 50},
    "final": {"seed": 2001, "severities": [0.5, 0.4, 0.3, 0.25], "per_leg": 100, "normal": 100},
    # 評価B の最終用。評価A の最終結果を見たあとで検知器を作るので、A とは別の未使用の表にする（強さと分布は同じ）
    "final_b": {"seed": 3001, "severities": [0.5, 0.4, 0.3, 0.25], "per_leg": 100, "normal": 100},
}
ONSET_RANGE_M = (4.0, 12.0)
JOINT_NOISE_RAD = 0.1
NUM_JOINTS = 12
TABLE_DIR = REPO_ROOT / "genesis_env" / "trial_tables"


def make_table(kind: str) -> dict:
    spec = TABLES[kind]
    rng = np.random.default_rng(spec["seed"])
    conditions = [(None, None)] * spec["normal"]
    conditions += [(s, leg) for s in spec["severities"] for leg in LEG_NAMES for _ in range(spec["per_leg"])]
    onsets = rng.uniform(*ONSET_RANGE_M, size=len(conditions))
    noises = rng.uniform(-JOINT_NOISE_RAD, JOINT_NOISE_RAD, size=(len(conditions), NUM_JOINTS))
    trials = [
        {"id": i, "s": s, "leg": leg, "onset_m": round(float(onset), 4),
         "joint_noise_rad": [round(float(x), 5) for x in noise]}
        for i, ((s, leg), onset, noise) in enumerate(zip(conditions, onsets, noises))
    ]
    return {
        "kind": kind, **spec, "onset_range_m": ONSET_RANGE_M, "joint_noise_rad": JOINT_NOISE_RAD,
        "initial_pose": "base (0, 0), heading +x", "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=TABLES.keys())
    args = parser.parse_args()
    TABLE_DIR.mkdir(exist_ok=True)
    out = TABLE_DIR / f"{args.kind}.json"
    table = make_table(args.kind)
    out.write_text(json.dumps(table, separators=(",", ":")))
    print(f"{len(table['trials'])} trials -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
