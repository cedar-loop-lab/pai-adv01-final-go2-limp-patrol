"""評価B（運用）: 巡回中に1脚が劣化したとき、帰還の通知の有無と時刻だけを変えて、任務の結果を比べる。

任務は通常どおり「コースを1周して基地へ戻り、停止して保持する」（評価A の3周は使わない）。
劣化はコースに沿った距離が試行表の onset_m に達した時点で入れる。歩行方策・追従器・帰還速度・成功判定は全条件で同じ。
通知の条件:
  none   : 通知しない。1周を終えれば自然に帰還する
  oracle : 劣化を入れた時点で帰還を通知する（正解通知。固定した帰還ルールの参照条件で、最適な行動の上限ではない）
成功 = 劣化（起点）から DEADLINE_S 秒以内に帰還完了（半径 r 以内かつ低速を連続 hold_s 秒）し、それまで転倒しないこと。
自然に1周を終えて帰還した場合も成功に数える。分母は試行表の全試行（通知前の転倒・未通知・誤報も含める）。
失敗は、転倒（局面つき）・停止保持の失敗（基地に着いたが完了せず）・巡回中・帰還途中・起点前の帰還（誤報）に分ける。
"""

import argparse
import json
import math
from collections import Counter

import torch

import genesis as gs

from eval_common import (
    LEG_NAMES, REPO_ROOT, degraded_limit, is_fallen, load_env_and_policy, motion_label, reset_from_table,
)
from patrol import HOLD, PATROL, RETURN, MissionConfig, PatrolMission

DEADLINE_S = 40.0
MAX_TIME_S = 80.0  # 起点は最大 12 m（約 26 s）なので、起点 + 40 s に余裕を持たせた打ち切り
MODE_NAMES = {PATROL: "patrol", RETURN: "return", HOLD: "hold"}
EXP_NAMES = {"proposed": "go2-degr-h10-s{}", "nominal": "go2-nominal-h10-s{}"}
CKPT = 999


def simulate(env, policy, trials: list[dict], notify: str) -> dict:
    n = len(trials)
    mission = PatrolMission(MissionConfig(), n, env.dt)
    cfg = mission.cfg
    limit = torch.stack([degraded_limit(env, t["s"], LEG_NAMES.index(t["leg"]) if t["leg"] else 0) for t in trials])
    onset_m = torch.tensor([t["onset_m"] for t in trials], device=gs.device)

    def full(value: float) -> torch.Tensor:
        return torch.full((n,), value, device=gs.device)

    onset_t, notify_t, arrive_t, done_t, fall_t, deadline_t = (full(math.nan) for _ in range(6))
    fall_mode = torch.full((n,), -1, dtype=torch.int64, device=gs.device)
    mode_at_deadline = torch.full((n,), -1, dtype=torch.int64, device=gs.device)
    fall_cmd = torch.zeros(n, 3, device=gs.device)
    obs = reset_from_table(env, trials)
    with torch.no_grad():
        for i_step in range(int(MAX_TIME_S / env.dt)):
            t = i_step * env.dt
            pos_xy = env.base_pos[:, :2]
            is_new = onset_t.isnan() & (mission.mode == PATROL) & (mission.distance_along(pos_xy) >= onset_m)
            if bool(is_new.any()):
                env.set_torque_limit(limit, is_new)
                onset_t = torch.where(is_new, t, onset_t)
                if notify == "oracle":
                    is_notify = is_new & (mission.mode == PATROL)
                    mission.request_return(is_notify)
                    notify_t = torch.where(is_notify, t, notify_t)
            is_alive = fall_t.isnan()
            is_open = is_alive & done_t.isnan()
            arrive_t = torch.where(is_open & arrive_t.isnan() & (mission.mode != PATROL)
                                   & (pos_xy.norm(dim=1) < cfg.home_radius), t, arrive_t)
            is_deadline = deadline_t.isnan() & ~onset_t.isnan() & ((t - onset_t) > DEADLINE_S)
            deadline_t = torch.where(is_deadline, t, deadline_t)
            mode_at_deadline = torch.where(is_deadline, mission.mode, mode_at_deadline)

            was_done = mission.is_done.clone()
            yaw = torch.deg2rad(env.base_euler[:, 2])
            cmd = mission.step(pos_xy, yaw, env.base_lin_vel[:, :2].norm(dim=1))
            env.commands.copy_(cmd)
            done_t = torch.where(is_alive & mission.is_done & ~was_done, t, done_t)
            obs, _, _, _ = env.step(policy(obs))

            is_new_fall = is_alive & done_t.isnan() & is_fallen(env)
            fall_t = torch.where(is_new_fall, t, fall_t)
            fall_mode = torch.where(is_new_fall, mission.mode, fall_mode)
            fall_cmd = torch.where(is_new_fall[:, None], cmd, fall_cmd)
            is_over = ~fall_t.isnan() | ~done_t.isnan() | ~deadline_t.isnan()
            if bool(is_over.all()):
                break
    return {"onset_t": onset_t, "notify_t": notify_t, "arrive_t": arrive_t, "done_t": done_t, "fall_t": fall_t,
            "fall_mode": fall_mode, "fall_cmd": fall_cmd, "mode_at_deadline": mode_at_deadline,
            "num_rearm": mission.num_rearm}


def classify(rec: dict, i: int) -> str:
    """1試行の結果。成功以外は失敗の理由で分ける。"""
    def at(key: str) -> float:
        return rec[key][i].item()

    onset, done, fall, arrive = at("onset_t"), at("done_t"), at("fall_t"), at("arrive_t")
    if math.isnan(onset):
        if not math.isnan(fall):
            return "fell_before_onset"
        return "returned_before_onset"  # 誤報で起点に着く前に帰還した（または起点に達しなかった）
    if not math.isnan(done) and done - onset <= DEADLINE_S:
        return "success"
    if not math.isnan(fall) and fall - onset <= DEADLINE_S:
        return f"fell_{MODE_NAMES[int(rec['fall_mode'][i])]}"
    if not math.isnan(arrive) and arrive - onset <= DEADLINE_S:
        return "timeout_hold"  # 基地に着いたが停止保持を完了できなかった
    mode = int(rec["mode_at_deadline"][i])
    return "timeout_patrolling" if mode == PATROL else "timeout_returning"


def per_trial_records(rec: dict, trials: list[dict]) -> list[dict]:
    records = []
    for i, trial in enumerate(trials):
        onset = rec["onset_t"][i].item()

        def since_onset(key: str) -> float | None:
            value = rec[key][i].item() - onset
            return None if math.isnan(value) else round(value, 2)

        outcome = classify(rec, i)
        notified = not math.isnan(rec["notify_t"][i].item())
        records.append({
            "id": trial["id"], "s": trial["s"], "leg": trial["leg"], "outcome": outcome,
            "notified": notified, "natural_return": outcome == "success" and not notified,
            "notify_after_onset_s": since_onset("notify_t"), "arrive_after_onset_s": since_onset("arrive_t"),
            "done_after_onset_s": since_onset("done_t"), "fall_after_onset_s": since_onset("fall_t"),
            "fall_motion": motion_label(rec["fall_cmd"][i]) if outcome.startswith("fell") else None,
            "num_rearm": int(rec["num_rearm"][i]),
        })
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["s"], r["leg"]), []).append(r)
        if r["s"] is not None:
            groups.setdefault((r["s"], "all"), []).append(r)
    rows = []
    for (s, leg), rs in groups.items():
        outcomes = Counter(r["outcome"] for r in rs)
        rows.append({
            "s": s, "leg": leg, "trials": len(rs),
            "success_rate": outcomes["success"] / len(rs),
            "fall_rate": sum(v for k, v in outcomes.items() if k.startswith("fell")) / len(rs),
            "natural_return": sum(r["natural_return"] for r in rs),
            "outcomes": dict(outcomes),
            "fall_motions": dict(Counter(f"{r['outcome']}/{r['fall_motion']}" for r in rs if r["fall_motion"])),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notify", choices=["none", "oracle"], required=True)
    parser.add_argument("--policy", choices=EXP_NAMES.keys(), default="proposed")
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--table", type=str, required=True)
    args = parser.parse_args()

    table = json.loads((REPO_ROOT / "genesis_env" / "trial_tables" / f"{args.table}.json").read_text())
    trials = table["trials"]
    exp = EXP_NAMES[args.policy].format(args.train_seed)
    env, policy, _ = load_env_and_policy(exp, CKPT, len(trials), MAX_TIME_S, seed=table["seed"])
    rec = simulate(env, policy, trials, args.notify)
    records = per_trial_records(rec, trials)
    rows = summarize(records)

    out_dir = REPO_ROOT / "logs" / "eval_b" / args.table
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.policy}-{args.notify}"
    out = out_dir / f"{tag}-s{args.train_seed}.json"
    meta = {"policy": args.policy, "notify": args.notify, "exp": exp, "ckpt": CKPT, "train_seed": args.train_seed,
            "table": args.table, "table_seed": table["seed"], "deadline_s": DEADLINE_S,
            "mission": MissionConfig().__dict__}
    out.write_text(json.dumps({**meta, "summary": rows, "trials": records}, indent=1))
    for row in rows:
        if row["leg"] in (None, "all"):
            print(f"{tag} s{args.train_seed} s={row['s']}: success {row['success_rate']:.3f} fall {row['fall_rate']:.3f} "
                  f"natural {row['natural_return']} {row['outcomes']}")
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
