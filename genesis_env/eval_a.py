"""評価A（歩行）: 巡回中に1脚が不調になっても、帰還させずに巡回を続けたとき、転ばずにどれだけ進めるか。

試行表（make_trial_table.py）の各試行を、全方式で同じ条件で走らせる。
- コースは1周 16 m の正方形を3周つなげたもの（観測時間の間に帰還区間へ入らない長さ）。基本速度 0.5 m/s。
- 不調はコースに沿った距離が試行表の onset_m に達した時点で入れる（正常条件も同じ時点を起点にする）。
- 方式: nominal（通常版）/ nominal_slow（通常版。不調と同時に前進・旋回の指令を --slow-factor 倍にする。
  不調発生の時点は正解として与える。正常条件では減速しない）/ proposed（提案版。速度は変えない）。
- 観測時間: 不調（起点）から OBSERVE_S 秒。転倒した時点で打ち切る。
指標: 転倒率、不調が起きた後に進めた距離（コースに沿った距離）、転倒した局面。正常条件では1周目の所要時間と速度追従も。
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
from patrol import PATROL, MissionConfig, PatrolMission, looped_waypoints

OBSERVE_S = 40.0
LAPS = 3
LAP_SEGMENTS = 4  # 1周の区間数（基地 → 3点 → 基地）
MAX_TIME_S = 80.0  # 起点は最大 12 m（約 26 s）なので、起点 + 40 s に余裕を持たせた打ち切り
EXP_NAMES = {"nominal": "go2-nominal-h10-s{}", "nominal_slow": "go2-nominal-h10-s{}", "proposed": "go2-degr-h10-s{}"}
CKPT = 999


def simulate(env, policy, trials: list[dict], slow_factor: float | None) -> dict:
    n = len(trials)
    mission = PatrolMission(MissionConfig(waypoints=looped_waypoints(LAPS)), n, env.dt)
    limit = torch.stack([degraded_limit(env, t["s"], LEG_NAMES.index(t["leg"]) if t["leg"] else 0) for t in trials])
    onset_m = torch.tensor([t["onset_m"] for t in trials], device=gs.device)
    # 減速は不調を入れた試行だけ（正常条件には故障の通知が無いので減速しない）
    has_fault = torch.tensor([t["s"] is not None for t in trials], device=gs.device)

    def full(value: float) -> torch.Tensor:
        return torch.full((n,), value, device=gs.device)

    onset_t, fall_t, lap_t = full(math.nan), full(math.nan), full(math.nan)
    dist_at_onset, dist_end = full(math.nan), full(math.nan)
    fall_cmd = torch.zeros(n, 3, device=gs.device)
    left_patrol = torch.zeros(n, dtype=torch.bool, device=gs.device)
    track_err, track_steps = full(0.0), full(0.0)
    obs = reset_from_table(env, trials)
    with torch.no_grad():
        for i_step in range(int(MAX_TIME_S / env.dt)):
            t = i_step * env.dt
            pos_xy = env.base_pos[:, :2]
            dist = mission.distance_along(pos_xy)
            is_new = onset_t.isnan() & (dist >= onset_m)
            if bool(is_new.any()):
                env.set_torque_limit(limit, is_new)
                if slow_factor is not None:
                    mission.set_speed_scale(is_new & has_fault, slow_factor)
                onset_t = torch.where(is_new, t, onset_t)
                dist_at_onset = torch.where(is_new, dist, dist_at_onset)
            is_alive = fall_t.isnan()
            is_observing = ~onset_t.isnan() & ((t - onset_t) <= OBSERVE_S) & is_alive
            dist_end = torch.where(is_observing, dist, dist_end)
            left_patrol |= is_observing & (mission.mode != PATROL)

            yaw = torch.deg2rad(env.base_euler[:, 2])
            cmd = mission.step(pos_xy, yaw, env.base_lin_vel[:, :2].norm(dim=1))
            lap_t = torch.where(lap_t.isnan() & (mission.i_seg >= LAP_SEGMENTS), t, lap_t)
            env.commands.copy_(cmd)
            # 速度追従: 直進の指令が出ているあいだの |実速度 - 指令|（転倒前のみ）
            is_straight = (cmd[:, 0] > 0) & (cmd[:, 2].abs() < 0.05) & is_alive
            track_err += torch.where(is_straight, (env.base_lin_vel[:, 0] - cmd[:, 0]).abs(), 0.0)
            track_steps += is_straight.float()
            obs, _, _, _ = env.step(policy(obs))

            is_new_fall = is_alive & is_fallen(env)
            fall_t = torch.where(is_new_fall, t, fall_t)
            fall_cmd = torch.where(is_new_fall[:, None], cmd, fall_cmd)
            is_over = ~fall_t.isnan() | (~onset_t.isnan() & ((t - onset_t) > OBSERVE_S))
            if bool(is_over.all()):
                break
    return {
        "onset_t": onset_t, "fall_t": fall_t, "lap_t": lap_t, "fall_cmd": fall_cmd,
        "progress_m": dist_end - dist_at_onset, "left_patrol": left_patrol,
        "track_err": track_err / track_steps.clamp(min=1),
    }


def per_trial_records(rec: dict, trials: list[dict]) -> list[dict]:
    records = []
    for i, trial in enumerate(trials):
        onset, fall = rec["onset_t"][i].item(), rec["fall_t"][i].item()
        if math.isnan(onset):
            outcome = "fell_before_onset" if not math.isnan(fall) else "no_onset"
        elif not math.isnan(fall) and fall - onset <= OBSERVE_S:
            outcome = "fell"
        else:
            outcome = "walked"
        records.append({
            "id": trial["id"], "s": trial["s"], "leg": trial["leg"], "outcome": outcome,
            "fall_after_onset_s": None if outcome != "fell" else round(fall - onset, 2),
            "fall_motion": motion_label(rec["fall_cmd"][i]) if outcome == "fell" else None,
            "progress_m": round(rec["progress_m"][i].item(), 3),
            "lap_time_s": round(rec["lap_t"][i].item(), 2),
            "track_err_mps": round(rec["track_err"][i].item(), 4),
            "left_patrol": bool(rec["left_patrol"][i]),
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
        progress = torch.tensor([r["progress_m"] for r in rs])
        rows.append({
            "s": s, "leg": leg, "trials": len(rs),
            "fall_rate": sum(r["outcome"] == "fell" for r in rs) / len(rs),
            "invalid": sum(r["outcome"] in ("fell_before_onset", "no_onset") or r["left_patrol"] for r in rs),
            "progress_mean_m": progress.mean().item(),
            "progress_p10_m": torch.quantile(progress, 0.1).item(),
            "fall_motions": dict(Counter(r["fall_motion"] for r in rs if r["fall_motion"])),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=EXP_NAMES.keys(), required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--table", choices=["val", "final"], required=True)
    parser.add_argument("--slow-factor", type=float, default=None, help="Required for nominal_slow")
    args = parser.parse_args()
    if (args.method == "nominal_slow") != (args.slow_factor is not None):
        parser.error("--slow-factor is required for nominal_slow and only for it")

    table = json.loads((REPO_ROOT / "genesis_env" / "trial_tables" / f"{args.table}.json").read_text())
    trials = table["trials"]
    exp = EXP_NAMES[args.method].format(args.train_seed)
    env, policy, _ = load_env_and_policy(exp, CKPT, len(trials), MAX_TIME_S, seed=table["seed"])
    rec = simulate(env, policy, trials, args.slow_factor)
    records = per_trial_records(rec, trials)
    rows = summarize(records)

    tag = args.method if args.slow_factor is None else f"{args.method}{args.slow_factor:g}"
    out_dir = REPO_ROOT / "logs" / "eval_a" / args.table
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}-s{args.train_seed}.json"
    meta = {"method": args.method, "slow_factor": args.slow_factor, "exp": exp, "ckpt": CKPT,
            "train_seed": args.train_seed, "table": args.table, "table_seed": table["seed"],
            "observe_s": OBSERVE_S, "laps": LAPS, "mission": MissionConfig().__dict__}
    out.write_text(json.dumps({**meta, "summary": rows, "trials": records}, indent=1))
    for row in rows:
        if row["leg"] in (None, "all"):
            print(f"{tag} s{args.train_seed} s={row['s']}: fall {row['fall_rate']:.3f} progress "
                  f"{row['progress_mean_m']:.1f} m (p10 {row['progress_p10_m']:.1f}) invalid {row['invalid']} "
                  f"falls {row['fall_motions']}")
    normal = [r for r in records if r["s"] is None]
    print(f"normal: lap {sum(r['lap_time_s'] for r in normal) / len(normal):.2f} s, "
          f"track err {sum(r['track_err_mps'] for r in normal) / len(normal):.3f} m/s -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
