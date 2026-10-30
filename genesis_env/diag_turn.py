"""開発用の診断: 劣化した脚で「減速」「その場旋回」「前進しながらの旋回」「追従器での帰還」ができるかを切り分ける。

指令テスト（停止 stop も含む）: PRE_S 秒間 pre 指令で歩かせ、PRE_S 秒の時点で劣化を入れると同時に post 指令へ切り替え、POST_S 秒続ける。
追従器テスト: 巡回 0.5 m/s で進み具合 ONSET_PROGRESS の時点で劣化と帰還の指示を入れる（劣化なしの条件も同じ時点で指示）。
この結果は原因の切り分けと追従器の調整に使う開発用の結果で、最終結果には使わない。
"""

import argparse
import json
import math

import torch

import genesis as gs

from configs import REFERENCE_TORQUE_NM
from eval_common import LEG_NAMES, is_fallen, load_env_and_policy, reset_with_noise
from patrol import MissionConfig, PatrolMission

CASES = {
    "keep_1.0": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "decel_1.0to0.5": ((1.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
    "keep_0.5": ((0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
    "stop": ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "spin_L": ((0.5, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "spin_R": ((0.5, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "arc_L": ((0.5, 0.0, 0.0), (0.5, 0.0, 1.0)),
    "arc_R": ((0.5, 0.0, 0.0), (0.5, 0.0, -1.0)),
}
PRE_S = 3.0
POST_S = 10.0
SETTLE_S = 1.0  # 切り替え直後の過渡を、速度の平均から外す
ONSET_PROGRESS = 0.4
TRACKER_TIME_S = 40.0
FACING_DEG = 45.0  # 追従器が前進を始める向きのずれ（MissionConfig.turn_in_place_deg と同じ）


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(angle.sin(), angle.cos())


def degraded_limit(env, leg: int, s: float | None) -> torch.Tensor:
    limit = env.nominal_torque_limit.clone()
    if s is not None:
        ref = torch.tensor(REFERENCE_TORQUE_NM, device=gs.device)
        cols = slice(leg * 3, leg * 3 + 3)
        limit[cols] = torch.minimum(s * ref[cols], env.nominal_torque_limit[cols])
    return limit


def run_commands(exp: str, leg: int, severities: list[float | None], trials: int, seed: int) -> dict:
    conditions = [(s, case) for s in severities for case in CASES]
    num_envs = len(conditions) * trials
    env, policy, _ = load_env_and_policy(exp, 999, num_envs, PRE_S + POST_S, seed)
    pre = torch.tensor([CASES[c][0] for _, c in conditions], device=gs.device).repeat_interleave(trials, 0)
    post = torch.tensor([CASES[c][1] for _, c in conditions], device=gs.device).repeat_interleave(trials, 0)
    limit = torch.stack([degraded_limit(env, leg, s) for s, _ in conditions]).repeat_interleave(trials, 0)

    obs = reset_with_noise(env, pre)
    onset_step, settle_step = int(PRE_S / env.dt), int((PRE_S + SETTLE_S) / env.dt)
    has_fallen = torch.zeros(num_envs, dtype=torch.bool, device=gs.device)
    vx_sum, wz_sum, n_avg = (torch.zeros(num_envs, device=gs.device) for _ in range(3))
    yaw_turned = torch.zeros(num_envs, device=gs.device)
    path = torch.zeros(num_envs, device=gs.device)
    prev_yaw = torch.deg2rad(env.base_euler[:, 2])
    prev_xy = env.base_pos[:, :2].clone()
    with torch.no_grad():
        for i_step in range(int((PRE_S + POST_S) / env.dt)):
            if i_step == onset_step:
                env.set_torque_limit(limit)
                env.commands.copy_(post)
            obs, _, _, _ = env.step(policy(obs))
            has_fallen |= is_fallen(env)
            alive = ~has_fallen
            yaw = torch.deg2rad(env.base_euler[:, 2])
            xy = env.base_pos[:, :2]
            if i_step >= onset_step:
                yaw_turned += torch.where(alive, wrap(yaw - prev_yaw), 0.0)
                path += torch.where(alive, (xy - prev_xy).norm(dim=1), 0.0)
            if i_step >= settle_step:
                vx_sum += torch.where(alive, env.base_lin_vel[:, 0], 0.0)
                wz_sum += torch.where(alive, env.base_ang_vel[:, 2], 0.0)
                n_avg += alive.float()
            prev_yaw, prev_xy = yaw, xy.clone()

    results = []
    for i_cond, (s, case) in enumerate(conditions):
        sl = slice(i_cond * trials, (i_cond + 1) * trials)
        n = n_avg[sl].clamp(min=1)
        results.append({
            "s": s, "case": case, "post_cmd": CASES[case][1],
            "fall_rate": has_fallen[sl].float().mean().item(),
            "vx_mean": (vx_sum[sl] / n).mean().item(),
            "yaw_rate_mean": (wz_sum[sl] / n).mean().item(),
            "yaw_turned_deg": math.degrees(yaw_turned[sl].mean().item()),
            "path_m": path[sl].mean().item(),
        })
    return {"exp": exp, "leg": LEG_NAMES[leg], "pre_s": PRE_S, "post_s": POST_S, "trials": trials, "results": results}


def run_tracker(exp: str, leg: int, severities: list[float | None], trials: int, seed: int) -> dict:
    num_envs = len(severities) * trials
    env, policy, _ = load_env_and_policy(exp, 999, num_envs, 120.0, seed)
    mission = PatrolMission(MissionConfig(), num_envs, env.dt)
    limit = torch.stack([degraded_limit(env, leg, s) for s in severities]).repeat_interleave(trials, 0)

    obs = reset_with_noise(env, torch.zeros(num_envs, 3, device=gs.device))
    is_triggered = torch.zeros(num_envs, dtype=torch.bool, device=gs.device)
    onset_t, facing_t, done_t = (torch.full((num_envs,), math.nan, device=gs.device) for _ in range(3))
    has_fallen = torch.zeros(num_envs, dtype=torch.bool, device=gs.device)
    yaw_turned = torch.zeros(num_envs, device=gs.device)
    trace = []  # 各条件の最初の試行の時系列（0.5 s ごと）
    prev_yaw = torch.deg2rad(env.base_euler[:, 2])
    with torch.no_grad():
        for i_step in range(int(120.0 / env.dt)):
            t = i_step * env.dt
            pos_xy = env.base_pos[:, :2]
            yaw = torch.deg2rad(env.base_euler[:, 2])
            is_new = ~is_triggered & (mission.progress(pos_xy) >= ONSET_PROGRESS)
            if bool(is_new.any()):
                env.set_torque_limit(limit, is_new)
                mission.request_return(is_new)
                onset_t = torch.where(is_new, t, onset_t)
                is_triggered |= is_new
            to_home = -pos_xy
            heading_err = wrap(torch.atan2(to_home[:, 1], to_home[:, 0]) - yaw)
            is_facing = is_triggered & (heading_err.abs() < math.radians(FACING_DEG)) & facing_t.isnan()
            facing_t = torch.where(is_facing, t, facing_t)
            was_done = mission.is_done.clone()
            cmd = mission.step(pos_xy, yaw, env.base_lin_vel[:, :2].norm(dim=1))
            env.commands.copy_(cmd)
            done_t = torch.where(mission.is_done & ~was_done, t, done_t)
            obs, _, _, _ = env.step(policy(obs))
            has_fallen |= is_fallen(env)
            new_yaw = torch.deg2rad(env.base_euler[:, 2])
            yaw_turned += torch.where(is_triggered & ~has_fallen, wrap(new_yaw - yaw), 0.0)
            if i_step % 25 == 0:
                rows = [i * trials for i in range(len(severities))]
                trace.append({
                    "t": round(t, 2),
                    "per_cond": [{
                        "heading_err_deg": round(math.degrees(heading_err[r].item()), 1),
                        "vx_cmd": round(cmd[r, 0].item(), 2), "yaw_cmd": round(cmd[r, 2].item(), 2),
                        "vx": round(env.base_lin_vel[r, 0].item(), 2), "yaw_rate": round(env.base_ang_vel[r, 2].item(), 2),
                    } for r in rows],
                })
            is_past = is_triggered & ((t - onset_t) > TRACKER_TIME_S)
            if bool((is_past | has_fallen | mission.is_done).all()):
                break

    results = []
    for i, s in enumerate(severities):
        sl = slice(i * trials, (i + 1) * trials)
        to_face, to_done = facing_t[sl] - onset_t[sl], done_t[sl] - onset_t[sl]
        results.append({
            "s": s,
            "fall_rate": has_fallen[sl].float().mean().item(),
            "faced_home_rate": (~to_face.isnan()).float().mean().item(),
            "time_to_face_mean_s": to_face.nanmean().item(),
            "done_within_T_rate": (to_done <= TRACKER_TIME_S).float().mean().item(),
            "yaw_turned_after_onset_deg": math.degrees(yaw_turned[sl].mean().item()),
        })
    return {"exp": exp, "leg": LEG_NAMES[leg], "trials": trials, "results": results, "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--mode", choices=["commands", "tracker"], required=True)
    parser.add_argument("--leg", type=str, default="FR", choices=LEG_NAMES)
    parser.add_argument("--severities", type=str, nargs="+", default=["none", "0.45", "0.28"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    severities = [None if s == "none" else float(s) for s in args.severities]
    run = run_commands if args.mode == "commands" else run_tracker
    result = run(args.exp_name, LEG_NAMES.index(args.leg), severities, args.trials, args.seed)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    for row in result["results"]:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
