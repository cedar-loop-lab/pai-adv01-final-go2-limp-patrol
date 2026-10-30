"""開発用の診断: 巡回中に1脚が不調になり、追従器で基地へ帰還するまでを、どこで失敗したか分かる形で記録する。

各試行は巡回 0.5 m/s で始め、進み具合が試行ごとの値（--onset-range から一様に抽選）に達した時点で
不調と帰還の指示を同時に入れる（検知器は使わない「正解通知」。不調なしの条件も同じ時点で指示する）。
期限は不調（指示）から T 秒で、再接近してもリセットしない。成功判定は MissionConfig のとおり。
記録: 基地（半径 r）への初回到達時刻、保持から帰還へ戻った回数、転倒した局面、最終結果（完了・転倒・時間切れ）。
この結果は追従器の調整と原因の切り分けに使う開発用の結果で、最終結果には使わない。
"""

import argparse
import json
import math
from collections import Counter

import torch

import genesis as gs

from eval_common import LEG_NAMES, degraded_limit, is_fallen, load_env_and_policy, motion_label, reset_with_noise
from patrol import HOLD, PATROL, RETURN, MissionConfig, PatrolMission

DEADLINE_S = 40.0
MAX_TIME_S = 100.0  # 不調が起きる前の巡回（最大でコースの 75%、約 25 s）＋ 期限に余裕を持たせた打ち切り
MODE_NAMES = {PATROL: "patrol", RETURN: "return", HOLD: "hold"}


def degraded_limits(env, conditions: list[tuple[float | None, int]], trials: int) -> torch.Tensor:
    return torch.stack([degraded_limit(env, s, leg) for s, leg in conditions]).repeat_interleave(trials, 0)


def simulate(exp: str, conditions: list, trials: int, onset_range: tuple[float, float], seed: int) -> dict:
    num_envs = len(conditions) * trials
    env, policy, _ = load_env_and_policy(exp, 999, num_envs, MAX_TIME_S, seed)
    mission = PatrolMission(MissionConfig(), num_envs, env.dt)
    cfg = mission.cfg
    limit = degraded_limits(env, conditions, trials)
    onset_at = torch.empty(num_envs, device=gs.device).uniform_(*onset_range)

    def nan_tensor() -> torch.Tensor:
        return torch.full((num_envs,), math.nan, device=gs.device)

    onset_t, arrive_t, done_t, fall_t = nan_tensor(), nan_tensor(), nan_tensor(), nan_tensor()
    fall_mode = torch.full((num_envs,), -1, dtype=torch.int64, device=gs.device)
    fall_cmd = torch.zeros(num_envs, 3, device=gs.device)
    is_triggered = torch.zeros(num_envs, dtype=torch.bool, device=gs.device)
    obs = reset_with_noise(env, torch.zeros(num_envs, 3, device=gs.device))
    with torch.no_grad():
        for i_step in range(int(MAX_TIME_S / env.dt)):
            t = i_step * env.dt
            pos_xy = env.base_pos[:, :2]
            is_new = ~is_triggered & (mission.progress(pos_xy) >= onset_at)
            if bool(is_new.any()):
                env.set_torque_limit(limit, is_new)
                mission.request_return(is_new)
                onset_t = torch.where(is_new, t, onset_t)
                is_triggered |= is_new
            is_active = is_triggered & fall_t.isnan() & done_t.isnan() & ((t - onset_t) <= DEADLINE_S)
            is_arrived = is_active & arrive_t.isnan() & (pos_xy.norm(dim=1) < cfg.home_radius)
            arrive_t = torch.where(is_arrived, t, arrive_t)

            was_done = mission.is_done.clone()
            yaw = torch.deg2rad(env.base_euler[:, 2])
            cmd = mission.step(pos_xy, yaw, env.base_lin_vel[:, :2].norm(dim=1))
            env.commands.copy_(cmd)
            done_t = torch.where(is_active & mission.is_done & ~was_done, t, done_t)
            obs, _, _, _ = env.step(policy(obs))

            is_new_fall = fall_t.isnan() & done_t.isnan() & is_fallen(env)
            fall_t = torch.where(is_new_fall, t, fall_t)
            fall_mode = torch.where(is_new_fall, mission.mode, fall_mode)
            fall_cmd = torch.where(is_new_fall[:, None], cmd, fall_cmd)
            is_over = ~fall_t.isnan() | ~done_t.isnan() | (is_triggered & ((t - onset_t) > DEADLINE_S))
            if bool(is_over.all()):
                break
    return {
        "onset_t": onset_t, "arrive_t": arrive_t, "done_t": done_t, "fall_t": fall_t,
        "fall_mode": fall_mode, "fall_cmd": fall_cmd, "num_rearm": mission.num_rearm.clone(),
    }


def summarize(rec: dict, conditions: list, trials: int) -> list[dict]:
    results = []
    for i_cond, (s, leg) in enumerate(conditions):
        sl = slice(i_cond * trials, (i_cond + 1) * trials)
        onset = rec["onset_t"][sl]
        to_done = rec["done_t"][sl] - onset
        to_arrive = rec["arrive_t"][sl] - onset
        is_fall_before = ~rec["fall_t"][sl].isnan() & ((rec["fall_t"][sl] - onset) <= DEADLINE_S)
        is_success = to_done <= DEADLINE_S
        is_fall = is_fall_before & ~is_success
        is_timeout = ~is_success & ~is_fall
        phases = Counter(
            f"{MODE_NAMES.get(int(m), 'before_onset')}/{motion_label(c)}"
            for m, c, f in zip(rec["fall_mode"][sl], rec["fall_cmd"][sl], is_fall)
            if bool(f)
        )
        results.append({
            "s": s, "leg": LEG_NAMES[leg], "trials": trials,
            "triggered_rate": (~onset.isnan()).float().mean().item(),
            "success_rate": is_success.float().mean().item(),
            "reached_rate": (to_arrive <= DEADLINE_S).float().mean().item(),
            "fall_rate": is_fall.float().mean().item(),
            "timeout_rate": is_timeout.float().mean().item(),
            "rearm_mean": rec["num_rearm"][sl].float().mean().item(),
            "rearm_max": int(rec["num_rearm"][sl].max().item()),
            "time_to_arrive_s": {"mean": to_arrive.nanmean().item(), "max": to_arrive.nan_to_num(-1).max().item()},
            "time_to_done_s": {"mean": to_done.nanmean().item(), "max": to_done.nan_to_num(-1).max().item()},
            "fall_phases": dict(phases),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--legs", type=str, nargs="+", default=LEG_NAMES, choices=LEG_NAMES)
    parser.add_argument("--severities", type=str, nargs="+", default=["none", "0.6", "0.45"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--onset-range", type=float, nargs=2, default=[0.25, 0.75])
    parser.add_argument("--seed", type=int, default=1300)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    severities = [None if s == "none" else float(s) for s in args.severities]
    legs = [LEG_NAMES.index(leg) for leg in args.legs]
    # 不調なしは脚によらないので1条件だけにする
    conditions = [(s, leg) for s in severities for leg in (legs if s is not None else legs[:1])]
    rec = simulate(args.exp_name, conditions, args.trials, tuple(args.onset_range), args.seed)
    results = summarize(rec, conditions, args.trials)
    with open(args.out, "w") as f:
        json.dump({"exp": args.exp_name, "deadline_s": DEADLINE_S, "onset_range": args.onset_range,
                   "mission": MissionConfig().__dict__, "results": results}, f, indent=1)
    for row in results:
        print(
            f"s={row['s']} {row['leg']}: success {row['success_rate']:.2f} reached {row['reached_rate']:.2f} "
            f"fall {row['fall_rate']:.2f} timeout {row['timeout_rate']:.2f} rearm {row['rearm_mean']:.1f}"
            f"(max {row['rearm_max']}) arrive {row['time_to_arrive_s']['mean']:.1f}s "
            f"done {row['time_to_done_s']['mean']:.1f}s falls {row['fall_phases']}"
        )


if __name__ == "__main__":
    main()
