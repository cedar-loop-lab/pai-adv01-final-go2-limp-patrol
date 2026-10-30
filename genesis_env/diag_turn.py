"""開発用の診断: 不調のある脚で「減速」「停止」「その場旋回」「前進しながらの旋回」ができるかを切り分ける。

PRE_S 秒間 pre 指令で歩かせ、PRE_S 秒の時点で不調を入れると同時に post 指令へ切り替え、POST_S 秒続ける。
追従器を使った帰還の診断は diag_return.py。
この結果は原因の切り分けと追従器の調整に使う開発用の結果で、最終結果には使わない。
"""

import argparse
import json
import math

import torch

import genesis as gs

from configs import REFERENCE_TORQUE_NM
from eval_common import LEG_NAMES, is_fallen, load_env_and_policy, reset_with_noise

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--leg", type=str, default="FR", choices=LEG_NAMES)
    parser.add_argument("--severities", type=str, nargs="+", default=["none", "0.45", "0.28"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    severities = [None if s == "none" else float(s) for s in args.severities]
    result = run_commands(args.exp_name, LEG_NAMES.index(args.leg), severities, args.trials, args.seed)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    for row in result["results"]:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
