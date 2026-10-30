"""確認①: 正常時（不調なし）の運転範囲を調べる。

前進速度 × 旋回速度の格子ごとに、初期姿勢を少しずつ変えた試行を並列で走らせ、
転倒率・速度追従誤差・関節トルク（不調の強さの共通基準に使う）を集計する。
結果は logs/<exp>/envelope.json に保存する。
"""

import argparse
import json

import torch

import genesis as gs

from eval_common import is_fallen, load_env_and_policy, reset_with_noise

VX_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
YAW_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
# 指令に追従するまでの立ち上がり時間は誤差の集計から外す
SETTLE_S = 1.0
TORQUE_QUANTILE_SAMPLES = 200_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--trials", type=int, default=40, help="Trials per grid cell")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per trial")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    cells = [(vx, yaw) for vx in VX_GRID for yaw in YAW_GRID]
    num_envs = len(cells) * args.trials
    env, policy, log_dir = load_env_and_policy(args.exp_name, args.ckpt, num_envs, args.duration, args.seed)

    commands = torch.tensor([[vx, 0.0, yaw] for vx, yaw in cells], device=gs.device).repeat_interleave(args.trials, 0)
    obs = reset_with_noise(env, commands)

    num_steps = int(args.duration / env.dt)
    settle_steps = int(SETTLE_S / env.dt)
    has_fallen = torch.zeros(num_envs, dtype=torch.bool, device=gs.device)
    vx_err_sum = torch.zeros(num_envs, device=gs.device)
    yaw_err_sum = torch.zeros(num_envs, device=gs.device)
    torques = torch.empty(num_steps, num_envs, 12, device=gs.device)
    with torch.no_grad():
        for i_step in range(num_steps):
            obs, _, _, _ = env.step(policy(obs))
            has_fallen |= is_fallen(env)
            torques[i_step] = env.robot.get_dofs_control_force(env.motors_dof_idx).abs()
            if i_step >= settle_steps:
                vx_err_sum += (env.base_lin_vel[:, 0] - commands[:, 0]).abs()
                yaw_err_sum += (env.base_ang_vel[:, 2] - commands[:, 2]).abs()

    n_err = num_steps - settle_steps
    joint_names = env.env_cfg["joint_names"]
    result = {"exp": args.exp_name, "ckpt": args.ckpt, "trials": args.trials, "duration_s": args.duration, "cells": []}
    print(f"{'vx':>5} {'yaw':>5} | {'fall':>5} | {'|vx err|':>8} {'|yaw err|':>9} (mean of non-fallen)")
    for i_cell, (vx, yaw) in enumerate(cells):
        sl = slice(i_cell * args.trials, (i_cell + 1) * args.trials)
        ok = ~has_fallen[sl]
        fall_rate = has_fallen[sl].float().mean().item()
        vx_err = (vx_err_sum[sl][ok] / n_err).mean().item() if ok.any() else float("nan")
        yaw_err = (yaw_err_sum[sl][ok] / n_err).mean().item() if ok.any() else float("nan")
        result["cells"].append({"vx": vx, "yaw": yaw, "fall_rate": fall_rate, "vx_err": vx_err, "yaw_err": yaw_err})
        print(f"{vx:5.2f} {yaw:5.2f} | {fall_rate:5.2f} | {vx_err:8.3f} {yaw_err:9.3f}")

    # 関節ごとの必要トルク（転倒しなかった試行、立ち上がり後）。不調の強さの共通基準に使う。
    t = torques[settle_steps:, ~has_fallen].reshape(-1, 12)
    sample = t[torch.randperm(t.shape[0], device=gs.device)[:TORQUE_QUANTILE_SAMPLES]]
    p95 = torch.quantile(sample, 0.95, dim=0)
    t_max = t.max(dim=0).values
    result["torque_p95"] = dict(zip(joint_names, p95.tolist()))
    result["torque_max"] = dict(zip(joint_names, t_max.tolist()))
    result["torque_limit_nominal"] = dict(zip(joint_names, env.nominal_torque_limit.tolist()))
    print("joint            p95[Nm]  max[Nm]  limit[Nm]")
    for name, a, b, c in zip(joint_names, p95.tolist(), t_max.tolist(), env.nominal_torque_limit.tolist()):
        print(f"{name:15s} {a:8.2f} {b:8.2f} {c:9.2f}")

    out = log_dir / "envelope.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
