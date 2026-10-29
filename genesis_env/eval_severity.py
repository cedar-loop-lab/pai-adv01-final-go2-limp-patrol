"""予備実験: 片脚のトルク上限を下げたとき、方策がどこから崩れるかを調べる（劣化条件を決めるため）。

劣化の強さ s は「上限 / 共通基準トルク」。共通基準は基準方策の正常時の関節別 p95 トルク
（eval_envelope.py の結果）で、全方式に同じ Nm をかけるために固定する。
劣化は片脚の 3 関節すべてに、歩き始めてから ONSET_S 秒後にかける。
この結果は最終テスト条件を決めるための予備実験で、最終結果には使わない。
"""

import argparse
import json

import torch

import genesis as gs

from eval_common import LEG_NAMES, REPO_ROOT, is_fallen, load_env_and_policy, reset_with_noise

SEVERITIES = [1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15]
COMMANDS = [(0.5, 0.0), (1.0, 0.0), (0.5, 1.0), (0.5, -1.0)]
ONSET_S = 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True, help="Policy under test")
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--ref-exp", type=str, required=True, help="Experiment whose envelope.json is the reference")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=200)
    args = parser.parse_args()

    ref = json.loads((REPO_ROOT / "logs" / args.ref_exp / "envelope.json").read_text())
    conditions = [(s, leg, cmd) for s in SEVERITIES for leg in range(4) for cmd in COMMANDS]
    num_envs = len(conditions) * args.trials
    env, policy, log_dir = load_env_and_policy(args.exp_name, args.ckpt, num_envs, args.duration, args.seed)

    joint_names = env.env_cfg["joint_names"]
    ref_torque = torch.tensor([ref["torque_p95"][name] for name in joint_names], device=gs.device)
    limit = env.nominal_torque_limit.repeat(num_envs, 1)
    for i_cond, (s, leg, _) in enumerate(conditions):
        sl = slice(i_cond * args.trials, (i_cond + 1) * args.trials)
        cols = slice(leg * 3, leg * 3 + 3)
        limit[sl, cols] = torch.minimum(s * ref_torque[cols], env.nominal_torque_limit[cols])
    commands = torch.tensor(
        [[vx, 0.0, yaw] for _, _, (vx, yaw) in conditions], device=gs.device
    ).repeat_interleave(args.trials, 0)

    obs = reset_with_noise(env, commands)
    num_steps = int(args.duration / env.dt)
    onset_step = int(ONSET_S / env.dt)
    has_fallen = torch.zeros(num_envs, dtype=torch.bool, device=gs.device)
    vx_sum = torch.zeros(num_envs, device=gs.device)
    with torch.no_grad():
        for i_step in range(num_steps):
            if i_step == onset_step:
                env.set_torque_limit(limit)
            obs, _, _, _ = env.step(policy(obs))
            has_fallen |= is_fallen(env)
            if i_step >= onset_step:
                vx_sum += torch.where(has_fallen, 0.0, env.base_lin_vel[:, 0])

    vx_ratio = vx_sum / (num_steps - onset_step) / commands[:, 0]
    fall = has_fallen.float().view(len(conditions), args.trials).mean(dim=1)
    ratio = vx_ratio.view(len(conditions), args.trials).mean(dim=1)
    result = {
        "exp": args.exp_name,
        "ckpt": args.ckpt,
        "ref_exp": args.ref_exp,
        "trials": args.trials,
        "onset_s": ONSET_S,
        "duration_s": args.duration,
        "conditions": [
            {"s": s, "leg": LEG_NAMES[leg], "vx": vx, "yaw": yaw, "fall_rate": f, "vx_ratio": r}
            for (s, leg, (vx, yaw)), f, r in zip(conditions, fall.tolist(), ratio.tolist())
        ],
    }

    n_cmd = len(COMMANDS)
    fall_grid = fall.view(len(SEVERITIES), 4, n_cmd)
    ratio_grid = ratio.view(len(SEVERITIES), 4, n_cmd)
    print("fall rate by severity x leg (mean over commands) | vx ratio (mean over legs/commands)")
    print(f"{'s':>5} | " + " ".join(f"{leg:>5}" for leg in LEG_NAMES) + " |  vx_ratio")
    for i_s, s in enumerate(SEVERITIES):
        legs = " ".join(f"{v:5.2f}" for v in fall_grid[i_s].mean(dim=1).tolist())
        print(f"{s:5.2f} | {legs} |  {ratio_grid[i_s].mean().item():6.2f}")
    print("fall rate by severity x command (mean over legs)")
    print(f"{'s':>5} | " + " ".join(f"{f'{vx}/{yaw}':>9}" for vx, yaw in COMMANDS))
    for i_s, s in enumerate(SEVERITIES):
        print(f"{s:5.2f} | " + " ".join(f"{v:9.2f}" for v in fall_grid[i_s].mean(dim=0).tolist()))

    out = log_dir / f"severity_ref-{args.ref_exp}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
