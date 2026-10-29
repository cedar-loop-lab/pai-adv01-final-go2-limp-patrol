"""確認①の続き: 正常時（劣化なし）に巡回・停止・帰還が通しでできるかを調べる。

半分の試行はコースを1周して基地に戻る。残り半分は、進み具合 25〜75% のどこかで帰還を指示し、
そこから基地へまっすぐ戻る（本番で劣化が起きる区間と同じ）。帰還の所要時間は T を決める材料にする。
"""

import argparse
import json
import math

import torch

import genesis as gs

from eval_common import is_fallen, load_env_and_policy, reset_with_noise
from patrol import MissionConfig, PatrolMission

TRIGGER_RANGE = (0.25, 0.75)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=300)
    args = parser.parse_args()

    env, policy, log_dir = load_env_and_policy(args.exp_name, args.ckpt, args.trials, args.max_time, args.seed)
    mission = PatrolMission(MissionConfig(), args.trials, env.dt)

    has_trigger = torch.arange(args.trials, device=gs.device) % 2 == 1
    trigger_at = torch.empty(args.trials, device=gs.device).uniform_(*TRIGGER_RANGE)
    is_triggered = torch.zeros(args.trials, dtype=torch.bool, device=gs.device)
    trigger_time = torch.full((args.trials,), float("nan"), device=gs.device)
    done_time = torch.full((args.trials,), float("nan"), device=gs.device)
    has_fallen = torch.zeros(args.trials, dtype=torch.bool, device=gs.device)

    obs = reset_with_noise(env, torch.zeros(args.trials, 3, device=gs.device))
    num_steps = int(args.max_time / env.dt)
    with torch.no_grad():
        for i_step in range(num_steps):
            t = i_step * env.dt
            pos_xy = env.base_pos[:, :2]
            yaw = torch.deg2rad(env.base_euler[:, 2])
            speed = env.base_lin_vel[:, :2].norm(dim=1)

            is_new = has_trigger & ~is_triggered & (mission.progress(pos_xy) >= trigger_at)
            mission.request_return(is_new)
            trigger_time = torch.where(is_new, t, trigger_time)
            is_triggered |= is_new

            was_done = mission.is_done.clone()
            env.commands.copy_(mission.step(pos_xy, yaw, speed))
            done_time = torch.where(mission.is_done & ~was_done, t, done_time)
            obs, _, _, _ = env.step(policy(obs))
            has_fallen |= is_fallen(env)
            if bool((mission.is_done | has_fallen).all()):
                break

    final_dist = env.base_pos[:, :2].norm(dim=1)
    is_success = mission.is_done & ~has_fallen
    lap = ~has_trigger
    ret = has_trigger
    lap_time = done_time[lap & is_success]
    return_time = (done_time - trigger_time)[ret & is_success]

    def stats(x: torch.Tensor) -> dict:
        if x.numel() == 0:
            return {"n": 0}
        return {
            "n": x.numel(),
            "mean": x.mean().item(),
            "p95": torch.quantile(x, 0.95).item(),
            "max": x.max().item(),
        }

    result = {
        "exp": args.exp_name,
        "ckpt": args.ckpt,
        "mission": MissionConfig().__dict__,
        "course_length_m": mission.course_length,
        "lap": {
            "trials": int(lap.sum()),
            "success_rate": is_success[lap].float().mean().item(),
            "fall_rate": has_fallen[lap].float().mean().item(),
            "time_s": stats(lap_time),
        },
        "return": {
            "trials": int(ret.sum()),
            "success_rate": is_success[ret].float().mean().item(),
            "fall_rate": has_fallen[ret].float().mean().item(),
            "time_from_trigger_s": stats(return_time),
        },
        "final_dist_success_max_m": final_dist[is_success].max().item() if is_success.any() else math.nan,
    }
    print(json.dumps(result, indent=2))
    (log_dir / "patrol_check.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
