# Derived from Genesis v1.4.5 examples/locomotion/go2_train.py (Apache-2.0, Copyright Genesis-Embodied-AI).
# Changes: --variant for the command range, explicit torch seed, logs under the repository root,
# --history-len and --degradation for the nominal / degradation-trained comparison.
import argparse
import os
import pickle
import shutil
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from configs import HISTORY_LEN, get_cfgs, get_train_cfg
from envs.go2_env import Go2Env

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-patrol", help="Log directory name under logs/")
    parser.add_argument("-v", "--variant", type=str, default="patrol", help="Command range variant (see configs.py)")
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history-len", type=int, default=HISTORY_LEN, help="Observation history steps")
    parser.add_argument("--degradation", action="store_true", help="Randomly lower one leg's torque limits")
    args = parser.parse_args()

    log_dir = REPO_ROOT / "logs" / args.exp_name
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.variant, args.history_len, args.degradation)
    train_cfg = get_train_cfg(args.exp_name)

    if log_dir.exists():
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    torch.manual_seed(args.seed)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
