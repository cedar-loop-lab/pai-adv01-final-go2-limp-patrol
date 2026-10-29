"""評価スクリプト共通: 学習済み方策と評価用の環境を読み込む、転倒を判定する。"""

import pickle
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from envs.go2_env import Go2Env

REPO_ROOT = Path(__file__).resolve().parent.parent
LEG_NAMES = ["FR", "FL", "RR", "RL"]
# 転倒の判定（学習時のリセット条件 10° より緩い、実際に倒れた状態）
FALL_TILT_DEG = 45.0
FALL_BASE_Z = 0.15
INIT_JOINT_NOISE_RAD = 0.1


def load_env_and_policy(exp_name: str, ckpt: int, num_envs: int, duration_s: float, seed: int):
    """評価用の環境と推論用の方策を返す。自動リセットと指令の再抽選は止める（転倒は is_fallen で判定）。"""
    log_dir = REPO_ROOT / "logs" / exp_name
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["episode_length_s"] = duration_s + 1.0
    env_cfg["resampling_time_s"] = 1e6

    torch.manual_seed(seed)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=seed)
    env = Go2Env(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.load(str(log_dir / f"model_{ckpt}.pt"))
    return env, runner.get_inference_policy(device=gs.device), log_dir


def reset_with_noise(env: Go2Env, commands: torch.Tensor):
    """全 env をリセットし、関節の初期角度に小さな乱れを入れて試行ごとに差を出す。"""
    obs = env.reset()
    noise = (torch.rand(env.num_envs, 12, device=gs.device) * 2 - 1) * INIT_JOINT_NOISE_RAD
    env.robot.set_dofs_position(env.default_dof_pos + noise, env.motors_dof_idx)
    env.commands.copy_(commands)
    return obs


def is_fallen(env: Go2Env) -> torch.Tensor:
    tilt = env.base_euler[:, :2].abs().max(dim=1).values
    return (tilt > FALL_TILT_DEG) | (env.base_pos[:, 2] < FALL_BASE_Z)
