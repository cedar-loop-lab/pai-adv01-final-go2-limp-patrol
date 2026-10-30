"""評価スクリプト共通: 学習済み方策と評価用の環境を読み込む、転倒を判定する。"""

import pickle
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from configs import REFERENCE_TORQUE_NM
from envs.go2_env import Go2Env

REPO_ROOT = Path(__file__).resolve().parent.parent
# 公開した重み（export_policies.py で actor だけを書き出したもの）。ここに無ければ学習の出力 logs/ を使う
POLICY_ROOT = REPO_ROOT / "policies"
LEG_NAMES = ["FR", "FL", "RR", "RL"]
# 転倒の判定（学習時のリセット条件 10° より緩い、実際に倒れた状態）
FALL_TILT_DEG = 45.0
FALL_BASE_Z = 0.15
INIT_JOINT_NOISE_RAD = 0.1
TURN_CMD_MIN = 0.05  # これより小さい旋回指令は直進とみなす [rad/s]


def load_env_and_policy(
    exp_name: str, ckpt: int, num_envs: int, duration_s: float, seed: int, camera_res: tuple[int, int] | None = None
):
    """評価用の環境と推論用の方策を返す。自動リセットと指令の再抽選は止める（転倒は is_fallen で判定）。"""
    log_dir = REPO_ROOT / "logs" / exp_name
    src_dir = POLICY_ROOT / exp_name if (POLICY_ROOT / exp_name / f"model_{ckpt}.pt").exists() else log_dir
    with open(src_dir / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["episode_length_s"] = duration_s + 1.0
    env_cfg["resampling_time_s"] = 1e6
    # 不調は評価スクリプトが set_torque_limit で決めた条件だけを入れる（学習時のランダムな不調は止める）
    env_cfg["degradation"] = None

    torch.manual_seed(seed)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=seed)
    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        camera_res=camera_res,
    )
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    # 推論には actor だけを使う（公開した重みには optimizer・critic の状態が無い）
    runner.load(str(src_dir / f"model_{ckpt}.pt"), load_cfg={"actor": True})
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


def motion_label(cmd: torch.Tensor) -> str:
    """転倒した瞬間の指令を、直進・停止・旋回（左右、その場か前進しながらか）に分類する。"""
    vx, yaw_rate = cmd[0].item(), cmd[2].item()
    if abs(yaw_rate) < TURN_CMD_MIN:
        return "straight" if vx > 0 else "stop"
    side = "L" if yaw_rate > 0 else "R"
    return f"spin_{side}" if vx == 0 else f"arc_{side}"


def degraded_limit(env: Go2Env, s: float | None, leg: int) -> torch.Tensor:
    """1脚（3関節）の上限を min(s × 共通基準, 元の上限) にした上限ベクトル。s が None なら元の上限。"""
    limit = env.nominal_torque_limit.clone()
    if s is not None:
        ref = torch.tensor(REFERENCE_TORQUE_NM, device=gs.device)
        cols = slice(leg * 3, leg * 3 + 3)
        limit[cols] = torch.minimum(s * ref[cols], env.nominal_torque_limit[cols])
    return limit


def reset_from_table(env: Go2Env, trials: list[dict]):
    """全 env をリセットし、試行表の関節角度の乱れを入れる（最初の制御ステップの前に1回）。"""
    obs = env.reset()
    noise = torch.tensor([t["joint_noise_rad"] for t in trials], device=gs.device)
    env.robot.set_dofs_position(env.default_dof_pos + noise, env.motors_dof_idx)
    env.commands.zero_()
    return obs
