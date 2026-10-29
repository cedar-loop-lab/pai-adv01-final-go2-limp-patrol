# Derived from Genesis v1.4.5 examples/locomotion/go2_train.py (Apache-2.0, Copyright Genesis-Embodied-AI).
# Changes: configs split out of the training script, command-range variants for patrol, observation history,
# torque-limit degradation during training.

# 指令範囲。official は公式サンプルのまま（前進 0.5 m/s 固定、旋回なし）。
# patrol は巡回・旋回・停止・帰還に必要な範囲（前進 0〜1.0 m/s、旋回 ±1.0 rad/s、1割は停止）。
COMMAND_VARIANTS = {
    "official": {
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "zero_command_prob": 0.0,
    },
    "patrol": {
        "lin_vel_x_range": [0.0, 1.0],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [-1.0, 1.0],
        "zero_command_prob": 0.1,
    },
}


# 劣化の強さ s の共通基準 [Nm]（joint_names の順）。go2-patrol-s1 の正常時の関節別 p95 トルク
# （logs/go2-patrol-s1/envelope.json、2026-10-30）を固定値として写したもの。全方式・全シードで同じ値を使う。
REFERENCE_TORQUE_NM = (
    8.775196075439453, 13.417834281921387, 15.17480754852295,  # FR hip / thigh / calf
    8.11659049987793, 15.104951858520508, 13.749022483825684,  # FL
    7.912249565124512, 8.110267639160156, 11.3601713180542,  # RR
    8.739553451538086, 7.935779571533203, 11.480952262878418,  # RL
)

# 学習時の劣化（8章で確定）: エピソードごとに確率 prob でランダムな1脚の3関節の上限を
# min(s × 共通基準, 元の上限) に下げる。s ~ U[s_range]、発生はエピソード中の一様な時刻。
TRAIN_DEGRADATION = {
    "prob": 0.5,
    "s_range": [0.35, 1.0],
    "reference_torque": list(REFERENCE_TORQUE_NM),
}

# 観測の履歴長（直近のステップ数）。通常版・提案版で同じ値を使う。
HISTORY_LEN = 10


def get_train_cfg(exp_name: str) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs(
    command_variant: str, history_len: int = 1, has_degradation: bool = False
) -> tuple[dict, dict, dict, dict]:
    if command_variant not in COMMAND_VARIANTS:
        raise ValueError(f"unknown command variant: {command_variant} (choose from {sorted(COMMAND_VARIANTS)})")

    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "degradation": dict(TRAIN_DEGRADATION) if has_degradation else None,
    }
    obs_cfg = {
        "history_len": history_len,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {"num_commands": 3, **COMMAND_VARIANTS[command_variant]}

    return env_cfg, obs_cfg, reward_cfg, command_cfg
