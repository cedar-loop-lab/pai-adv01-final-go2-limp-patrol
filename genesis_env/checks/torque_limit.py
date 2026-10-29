"""確認②: トルク上限（劣化モデル）が実際の制御出力と物理にかかるかを確かめる。

4 つの env を既定の姿勢で PD 保持（行動ゼロ）させ、2 秒進める。
  env0: 正常
  env1: 右前脚の膝下（FR_calf）だけ上限を下げる
  env2: 全関節の上限を下げる
  env3: 上限を下げたあとリセットする（劣化が次のエピソードに残らないか）
確認すること:
  1. 上限を下げた関節で、PD の指令トルクが上限を超え、実際に出た力は上限で止まる
  2. 上限を下げていない関節・env はそのまま
  3. 物理に効く（姿勢が崩れる）
  4. リセットで上限が正常に戻る
"""

import argparse
import sys
from pathlib import Path

import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import get_cfgs  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402

NUM_STEPS = 100
TOL_NM = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=float, default=2.0, help="Lowered torque limit [Nm]")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs("official")
    # 姿勢が崩れても自動リセットで上限が戻らないよう、転倒判定を外す
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env = Go2Env(num_envs=4, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
    env.reset()

    names = env_cfg["joint_names"]
    i_fr_calf = names.index("FR_calf_joint")
    nominal = env.nominal_torque_limit
    limit = nominal.repeat(4, 1)
    limit[1, i_fr_calf] = args.limit
    limit[2, :] = args.limit
    limit[3, :] = args.limit
    env.set_torque_limit(limit)

    is_reset = torch.tensor([False, False, False, True], device=gs.device)
    env._reset_idx(is_reset)
    _, upper_after_reset = env.robot.get_dofs_force_range(env.motors_dof_idx)

    kp, kd = env_cfg["kp"], env_cfg["kd"]
    max_commanded = torch.zeros(4, 12, device=gs.device)
    max_applied = torch.zeros(4, 12, device=gs.device)
    actions = torch.zeros(4, 12, device=gs.device)
    for _ in range(NUM_STEPS):
        env.step(actions)
        dof_pos = env.robot.get_dofs_position(env.motors_dof_idx)
        dof_vel = env.robot.get_dofs_velocity(env.motors_dof_idx)
        commanded = kp * (env.default_dof_pos - dof_pos) - kd * dof_vel
        applied = env.robot.get_dofs_control_force(env.motors_dof_idx)
        max_commanded = torch.maximum(max_commanded, commanded.abs())
        max_applied = torch.maximum(max_applied, applied.abs())

    _, upper_now = env.robot.get_dofs_force_range(env.motors_dof_idx)
    base_z = env.robot.get_pos()[:, 2]
    fr_calf_err = (env.robot.get_dofs_position(env.motors_dof_idx) - env.default_dof_pos)[:, i_fr_calf].abs()

    print(f"nominal limit [Nm]: {[round(v, 2) for v in nominal.tolist()]}")
    for i_env, label in enumerate(["normal", "FR_calf limited", "all limited", "limited then reset"]):
        print(
            f"env{i_env} {label:20s} base_z={base_z[i_env]:.3f} m  FR_calf |err|={fr_calf_err[i_env]:.3f} rad  "
            f"FR_calf cmd max={max_commanded[i_env, i_fr_calf]:.2f} applied max={max_applied[i_env, i_fr_calf]:.2f} "
            f"limit={upper_now[i_env, i_fr_calf]:.2f}"
        )

    is_limited = limit < nominal
    is_limited[3] = False
    checks = {
        "1a limited joints saturate (commanded > limit)": bool((max_commanded[1, i_fr_calf] > args.limit).item()),
        "1b applied force never exceeds the lowered limit": bool(
            (max_applied[is_limited] <= args.limit + TOL_NM).all().item()
        ),
        "2a untouched joints keep nominal limit": bool(
            torch.isclose(upper_now[~is_limited], nominal.repeat(4, 1)[~is_limited], atol=TOL_NM).all().item()
        ),
        "3a single-joint limit changes posture (FR_calf err env1 > env0 + 0.5 rad, base_z drop > 3 cm)": bool(
            ((fr_calf_err[1] > fr_calf_err[0] + 0.5) & (base_z[1] < base_z[0] - 0.03)).item()
        ),
        "3b all-joint limit collapses the body (base_z env2 < env0 - 0.05 m)": bool(
            (base_z[2] < base_z[0] - 0.05).item()
        ),
        "4a reset restores nominal limit": bool(
            torch.isclose(upper_after_reset[3], nominal, atol=TOL_NM).all().item()
        ),
        "4b reset env behaves like normal (base_z within 1 cm of env0)": bool(
            ((base_z[3] - base_z[0]).abs() < 0.01).item()
        ),
    }
    for name, is_ok in checks.items():
        print(f"[{'PASS' if is_ok else 'FAIL'}] {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
