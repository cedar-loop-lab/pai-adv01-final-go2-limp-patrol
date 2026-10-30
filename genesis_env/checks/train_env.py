"""学習用の環境の確認: 観測の履歴と、エピソード中にランダムに入る不調が仕様どおりか。

仕様（8章）: 確率 0.5 でランダムな1脚の3関節の上限を min(s × 共通基準, 元の上限) に下げる。
s ~ U[0.35, 1.0]、発生はエピソード中の一様な時刻。リセットで元の上限に戻る。
履歴は直近 history_len ステップの観測を古い順につなぐ。リセットした env は現在の観測で埋める。
"""

import sys
from pathlib import Path

import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import HISTORY_LEN, get_cfgs  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402

NUM_ENVS = 2048
OBS_DIM = 45


def main() -> None:
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs("patrol", HISTORY_LEN, has_degradation=True)
    env = Go2Env(NUM_ENVS, env_cfg, obs_cfg, reward_cfg, command_cfg)
    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    obs = env.get_observations()["policy"]
    check("obs shape", obs.shape == (NUM_ENVS, OBS_DIM * HISTORY_LEN), str(tuple(obs.shape)))
    hist = obs.view(NUM_ENVS, HISTORY_LEN, OBS_DIM)
    check("history filled on reset", bool((hist == hist[:, -1:]).all()), "all slots equal the current obs")

    # 抽選の分布
    has_fault = env.fault_onset_step >= 0
    rate = has_fault.float().mean().item()
    check("fault probability", abs(rate - 0.5) < 0.05, f"{rate:.3f} (spec 0.5)")
    nominal = env.nominal_torque_limit
    is_lowered = env.fault_limit < nominal
    legs_lowered = is_lowered.view(NUM_ENVS, 4, 3).any(dim=2).sum(dim=1)
    check("at most one leg", bool((legs_lowered <= 1).all()), f"max legs lowered = {legs_lowered.max().item()}")
    check("no fault -> nominal", bool((env.fault_limit[~has_fault] == nominal).all()), "")
    ref = env.reference_torque
    s = env.fault_limit / ref
    lowered_s = s[is_lowered]
    check(
        "s within [0.35, 1.0]",
        bool((lowered_s >= 0.35 - 1e-5).all() and (lowered_s <= 1.0 + 1e-5).all()),
        f"min {lowered_s.min().item():.3f}, max {lowered_s.max().item():.3f}",
    )
    onset = env.fault_onset_step[has_fault].float()
    check(
        "onset spread over the episode",
        onset.min().item() < 0.05 * env.max_episode_length and onset.max().item() > 0.95 * env.max_episode_length,
        f"min {onset.min().item():.0f}, max {onset.max().item():.0f} of {env.max_episode_length}",
    )

    # 発生ステップの前は元の上限、発生後は不調になった上限になること
    env.episode_length_buf.zero_()
    before = env.torque_limit.clone()
    actions = torch.zeros(NUM_ENVS, env.num_actions, device=gs.device)
    target = env.fault_onset_step.clone()
    target_limit = env.fault_limit.clone()
    applied = torch.zeros(NUM_ENVS, dtype=torch.bool, device=gs.device)
    wrong_early = torch.zeros(NUM_ENVS, dtype=torch.bool, device=gs.device)
    # 途中でリセットされた env は抽選し直されるので、以後は集計から外す
    was_reset = torch.zeros(NUM_ENVS, dtype=torch.bool, device=gs.device)
    for _ in range(60):
        step_now = env.episode_length_buf.clone()
        env.step(actions)
        was_reset |= env.reset_buf
        alive = ~was_reset
        is_due = (target >= 0) & (step_now >= target) & alive
        is_early = (step_now < target) | (target < 0)
        applied |= is_due & (env.torque_limit == target_limit).all(dim=1)
        wrong_early |= is_early & alive & (env.torque_limit != nominal).any(dim=1)
    due_early = (target >= 0) & (target < 50) & ~was_reset
    check("limit nominal at start", bool((before == nominal).all()), "")
    check("no limit change before onset", not bool(wrong_early.any()), f"{int(wrong_early.sum())} envs changed early")
    check(
        "limit applied at onset",
        bool(applied[due_early].float().mean() > 0.9),
        f"{applied[due_early].float().mean().item():.3f} of {int(due_early.sum())} envs with onset < 50 steps",
    )

    # リセットで元の上限に戻ること
    lowered_now = (env.torque_limit < nominal).any(dim=1)
    mask = lowered_now.clone()
    env._reset_idx(mask)
    env._update_observation()
    check("reset restores nominal", bool((env.torque_limit[mask] == nominal).all()), f"{int(mask.sum())} envs reset")
    hist = env.obs_buf.view(NUM_ENVS, HISTORY_LEN, OBS_DIM)
    check("history refilled after reset", bool((hist[mask] == hist[mask][:, -1:]).all()), "")

    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
