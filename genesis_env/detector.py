"""劣化の検知器: 実機でも測れる値（PD 制御へ送った関節の目標角度と、実際の関節角度）だけで、脚の劣化を判定する。

入力に劣化の強さ s・設定したトルク上限・発生時刻・正解の脚は使わない。
1. 脚ごとに 3 関節の |目標角度 - 実際の角度| の平均をとる（leg_error）。
2. is_relative なら、その脚の値から他の 3 脚の中央値を引く（歩き方や旋回で全脚が一緒に増える分を消す）。
3. 一次遅れ（時定数 tau_s）でならした値を score とする。
4. score が threshold を hold_s 秒続けて超えたら、その脚の劣化として通知する（1 試行で 1 回だけ）。
判定するのは巡回中（呼び出し側が is_active で渡す）かつ歩き始めから warmup_s 秒以降だけ。
設定は検証用の表（val）だけで決める（select_detector.py）。
"""

from dataclasses import dataclass

import torch

NUM_LEGS = 4
# 評価B で使う設定。select_detector.py で val だけから決めて固定する（final_b を回す前に）
DETECTOR_CFG: "DetectorConfig | None" = None


@dataclass(frozen=True)
class DetectorConfig:
    tau_s: float = 0.5
    hold_s: float = 0.5
    is_relative: bool = True
    threshold: float = float("inf")
    warmup_s: float = 2.0  # 立った状態から歩き出す過渡を判定に使わない


def leg_error(target_dof_pos: torch.Tensor, dof_pos: torch.Tensor) -> torch.Tensor:
    """(n, 12) の目標角度と実際の角度から、脚ごとの追従誤差 (n, 4) [rad]。関節の並びは FR, FL, RR, RL × 3。"""
    return (target_dof_pos - dof_pos).abs().view(-1, NUM_LEGS, 3).mean(dim=2)


def relative_to_others(err: torch.Tensor) -> torch.Tensor:
    """各脚の値から、他の 3 脚の中央値を引く。err は (..., 4)。"""
    others = torch.stack([err[..., [j for j in range(NUM_LEGS) if j != i]] for i in range(NUM_LEGS)], dim=-2)
    return err - others.median(dim=-1).values


class LegFaultDetector:
    def __init__(self, cfg: DetectorConfig, num_envs: int, dt: float, device) -> None:
        self.cfg = cfg
        self.dt = dt
        self.alpha = dt / max(cfg.tau_s, dt)
        self.hold_steps = max(1, round(cfg.hold_s / dt))
        self.score = torch.zeros(num_envs, NUM_LEGS, device=device)
        self.over_steps = torch.zeros(num_envs, NUM_LEGS, dtype=torch.int64, device=device)
        self.is_fired = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.leg = torch.full((num_envs,), -1, dtype=torch.int64, device=device)

    def update(self, err: torch.Tensor, is_active: torch.Tensor, t: float) -> torch.Tensor:
        """1 制御ステップ分の脚ごとの誤差 (n, 4) で更新し、このステップで新たに通知した env の mask を返す。"""
        cfg = self.cfg
        signal = relative_to_others(err) if cfg.is_relative else err
        self.score += self.alpha * (signal - self.score)
        is_judged = (is_active & ~self.is_fired)[:, None] & (t >= cfg.warmup_s)
        is_over = is_judged & (self.score > cfg.threshold)
        self.over_steps = torch.where(is_over, self.over_steps + 1, 0)
        is_leg_fired = self.over_steps >= self.hold_steps
        is_new = is_leg_fired.any(dim=1) & ~self.is_fired
        leg = torch.where(is_leg_fired, self.score, -torch.inf).argmax(dim=1)
        self.leg = torch.where(is_new, leg, self.leg)
        self.is_fired |= is_new
        return is_new
