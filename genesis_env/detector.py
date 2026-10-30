"""劣化の検知器: 実機でも測れる値（PD 制御へ送った関節の目標角度と、実際の関節角度）だけで、脚の劣化を判定する。

入力に劣化の強さ s・設定したトルク上限・発生時刻・正解の脚は使わない。
1. 脚ごとに 3 関節の |目標角度 - 実際の角度| の平均をとる（leg_error）。
2. 劣化のない巡回で測った脚ごとの平均（baseline。方策ごとの較正値）を引く。
3. 一次遅れ（時定数 tau_s）でならし、4 脚のうち絶対値が最大のものを score とする。
4. score が threshold を hold_s 秒続けて超えたら通知する（1 試行で 1 回だけ）。そのときの最大の脚も記録する。
判定するのは巡回中（呼び出し側が is_active で渡す）かつ歩き始めから warmup_s 秒以降だけ。

劣化した脚そのものの誤差はほとんど増えず、方策が補うために他の脚（対角・反対側）の誤差が増える（val で確認）。
そのため「どの脚か」の判定は当てにならず、通知は「どこかの脚がいつもと違う」という意味で使う。
設定と較正値は検証用の表（val）だけで決める（select_detector.py。結果は detector_calib.json）。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

NUM_LEGS = 4
CALIB_PATH = Path(__file__).resolve().parent / "detector_calib.json"


@dataclass(frozen=True)
class DetectorConfig:
    tau_s: float = 4.0
    hold_s: float = 0.25
    threshold: float = float("inf")
    warmup_s: float = 8.0  # 立った状態から歩き出す過渡と、一次遅れの立ち上がりを判定に使わない


def leg_error(target_dof_pos: torch.Tensor, dof_pos: torch.Tensor) -> torch.Tensor:
    """(n, 12) の目標角度と実際の角度から、脚ごとの追従誤差 (n, 4) [rad]。関節の並びは FR, FL, RR, RL × 3。"""
    return (target_dof_pos - dof_pos).abs().view(-1, NUM_LEGS, 3).mean(dim=2)


def load_calibration(exp_name: str) -> tuple[DetectorConfig, list[float]]:
    """val で決めた設定と、その方策の較正値（脚ごとの平均誤差）を読む。"""
    calib = json.loads(CALIB_PATH.read_text())
    return DetectorConfig(**calib["cfg"]), calib["baseline"][exp_name]


class LegFaultDetector:
    def __init__(self, cfg: DetectorConfig, baseline, num_envs: int, dt: float, device) -> None:
        self.cfg = cfg
        self.baseline = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        self.alpha = dt / max(cfg.tau_s, dt)
        self.hold_steps = max(1, round(cfg.hold_s / dt))
        self.dev = torch.zeros(num_envs, NUM_LEGS, device=device)
        self.over_steps = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.is_fired = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.leg = torch.full((num_envs,), -1, dtype=torch.int64, device=device)

    def update(self, err: torch.Tensor, is_active: torch.Tensor, t: float) -> torch.Tensor:
        """1 制御ステップ分の脚ごとの誤差 (n, 4) で更新し、このステップで新たに通知した env の mask を返す。"""
        cfg = self.cfg
        self.dev += self.alpha * ((err - self.baseline) - self.dev)
        score, leg = self.dev.abs().max(dim=1)
        is_over = is_active & ~self.is_fired & (t >= cfg.warmup_s) & (score > cfg.threshold)
        self.over_steps = torch.where(is_over, self.over_steps + 1, 0)
        is_new = self.over_steps >= self.hold_steps
        self.leg = torch.where(is_new, leg, self.leg)
        self.is_fired |= is_new
        return is_new
