"""巡回ミッション: ウェイポイント追従と帰還を、歩行方策への指令（前進速度・旋回速度）に変換する。

全 env をまとめて扱う（torch のテンソル演算）。モードは env ごとに
  PATROL: コースのウェイポイントを順にたどる
  RETURN: 基地へまっすぐ戻る（コースを1周し終えたとき、または帰還を指示されたとき）
  HOLD:   基地の半径 r 以内に入ったので停止指令を出し続ける
帰還完了は「HOLD のまま、半径 r 以内かつ速度が閾値以下の状態を hold_s 秒保った」こと。
"""

import math
from dataclasses import dataclass

import torch

import genesis as gs

PATROL, RETURN, HOLD = 0, 1, 2


@dataclass(frozen=True)
class MissionConfig:
    # 基地は原点。コースは基地 → 各ウェイポイント → 基地。1辺 4 m の正方形を仮置き。
    waypoints: tuple[tuple[float, float], ...] = ((4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    v_cruise: float = 0.5
    # 帰還中の前進速度。None なら v_cruise と同じ
    v_return: float | None = None
    yaw_gain: float = 2.0
    yaw_max: float = 1.0
    # 向きのずれがこれより大きいと前進せずに旋回する
    turn_in_place_deg: float = 45.0
    reach_radius: float = 0.3
    home_radius: float = 0.5
    # 基地に近づいたら減速する（距離 [m] × この値 [1/s] を速度の上限にする）
    approach_gain: float = 1.0
    hold_speed: float = 0.1
    hold_s: float = 1.0


class PatrolMission:
    def __init__(self, cfg: MissionConfig, num_envs: int, dt: float):
        self.cfg = cfg
        self.dt = dt
        self.waypoints = torch.tensor(cfg.waypoints, dtype=gs.tc_float, device=gs.device)
        self.home = torch.zeros(2, dtype=gs.tc_float, device=gs.device)
        # コースの区間（基地から始まり基地で終わる）と、各区間の始点までの累積距離
        points = torch.cat([self.home[None], self.waypoints, self.home[None]])
        self.seg_start, self.seg_end = points[:-1], points[1:]
        seg_len = (self.seg_end - self.seg_start).norm(dim=1)
        self.seg_len = seg_len
        self.cum_len = torch.cat([torch.zeros(1, device=gs.device), seg_len.cumsum(0)])
        self.course_length = self.cum_len[-1].item()

        self.mode = torch.full((num_envs,), PATROL, dtype=torch.int64, device=gs.device)
        self.i_seg = torch.zeros(num_envs, dtype=torch.int64, device=gs.device)
        self.hold_steps = torch.zeros(num_envs, dtype=torch.int64, device=gs.device)
        self.is_done = torch.zeros(num_envs, dtype=torch.bool, device=gs.device)

    def progress(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """コースの進み具合（0〜1）。PATROL 中は現在の区間への射影で、それ以外は 1。"""
        start, end = self.seg_start[self.i_seg], self.seg_end[self.i_seg]
        seg_len = self.seg_len[self.i_seg]
        along = ((pos_xy - start) * (end - start)).sum(dim=1) / seg_len
        dist = self.cum_len[self.i_seg] + along.clamp(min=0.0).minimum(seg_len)
        return torch.where(self.mode == PATROL, dist / self.course_length, 1.0)

    def request_return(self, envs_mask: torch.Tensor) -> None:
        self.mode = torch.where(envs_mask & (self.mode == PATROL), RETURN, self.mode)

    def step(self, pos_xy: torch.Tensor, yaw: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """現在の位置・向き [rad]・速さから、歩行方策への指令 (vx, vy, yaw_rate) を返す。"""
        cfg = self.cfg
        # PATROL の最後の区間（基地へ戻る区間）は RETURN と同じ扱い
        is_last_seg = self.i_seg == len(self.seg_len) - 1
        self.mode = torch.where((self.mode == PATROL) & is_last_seg, RETURN, self.mode)

        target = torch.where((self.mode == PATROL)[:, None], self.seg_end[self.i_seg], self.home)
        delta = target - pos_xy
        dist = delta.norm(dim=1)

        # ウェイポイントに着いたら次の区間へ
        is_reached = (self.mode == PATROL) & (dist < cfg.reach_radius)
        self.i_seg = torch.where(is_reached, self.i_seg + 1, self.i_seg)

        # 基地の半径内に入ったら停止して保持
        is_home = dist < cfg.home_radius
        self.mode = torch.where((self.mode == RETURN) & is_home, HOLD, self.mode)
        is_still = (self.mode == HOLD) & is_home & (speed < cfg.hold_speed)
        self.hold_steps = torch.where(is_still, self.hold_steps + 1, 0)
        self.is_done |= self.hold_steps >= round(cfg.hold_s / self.dt)

        heading_err = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
        heading_err = torch.atan2(heading_err.sin(), heading_err.cos())
        yaw_rate = (cfg.yaw_gain * heading_err).clamp(-cfg.yaw_max, cfg.yaw_max)
        v_return = cfg.v_cruise if cfg.v_return is None else cfg.v_return
        vx = torch.where(self.mode == RETURN, v_return, cfg.v_cruise).to(dist.dtype)
        vx = torch.where(heading_err.abs() > math.radians(cfg.turn_in_place_deg), 0.0, vx)
        vx = torch.where(self.mode == RETURN, vx.minimum(dist * cfg.approach_gain), vx)

        is_hold = self.mode == HOLD
        vx = torch.where(is_hold, 0.0, vx)
        yaw_rate = torch.where(is_hold, 0.0, yaw_rate)
        return torch.stack([vx, torch.zeros_like(vx), yaw_rate], dim=1)
