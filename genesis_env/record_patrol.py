"""動画用: 1台の Go2 が巡回中に1脚のトルク上限が下がり、帰還するまでを録画する。

劣化はコースの進み具合 --onset-progress で入れ、同時に帰還を指示する（検知器はまだ無いので「正解通知」）。
劣化した脚は、3D の赤い目印（股・膝・足先と、脚の上の矢印）で示す。
1回の試行を録るだけで、評価の結果ではない。出力は video/（Git 管理外）。
"""

import argparse
import math

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import genesis as gs
from genesis.utils.geom import transform_by_quat

from configs import REFERENCE_TORQUE_NM
from eval_common import LEG_NAMES, REPO_ROOT, is_fallen, load_env_and_policy, reset_with_noise
from patrol import MissionConfig, PatrolMission

RES = (640, 480)
FRAME_EVERY = 2  # 50 Hz の制御から 25 fps で録る
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CAMERA_OFFSET = np.array([-1.6, -1.6, 1.0])
AFTER_DONE_S = 1.5
CALF_LENGTH_M = 0.213  # Go2 の膝から足先まで（URDF の foot_joint の位置）
MARKER_COLOR = (1.0, 0.1, 0.1, 0.85)
# 脚の上に立てる下向きの矢印（胴体に隠れても見えるように）
ARROW_HEIGHT_M = 0.45
ARROW_LENGTH_M = 0.25


def leg_points(env, link_idx: list[int]) -> np.ndarray:
    """劣化した脚の股（thigh の原点）・膝（calf の原点）・足先の位置。足先は calf に固定されているので計算する。"""
    pos = env.robot.get_links_pos(link_idx)[0]
    quat = env.robot.get_links_quat(link_idx)[0]
    foot = pos[1] + transform_by_quat(torch.tensor([0.0, 0.0, -CALF_LENGTH_M], device=gs.device), quat[1])
    return torch.stack([pos[0], pos[1], foot]).cpu().numpy()


def draw_leg_marker(scene, points: np.ndarray) -> None:
    scene.clear_debug_objects()
    for point in points:
        scene.draw_debug_sphere(point, radius=0.055, color=MARKER_COLOR)
    for start, end in zip(points[:-1], points[1:]):
        scene.draw_debug_line(start, end, radius=0.022, color=MARKER_COLOR)
    arrow_start = points[0] + np.array([0.0, 0.0, ARROW_HEIGHT_M])
    scene.draw_debug_arrow(arrow_start, vec=(0.0, 0.0, -ARROW_LENGTH_M), radius=0.02, color=MARKER_COLOR)


def draw_overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    y = 8
    for text, color in lines:
        draw.text((10, y), text, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += 26
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--leg", type=str, default="FR", choices=LEG_NAMES)
    parser.add_argument("--s", type=float, default=0.45)
    parser.add_argument("--onset-progress", type=float, default=0.4)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    env, policy, _ = load_env_and_policy(args.exp_name, args.ckpt, 1, args.max_time, args.seed, camera_res=RES)
    mission = PatrolMission(MissionConfig(), 1, env.dt)
    font = ImageFont.truetype(FONT_PATH, 20)

    leg = LEG_NAMES.index(args.leg)
    leg_link_idx = [env.robot.get_link(f"{args.leg}_{part}").idx_local for part in ("thigh", "calf")]
    ref = torch.tensor(REFERENCE_TORQUE_NM, device=gs.device)
    limit = env.nominal_torque_limit.clone()
    cols = slice(leg * 3, leg * 3 + 3)
    limit[cols] = torch.minimum(args.s * ref[cols], env.nominal_torque_limit[cols])

    obs = reset_with_noise(env, torch.zeros(1, 3, device=gs.device))
    onset_t, done_t, fallen_t = math.nan, math.nan, math.nan
    frames = []
    num_steps = int(args.max_time / env.dt)
    with torch.no_grad():
        for i_step in range(num_steps):
            t = i_step * env.dt
            pos_xy = env.base_pos[:, :2]
            if math.isnan(onset_t) and mission.progress(pos_xy)[0].item() >= args.onset_progress:
                env.set_torque_limit(limit[None])
                mission.request_return(torch.ones(1, dtype=torch.bool, device=gs.device))
                onset_t = t
            yaw = torch.deg2rad(env.base_euler[:, 2])
            speed = env.base_lin_vel[:, :2].norm(dim=1)
            was_done = bool(mission.is_done[0])
            env.commands.copy_(mission.step(pos_xy, yaw, speed))
            if mission.is_done[0] and not was_done:
                done_t = t
            obs, _, _, _ = env.step(policy(obs))
            if math.isnan(fallen_t) and bool(is_fallen(env)[0]):
                fallen_t = t

            if i_step % FRAME_EVERY == 0:
                base = env.base_pos[0].cpu().numpy()
                if not math.isnan(onset_t):
                    draw_leg_marker(env.scene, leg_points(env, leg_link_idx))
                env.camera.set_pose(pos=base + CAMERA_OFFSET, lookat=base)
                rgb = env.camera.render()[0]
                lines = [(args.label, (255, 255, 255)), (f"t = {t:5.1f} s", (255, 255, 255))]
                if math.isnan(onset_t):
                    lines.append(("巡回中（劣化なし）", (180, 255, 180)))
                else:
                    lines.append((f"{args.leg} 脚のトルク上限 s = {args.s}（{onset_t:.1f} s から）→ 帰還", (255, 200, 80)))
                if not math.isnan(fallen_t):
                    lines.append((f"転倒（{fallen_t:.1f} s）", (255, 90, 90)))
                elif not math.isnan(done_t):
                    lines.append((f"帰還完了（劣化から {done_t - onset_t:.1f} s）", (120, 200, 255)))
                frames.append(draw_overlay(rgb, lines, font))

            is_over = not math.isnan(fallen_t) and t > fallen_t + AFTER_DONE_S
            is_over |= not math.isnan(done_t) and t > done_t + AFTER_DONE_S
            if is_over:
                break

    out = REPO_ROOT / "video" / args.out
    out.parent.mkdir(exist_ok=True)
    imageio.mimwrite(out, frames, fps=50 // FRAME_EVERY, quality=8, macro_block_size=8)
    print(f"onset {onset_t:.2f} s, done {done_t:.2f} s, fallen {fallen_t:.2f} s -> {out}")


if __name__ == "__main__":
    main()
