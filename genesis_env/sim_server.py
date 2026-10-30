"""ROS2 結合用のシミュレーション側: Genesis で 1 台の Go2 を実時間（50 Hz）で動かし、ROS2 のブリッジとローカルの UDP でつなぐ。

Genesis の環境（Python 3.12）と ROS2 Lyrical の rclpy（Python 3.14）は同じプロセスで動かせないので、プロセスを分ける。
実機の Go2 でも、メーカーの SDK（ROS ではない）をドライバが ROS2 につなぐ形になる。

送る（→ ブリッジ、SIM_PORT）: 時刻・位置・向き・速度・脚ごとの追従誤差（検知器の入力）・転倒、
  評価用の正解（不調の時刻・脚・強さ。ブリッジは /sim/ground_truth にだけ出し、判断するノードは読まない）。
受け取る（← ブリッジ、CMD_PORT）: 速度指令 (vx, vy, yaw_rate)、ミッションの状態（モード・通知・完了。記録と画面表示用）。
不調は試行表の試行（--table / --trial-id）の脚・強さで、コースに沿った距離が onset_m に達したら入れる（評価と同じ定義）。
最初の指令が届くまでは止まった状態を送り続け、届いたら歩き始める。
終了: 転倒・帰還完了の 1.5 s 後、不調から DEADLINE_S 秒、または --max-time。結果は --out の JSON（と --video の MP4）。
"""

import argparse
import json
import math
import socket
import time

import imageio.v2 as imageio
import torch
from PIL import ImageFont

import genesis as gs

from eval_b import DEADLINE_S
from eval_common import LEG_NAMES, REPO_ROOT, degraded_limit, is_fallen, load_env_and_policy, reset_from_table
from detector import leg_error
from patrol import MissionConfig, PatrolMission
from record_patrol import CAMERA_OFFSET, FONT_PATH, FRAME_EVERY, LEG_LABELS, RES, draw_leg_marker, draw_overlay, leg_points
from sim_link import CMD_PORT, HOST, SIM_PORT
AFTER_END_S = 1.5
WAIT_PERIOD_S = 0.1


def load_trial(table: str, trial_id: int) -> dict:
    return json.loads((REPO_ROOT / "genesis_env" / "trial_tables" / f"{table}.json").read_text())["trials"][trial_id]


class Link:
    """ブリッジとの UDP。指令は最後に届いたものを使う。"""

    def __init__(self) -> None:
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind((HOST, CMD_PORT))
        self.rx.setblocking(False)
        self.last: dict | None = None

    def send(self, state: dict) -> None:
        self.tx.sendto(json.dumps(state).encode(), (HOST, SIM_PORT))

    def poll(self) -> dict | None:
        while True:
            try:
                data, _ = self.rx.recvfrom(4096)
            except BlockingIOError:
                return self.last
            self.last = json.loads(data)


def state_message(env, t: float, err: torch.Tensor, is_fall: bool, truth: dict) -> dict:
    yaw = math.radians(env.base_euler[0, 2].item())
    return {
        "t": round(t, 4), "x": env.base_pos[0, 0].item(), "y": env.base_pos[0, 1].item(), "yaw": yaw,
        "vx": env.base_lin_vel[0, 0].item(), "vy": env.base_lin_vel[0, 1].item(), "wz": env.base_ang_vel[0, 2].item(),
        "leg_err": [round(x, 6) for x in err[0].tolist()], "fallen": is_fall, "truth": truth,
    }


def overlay_lines(label: str, t: float, trial: dict, truth: dict, cmd: dict, events: dict) -> list:
    lines = [(label, (255, 255, 255)),
             (f"t = {t:5.1f} s   ROS2 mission_node: {cmd.get('mode', '-')}   速度指令 {cmd.get('vx', 0.0):.2f} m/s",
              (255, 255, 255))]
    if truth["onset_t"] is not None:
        lines.append((f"{truth['onset_t']:.1f} s: {LEG_LABELS[trial['leg']]}のトルク上限を下げた（ROS2 側には知らせない）",
                      (255, 200, 80)))
    if events["notify_t"] is not None:
        lines.append((f"{events['notify_t']:.1f} s: health_monitor が異常を検知 → 帰還", (255, 140, 200)))
    if events["fall_t"] is not None:
        lines.append((f"{events['fall_t']:.1f} s: 転倒", (255, 90, 90)))
    elif events["done_t"] is not None:
        lines.append((f"{events['done_t']:.1f} s: 帰還完了", (120, 200, 255)))
    return lines


def is_finished(t: float, events: dict, truth: dict, max_time: float) -> bool:
    ends = [events["fall_t"], events["done_t"]]
    if any(e is not None and t > e + AFTER_END_S for e in ends):
        return True
    return (truth["onset_t"] is not None and t > truth["onset_t"] + DEADLINE_S) or t > max_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=999)
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--max-time", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True, help="Result JSON path")
    parser.add_argument("--video", type=str, default=None, help="Optional MP4 path (rendering slows the loop)")
    parser.add_argument("--label", type=str, default="Genesis + ROS2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial = load_trial(args.table, args.trial_id)
    camera_res = RES if args.video else None
    env, policy, _ = load_env_and_policy(args.exp_name, args.ckpt, 1, args.max_time, args.seed, camera_res=camera_res)
    probe = PatrolMission(MissionConfig(), 1, env.dt)  # コースに沿った距離を測るだけ（指令には使わない）
    limit = degraded_limit(env, trial["s"], LEG_NAMES.index(trial["leg"]) if trial["leg"] else 0)
    leg_link_idx = [env.robot.get_link(f"{trial['leg']}_{p}").idx_local for p in ("thigh", "calf")] if trial["leg"] else []
    font = ImageFont.truetype(FONT_PATH, 18)
    link = Link()
    obs = reset_from_table(env, [trial])
    truth = {"onset_t": None, "leg": trial["leg"], "s": trial["s"]}
    events = {"notify_t": None, "done_t": None, "fall_t": None}
    zero_err = torch.zeros(1, 4)

    while link.poll() is None:  # ROS2 側が指令を出すまで、止まった状態を送り続ける
        link.send(state_message(env, 0.0, zero_err, False, truth))
        time.sleep(WAIT_PERIOD_S)

    frames, overruns, t = [], 0, 0.0
    start = time.monotonic()
    with torch.no_grad():
        for i_step in range(int(args.max_time / env.dt) + 1):
            t = i_step * env.dt
            cmd = link.poll()
            if truth["onset_t"] is None and probe.distance_along(env.base_pos[:, :2])[0].item() >= trial["onset_m"]:
                env.set_torque_limit(limit[None])
                truth["onset_t"] = t
            for key, flag in (("notify_t", "fault"), ("done_t", "done")):
                if events[key] is None and cmd.get(flag):
                    events[key] = t
            probe.step(env.base_pos[:, :2], torch.deg2rad(env.base_euler[:, 2]), env.base_lin_vel[:, :2].norm(dim=1))
            env.commands.copy_(torch.tensor([[cmd["vx"], cmd["vy"], cmd["wz"]]], device=gs.device))
            obs, _, _, _ = env.step(policy(obs))
            if events["fall_t"] is None and bool(is_fallen(env)[0]):
                events["fall_t"] = t
            link.send(state_message(env, t, leg_error(env.target_dof_pos, env.dof_pos), events["fall_t"] is not None,
                                    truth))

            if args.video and i_step % FRAME_EVERY == 0:
                base = env.base_pos[0].cpu().numpy()
                if truth["onset_t"] is not None:
                    draw_leg_marker(env.scene, leg_points(env, leg_link_idx))
                env.camera.set_pose(pos=base + CAMERA_OFFSET, lookat=base)
                lines = overlay_lines(args.label, t, trial, truth, cmd, events)
                frames.append(draw_overlay(env.camera.render()[0], lines, font))
            if is_finished(t, events, truth, args.max_time):
                break
            lag = start + (i_step + 1) * env.dt - time.monotonic()  # 実時間に合わせる
            if lag > 0:
                time.sleep(lag)
            else:
                overruns += 1

    result = {"exp": args.exp_name, "table": args.table, "trial_id": args.trial_id, "trial_s": trial["s"],
              "trial_leg": trial["leg"], "end_t": t, "overruns": overruns, "steps": i_step + 1, **truth,
              **{k: events[k] for k in events}, "last_mode": link.last.get("mode") if link.last else None}
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    if args.video:
        video = REPO_ROOT / args.video
        video.parent.mkdir(exist_ok=True)
        imageio.mimwrite(video, frames, fps=50 // FRAME_EVERY, quality=8, macro_block_size=8)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
