"""巡回ミッションの判定の確認（シミュレーションなし）: 帰還完了の数え方と、保持から帰還への戻り。

仕様（8章）: 帰還を始めたあと、半径 0.5 m 以内かつ速度 0.1 m/s 以下を連続 1 s 保ったら完了。
どちらかを外れたら数え直す。保持中に半径の外へ出たら帰還に戻る（完了後は戻らない）。
"""

import math
import sys
from pathlib import Path

import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patrol import HOLD, PATROL, RETURN, MissionConfig, PatrolMission, looped_waypoints  # noqa: E402

DT = 0.02
HOLD_STEPS = round(MissionConfig().hold_s / DT)


def run(mission: PatrolMission, pos: tuple[float, float], speed: float, steps: int) -> None:
    pos_xy = torch.tensor([pos], device=gs.device)
    yaw = torch.zeros(1, device=gs.device)
    for _ in range(steps):
        mission.step(pos_xy, yaw, torch.tensor([speed], device=gs.device))


def returning_mission() -> PatrolMission:
    mission = PatrolMission(MissionConfig(), 1, DT)
    mission.request_return(torch.ones(1, dtype=torch.bool, device=gs.device))
    return mission


def main() -> None:
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    mission = PatrolMission(MissionConfig(), 1, DT)
    run(mission, (0.0, 0.0), 0.0, 3 * HOLD_STEPS)
    check("no completion while patrolling", not bool(mission.is_done[0]) and int(mission.mode[0]) == PATROL)

    mission = returning_mission()
    run(mission, (0.2, 0.0), 0.0, HOLD_STEPS - 1)
    check("not done before hold_s", not bool(mission.is_done[0]), f"{HOLD_STEPS - 1} steps")
    run(mission, (0.2, 0.0), 0.0, 1)
    check("done after hold_s", bool(mission.is_done[0]), f"{HOLD_STEPS} steps")

    mission = returning_mission()
    run(mission, (0.2, 0.0), 0.0, HOLD_STEPS - 1)
    run(mission, (0.2, 0.0), 0.2, 1)
    run(mission, (0.2, 0.0), 0.0, HOLD_STEPS - 1)
    check("speed violation restarts the count", not bool(mission.is_done[0]))
    run(mission, (0.2, 0.0), 0.0, 1)
    check("done after a full hold following the violation", bool(mission.is_done[0]))

    mission = returning_mission()
    run(mission, (0.2, 0.0), 0.0, HOLD_STEPS - 1)
    run(mission, (0.6, 0.0), 0.0, 1)
    run(mission, (0.2, 0.0), 0.0, HOLD_STEPS - 1)
    check("radius violation restarts the count", not bool(mission.is_done[0]))

    mission = returning_mission()
    run(mission, (0.45, 0.0), 0.0, HOLD_STEPS)
    check("still in the radius counts even before the hold mode", bool(mission.is_done[0]),
          f"mode {int(mission.mode[0])} (RETURN = {RETURN})")

    mission = returning_mission()
    run(mission, (0.2, 0.0), 0.0, 5)
    is_hold = int(mission.mode[0]) == HOLD
    run(mission, (0.6, 0.0), 0.0, 1)
    check("leaving the radius in hold returns", is_hold and int(mission.mode[0]) == RETURN,
          f"rearm {int(mission.num_rearm[0])}")
    check("rearm counted once", int(mission.num_rearm[0]) == 1)
    run(mission, (0.2, 0.0), 0.0, HOLD_STEPS)
    check("done after re-approach", bool(mission.is_done[0]))
    run(mission, (0.6, 0.0), 0.0, 1)
    check("no rearm after done", int(mission.num_rearm[0]) == 1 and int(mission.mode[0]) == HOLD)

    looped = PatrolMission(MissionConfig(waypoints=looped_waypoints(3)), 1, DT)
    check("3 laps = 48 m", abs(looped.course_length - 48.0) < 1e-4, f"{looped.course_length:.1f} m")

    mission = PatrolMission(MissionConfig(), 2, DT)
    mission.set_speed_scale(torch.tensor([False, True], device=gs.device), 0.6)
    # 1台目は正面、2台目は真横にウェイポイント（その場旋回で旋回指令が上限に張り付く）
    cmd = mission.step(torch.zeros(2, 2, device=gs.device), torch.tensor([0.0, -math.pi / 2], device=gs.device),
                       torch.zeros(2, device=gs.device))
    unscaled = PatrolMission(MissionConfig(), 2, DT).step(
        torch.zeros(2, 2, device=gs.device), torch.tensor([0.0, -math.pi / 2], device=gs.device),
        torch.zeros(2, device=gs.device))
    check("speed scale leaves other envs", torch.allclose(cmd[0], unscaled[0]), f"{cmd[0].tolist()}")
    check("speed scale scales vx and yaw rate", torch.allclose(cmd[1], unscaled[1] * 0.6), f"{cmd[1].tolist()}")

    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
