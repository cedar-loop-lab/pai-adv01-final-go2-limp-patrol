"""検知器の確認（シミュレーションなし）: 合成した追従誤差で、発報の条件と脚の判定を確かめる。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import DetectorConfig, LegFaultDetector  # noqa: E402

DT = 0.02
BASE_ERR = 0.05
CFG = DetectorConfig(tau_s=0.1, hold_s=0.2, is_relative=True, threshold=0.03, warmup_s=1.0)


def run(cfg: DetectorConfig, err: torch.Tensor, is_active: torch.Tensor, steps: int, t0: float = 0.0):
    """同じ誤差 (n, 4) を steps 回入れ、最初に発報したステップ（なければ -1）と判定した脚を返す。"""
    detector = LegFaultDetector(cfg, err.shape[0], DT, "cpu")
    first = torch.full((err.shape[0],), -1)
    for k in range(steps):
        is_new = detector.update(err, is_active, t0 + k * DT)
        first = torch.where(is_new, k, first)
    return first, detector.leg


def main() -> None:
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    active = torch.ones(1, dtype=torch.bool)
    rl_high = torch.tensor([[BASE_ERR, BASE_ERR, BASE_ERR, BASE_ERR + 0.1]])
    first, leg = run(CFG, rl_high, active, 200, t0=CFG.warmup_s)
    check("one leg with sustained error fires", first.item() >= 0, f"step {first.item()}")
    check("fired leg is RL", leg.item() == 3, f"leg {leg.item()}")

    all_high = torch.full((1, 4), BASE_ERR + 0.1)
    first, _ = run(CFG, all_high, active, 200, t0=CFG.warmup_s)
    check("relative: all legs up together does not fire", first.item() == -1)

    absolute = DetectorConfig(CFG.tau_s, CFG.hold_s, False, BASE_ERR + 0.05, CFG.warmup_s)
    first, _ = run(absolute, all_high, active, 200, t0=CFG.warmup_s)
    check("absolute: all legs up together fires", first.item() >= 0)

    first, _ = run(CFG, rl_high, torch.zeros(1, dtype=torch.bool), 200, t0=CFG.warmup_s)
    check("inactive (not patrolling) never fires", first.item() == -1)

    first, _ = run(CFG, rl_high, active, round(CFG.warmup_s / DT) - 1)
    check("no fire before warmup", first.item() == -1)

    hold_steps = round(CFG.hold_s / DT)
    instant = DetectorConfig(1e-9, CFG.hold_s, True, CFG.threshold, 0.0)
    first, _ = run(instant, rl_high, active, 200)
    check("fires exactly after hold_s", first.item() == hold_steps - 1, f"step {first.item()} (hold {hold_steps})")

    detector = LegFaultDetector(instant, 1, DT, "cpu")
    fired = 0
    for k in range(200):
        err = rl_high if (k // (hold_steps - 1)) % 2 == 0 else torch.full((1, 4), BASE_ERR)
        fired += int(detector.update(err, active, k * DT).item())
    check("interrupted excess restarts the hold count", fired == 0)

    detector = LegFaultDetector(instant, 1, DT, "cpu")
    fired = sum(int(detector.update(rl_high, active, k * DT).item()) for k in range(200))
    check("fires only once per trial", fired == 1, f"{fired}")

    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
