"""検知器の確認（シミュレーションなし）: 合成した追従誤差で、発報の条件と脚の記録を確かめる。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import DetectorConfig, LegFaultDetector  # noqa: E402

DT = 0.02
BASELINE = (0.25, 0.24, 0.23, 0.21)
CFG = DetectorConfig(tau_s=0.1, hold_s=0.2, threshold=0.03, warmup_s=1.0)
INSTANT = DetectorConfig(tau_s=1e-9, hold_s=CFG.hold_s, threshold=CFG.threshold, warmup_s=0.0)


def shifted(leg: int, delta: float) -> torch.Tensor:
    err = torch.tensor([BASELINE])
    err[0, leg] += delta
    return err


def run(cfg: DetectorConfig, err: torch.Tensor, is_active: torch.Tensor, steps: int, t0: float = 0.0):
    """同じ誤差 (1, 4) を steps 回入れ、最初に発報したステップ（なければ -1）と記録した脚を返す。"""
    detector = LegFaultDetector(cfg, BASELINE, 1, DT, "cpu")
    first = -1
    for k in range(steps):
        if detector.update(err, is_active, t0 + k * DT).item() and first < 0:
            first = k
    return first, int(detector.leg.item())


def main() -> None:
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    active = torch.ones(1, dtype=torch.bool)
    first, _ = run(CFG, torch.tensor([BASELINE]), active, 200, t0=CFG.warmup_s)
    check("baseline error never fires", first == -1)

    first, leg = run(CFG, shifted(2, 0.1), active, 200, t0=CFG.warmup_s)
    check("one leg above baseline fires", first >= 0, f"step {first}")
    check("records the leg with the largest deviation (RR)", leg == 2, f"leg {leg}")

    first, leg = run(CFG, shifted(1, -0.1), active, 200, t0=CFG.warmup_s)
    check("a drop below baseline also fires", first >= 0 and leg == 1, f"step {first}, leg {leg}")

    first, _ = run(CFG, shifted(2, 0.02), active, 200, t0=CFG.warmup_s)
    check("deviation under the threshold does not fire", first == -1)

    first, _ = run(CFG, shifted(2, 0.1), torch.zeros(1, dtype=torch.bool), 200, t0=CFG.warmup_s)
    check("inactive (not patrolling) never fires", first == -1)

    first, _ = run(CFG, shifted(2, 0.1), active, round(CFG.warmup_s / DT) - 1)
    check("no fire before warmup", first == -1)

    hold_steps = round(CFG.hold_s / DT)
    first, _ = run(INSTANT, shifted(2, 0.1), active, 200)
    check("fires exactly after hold_s", first == hold_steps - 1, f"step {first} (hold {hold_steps})")

    detector = LegFaultDetector(INSTANT, BASELINE, 1, DT, "cpu")
    fired = 0
    for k in range(200):
        err = shifted(2, 0.1) if (k // (hold_steps - 1)) % 2 == 0 else torch.tensor([BASELINE])
        fired += int(detector.update(err, active, k * DT).item())
    check("interrupted excess restarts the hold count", fired == 0)

    detector = LegFaultDetector(INSTANT, BASELINE, 1, DT, "cpu")
    fired = sum(int(detector.update(shifted(2, 0.1), active, k * DT).item()) for k in range(200))
    check("fires only once per trial", fired == 1, f"{fired}")

    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
