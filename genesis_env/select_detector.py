"""検知器の設定を、検証用の表（val）だけで決める。final_b は使わない。

record: 提案版の各学習シードで val を「通知なし」で走らせ、ステップごとの脚の追従誤差と巡回中かどうかを保存する。
        通知なしの走行なので、検知器が通知するまでの軌跡は検知器ありの走行と同じになる（発報時刻はこの記録から決まる）。
select: 記録から候補の設定ごとに発報を計算し、下の規則で 1 つ選ぶ（この規則は結果を見る前に決めた）。
  候補: tau_s ∈ TAUS × hold_s ∈ HOLDS × is_relative ∈ {True, False}
  閾値: 誤報が 0 になる最小の閾値 × MARGIN（3 シード分の、正常条件の巡回中すべてと、劣化条件の発生前の巡回中）
  選び方: 劣化条件で巡回中に発報できた割合（検知率）が最大のもの。最大から DETECT_TIE 以内が複数なら
          発生から発報までの時間の中央値が短いもの。
  最後に、選んだ設定で LegFaultDetector を記録に対して実際に動かし、計算と一致することを確かめる。
"""

import argparse
import json
from itertools import product

import torch

from detector import DetectorConfig, LegFaultDetector, relative_to_others
from eval_b import CKPT, EXP_NAMES, MAX_TIME_S, simulate
from eval_common import LEG_NAMES, REPO_ROOT, load_env_and_policy

TAUS = (0.25, 0.5, 1.0)
HOLDS = (0.25, 0.5, 1.0)
MARGIN = 1.25
DETECT_TIE = 0.02
SEEDS = (1, 2, 3)
OUT_DIR = REPO_ROOT / "logs" / "detector"


def trace_path(seed: int):
    return OUT_DIR / f"val-proposed-s{seed}.pt"


def record(seed: int) -> None:
    table = json.loads((REPO_ROOT / "genesis_env" / "trial_tables" / "val.json").read_text())
    trials = table["trials"]
    env, policy, _ = load_env_and_policy(EXP_NAMES["proposed"].format(seed), CKPT, len(trials), MAX_TIME_S,
                                         seed=table["seed"])
    trace: dict = {}
    rec = simulate(env, policy, trials, "none", trace=trace)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dt": env.dt, "leg_err": torch.stack(trace["leg_err"]).cpu(), "is_active": torch.stack(trace["is_active"]).cpu(),
        "onset_t": rec["onset_t"].cpu(), "s": [t["s"] for t in trials], "leg": [t["leg"] for t in trials],
    }, trace_path(seed))
    print(f"seed {seed}: {len(trace['leg_err'])} steps -> {trace_path(seed).relative_to(REPO_ROOT)}")


def scores(leg_err: torch.Tensor, cfg: DetectorConfig, dt: float) -> torch.Tensor:
    """(T, n, 4) の誤差から、LegFaultDetector と同じ一次遅れの score (T, n, 4)。"""
    signal = relative_to_others(leg_err) if cfg.is_relative else leg_err
    alpha = dt / max(cfg.tau_s, dt)
    out = torch.empty_like(signal)
    score = torch.zeros_like(signal[0])
    for k in range(signal.shape[0]):
        score = score + alpha * (signal[k] - score)
        out[k] = score
    return out


def held_level(data: dict, cfg: DetectorConfig) -> torch.Tensor:
    """各ステップで「直近 hold_s 秒ずっと超えていた」と言える最大の閾値 (T, n)。判定しないステップは -inf。"""
    dt = data["dt"]
    score = scores(data["leg_err"], cfg, dt)
    n_steps = score.shape[0]
    is_judged = data["is_active"] & (torch.arange(n_steps)[:, None] * dt >= cfg.warmup_s)
    score = torch.where(is_judged[..., None], score, -torch.inf)
    hold = max(1, round(cfg.hold_s / dt))
    padded = torch.cat([torch.full((hold - 1, *score.shape[1:]), -torch.inf), score])
    rolling_min = padded.unfold(0, hold, 1).min(dim=-1).values  # (T, n, 4)
    return rolling_min.max(dim=2).values


def evaluate(datas: list[dict], cfg: DetectorConfig) -> dict:
    """閾値を決め（誤報 0 の最小値 × MARGIN）、検知率と発報までの時間を返す。"""
    levels = [held_level(d, cfg) for d in datas]
    fa_level = -torch.inf
    for d, level in zip(datas, levels):
        t = torch.arange(level.shape[0])[:, None] * d["dt"]
        is_normal = torch.tensor([s is None for s in d["s"]])[None, :]
        is_before = t < d["onset_t"][None, :].nan_to_num(torch.inf)
        fa_level = max(fa_level, level[is_normal | is_before].max().item())
    assert fa_level > 0, f"false-alarm level {fa_level} is not positive; MARGIN would lower the threshold"
    threshold = fa_level * MARGIN
    delays, n_fault = [], 0
    for d, level in zip(datas, levels):
        for i, s in enumerate(d["s"]):
            if s is None:
                continue
            n_fault += 1
            fired = (level[:, i] > threshold).nonzero()
            if len(fired):
                delays.append(fired[0, 0].item() * d["dt"] - d["onset_t"][i].item())
    delays_t = torch.tensor(delays) if delays else torch.tensor([torch.nan])
    return {"cfg": DetectorConfig(cfg.tau_s, cfg.hold_s, cfg.is_relative, threshold, cfg.warmup_s),
            "fa_level": fa_level, "detect_rate": len(delays) / n_fault, "delay_median_s": delays_t.median().item(),
            "delay_p90_s": delays_t.quantile(0.9).item()}


def replay(data: dict, cfg: DetectorConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """記録に対して LegFaultDetector を実際に動かし、発報時刻と判定した脚を返す。"""
    n = data["leg_err"].shape[1]
    detector = LegFaultDetector(cfg, n, data["dt"], "cpu")
    fire_t = torch.full((n,), torch.nan)
    for k in range(data["leg_err"].shape[0]):
        is_new = detector.update(data["leg_err"][k].float(), data["is_active"][k], k * data["dt"])
        fire_t = torch.where(is_new, k * data["dt"], fire_t)
    return fire_t, detector.leg, detector.is_fired


def check_replay(datas: list[dict], best: dict) -> None:
    n_fault = n_hit = n_fa = n_leg_ok = 0
    for d in datas:
        fire_t, leg, is_fired = replay(d, best["cfg"])
        for i, s in enumerate(d["s"]):
            is_before = bool(is_fired[i]) and fire_t[i].item() < d["onset_t"][i].nan_to_num(torch.inf).item()
            if s is None or is_before:
                n_fa += bool(is_fired[i])
                continue
            n_fault += 1
            n_hit += bool(is_fired[i])
            n_leg_ok += bool(is_fired[i]) and LEG_NAMES[int(leg[i])] == d["leg"][i]
    print(f"replay: false alarms {n_fa}, detect {n_hit}/{n_fault} ({n_hit / n_fault:.3f}), leg correct {n_leg_ok}/{n_hit}")
    assert n_fa == 0 and abs(n_hit / n_fault - best["detect_rate"]) < 1e-9, "replay does not match the offline sweep"


def select() -> None:
    datas = []
    for seed in SEEDS:
        d = torch.load(trace_path(seed))
        d["leg_err"] = d["leg_err"].float()
        datas.append(d)
    results = []
    for tau, hold, is_relative in product(TAUS, HOLDS, (True, False)):
        r = evaluate(datas, DetectorConfig(tau, hold, is_relative))
        results.append(r)
        c = r["cfg"]
        print(f"tau {tau:<4} hold {hold:<4} rel {is_relative!s:<5}: threshold {c.threshold:.4f} rad, "
              f"detect {r['detect_rate']:.3f}, delay median {r['delay_median_s']:.2f} s (p90 {r['delay_p90_s']:.2f})")
    best_rate = max(r["detect_rate"] for r in results)
    near = [r for r in results if r["detect_rate"] >= best_rate - DETECT_TIE]
    best = min(near, key=lambda r: r["delay_median_s"])
    print(f"selected: {best['cfg']} (detect {best['detect_rate']:.3f}, delay median {best['delay_median_s']:.2f} s)")
    check_replay(datas, best)
    out = OUT_DIR / "selected.json"
    out.write_text(json.dumps({**best, "cfg": best["cfg"].__dict__, "rule": __doc__,
                               "candidates": [{**r, "cfg": r["cfg"].__dict__} for r in results]}, indent=1))
    print(f"-> {out.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["record", "select"])
    parser.add_argument("--train-seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.step == "record":
        if args.train_seed is None:
            parser.error("record needs --train-seed")
        record(args.train_seed)
    else:
        select()


if __name__ == "__main__":
    main()
