"""検知器の設定と較正値を、検証用の表（val）だけで決める。final_b は使わない。

record: 提案版の各学習シードで val を「通知なし」で走らせ、ステップごとの脚の追従誤差と巡回中かどうかを保存する。
        通知なしの走行なので、検知器が通知するまでの軌跡は検知器ありの走行と同じになる（発報時刻はこの記録から決まる）。
select: 較正値（方策ごとに、val の正常条件の巡回中・歩き始め 2 s 以降の、脚ごとの平均誤差）を求め、
        候補の設定ごとに発報を計算し、下の規則で 1 つ選ぶ。
  候補: tau_s ∈ TAUS × hold_s ∈ HOLDS（warmup_s = max(2, 2 tau_s)）
  閾値: 誤報が 0 になる最小の閾値 × MARGIN（3 シード分の、正常条件の巡回中すべてと、不調条件の発生前の巡回中）
  選び方: 不調条件で巡回中に発報できた割合（検知率）が最大のもの。最大から DETECT_TIE 以内が複数なら
          発生から発報までの時間の中央値が短いもの。
  最後に、選んだ設定で LegFaultDetector を記録に対して実際に動かし、計算と一致することを確かめる。
経緯: 最初の方式（各脚を他の 3 脚の中央値と比べる。時定数 0.25〜1 s）は val で検知率 4%、脚の判定 0/52 だった。
      不調のある脚ではなく他の脚の誤差が増えるためで、この方式（正常時の平均との差を長めにならす）に変えた。
      候補の範囲はこの val の結果を見ながら決めた（final_b は見ていない）。
"""

import argparse
import json
from itertools import product

import torch

from detector import CALIB_PATH, DetectorConfig, LegFaultDetector
from eval_b import CKPT, EXP_NAMES, MAX_TIME_S, simulate
from eval_common import LEG_NAMES, REPO_ROOT, load_env_and_policy

TAUS = (1.0, 2.0, 4.0, 8.0)
HOLDS = (0.25, 1.0)
MARGIN = 1.25
DETECT_TIE = 0.02
BASELINE_FROM_S = 2.0
SEEDS = (1, 2, 3)
OUT_DIR = REPO_ROOT / "logs" / "detector"


def trace_path(seed: int):
    return OUT_DIR / f"val-proposed-s{seed}.pt"


def exp_name(seed: int) -> str:
    return EXP_NAMES["proposed"].format(seed)


def record(seed: int) -> None:
    table = json.loads((REPO_ROOT / "genesis_env" / "trial_tables" / "val.json").read_text())
    trials = table["trials"]
    env, policy, _ = load_env_and_policy(exp_name(seed), CKPT, len(trials), MAX_TIME_S,
                                         seed=table["seed"])
    trace: dict = {}
    rec = simulate(env, policy, trials, "none", trace=trace)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dt": env.dt, "leg_err": torch.stack(trace["leg_err"]).cpu(), "is_active": torch.stack(trace["is_active"]).cpu(),
        "onset_t": rec["onset_t"].cpu(), "s": [t["s"] for t in trials], "leg": [t["leg"] for t in trials],
    }, trace_path(seed))
    print(f"seed {seed}: {len(trace['leg_err'])} steps -> {trace_path(seed).relative_to(REPO_ROOT)}")


def baseline(data: dict) -> torch.Tensor:
    """正常条件の巡回中（歩き始め BASELINE_FROM_S 秒以降）の脚ごとの平均誤差 (4,)。"""
    t = torch.arange(data["leg_err"].shape[0])[:, None] * data["dt"]
    is_normal = torch.tensor([s is None for s in data["s"]])[None, :]
    mask = data["is_active"] & (t >= BASELINE_FROM_S) & is_normal
    return (data["leg_err"] * mask[..., None]).sum(dim=(0, 1)) / mask.sum()


def with_warmup(tau: float, hold: float) -> DetectorConfig:
    return DetectorConfig(tau_s=tau, hold_s=hold, warmup_s=max(2.0, 2 * tau))


def held_level(data: dict, cfg: DetectorConfig) -> torch.Tensor:
    """各ステップで「直近 hold_s 秒ずっと超えていた」と言える最大の閾値 (T, n)。判定しないステップは -inf。
    LegFaultDetector と同じ一次遅れを、記録の全ステップにまとめてかける。"""
    dt = data["dt"]
    alpha = dt / max(cfg.tau_s, dt)
    signal = data["leg_err"] - data["baseline"]
    dev = torch.zeros_like(signal[0])
    score = torch.empty(signal.shape[:2])
    for k in range(signal.shape[0]):
        dev = dev + alpha * (signal[k] - dev)
        score[k] = dev.abs().max(dim=1).values
    n_steps = score.shape[0]
    is_judged = data["is_active"] & (torch.arange(n_steps)[:, None] * dt >= cfg.warmup_s)
    score = torch.where(is_judged, score, -torch.inf)
    hold = max(1, round(cfg.hold_s / dt))
    padded = torch.cat([torch.full((hold - 1, score.shape[1]), -torch.inf), score])
    return padded.unfold(0, hold, 1).min(dim=-1).values


def evaluate(datas: list[dict], cfg: DetectorConfig) -> dict:
    """閾値を決め（誤報 0 の最小値 × MARGIN）、検知率と発報までの時間を返す。"""
    levels = [held_level(d, cfg) for d in datas]
    fa_level = -torch.inf
    for d, level in zip(datas, levels):
        t = torch.arange(level.shape[0])[:, None] * d["dt"]
        is_normal = torch.tensor([s is None for s in d["s"]])[None, :]
        is_before = t < d["onset_t"][None, :].nan_to_num(torch.inf)
        fa_level = max(fa_level, level[is_normal | is_before].max().item())
    assert fa_level > 0, f"false-alarm level {fa_level} is not positive"
    threshold = fa_level * MARGIN
    delays, hits = [], {}
    for d, level in zip(datas, levels):
        for i, s in enumerate(d["s"]):
            if s is None:
                continue
            fired = (level[:, i] > threshold).nonzero()
            hit = hits.setdefault(s, [0, 0])
            hit[1] += 1
            if len(fired):
                hit[0] += 1
                delays.append(fired[0, 0].item() * d["dt"] - d["onset_t"][i].item())
    delays_t = torch.tensor(delays) if delays else torch.tensor([torch.nan])
    n_fault = sum(h[1] for h in hits.values())
    return {"cfg": DetectorConfig(cfg.tau_s, cfg.hold_s, threshold, cfg.warmup_s), "fa_level": fa_level,
            "detect_rate": len(delays) / n_fault, "detect_by_s": {str(k): v for k, v in hits.items()},
            "delay_median_s": delays_t.median().item(), "delay_p90_s": delays_t.quantile(0.9).item()}


def replay(data: dict, cfg: DetectorConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """記録に対して LegFaultDetector を実際に動かし、発報時刻と最大だった脚を返す。"""
    n = data["leg_err"].shape[1]
    detector = LegFaultDetector(cfg, data["baseline"], n, data["dt"], "cpu")
    fire_t = torch.full((n,), torch.nan)
    for k in range(data["leg_err"].shape[0]):
        is_new = detector.update(data["leg_err"][k], data["is_active"][k], k * data["dt"])
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
    print(f"replay: false alarms {n_fa}, detect {n_hit}/{n_fault} ({n_hit / n_fault:.3f}), "
          f"largest leg = degraded leg {n_leg_ok}/{n_hit}")
    assert n_fa == 0 and abs(n_hit / n_fault - best["detect_rate"]) < 1e-9, "replay does not match the offline sweep"


def select() -> None:
    datas = []
    for seed in SEEDS:
        d = torch.load(trace_path(seed))
        d["leg_err"] = d["leg_err"].float()
        d["baseline"] = baseline(d)
        datas.append(d)
    results = []
    for tau, hold in product(TAUS, HOLDS):
        r = evaluate(datas, with_warmup(tau, hold))
        results.append(r)
        print(f"tau {tau:<4} hold {hold:<4}: threshold {r['cfg'].threshold:.4f} rad, detect {r['detect_rate']:.3f} "
              f"{r['detect_by_s']}, delay median {r['delay_median_s']:.2f} s (p90 {r['delay_p90_s']:.2f})")
    best_rate = max(r["detect_rate"] for r in results)
    near = [r for r in results if r["detect_rate"] >= best_rate - DETECT_TIE]
    best = min(near, key=lambda r: r["delay_median_s"])
    print(f"selected: {best['cfg']} (detect {best['detect_rate']:.3f}, delay median {best['delay_median_s']:.2f} s)")
    check_replay(datas, best)
    calib = {"cfg": best["cfg"].__dict__, "baseline": {exp_name(seed): [round(x, 6) for x in d["baseline"].tolist()]
                                                       for seed, d in zip(SEEDS, datas)},
             "val": {k: v for k, v in best.items() if k != "cfg"}}
    CALIB_PATH.write_text(json.dumps(calib, indent=1) + "\n")
    (OUT_DIR / "selected.json").write_text(json.dumps(
        {"calibration": calib, "candidates": [{**r, "cfg": r["cfg"].__dict__} for r in results]}, indent=1))
    print(f"-> {CALIB_PATH.relative_to(REPO_ROOT)}")


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
