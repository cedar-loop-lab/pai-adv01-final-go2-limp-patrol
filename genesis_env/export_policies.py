"""公開用の重みを書き出す: logs/<exp>/model_999.pt から推論に使う actor だけを policies/<exp>/ へ。

学習を続けるための optimizer・critic の状態は外す（1 本 9.5 MB → 1.6 MB）。評価に要る設定（cfgs.pkl）も一緒に写す。
書き出したあと、actor の全テンソルが元と完全に一致することを確かめる。
"""

import shutil

import torch

from eval_common import POLICY_ROOT, REPO_ROOT

CKPT = 999
EXP_NAMES = [f"go2-{variant}-h10-s{seed}" for variant in ("nominal", "degr") for seed in (1, 2, 3)]


def export(exp_name: str) -> None:
    src_dir = REPO_ROOT / "logs" / exp_name
    dst_dir = POLICY_ROOT / exp_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    full = torch.load(src_dir / f"model_{CKPT}.pt", map_location="cpu", weights_only=False)
    dst = dst_dir / f"model_{CKPT}.pt"
    torch.save({"actor_state_dict": full["actor_state_dict"], "iter": full["iter"], "infos": None}, dst)
    shutil.copyfile(src_dir / "cfgs.pkl", dst_dir / "cfgs.pkl")

    slim = torch.load(dst, map_location="cpu", weights_only=True)
    original, exported = full["actor_state_dict"], slim["actor_state_dict"]
    if original.keys() != exported.keys() or not all(torch.equal(original[k], exported[k]) for k in original):
        raise SystemExit(f"{exp_name}: exported actor differs from the original")
    print(f"{exp_name}: {dst.stat().st_size / 1e6:.2f} MB, {len(exported)} tensors identical")


if __name__ == "__main__":
    for name in EXP_NAMES:
        export(name)
