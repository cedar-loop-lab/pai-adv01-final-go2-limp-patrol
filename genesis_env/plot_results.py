"""レポート用の図: 評価A（転倒率）と評価B（帰還成功率）を、劣化の強さごとに方式で並べた棒グラフ。

入力は logs/eval_a/final と logs/eval_b/final_b の結果（3 シード合計）。出力は --out-dir の PNG。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from eval_common import REPO_ROOT  # noqa: E402

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # 3 色は検証済み（色覚の違いでも区別できる）
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEVERITIES = (0.5, 0.4, 0.3, 0.25)


def pooled(directory: Path, pattern: str, key) -> dict[str, dict[float, float]]:
    """ファイル名の方式ごとに、強さ → 3 シード合計の割合。"""
    hits: dict[str, dict[float, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for path in directory.glob(pattern):
        tag = path.stem.rsplit("-s", 1)[0]
        for t in json.loads(path.read_text())["trials"]:
            if t["s"] is not None:
                hits[tag][t["s"]][0] += key(t)
                hits[tag][t["s"]][1] += 1
    return {tag: {s: h / n for s, (h, n) in by_s.items()} for tag, by_s in hits.items()}


def grouped_bars(ax, values: dict[str, dict[float, float]], labels: dict[str, str], ylabel: str) -> None:
    width = 0.26
    for i, (tag, label) in enumerate(labels.items()):
        xs = [j + (i - 1) * width for j in range(len(SEVERITIES))]
        ys = [values[tag][s] for s in SEVERITIES]
        ax.bar(xs, ys, width=width - 0.03, color=SERIES_COLORS[i], label=label)
        for x, y in zip(xs, ys):
            if y == 0:  # 高さ 0 の棒は見えないので、値を書いて「無い」のではなく「0」だと示す
                ax.text(x, 0.015, "0", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_xticks(range(len(SEVERITIES)), [f"s = {s}" + ("\n（学習範囲内）" if s >= 0.35 else "\n（範囲外）")
                                           for s in SEVERITIES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8, labelcolor=TEXT,
              handlelength=1.2, columnspacing=1.0)


def save(fig, out: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--term", type=str, default="劣化", help="Word used for the fault in labels (e.g. 不調 for the video)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()

    fall = pooled(REPO_ROOT / "logs" / "eval_a" / "final", "*.json", lambda t: t["outcome"] == "fell")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    term = args.term
    grouped_bars(ax, fall, {"nominal": "通常版", "nominal_slow0.4": f"通常版 + {term}時に減速（× 0.4）",
                            "proposed": f"提案版（{term}を入れて学習）"}, f"転倒率（{term}から 40 s）")
    ax.set_title("評価A: 帰還させずに巡回を続けたときの転倒率（3 シード × 4 脚 × 100 試行）", color=TEXT, fontsize=11, pad=24)
    save(fig, out_dir / "eval_a_fall.png")

    success = pooled(REPO_ROOT / "logs" / "eval_b" / "final_b", "proposed-*.json", lambda t: t["outcome"] == "success")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    grouped_bars(ax, success, {"proposed-none": "検知なし（1 周して自然に帰還）", "proposed-detector": "検知器で帰還",
                               "proposed-oracle": "正解通知で即帰還"}, f"帰還成功率（{term}から 40 s 以内）")
    ax.set_title("評価B: 提案版の帰還成功率（3 シード × 4 脚 × 100 試行）", color=TEXT, fontsize=11, pad=24)
    save(fig, out_dir / "eval_b_success.png")


if __name__ == "__main__":
    main()
