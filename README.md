# 片脚のモーターが弱っても巡回を続ける四足ロボット（Go2 / Genesis / ROS2）

Physical AI 応用編1（2026 Summer）最終課題「AI×ロボティクスによる創造的タスクの実現」の提出リポジトリ。

四足ロボット Unitree Go2 が、基地から正方形のコース（1周 16 m）を巡回して戻る。
巡回の途中で1脚のモーターが弱る（関節トルクの上限が下がる）。それでも転ばずに歩き、任務を終えて基地に戻れるかを扱う。

- **歩き方（強化学習）**: Genesis で PPO（rsl-rl）により学習。学習中にランダムな1脚のトルク上限を下げた「提案版」と、下げない「通常版」を、同じ観測（直近 10 ステップの履歴）・同じネットワーク・同じ学習量で、学習シード 3 つずつ用意した。
- **判断（ROS2）**: `health_monitor` が脚の追従誤差から異常を検知し、`mission_node` が巡回と帰還を切り替える。シミュレーション（Genesis）とは UDP のブリッジでつなぐ。
- **評価**: 事前に作った試行表（劣化する脚・強さ・発生地点・初期姿勢の乱れ）を全方式で共有し、検証用の表で決めた設定を固定してから、最終テスト用の未使用の表で評価した。

## 結果（最終テスト。詳細はレポート）

劣化の強さ s は、その脚の 3 関節のトルク上限を、正常に歩くときのトルク（関節ごとの p95）の何倍にするか。
学習では s ∈ [0.35, 1.0] を使ったので、s = 0.5, 0.4 が学習範囲内、s = 0.3, 0.25 が範囲外。

**評価A（帰還させずに巡回を続けたとき、劣化から 40 s の転倒率。3 シード × 4 脚 × 100 試行）**

| 方式 | 範囲内 (s = 0.5, 0.4) | 範囲外 (s = 0.3, 0.25) |
|---|---|---|
| 通常版 | 0.68 | 0.91 |
| 通常版 + 劣化時に減速（指令 × 0.4。劣化の時刻は正解を与える） | 0.64 | 0.92 |
| 提案版 | **0**（2400 試行で転倒なし） | **0.16** |

範囲外では、特定の脚と学習シードの組み合わせ（例: seed 3 の左前脚）に転倒が集中した。正常時の1周の所要時間に悪化は見られなかった。

**評価B（任務: 1 周して基地に戻り停止。劣化から 40 s 以内の帰還成功率。提案版、3 シード）**

| s | 検知なし（1 周して自然に帰還） | 検知器で帰還 | 正解通知で即帰還 |
|---|---|---|---|
| 0.5 | 1.000 | 0.995 | 0.994 |
| 0.4 | 1.000 | 0.974 | 0.978 |
| 0.3 | 0.917 | 0.885 | 0.894 |
| 0.25 | 0.535 | 0.527 | 0.478 |

帰還を早めても成功率は上がらず、劣化直後に基地へ向きを変える旋回での転倒が増えた（seed 2 に集中）。
帰還を早める効果は完了までの時間の短縮（中央値 約 19.6 s → 検知器 約 15 s・正解通知 約 12 s）に限られた。
検知器は正常条件 300 試行で誤報 0、劣化条件の 62〜83% を検知した。

**ROS2 結合試験（検証用の表の 20 試行、実時間）**: 同じ試行のバッチ評価と結果が 20/20 で一致（すべて帰還成功、正常条件で誤報 0）。
通知の時刻の差は 0.1〜0.3 s。通知の有無が食い違ったのは、巡回が終わる直前に検知する境界の 2 試行。

## 構成

| パス | 内容 |
|---|---|
| `genesis_env/envs/go2_env.py` | Go2 の環境（Genesis の例 `go2_env.py` を改変。関節ごとのトルク上限、観測の履歴、学習中の劣化） |
| `genesis_env/configs.py` | 学習・環境の設定（劣化の基準トルクを含む） |
| `genesis_env/train.py` | 学習 |
| `genesis_env/patrol.py` | 巡回・帰還・停止の追従器（評価と ROS2 で共通） |
| `genesis_env/detector.py`, `detector_calib.json` | 劣化の検知器と、検証用の表で決めた設定・較正値（評価と ROS2 で共通） |
| `genesis_env/make_trial_table.py`, `trial_tables/` | 試行表（val: 開発・検証、final: 評価A、final_b: 評価B） |
| `genesis_env/eval_a.py`, `eval_b.py`, `summarize_eval_*.py` | 評価A・B と集計 |
| `genesis_env/select_detector.py` | 検知器の設定を検証用の表だけで決める |
| `genesis_env/sim_server.py`, `sim_link.py` | ROS2 結合用のシミュレーション（実時間 50 Hz、UDP） |
| `ros2/` | ROS2 ノード（`genesis_bridge`, `health_monitor`, `mission_node`）と結合試験のスクリプト |
| `genesis_env/checks/` | シミュレーションを使わない確認（追従器・検知器）と、環境の確認 |
| `genesis_env/diag_*.py`, `eval_patrol.py` など | 開発中の診断（最終結果には使っていない） |

## 動作環境

- Ubuntu 26.04、NVIDIA GPU（RTX 5070 Ti で確認）
- Genesis 側: Python 3.12（Genesis 1.4.5 は Python 3.14 に未対応）
- ROS2 側: ROS 2 Lyrical（Python 3.14）。Genesis と同じプロセスでは動かせないので、プロセスを分けて UDP でつなぐ

## セットアップ

```bash
# Genesis 側
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# ROS2 側（rclpy は /opt/ros/lyrical から。追従器と検知器のために CPU 版の torch を入れる）
uv venv --python /usr/bin/python3.14 --system-site-packages .venv-ros
uv pip install --python .venv-ros/bin/python torch --index-url https://download.pytorch.org/whl/cpu
```

## 再現手順

以下は `genesis_env/` で実行する（画面の無い環境では `PYOPENGL_PLATFORM=egl`）。

```bash
# 学習（通常版 / 提案版 × seed 1〜3）
../.venv/bin/python train.py -e go2-nominal-h10-s1 --seed 1 --max-iterations 1000
../.venv/bin/python train.py -e go2-degr-h10-s1 --seed 1 --max-iterations 1000 --degradation

# 評価A（減速率 k は val で決めた 0.4）
../.venv/bin/python eval_a.py --method proposed --train-seed 1 --table final
../.venv/bin/python eval_a.py --method nominal_slow --slow-factor 0.4 --train-seed 1 --table final
../.venv/bin/python summarize_eval_a.py final

# 検知器の設定（val だけで決める）と評価B
../.venv/bin/python select_detector.py record --train-seed 1   # seed 2, 3 も
../.venv/bin/python select_detector.py select
../.venv/bin/python eval_b.py --notify detector --policy proposed --train-seed 1 --table final_b
../.venv/bin/python summarize_eval_b.py final_b

# シミュレーションを使わない確認
../.venv/bin/python checks/mission.py && ../.venv/bin/python checks/detector.py
```

ROS2 との結合（リポジトリの直下で実行。ノード 3 つと rosbag2 を起動し、試行表の 1 試行を実時間で走らせる）:

```bash
ros2/run_trial.sh go2-degr-h10-s1 val 300 demo        # 結果は logs/ros2/demo/
ros2/run_batch.sh && .venv/bin/python ros2/summarize_runs.py   # 結合試験 20 回
```

学習済みの重みは容量が大きいためリポジトリに含めていない。

## ライセンスと出典

- `genesis_env/envs/go2_env.py` は [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) v1.4.5 の `examples/locomotion/go2_env.py`（Apache-2.0）を改変したもの。変更点はファイル冒頭に記載。
- それ以外のコードは本リポジトリの作成物。ライセンスは [LICENSE](LICENSE)（Apache-2.0）。
