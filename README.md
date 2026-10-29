# pai2026-adv01-final

Physical AI 応用編1（2026 Summer）最終課題 — AI×ロボティクスによる創造的タスクの実現。

## 概要

Genesis シミュレータ上で強化学習したポリシーを、ROS2 ノードから推論・制御して
ロボットにタスクを実行させるプロジェクト。

> タスクの詳細・工夫点はレポート（提出物）に記載。

## 構成

| ディレクトリ | 内容 |
|---|---|
| `genesis_env/` | Genesis の環境定義・RL 学習/評価スクリプト |
| `ros2_ws/` | ROS2 ワークスペース（推論ノードでシミュ/実機を制御） |
| `assets/` | URDF・MJCF・モデル等 |
| `policies/` | 学習済みポリシー（大容量の重みは配布対象外） |
| `scripts/` | 補助スクリプト（録画の MP4 化など） |
| `docs/` | レポート素材 |

## 動作環境

- Ubuntu 26.04 / Python 3.x
- NVIDIA GPU（CUDA）
- Genesis, ROS2

## セットアップ

```bash
# 依存のインストール（追って requirements.txt に整理）
pip install -r requirements.txt
```

## 学習・実行

```bash
# 学習（Genesis）
python genesis_env/train.py

# 推論・制御（ROS2 ノード）
# 追記予定
```

## ライセンス

TBD
