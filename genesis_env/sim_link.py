"""シミュレーション（sim_server.py）と ROS2 のブリッジ（ros2/genesis_bridge.py）がつなぐ UDP のアドレス。両方から読む。"""

HOST = "127.0.0.1"
SIM_PORT = 47101  # シミュレーション → ブリッジ（状態、50 Hz）
CMD_PORT = 47102  # ブリッジ → シミュレーション（指令とミッションの状態）
