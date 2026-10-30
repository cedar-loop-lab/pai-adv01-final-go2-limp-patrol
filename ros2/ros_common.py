"""ROS2 ノード共通: トピック名、QoS、評価コード（genesis_env/）の読み込み。

判断に使うロジック（追従器 patrol.py・検知器 detector.py）は評価と同じファイルを使う（ここで書き直さない）。
"""

import sys
from pathlib import Path

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "genesis_env"))

ODOM = "/odom"  # nav_msgs/Odometry。header.stamp はシミュレーションの時刻
LEG_ERROR = "/leg_tracking_error"  # std_msgs/Float32MultiArray。脚ごとの |目標角度 - 実際の角度| の平均 [rad]（FR, FL, RR, RL）
FALLEN = "/sim/fallen"  # std_msgs/Bool
GROUND_TRUTH = "/sim/ground_truth"  # std_msgs/String（JSON）。記録専用で、判断するノードは購読しない
CMD_VEL = "/cmd_vel"  # geometry_msgs/Twist
MISSION_MODE = "/mission/mode"  # std_msgs/String: patrol / return / hold
MISSION_STATUS = "/mission/status"  # std_msgs/String（JSON）: 完了・保持から戻った回数
HEALTH_FAULT = "/health/fault"  # std_msgs/Bool。異常を検知したら True（後から起動したノードにも届くよう transient local）
HEALTH_STATUS = "/health/status"  # std_msgs/String（JSON）: score と最大の脚

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
SENSOR = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
MODE_NAMES = ("patrol", "return", "hold")  # patrol.py の PATROL, RETURN, HOLD の順
LEG_NAMES = ("FR", "FL", "RR", "RL")  # eval_common.LEG_NAMES と同じ順（eval_common は Genesis を読むのでここに置く）
CONTROL_DT = 0.02  # シミュレーションの制御周期（Go2Env.dt）。検知器はこの周期で更新する


def spin(node) -> None:
    """ノードを回し、Ctrl-C や run_trial.sh からの SIGINT で静かに止める（終了処理中のエラーは無視する）。"""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        try:
            node.destroy_node()
        finally:
            rclpy.try_shutdown()
