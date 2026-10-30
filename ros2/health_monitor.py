"""health_monitor ノード: 脚ごとの追従誤差から劣化を検知し、/health/fault に出す。

検知器は評価B と同じ detector.py・同じ設定と較正値（detector_calib.json）。判定は巡回中（/mission/mode が patrol）だけ。
時刻は /odom の header.stamp（シミュレーションの時刻）を使う。/sim/ground_truth は購読しない。
"""

import json

import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String

from ros_common import (
    CONTROL_DT, HEALTH_FAULT, HEALTH_STATUS, LATCHED, LEG_ERROR, LEG_NAMES, MISSION_MODE, ODOM, SENSOR, spin,
)

from detector import LegFaultDetector, load_calibration  # ros_common が genesis_env を import パスに足す

STATUS_EVERY = 5  # 50 Hz のうち 10 Hz で状態を出す


class HealthMonitor(Node):
    def __init__(self) -> None:
        super().__init__("health_monitor")
        exp_name = self.declare_parameter("exp_name", "go2-degr-h10-s1").value
        cfg, baseline = load_calibration(exp_name)
        self.detector = LegFaultDetector(cfg, baseline, 1, CONTROL_DT, "cpu")
        self.sim_t = 0.0
        self.is_patrol = False
        self.count = 0
        self.pub_fault = self.create_publisher(Bool, HEALTH_FAULT, LATCHED)
        self.pub_status = self.create_publisher(String, HEALTH_STATUS, 10)
        self.pub_fault.publish(Bool(data=False))
        self.create_subscription(Odometry, ODOM, self.on_odom, SENSOR)
        self.create_subscription(String, MISSION_MODE, self.on_mode, 10)
        self.create_subscription(Float32MultiArray, LEG_ERROR, self.on_error, SENSOR)
        self.get_logger().info(f"detector {cfg} baseline {baseline}")

    def on_odom(self, msg: Odometry) -> None:
        self.sim_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def on_mode(self, msg: String) -> None:
        self.is_patrol = msg.data == "patrol"

    def on_error(self, msg: Float32MultiArray) -> None:
        err = torch.tensor([list(msg.data)])
        is_new = bool(self.detector.update(err, torch.tensor([self.is_patrol]), self.sim_t)[0])
        if is_new:
            leg = LEG_NAMES[int(self.detector.leg[0])]
            self.pub_fault.publish(Bool(data=True))
            self.get_logger().warning(f"fault detected at t={self.sim_t:.2f} s (largest deviation: {leg})")
        self.count += 1
        if self.count % STATUS_EVERY == 0:
            score, leg = self.detector.dev[0].abs().max(dim=0)
            self.pub_status.publish(String(data=json.dumps({
                "t": round(self.sim_t, 3), "score": round(score.item(), 5), "threshold": self.detector.cfg.threshold,
                "largest_leg": LEG_NAMES[int(leg)], "fired": bool(self.detector.is_fired[0]),
            })))


def main() -> None:
    rclpy.init()
    node = HealthMonitor()
    spin(node)


if __name__ == "__main__":
    main()
