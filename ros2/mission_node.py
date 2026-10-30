"""mission_node: 巡回（1 周して基地へ戻り、停止して保持）を進め、/cmd_vel を出す。/health/fault が True なら帰還する。

追従器は評価と同じ patrol.py（MissionConfig の既定値）。周期は rate_hz（既定 10 Hz。評価は 50 Hz で回している）。
"""

import json
import math

import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String

from ros_common import CMD_VEL, HEALTH_FAULT, LATCHED, MISSION_MODE, MISSION_STATUS, MODE_NAMES, ODOM, SENSOR, spin

from patrol import MissionConfig, PatrolMission  # ros_common が genesis_env を import パスに足す


class MissionNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_node")
        rate_hz = self.declare_parameter("rate_hz", 10.0).value
        self.mission = PatrolMission(MissionConfig(), 1, 1.0 / rate_hz, "cpu")
        self.odom: Odometry | None = None
        self.pub_cmd = self.create_publisher(Twist, CMD_VEL, 10)
        self.pub_mode = self.create_publisher(String, MISSION_MODE, 10)
        self.pub_status = self.create_publisher(String, MISSION_STATUS, 10)
        self.create_subscription(Odometry, ODOM, self.on_odom, SENSOR)
        self.create_subscription(Bool, HEALTH_FAULT, self.on_fault, LATCHED)
        self.create_timer(1.0 / rate_hz, self.tick)

    def on_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def on_fault(self, msg: Bool) -> None:
        if msg.data:
            self.mission.request_return(torch.tensor([True]))
            self.get_logger().warning("fault reported by health_monitor -> return to base")

    def tick(self) -> None:
        if self.odom is None:
            return
        pose, twist = self.odom.pose.pose, self.odom.twist.twist
        yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
        cmd = self.mission.step(torch.tensor([[pose.position.x, pose.position.y]]), torch.tensor([yaw]),
                                torch.tensor([math.hypot(twist.linear.x, twist.linear.y)]))[0]
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = (float(v) for v in cmd)
        self.pub_cmd.publish(msg)
        self.pub_mode.publish(String(data=MODE_NAMES[int(self.mission.mode[0])]))
        stamp = self.odom.header.stamp
        self.pub_status.publish(String(data=json.dumps({
            "t": round(stamp.sec + stamp.nanosec * 1e-9, 3), "done": bool(self.mission.is_done[0]),
            "num_rearm": int(self.mission.num_rearm[0]),
        })))


def main() -> None:
    rclpy.init()
    node = MissionNode()
    spin(node)


if __name__ == "__main__":
    main()
