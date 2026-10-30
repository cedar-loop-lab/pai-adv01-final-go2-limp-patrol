"""ROS2 ブリッジ: Genesis のシミュレーション（genesis_env/sim_server.py）と UDP でつなぎ、ROS2 のトピックに変換する。

シミュレーション → /odom・/leg_tracking_error・/sim/fallen・/sim/ground_truth（記録専用）
/cmd_vel と、ミッションの状態（/mission/mode・/mission/status・/health/fault）→ シミュレーション（指令と画面表示・記録）
"""

import json
import math
import socket

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String

from ros_common import (
    CMD_VEL, FALLEN, GROUND_TRUTH, HEALTH_FAULT, LATCHED, LEG_ERROR, MISSION_MODE, MISSION_STATUS, ODOM, SENSOR, spin,
)
from sim_link import CMD_PORT, HOST, SIM_PORT

POLL_PERIOD_S = 0.004  # 50 Hz の状態を取りこぼさないよう、短い周期で UDP を読む


def to_odometry(state: dict) -> Odometry:
    msg = Odometry()
    sec = int(state["t"])
    msg.header.stamp.sec, msg.header.stamp.nanosec = sec, min(int((state["t"] - sec) * 1e9), 999_999_999)
    msg.header.frame_id, msg.child_frame_id = "odom", "base"
    msg.pose.pose.position.x, msg.pose.pose.position.y = state["x"], state["y"]
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(state["yaw"] / 2), math.cos(state["yaw"] / 2)
    msg.twist.twist.linear.x, msg.twist.twist.linear.y = state["vx"], state["vy"]
    msg.twist.twist.angular.z = state["wz"]
    return msg


class GenesisBridge(Node):
    def __init__(self) -> None:
        super().__init__("genesis_bridge")
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind((HOST, SIM_PORT))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pub_odom = self.create_publisher(Odometry, ODOM, SENSOR)
        self.pub_err = self.create_publisher(Float32MultiArray, LEG_ERROR, SENSOR)
        self.pub_fallen = self.create_publisher(Bool, FALLEN, SENSOR)
        self.pub_truth = self.create_publisher(String, GROUND_TRUTH, LATCHED)
        self.to_sim = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "mode": None, "fault": False, "done": False}
        self.has_cmd = False
        self.last_truth = None
        self.create_subscription(Twist, CMD_VEL, self.on_cmd, 10)
        self.create_subscription(String, MISSION_MODE, lambda m: self.to_sim.update(mode=m.data), 10)
        self.create_subscription(String, MISSION_STATUS, self.on_status, 10)
        self.create_subscription(Bool, HEALTH_FAULT, lambda m: self.to_sim.update(fault=m.data), LATCHED)
        self.create_timer(POLL_PERIOD_S, self.poll)

    def on_cmd(self, msg: Twist) -> None:
        self.to_sim.update(vx=msg.linear.x, vy=msg.linear.y, wz=msg.angular.z)
        self.has_cmd = True
        self.send()

    def on_status(self, msg: String) -> None:
        self.to_sim["done"] = bool(json.loads(msg.data)["done"])
        self.send()

    def send(self) -> None:
        if self.has_cmd:
            self.tx.sendto(json.dumps(self.to_sim).encode(), (HOST, CMD_PORT))

    def poll(self) -> None:
        while True:
            try:
                data, _ = self.rx.recvfrom(8192)
            except BlockingIOError:
                return
            state = json.loads(data)
            self.pub_odom.publish(to_odometry(state))
            self.pub_err.publish(Float32MultiArray(data=state["leg_err"]))
            self.pub_fallen.publish(Bool(data=state["fallen"]))
            truth = json.dumps(state["truth"])
            if truth != self.last_truth:
                self.pub_truth.publish(String(data=truth))
                self.last_truth = truth


def main() -> None:
    rclpy.init()
    node = GenesisBridge()
    spin(node)


if __name__ == "__main__":
    main()
