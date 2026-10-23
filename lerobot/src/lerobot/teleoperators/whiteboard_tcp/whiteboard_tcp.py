from __future__ import annotations

import threading
import time
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_whiteboard_tcp import WhiteboardTcpConfig


class WhiteboardTcp(Teleoperator):
    """Reads the TCP target published by the VR-to-TCP bridge."""

    config_class = WhiteboardTcpConfig
    name = "whiteboard_tcp"

    def __init__(self, config: WhiteboardTcpConfig):
        super().__init__(config)
        self.config = config
        self._node = None
        self._subscription = None
        self._lock = threading.Lock()
        self._action: RobotAction | None = None
        self._last_update = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in (
            "tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw"
        )}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._node is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            import rclpy
            from whiteboard_msgs.msg import TcpCommand
        except ImportError as exc:
            raise ImportError("whiteboard_tcp requires sourced ROS 2 whiteboard_msgs and rclpy") from exc
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("lerobot_whiteboard_tcp")
        self._subscription = self._node.create_subscription(TcpCommand, self.config.command_topic, self._callback, 20)

    def _callback(self, msg: Any) -> None:
        with self._lock:
            self._action = {
                "tcp_x": float(msg.tcp_x_m), "tcp_y": float(msg.tcp_y_m), "tcp_z": float(msg.tcp_z_m),
                "tcp_roll": float(msg.tcp_roll_rad), "tcp_pitch": float(msg.tcp_pitch_rad),
                "tcp_yaw": float(msg.tcp_yaw_rad),
            }
            self._last_update = time.monotonic()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=0.0)
        with self._lock:
            action = None if self._action is None else dict(self._action)
            age = time.monotonic() - self._last_update
        if action is None:
            raise RuntimeError(f"No TCP command received; check {self.config.command_topic}")
        if age > self.config.stale_after_s:
            raise RuntimeError(f"TCP teleoperation target is stale ({age:.3f}s)")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._node.destroy_node()
        self._node = self._subscription = None
