from __future__ import annotations

import threading
import time
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_encos_feedback import EncosFeedbackConfig


def _sides(arm_mode: str) -> tuple[str, ...]:
    if arm_mode == "both":
        return ("left", "right")
    return (arm_mode,)


class EncosFeedback(Teleoperator):
    """Reads the commanded joint target from ENCOS feedback.

    This is useful with the existing VR upper node: VR remains the source of
    commands, while LeRobot records exactly the target that ENCOS executed.
    """

    config_class = EncosFeedbackConfig
    name = "encos_feedback"

    def __init__(self, config: EncosFeedbackConfig):
        super().__init__(config)
        self.config = config
        self._node = None
        self._subscription = None
        self._lock = threading.Lock()
        self._action: RobotAction | None = None
        self._last_update = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{side}_joint_{i}.pos": float for side in _sides(self.config.arm_mode) for i in range(1, 8)}

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
        """ENCOS target feedback has no device-side configuration step."""
        return

    def _require_ros(self):
        try:
            import rclpy
            from ctl_msgs_hw.msg import MidNodeCtlMsgRet
        except ImportError as exc:
            raise ImportError(
                "ENCOS feedback teleoperator requires a sourced ROS 2 workspace with ctl_msgs_hw"
            ) from exc
        return rclpy, MidNodeCtlMsgRet

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        rclpy, msg_type = self._require_ros()
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("lerobot_encos_feedback")
        self._subscription = self._node.create_subscription(
            msg_type, self.config.feedback_topic, self._callback, 20
        )

    def _callback(self, msg: Any) -> None:
        action: RobotAction = {}
        positions_by_side = {"left": msg.l_arm_jntpos_des, "right": msg.r_arm_jntpos_des}
        for side in _sides(self.config.arm_mode):
            positions = positions_by_side[side]
            if len(positions) < 7:
                return
            for index, position in enumerate(positions[:7], start=1):
                action[f"{side}_joint_{index}.pos"] = float(position)
        with self._lock:
            self._action = action
            self._last_update = time.monotonic()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=0.0)
        with self._lock:
            action = None if self._action is None else dict(self._action)
            age = time.monotonic() - self._last_update
        if action is None:
            raise RuntimeError("No ENCOS target received; check /dexbotCtl/midMonitor_msg_send")
        if age > self.config.stale_after_s:
            raise RuntimeError(f"ENCOS teleoperation target is stale ({age:.3f}s)")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._node.destroy_node()
        self._node = self._subscription = None
