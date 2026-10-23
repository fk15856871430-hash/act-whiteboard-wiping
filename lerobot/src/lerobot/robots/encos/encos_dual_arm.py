"""ROS 2 transport adapter for the ENCOS dual-arm controller."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_encos_dual_arm import EncosDualArmConfig

logger = logging.getLogger(__name__)


def _sides(arm_mode: str) -> tuple[str, ...]:
    if arm_mode == "both":
        return ("left", "right")
    return (arm_mode,)


def _joints(arm_mode: str) -> tuple[str, ...]:
    return tuple(f"{side}_joint_{index}" for side in _sides(arm_mode) for index in range(1, 8))


def _values(action: dict[str, Any], side: str, suffix: str = ".pos") -> list[float]:
    return [float(action[f"{side}_joint_{i}{suffix}"]) for i in range(1, 8)]


class EncosDualArm(Robot):
    config_class = EncosDualArmConfig
    name = "encos_dual_arm"

    def __init__(self, config: EncosDualArmConfig):
        super().__init__(config)
        self.config = config
        self._node = None
        self._publisher = None
        self._subscription = None
        self._lock = threading.Lock()
        self._feedback: dict[str, list[float]] | None = None
        self._last_feedback_time = 0.0
        self._connected = False
        self._cameras = make_cameras_from_configs(config.cameras)

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for joint in _joints(self.config.arm_mode):
            features[f"{joint}.pos"] = float
            features[f"{joint}.vel"] = float
            features[f"{joint}.iq"] = float
        for name, camera in self.config.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{name}_depth"] = (camera.height, camera.width, 1)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in _joints(self.config.arm_mode)}

    @property
    def is_connected(self) -> bool:
        return self._connected and all(camera.is_connected for camera in self._cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _require_ros(self):
        try:
            import rclpy
            from ctl_msgs_hw.msg import MidNodeCtlMsgRet, MidNodeCtlMsgSend
        except ImportError as exc:
            raise ImportError(
                "ENCOS ROS bridge requires a sourced ROS 2 workspace with ctl_msgs_hw "
                "and the rclpy Python package."
            ) from exc
        return rclpy, MidNodeCtlMsgRet, MidNodeCtlMsgSend

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        rclpy, ret_type, send_type = self._require_ros()
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("lerobot_encos_dual_arm")
        self._subscription = self._node.create_subscription(
            ret_type, self.config.feedback_topic, self._feedback_callback, 20
        )
        if self.config.command_mode == "direct":
            self._publisher = self._node.create_publisher(send_type, self.config.command_topic, 20)
        for camera in self._cameras.values():
            camera.connect()
        self._connected = True
        deadline = time.monotonic() + self.config.startup_timeout_s
        while self._feedback is None and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if self._feedback is None:
            self.disconnect()
            raise RuntimeError(
                f"No ENCOS feedback received on {self.config.feedback_topic} "
                f"within {self.config.startup_timeout_s:.1f}s"
            )

    def _feedback_callback(self, msg: Any) -> None:
        fields = (
            msg.l_arm_pos_fdback,
            msg.l_arm_spd_fdback,
            msg.l_arm_iq_fdback,
            msg.r_arm_pos_fdback,
            msg.r_arm_spd_fdback,
            msg.r_arm_iq_fdback,
        )
        if any(len(values) < 7 for values in fields):
            logger.warning("Ignoring malformed ENCOS feedback with fewer than 7 joints")
            return
        with self._lock:
            self._feedback = {
                "left_pos": list(msg.l_arm_pos_fdback),
                "left_vel": list(msg.l_arm_spd_fdback),
                "left_iq": list(msg.l_arm_iq_fdback),
                "right_pos": list(msg.r_arm_pos_fdback),
                "right_vel": list(msg.r_arm_spd_fdback),
                "right_iq": list(msg.r_arm_iq_fdback),
            }
            self._last_feedback_time = time.monotonic()

    def _spin_once(self) -> None:
        if self._node is not None:
            import rclpy

            rclpy.spin_once(self._node, timeout_sec=0.0)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._spin_once()
        with self._lock:
            feedback = None if self._feedback is None else {key: value[:] for key, value in self._feedback.items()}
        if feedback is None:
            raise RuntimeError("No ENCOS feedback received; check the ROS topics and controller nodes")
        if time.monotonic() - self._last_feedback_time > self.config.command_timeout_s:
            logger.warning("ENCOS feedback is %.3fs old", time.monotonic() - self._last_feedback_time)
        observation: RobotObservation = {}
        for side in _sides(self.config.arm_mode):
            for index in range(7):
                joint = f"{side}_joint_{index + 1}"
                observation[f"{joint}.pos"] = feedback[f"{side}_pos"][index]
                observation[f"{joint}.vel"] = feedback[f"{side}_vel"][index]
                observation[f"{joint}.iq"] = feedback[f"{side}_iq"][index]
        for name, camera in self.config.cameras.items():
            cam = self._cameras.get(name)
            if cam is not None and getattr(camera, "use_rgb", True):
                observation[name] = cam.read_latest()
            if cam is not None and getattr(camera, "use_depth", False):
                observation[f"{name}_depth"] = cam.read_latest_depth()
        return observation

    def _make_command(self, left: list[float], right: list[float]) -> Any:
        _, _, send_type = self._require_ros()
        msg = send_type()
        # ENCOS uses 1/3 for joint-space tracking (2/4 are Cartesian modes).
        msg.ctl_trans_mod = 1
        msg.runin_mod = 3
        msg.tor_feedforward_ctl = 0
        msg.if_use_ret = 0
        msg.if_use_temp = False
        msg.rigidity_ctl_mod_l = [1] * 7
        msg.rigidity_ctl_mod_r = [1] * 7
        msg.kp_trans_l = [0.0] * 7
        msg.kd_trans_l = [0.0] * 7
        msg.kp_trans_r = [0.0] * 7
        msg.kd_trans_r = [0.0] * 7
        msg.upper_monitor_time = 0.02
        msg.traj_time_l_arm = [0.05] * 7
        msg.traj_time_r_arm = [0.05] * 7
        msg.acc_rate_base_l_arm = [1.0] * 7
        msg.acc_rate_base_r_arm = [1.0] * 7
        msg.l_arm_if_use_stab = [False] * 7
        msg.r_arm_if_use_stab = [False] * 7
        msg.l_arm_start_stab_times = [0] * 7
        msg.r_arm_start_stab_times = [0] * 7
        msg.l_arm_des_stab_times = [0] * 7
        msg.r_arm_des_stab_times = [0] * 7
        msg.l_arm_if_jnt_stab = [0] * 7
        msg.r_arm_if_jnt_stab = [0] * 7
        msg.if_redefine_stab_jnt_ang = False
        msg.l_arm_if_use_jntlim = False
        msg.r_arm_if_use_jntlim = False
        msg.l_arm_jntlim_num = 0
        msg.r_arm_jntlim_num = 0
        msg.l_arm_limjnt_label = []
        msg.r_arm_limjnt_label = []
        msg.l_arm_limjnt_ang = []
        msg.r_arm_limjnt_ang = []
        msg.if_update_lim = False
        msg.l_arm_des_pos_descartes = [0.0] * 6
        msg.r_arm_des_pos_descartes = [0.0] * 6
        msg.l_arm_des_tor_descartes = [0.0] * 6
        msg.r_arm_des_tor_descartes = [0.0] * 6
        msg.l_arm_des_pos_jntspace = left
        msg.r_arm_des_pos_jntspace = right
        msg.l_arm_des_tor_jntspace = [0.0] * 7
        msg.r_arm_des_tor_jntspace = [0.0] * 7
        msg.l_arm_des_vel_jntspace = [0.0] * 7
        msg.r_arm_des_vel_jntspace = [0.0] * 7
        return msg

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.command_mode == "disabled":
            return dict(action)
        with self._lock:
            current = None if self._feedback is None else {key: value[:] for key, value in self._feedback.items()}
        if current is None:
            raise RuntimeError("No ENCOS feedback received; cannot safely preserve the inactive arm")
        left = _values(action, "left") if "left" in _sides(self.config.arm_mode) else current["left_pos"]
        right = _values(action, "right") if "right" in _sides(self.config.arm_mode) else current["right_pos"]
        if self.config.max_relative_target is not None and current is not None:
            limit = self.config.max_relative_target
            if "left" in _sides(self.config.arm_mode):
                left = [max(p - limit, min(p + limit, g)) for g, p in zip(left, current["left_pos"], strict=True)]
            if "right" in _sides(self.config.arm_mode):
                right = [max(p - limit, min(p + limit, g)) for g, p in zip(right, current["right_pos"], strict=True)]
        for side, values in (("left", left), ("right", right)):
            if side not in _sides(self.config.arm_mode):
                continue
            for index, value in enumerate(values, start=1):
                bounds = self.config.joint_limits.get(f"{side}_joint_{index}")
                if bounds is not None:
                    values[index - 1] = max(bounds[0], min(bounds[1], value))
        if self._publisher is None:
            raise RuntimeError("ENCOS command publisher is not initialized")
        self._publisher.publish(self._make_command(left, right))
        sent = {"left": left, "right": right}
        return {
            f"{side}_joint_{index + 1}.pos": sent[side][index]
            for side in _sides(self.config.arm_mode)
            for index in range(7)
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        for camera in self._cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self._node = self._publisher = self._subscription = None
        self._connected = False
