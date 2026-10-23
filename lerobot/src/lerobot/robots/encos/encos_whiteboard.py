from __future__ import annotations

import math
import threading
import time
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_encos_whiteboard import EncosWhiteboardConfig


class EncosWhiteboard(Robot):
    """ENCOS right-arm adapter using the existing feedback and VR delta topics."""

    config_class = EncosWhiteboardConfig
    name = "encos_whiteboard"

    def __init__(self, config: EncosWhiteboardConfig):
        super().__init__(config)
        self.config = config
        self._node = None
        self._publisher = None
        self._tcp_command_publisher = None
        self._orientation_publisher = None
        self._last_commanded_target: list[float] | None = None
        self._lock = threading.Lock()
        self._observation: dict[str, Any] | None = None
        self._last_observation = 0.0
        self._connected = False
        self._cameras = make_cameras_from_configs(config.cameras)
        # LeRobot recording/rollout code discovers camera streams through this
        # public attribute. Keep the private alias for the adapter internals.
        self.cameras = self._cameras

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for index in range(1, 8):
            features[f"right_joint_{index}.pos"] = float
        for index in range(1, 8):
            features[f"right_joint_{index}.vel"] = float
        for name in ("tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw"):
            features[name] = float
        for name, camera in self.config.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{name}_depth"] = (camera.height, camera.width, 1)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in (
            "tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw"
        )}

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
            from ctl_msgs_hw.msg import FkDataPub, MidNodeCtlMsgRet
            from geometry_msgs.msg import Vector3
            from whiteboard_msgs.msg import TcpCommand, TcpObservation
        except ImportError as exc:
            raise ImportError("ENCOS whiteboard adapter requires sourced ROS 2 ctl_msgs_hw and rclpy") from exc
        return rclpy, FkDataPub, MidNodeCtlMsgRet, Vector3, TcpCommand, TcpObservation

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        rclpy, fk_type, mid_type, vector_type, command_type, observation_type = self._require_ros()
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("lerobot_encos_whiteboard")
        self._node.create_subscription(fk_type, self.config.fk_feedback_topic, self._fk_callback, 20)
        self._node.create_subscription(mid_type, self.config.feedback_topic, self._mid_callback, 20)
        self._node.create_subscription(observation_type, "/wiping/tcp_observation", self._observation_callback, 20)
        if self.config.command_mode == "direct":
            self._publisher = self._node.create_publisher(vector_type, self.config.delta_topic, 20)
            self._orientation_publisher = self._node.create_publisher(
                vector_type, self.config.euler_delta_topic, 20
            )
            self._tcp_command_publisher = self._node.create_publisher(command_type, "/wiping/tcp_command", 20)
        for camera in self._cameras.values():
            camera.connect()
        self._connected = True
        deadline = time.monotonic() + self.config.startup_timeout_s
        while self._observation is None and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if self._observation is None:
            self.disconnect()
            raise RuntimeError(f"No ENCOS feedback received on {self.config.fk_feedback_topic}")

    def _update(self, **values):
        with self._lock:
            if self._observation is None:
                self._observation = {}
            self._observation.update(values)
            required = [f"right_joint_{i}.pos" for i in range(1, 8)]
            required += [f"right_joint_{i}.vel" for i in range(1, 8)]
            required += ["tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw"]
            if all(key in self._observation for key in required):
                self._last_observation = time.monotonic()

    def _fk_callback(self, msg: Any) -> None:
        self._update(
            tcp_x=float(msg.x_r), tcp_y=float(msg.y_r), tcp_z=float(msg.z_r),
            tcp_roll=float(msg.roll_r), tcp_pitch=float(msg.pitch_r), tcp_yaw=float(msg.yaw_r),
        )

    def _mid_callback(self, msg: Any) -> None:
        q, dq = list(msg.r_arm_pos_fdback), list(msg.r_arm_spd_fdback)
        if len(q) < 7 or len(dq) < 7:
            return
        self._update(**{
            **{f"right_joint_{i + 1}.pos": float(q[i]) for i in range(7)},
            **{f"right_joint_{i + 1}.vel": float(dq[i]) for i in range(7)},
        })

    def _observation_callback(self, msg: Any) -> None:
        """Consume the controller's synchronized TCP/admittance observation when available."""
        q, dq = list(msg.q), list(msg.dq)
        if len(q) < 7 or len(dq) < 7:
            return
        self._update(
            **{
                **{f"right_joint_{i + 1}.pos": float(q[i]) for i in range(7)},
                **{f"right_joint_{i + 1}.vel": float(dq[i]) for i in range(7)},
                **dict(zip(("tcp_x", "tcp_y", "tcp_z"), map(float, msg.tcp_xyz_base_m), strict=True)),
                **dict(zip(("tcp_roll", "tcp_pitch", "tcp_yaw"), map(float, msg.tcp_rpy_base_rad), strict=True)),
            }
        )

    def _spin_once(self) -> None:
        if self._node is not None:
            import rclpy
            rclpy.spin_once(self._node, timeout_sec=0.0)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._spin_once()
        with self._lock:
            observation = None if self._observation is None else dict(self._observation)
            age = time.monotonic() - self._last_observation
        if observation is None or not all(name in observation for name in self.observation_features if isinstance(self.observation_features[name], type)):
            raise RuntimeError("ENCOS feedback is incomplete")
        if age > self.config.command_timeout_s:
            raise RuntimeError(f"ENCOS feedback is stale ({age:.3f}s)")
        for name, camera in self.config.cameras.items():
            cam = self._cameras[name]
            if getattr(camera, "use_rgb", True):
                observation[name] = cam.read_latest()
            if getattr(camera, "use_depth", False):
                observation[f"{name}_depth"] = cam.read_latest_depth()
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        names = ("tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw")
        target = [float(action[name]) for name in names]
        if self.config.command_mode == "disabled":
            return dict(zip(names, target, strict=True))
        with self._lock:
            current = None if self._observation is None else [self._observation.get(name) for name in names]
            previous = self._last_commanded_target
        if current is None or any(value is None for value in current) or self._publisher is None:
            raise RuntimeError("No TCP feedback or delta publisher")
        if previous is None:
            previous = current
        delta = [target[i] - float(previous[i]) for i in range(3)]
        angle_delta = [self._wrap_angle(target[i] - float(previous[i])) for i in range(3, 6)]
        if self.config.max_relative_target_m is not None:
            limit = self.config.max_relative_target_m
            delta = [max(-limit, min(limit, value)) for value in delta]
        if self.config.max_relative_target_rad is not None:
            limit = self.config.max_relative_target_rad
            angle_delta = [max(-limit, min(limit, value)) for value in angle_delta]
        # robot_control_node multiplies incoming VR deltas by 0.75 internally.
        scale = self.config.robot_control_delta_scale
        if scale <= 0:
            raise RuntimeError("robot_control_delta_scale must be positive")
        _, _, _, vector_type, command_type, _ = self._require_ros()
        tcp_command = command_type()
        tcp_command.tcp_x_m = target[0]
        tcp_command.tcp_y_m = target[1]
        tcp_command.tcp_z_m = target[2]
        tcp_command.tcp_roll_rad = target[3]
        tcp_command.tcp_pitch_rad = target[4]
        tcp_command.tcp_yaw_rad = target[5]
        if self._tcp_command_publisher is not None:
            self._tcp_command_publisher.publish(tcp_command)
        command = vector_type()
        command.x, command.y, command.z = [value / scale for value in delta]
        self._publisher.publish(command)
        euler_command = vector_type()
        # VR euler interface is [yaw, pitch, roll]; state/action are RPY.
        euler_command.x = angle_delta[2] / scale
        euler_command.y = angle_delta[1] / scale
        euler_command.z = angle_delta[0] / scale
        self._orientation_publisher.publish(euler_command)
        with self._lock:
            # Store the target represented by the command actually sent. If a
            # policy jump is clipped, subsequent calls continue stepping toward
            # the original ACT target instead of stopping after one clipped step.
            self._last_commanded_target = [
                float(previous[i]) + delta[i] for i in range(3)
            ] + [
                float(previous[i + 3]) + angle_delta[i] for i in range(3)
            ]
        return dict(zip(names, target, strict=True))

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        for camera in self._cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self._node = None
        self._publisher = None
        self._tcp_command_publisher = None
        self._orientation_publisher = None
        self._last_commanded_target = None
        self._connected = False
