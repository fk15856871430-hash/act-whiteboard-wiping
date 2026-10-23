from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


def whiteboard_cameras_config() -> dict[str, CameraConfig]:
    """Default RGB inputs used by the whiteboard ACT policy and recorder."""
    common = {"fps": 30, "width": 640, "height": 480}
    return {
        "head": RealSenseCameraConfig(serial_number_or_name="261222075640", **common),
        "wrist": RealSenseCameraConfig(serial_number_or_name="260422274993", **common),
    }


@RobotConfig.register_subclass("encos_whiteboard")
@dataclass(kw_only=True)
class EncosWhiteboardConfig(RobotConfig):
    """Adapter for the existing ENCOS five-process teleoperation chain."""

    fk_feedback_topic: str = "/dexbotCtl/DescPub/Fk_ret_data_pub"
    feedback_topic: str = "/dexbotCtl/midMonitor_msg_send"
    delta_topic: str = "/vr/right_arm_delta"
    euler_delta_topic: str = "/vr/right_arm_euler_delta"
    command_mode: str = "disabled"
    startup_timeout_s: float = 5.0
    command_timeout_s: float = 0.5
    max_relative_target_m: float | None = None
    max_relative_target_rad: float | None = 0.08
    robot_control_delta_scale: float = 0.75
    cameras: dict[str, CameraConfig] = field(default_factory=whiteboard_cameras_config)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.command_mode not in {"direct", "disabled"}:
            raise ValueError("command_mode must be 'direct' or 'disabled'")
        if self.startup_timeout_s <= 0 or self.command_timeout_s <= 0:
            raise ValueError("startup_timeout_s and command_timeout_s must be positive")
        if self.max_relative_target_m is not None and self.max_relative_target_m <= 0:
            raise ValueError("max_relative_target_m must be positive")
        if self.max_relative_target_rad is not None and self.max_relative_target_rad <= 0:
            raise ValueError("max_relative_target_rad must be positive")
        if self.robot_control_delta_scale <= 0:
            raise ValueError("robot_control_delta_scale must be positive")
