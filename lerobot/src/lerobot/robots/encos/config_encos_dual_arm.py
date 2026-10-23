from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("encos_dual_arm")
@dataclass(kw_only=True)
class EncosDualArmConfig(RobotConfig):
    """Configuration for the ENCOS 7+7 DOF ROS 2 arm bridge.

    ``command_mode=disabled`` is intended for recording while the existing
    ``robot_control_hw`` VR node owns the command topic.  In ``direct`` mode
    this bridge publishes joint targets directly to ``dual_arm_control_hw``.
    """

    feedback_topic: str = "/dexbotCtl/midMonitor_msg_send"
    command_topic: str = "/dexbotCtl/upperMonitor_msg_send"
    command_mode: str = "direct"
    arm_mode: str = "both"
    command_timeout_s: float = 0.5
    startup_timeout_s: float = 5.0
    max_relative_target: float | None = None
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.command_mode not in {"direct", "disabled"}:
            raise ValueError("command_mode must be 'direct' or 'disabled'")
        if self.arm_mode not in {"left", "right", "both"}:
            raise ValueError("arm_mode must be 'left', 'right', or 'both'")
        if self.command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive")
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.max_relative_target is not None and self.max_relative_target <= 0:
            raise ValueError("max_relative_target must be positive")
        unknown_limits = set(self.joint_limits) - {
            f"{side}_joint_{index}" for side in ("left", "right") for index in range(1, 8)
        }
        if unknown_limits:
            raise ValueError(f"Unknown ENCOS joint limit names: {sorted(unknown_limits)}")
        for name, bounds in self.joint_limits.items():
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid joint limit for {name}: {bounds}")
