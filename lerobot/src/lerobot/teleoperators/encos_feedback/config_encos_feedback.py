from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("encos_feedback")
@dataclass(kw_only=True)
class EncosFeedbackConfig(TeleoperatorConfig):
    feedback_topic: str = "/dexbotCtl/midMonitor_msg_send"
    stale_after_s: float = 0.5
    arm_mode: str = "both"

    def __post_init__(self) -> None:
        if self.stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        if self.arm_mode not in {"left", "right", "both"}:
            raise ValueError("arm_mode must be 'left', 'right', or 'both'")
