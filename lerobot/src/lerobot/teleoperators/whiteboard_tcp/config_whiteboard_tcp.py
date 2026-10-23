from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("whiteboard_tcp")
@dataclass(kw_only=True)
class WhiteboardTcpConfig(TeleoperatorConfig):
    command_topic: str = "/wiping/tcp_command"
    stale_after_s: float = 0.5

    def __post_init__(self) -> None:
        if self.stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
