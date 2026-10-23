"""Serve a LeRobot 0.5.x ACT or Diffusion checkpoint over WebSocket."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

from policy_servers.websocket_server import WebsocketPolicyServer


IMAGE_KEYS = ("observation.images.primary", "observation.images.wrist")
STATE_KEY = "observation.state"


def _image_to_tensor(value: np.ndarray) -> torch.Tensor:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an HWC RGB(A) image, got {image.shape}")
    image = np.ascontiguousarray(image[..., :3].transpose(2, 0, 1))
    tensor = torch.from_numpy(image).to(torch.float32)
    if image.dtype == np.uint8 or tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    return tensor


class LeRobotPolicyAdapter:
    def __init__(self, model_path: str, dataset_root: str, repo_id: str, device: str) -> None:
        model_dir = Path(model_path).expanduser().resolve()
        config = PreTrainedConfig.from_pretrained(model_dir)
        config.pretrained_path = model_dir
        config.device = device

        dataset_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
        self.policy = make_policy(config, ds_meta=dataset_meta).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(model_dir),
        )
        self.policy_type = config.type
        self.state_dim = config.input_features[STATE_KEY].shape[0]
        self.action_dim = config.output_features["action"].shape[0]
        self.parameter_count = sum(parameter.numel() for parameter in self.policy.parameters())
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def infer(self, observation: dict) -> dict:
        if observation.get("reset", False):
            self.reset()
            return {"action": None}

        state = np.asarray(observation[STATE_KEY], dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape ({self.state_dim},), got {state.shape}")

        policy_observation = {
            IMAGE_KEYS[0]: _image_to_tensor(observation[IMAGE_KEYS[0]]),
            IMAGE_KEYS[1]: _image_to_tensor(observation[IMAGE_KEYS[1]]),
            STATE_KEY: torch.from_numpy(state),
        }
        batch = self.preprocessor(policy_observation)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        return {"action": action.detach().cpu().numpy()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--repo_id", default="wipe_board_lerobot_locked")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    adapter = LeRobotPolicyAdapter(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        device=args.device,
    )
    metadata = {
        "policy_type": adapter.policy_type,
        "state_dim": adapter.state_dim,
        "action_dim": adapter.action_dim,
        "parameter_count": adapter.parameter_count,
        "protocol": "wipe_board_v1",
    }
    WebsocketPolicyServer(adapter, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
