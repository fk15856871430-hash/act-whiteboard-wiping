"""Convert the whiteboard HDF5 demonstrations to a finalized LeRobot v3 dataset."""

import argparse
import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


cv2.setNumThreads(1)


def _decode_image(encoded: np.ndarray, size: int, crop_type: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image")
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = width - crop_size if crop_type == "right" else (width - crop_size) // 2
    image = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_pair(item: tuple[np.ndarray, np.ndarray, int]) -> tuple[np.ndarray, np.ndarray]:
    primary, wrist, size = item
    return _decode_image(primary, size, "right"), _decode_image(wrist, size, "center")


def _features(size: int) -> dict:
    return {
        "observation.images.primary": {
            "dtype": "image",
            "shape": (size, size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (size, size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (20,)},
        "action": {"dtype": "float32", "shape": (14,)},
    }


def convert_dataset(
    input_dir: str,
    output_path: str,
    repo_id: str,
    episode_count: int,
    image_size: int,
    fps: int,
    workers: int,
    overwrite: bool,
) -> None:
    paths = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    if len(paths) < episode_count:
        raise ValueError(f"Expected at least {episode_count} HDF5 files, found {len(paths)}")
    # The official DP conversion used this same sorted 777-file snapshot. Newer files are excluded.
    paths = paths[:episode_count]

    output = Path(output_path).expanduser().resolve()
    partial = output.with_name(f"{output.name}.partial")
    for path in (output, partial):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=partial,
        fps=fps,
        robot_type="dual_arm_ur5",
        features=_features(image_size),
        use_videos=False,
        image_writer_threads=max(1, workers),
        metadata_buffer_size=10,
    )

    total_frames = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for path in tqdm(paths, desc="Converting ACT episodes"):
                with h5py.File(path, "r") as episode:
                    required = (
                        "obs/image_primary",
                        "obs/image_wrist",
                        "obs/pose7",
                        "obs/joint_pos",
                        "obs/wrench",
                        "action/target_pose",
                    )
                    lengths = []
                    for key in required:
                        if key not in episode:
                            raise ValueError(f"{path}: missing {key}")
                        lengths.append(len(episode[key]))
                    if len(set(lengths)) != 1:
                        raise ValueError(f"{path}: inconsistent feature lengths {dict(zip(required, lengths))}")

                    length = lengths[0]
                    primary = episode["obs/image_primary"][:]
                    wrist = episode["obs/image_wrist"][:]
                    pose = np.asarray(episode["obs/pose7"][:], dtype=np.float32)
                    joints = np.asarray(episode["obs/joint_pos"][:], dtype=np.float32)
                    wrench = np.asarray(episode["obs/wrench"][:], dtype=np.float32)
                    source_action = np.asarray(episode["action/target_pose"][:], dtype=np.float32)
                    instruction = episode.get("language_instruction")
                    if instruction is None:
                        task = "wipe the whiteboard clean"
                    else:
                        task = instruction[()]
                        task = task.decode("utf-8") if isinstance(task, bytes) else str(task)

                state = np.concatenate((pose, joints, wrench), axis=-1).astype(np.float32)
                if state.shape != (length, 20):
                    raise ValueError(f"{path}: expected state shape ({length}, 20), got {state.shape}")
                action = np.zeros((length, 14), dtype=np.float32)
                action[:, : min(7, source_action.shape[1])] = source_action[:, :7]
                action[:, 3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

                pairs = ((primary[i], wrist[i], image_size) for i in range(length))
                for index, (primary_rgb, wrist_rgb) in enumerate(executor.map(_decode_pair, pairs)):
                    dataset.add_frame(
                        {
                            "observation.images.primary": primary_rgb,
                            "observation.images.wrist": wrist_rgb,
                            "observation.state": state[index],
                            "action": action[index],
                            "task": task,
                        }
                    )
                dataset.save_episode(parallel_encoding=False)
                total_frames += length
    finally:
        # LeRobot parquet writers do not write valid footers until finalize() is called.
        dataset.finalize()

    partial.rename(output)
    print(f"Created {output}: {len(paths)} episodes, {total_frames} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="/home/user/IsaacLab/vla_dataset")
    parser.add_argument(
        "--output_path", default="/mnt/bigdrive/datasets/wipe_board_act_lerobot_v3"
    )
    parser.add_argument("--repo_id", default="wipe_board_act_lerobot_v3")
    parser.add_argument("--episode_count", type=int, default=777)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_dataset(**vars(args))


if __name__ == "__main__":
    main()
