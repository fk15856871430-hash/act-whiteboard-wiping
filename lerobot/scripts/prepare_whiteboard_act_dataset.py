#!/usr/bin/env python3
"""Prepare a clean, deterministic whiteboard ACT dataset without re-encoding video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path("/home/user/whiteboard_wipe_20260916_17_rotvec_v2")
DEFAULT_OUTPUT = Path("/home/user/whiteboard_wipe_20260916_17_rotvec_v3_clean")
TRIM_LENGTHS = {99: 3227, 100: 909}
N_EVAL_EPISODES = 12
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_parquet_tree(path: Path) -> tuple[pa.Table, pa.Schema]:
    files = sorted(path.glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    tables = [pq.read_table(file) for file in files]
    return pa.concat_tables(tables), tables[0].schema


def array_column(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(np.float64, copy=False)


def statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    result = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.asarray([len(values)], dtype=np.int64),
    }
    for name, q in QUANTILES.items():
        result[name] = np.quantile(values, q, axis=0)
    return result


def frame_feature_values(frames: pd.DataFrame, feature: str) -> np.ndarray:
    if feature in {"action", "observation.state"}:
        return array_column(frames[feature])
    return frames[feature].to_numpy(dtype=np.float64)[:, None]


def choose_eval_episodes(episode_frames: dict[int, pd.DataFrame]) -> list[int]:
    """Choose one representative episode from each chronological stratum."""
    episode_ids = sorted(episode_frames)
    descriptors = []
    for episode_id in episode_ids:
        frames = episode_frames[episode_id]
        action = array_column(frames["action"])
        tcp_state = array_column(frames["observation.state"])[:, -6:]
        descriptors.append(
            np.concatenate(
                [
                    [np.log1p(len(frames))],
                    action.mean(axis=0),
                    action.std(axis=0),
                    np.quantile(action, 0.1, axis=0),
                    np.quantile(action, 0.9, axis=0),
                    tcp_state.mean(axis=0),
                    tcp_state.std(axis=0),
                ]
            )
        )

    descriptor = np.stack(descriptors)
    median = np.median(descriptor, axis=0)
    scale = np.median(np.abs(descriptor - median), axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (descriptor - median) / scale

    selected: list[int] = []
    for positions in np.array_split(np.arange(len(episode_ids)), N_EVAL_EPISODES):
        center = np.median(normalized[positions], axis=0)
        temporal_center = positions.mean()
        candidates = [pos for pos in positions if episode_ids[pos] not in TRIM_LENGTHS]
        if not candidates:
            candidates = positions.tolist()
        scores = [
            np.linalg.norm(normalized[pos] - center) + 0.05 * abs(pos - temporal_center)
            for pos in candidates
        ]
        selected.append(episode_ids[candidates[int(np.argmin(scores))]])
    return sorted(selected)


def link_videos(source: Path, output: Path) -> None:
    for source_file in sorted((source / "videos").rglob("*.mp4")):
        relative = source_file.relative_to(source)
        output_file = output / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_file, output_file)
        except OSError:
            output_file.symlink_to(source_file)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    with (source / "meta/info.json").open() as file:
        info = json.load(file)
    fps = int(info["fps"])
    expected_episodes = int(info["total_episodes"])
    if expected_episodes != 111:
        raise ValueError(f"Expected 111 episodes, found {expected_episodes}")

    data_table, data_schema = load_parquet_tree(source / "data")
    episode_table, episode_schema = load_parquet_tree(source / "meta/episodes")
    source_data = data_table.to_pandas()
    source_episodes = episode_table.to_pandas().set_index("episode_index", drop=False)

    episode_frames: dict[int, pd.DataFrame] = {}
    for episode_id in range(expected_episodes):
        frames = source_data[source_data["episode_index"] == episode_id].sort_values("frame_index")
        expected_length = int(source_episodes.at[episode_id, "length"])
        if len(frames) != expected_length:
            raise ValueError(
                f"Episode {episode_id}: metadata length {expected_length}, data length {len(frames)}"
            )
        keep_length = TRIM_LENGTHS.get(episode_id, expected_length)
        if keep_length <= 0 or keep_length > expected_length:
            raise ValueError(f"Invalid trim length for episode {episode_id}: {keep_length}")
        episode_frames[episode_id] = frames.iloc[:keep_length].copy().reset_index(drop=True)

    eval_old_ids = choose_eval_episodes(episode_frames)
    eval_set = set(eval_old_ids)
    train_old_ids = [episode_id for episode_id in range(expected_episodes) if episode_id not in eval_set]
    ordered_old_ids = train_old_ids + eval_old_ids
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(ordered_old_ids)}

    output.mkdir(parents=True)
    try:
        link_videos(source, output)
        (output / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "meta/tasks.parquet", output / "meta/tasks.parquet")

        rewritten_frames = []
        rewritten_episode_rows = []
        global_index = 0
        tabular_features = [
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]

        for old_id in ordered_old_ids:
            new_id = old_to_new[old_id]
            frames = episode_frames[old_id]
            length = len(frames)
            frames["episode_index"] = new_id
            frames["frame_index"] = np.arange(length, dtype=np.int64)
            frames["timestamp"] = (np.arange(length, dtype=np.float32) / fps).astype(np.float32)
            frames["index"] = np.arange(global_index, global_index + length, dtype=np.int64)

            row = source_episodes.loc[old_id].copy()
            row["episode_index"] = new_id
            row["length"] = length
            row["data/chunk_index"] = 0
            row["data/file_index"] = 0
            row["dataset_from_index"] = global_index
            row["dataset_to_index"] = global_index + length
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            if old_id in TRIM_LENGTHS:
                for camera in ("observation.images.head", "observation.images.wrist"):
                    start_key = f"videos/{camera}/from_timestamp"
                    end_key = f"videos/{camera}/to_timestamp"
                    row[end_key] = float(row[start_key]) + length / fps

            for feature in tabular_features:
                feature_stats = statistics(frame_feature_values(frames, feature))
                for stat_name, value in feature_stats.items():
                    row[f"stats/{feature}/{stat_name}"] = value.tolist()

            rewritten_frames.append(frames)
            rewritten_episode_rows.append(row)
            global_index += length

        data = pd.concat(rewritten_frames, ignore_index=True)
        episodes = pd.DataFrame(rewritten_episode_rows).reset_index(drop=True)
        episodes = episodes[episode_schema.names]

        data_path = output / "data/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(data, schema=data_schema, preserve_index=False),
            data_path,
            compression="snappy",
        )
        episodes_path = output / "meta/episodes/chunk-000/file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(episodes, schema=episode_schema, preserve_index=False),
            episodes_path,
            compression="snappy",
        )

        info["total_frames"] = len(data)
        info["total_episodes"] = len(episodes)
        info["splits"] = {"train": f"0:{len(episodes)}"}
        with (output / "meta/info.json").open("w") as file:
            json.dump(info, file, indent=4)
            file.write("\n")

        with (source / "meta/stats.json").open() as file:
            global_stats = json.load(file)
        for feature in tabular_features:
            global_stats[feature] = statistics(frame_feature_values(data, feature))
        with (output / "meta/stats.json").open("w") as file:
            json.dump(json_ready(global_stats), file, indent=4)
            file.write("\n")

        manifest = {
            "source": str(source),
            "trim_lengths": {str(key): value for key, value in TRIM_LENGTHS.items()},
            "removed_frames": {
                str(key): int(source_episodes.at[key, "length"]) - value
                for key, value in TRIM_LENGTHS.items()
            },
            "eval_old_episode_ids": eval_old_ids,
            "eval_new_episode_ids": list(range(len(train_old_ids), len(ordered_old_ids))),
            "old_to_new_episode_id": {str(key): value for key, value in old_to_new.items()},
            "video_storage": "hardlinks, with symlink fallback",
        }
        with (output / "meta/whiteboard_preparation.json").open("w") as file:
            json.dump(manifest, file, indent=4)
            file.write("\n")

        # Import only after writing so this script also gives a clear structural error
        # when run from the wrong environment.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset("local/whiteboard_wipe_rotvec_clean", root=output)
        if dataset.num_frames != len(data) or dataset.num_episodes != len(episodes):
            raise RuntimeError("LeRobot dataset validation returned inconsistent sizes")
        for new_id in manifest["eval_new_episode_ids"][:1]:
            start = int(episodes.loc[new_id, "dataset_from_index"])
            _ = dataset[start]

        print(f"Prepared dataset: {output}")
        print(f"Frames: {len(source_data)} -> {len(data)} (removed {len(source_data) - len(data)})")
        print(f"Train/eval episodes: {len(train_old_ids)}/{len(eval_old_ids)}")
        print(f"Held-out original episode IDs: {eval_old_ids}")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
