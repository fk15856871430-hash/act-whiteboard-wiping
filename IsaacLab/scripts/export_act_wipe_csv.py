#!/usr/bin/env python3
"""Export a schema-v5 ACT whiteboard trajectory JSON to a flat CSV table."""

import argparse
import csv
import json
from pathlib import Path


def flatten_sample(sample, joint_names):
    row = {"time_s": sample["time_s"]}

    for matrix_name, matrix in (("world_T_base", sample["world_T_base"]),):
        for r in range(4):
            for c in range(4):
                row[f"{matrix_name}_{r}{c}"] = matrix[r][c]

    for pose_name in ("tcp_pose_base", "tcp_pose_world", "command_tcp_pose_base"):
        pose = sample[pose_name]
        for axis, value in zip(("x", "y", "z"), pose["position_m"]):
            row[f"{pose_name}_position_{axis}_m"] = value
        for component, value in zip(("w", "x", "y", "z"), pose["quaternion_wxyz"]):
            row[f"{pose_name}_quaternion_{component}"] = value
        for r in range(4):
            for c in range(4):
                row[f"{pose_name}_matrix_{r}{c}"] = pose["matrix"][r][c]

    vector_fields = {
        "joint_position_rad": "joint_position_rad",
        "joint_velocity_rad_s": "joint_velocity_rad_s",
        "joint_acc_rad_s2": "joint_acc_rad_s2",
        "joint_torque_nm": "joint_torque_nm",
        "joint_jerk_rad_s3": "joint_jerk_rad_s3",
    }
    for field, prefix in vector_fields.items():
        for index, value in enumerate(sample[field]):
            joint = joint_names[index] if index < len(joint_names) else f"joint_{index + 1}"
            row[f"{prefix}_{joint}"] = value

    for prefix, values in (
        ("contact_force_vector_n", sample["contact_force_vector_n"]),
        ("tcp_linear_velocity_base_m_s", sample["tcp_linear_velocity_base_m_s"]),
        ("tcp_linear_acceleration_base_m_s2", sample["tcp_linear_acceleration_base_m_s2"]),
        ("tcp_linear_jerk_base_m_s3", sample["tcp_linear_jerk_base_m_s3"]),
    ):
        for axis, value in zip(("x", "y", "z"), values):
            row[f"{prefix}_{axis}"] = value

    for field in (
        "contact_force_n",
        "remaining_stroke_points",
        "erased_ratio",
        "policy_round_trip_ms",
        "server_infer_ms",
    ):
        row[field] = sample[field]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="trajectory_results/act_wipe_episode.json")
    parser.add_argument("output", nargs="?", default="trajectory_results/act_wipe_episode.csv")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    with input_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema_version") != 5:
        raise ValueError(f"Expected schema-v5 trajectory JSON, got {record.get('schema_version')}")

    joint_names = record.get("joint_names", [])
    rows = [flatten_sample(sample, joint_names) for sample in record["samples"]]
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples and {len(fieldnames)} columns to {output_path}")


if __name__ == "__main__":
    main()
