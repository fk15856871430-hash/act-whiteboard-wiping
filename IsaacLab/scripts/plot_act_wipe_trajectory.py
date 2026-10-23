#!/usr/bin/env python3
"""Plot one normal ACT whiteboard-wiping episode (schema-v5 JSON)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d")


def load_episode(path):
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema_version") != 5:
        raise ValueError(f"Expected schema-v5 trajectory JSON, got {record.get('schema_version')}")
    samples = record.get("samples", [])
    if not samples:
        raise ValueError("Trajectory contains no samples")

    def values(key):
        return np.asarray([sample[key] for sample in samples], dtype=np.float64)

    tcp_position = np.asarray(
        [sample["tcp_pose_base"]["position_m"] for sample in samples], dtype=np.float64
    )
    tcp_command = np.asarray(
        [sample["command_tcp_pose_base"]["position_m"] for sample in samples], dtype=np.float64
    )
    quaternion = np.asarray(
        [sample["tcp_pose_base"]["quaternion_wxyz"] for sample in samples], dtype=np.float64
    )
    return record, {
        "time": values("time_s"),
        "tcp_position": tcp_position,
        "tcp_command": tcp_command,
        "quaternion": quaternion,
        "joint_position": values("joint_position_rad"),
        "joint_velocity": values("joint_velocity_rad_s"),
        "joint_torque": values("joint_torque_nm"),
        "joint_acceleration": values("joint_acc_rad_s2"),
        "joint_jerk": values("joint_jerk_rad_s3"),
        "tcp_velocity": values("tcp_linear_velocity_base_m_s"),
        "tcp_acceleration": values("tcp_linear_acceleration_base_m_s2"),
        "tcp_jerk": values("tcp_linear_jerk_base_m_s3"),
        "contact_force": values("contact_force_n"),
        "erase_progress": values("erased_ratio") * 100.0,
        "remaining_points": values("remaining_stroke_points"),
    }


def save_figure(figure, output_dir, name):
    figure.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def line_plot(time, data, labels, ylabel, title, output_dir, filename, ylim=None):
    data = np.asarray(data)
    figure, axes = plt.subplots(
        data.shape[1], 1, figsize=(11, max(4.0, 1.8 * data.shape[1])),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    axes = axes[:, 0]
    for index, axis in enumerate(axes):
        axis.plot(time, data[:, index], color=COLORS[index % len(COLORS)], linewidth=1.25)
        axis.set_ylabel(labels[index])
        axis.grid(True, alpha=0.28)
        if ylim is not None:
            axis.set_ylim(*ylim)
    axes[0].set_title(title, loc="left", fontweight="bold")
    axes[-1].set_xlabel("time (s)")
    save_figure(figure, output_dir, filename)


def plot_tcp_3d(time, tcp, command, output_dir):
    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    progress = (time - time[0]) / max(time[-1] - time[0], 1e-9)
    axis.scatter(tcp[0, 0], tcp[0, 1], tcp[0, 2], color="#16a34a", s=45, label="start")
    axis.scatter(tcp[-1, 0], tcp[-1, 1], tcp[-1, 2], color="#dc2626", s=45, label="end")
    axis.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color="#2563eb", linewidth=1.8, label="actual TCP")
    axis.plot(command[:, 0], command[:, 1], command[:, 2], "--", color="#64748b", linewidth=1.0, label="command TCP")
    axis.set_xlabel("X in robot base (m)")
    axis.set_ylabel("Y in robot base (m)")
    axis.set_zlabel("Z in robot base (m)")
    axis.set_title("ACT TCP wiping trajectory", loc="left", fontweight="bold")
    axis.legend(loc="best")
    save_figure(figure, output_dir, "tcp_trajectory_3d.png")

    figure, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.plot(time, tcp[:, index], label="actual", color="#2563eb", linewidth=1.3)
        axis.plot(time, command[:, index], label="command", color="#64748b", linestyle="--", linewidth=1.0)
        axis.set_ylabel(("X", "Y", "Z")[index] + " (m)")
        axis.grid(True, alpha=0.28)
    axes[0].set_title("TCP position in robot base", loc="left", fontweight="bold")
    axes[0].legend()
    axes[-1].set_xlabel("time (s)")
    save_figure(figure, output_dir, "tcp_position_base.png")


def plot_contact_and_progress(time, contact, progress, remaining, output_dir):
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.plot(time, contact, color="#dc2626", linewidth=1.3, label="contact force")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("contact force (N)", color="#dc2626")
    axis.tick_params(axis="y", labelcolor="#dc2626")
    axis.grid(True, alpha=0.28)
    right = axis.twinx()
    right.plot(time, progress, color="#059669", linewidth=1.5, label="erased ratio")
    right.set_ylabel("erased ratio (%)", color="#059669")
    right.set_ylim(0, 105)
    right.tick_params(axis="y", labelcolor="#059669")
    axis.set_title("Contact force and wiping progress", loc="left", fontweight="bold")
    lines = axis.lines + right.lines
    axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    save_figure(figure, output_dir, "contact_force_and_progress.png")

    figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    axis.plot(time, remaining, color="#7c3aed", linewidth=1.3)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("remaining stroke points")
    axis.set_title("Remaining whiteboard stroke points", loc="left", fontweight="bold")
    axis.grid(True, alpha=0.28)
    save_figure(figure, output_dir, "remaining_stroke_points.png")


def plot_dashboard(time, data, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col", constrained_layout=True)
    axes[0, 0].plot(time, data["tcp_position"])
    axes[0, 0].set_title("TCP position (base)")
    axes[0, 0].set_ylabel("m")
    axes[0, 0].legend(("X", "Y", "Z"), fontsize=8)
    axes[0, 1].plot(time, data["joint_position"])
    axes[0, 1].set_title("Joint angles")
    axes[0, 1].set_ylabel("rad")
    axes[0, 1].legend(tuple(f"J{i}" for i in range(1, 8)), fontsize=7, ncol=2)
    axes[0, 2].plot(time, data["joint_torque"])
    axes[0, 2].set_title("Joint torque")
    axes[0, 2].set_ylabel("N m")
    axes[0, 2].legend(tuple(f"J{i}" for i in range(1, 8)), fontsize=7, ncol=2)
    axes[1, 0].plot(time, np.linalg.norm(data["tcp_velocity"], axis=1), color="#2563eb")
    axes[1, 0].set_title("TCP linear speed")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 1].plot(time, data["contact_force"], color="#dc2626")
    axes[1, 1].set_title("Contact force")
    axes[1, 1].set_ylabel("N")
    axes[1, 2].plot(time, data["erase_progress"], color="#059669")
    axes[1, 2].set_title("Erased ratio")
    axes[1, 2].set_ylabel("%")
    axes[1, 2].set_ylim(0, 105)
    for row in axes:
        for axis in row:
            axis.grid(True, alpha=0.25)
            axis.set_xlabel("time (s)")
    figure.suptitle("Normal ACT whiteboard-wiping episode", fontsize=15, fontweight="bold")
    save_figure(figure, output_dir, "act_wipe_dashboard.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="trajectory_results/act_wipe_episode.json")
    parser.add_argument("--output-dir", default="trajectory_results/act_wipe_episode_curves")
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    record, data = load_episode(input_path)
    joint_labels = tuple(f"J{i}" for i in range(1, data["joint_position"].shape[1] + 1))
    tcp_labels = ("X", "Y", "Z")
    quaternion_labels = ("qw", "qx", "qy", "qz")
    line_plot(data["time"], data["quaternion"], quaternion_labels, "value", "TCP orientation (quaternion wxyz)", output_dir, "tcp_orientation_quaternion.png")
    line_plot(data["time"], data["joint_position"], joint_labels, "rad", "Joint angles", output_dir, "joint_position.png")
    line_plot(data["time"], data["joint_velocity"], joint_labels, "rad/s", "Joint velocities", output_dir, "joint_velocity.png")
    line_plot(data["time"], data["joint_torque"], joint_labels, "N m", "Joint torques", output_dir, "joint_torque.png")
    line_plot(data["time"], data["joint_acceleration"], joint_labels, "rad/s^2", "Joint accelerations", output_dir, "joint_acceleration.png")
    line_plot(data["time"], data["joint_jerk"], joint_labels, "rad/s^3", "Joint jerk", output_dir, "joint_jerk.png")
    line_plot(data["time"], data["tcp_velocity"], tcp_labels, "m/s", "TCP linear velocity in robot base", output_dir, "tcp_linear_velocity_base.png")
    line_plot(data["time"], data["tcp_acceleration"], tcp_labels, "m/s^2", "TCP linear acceleration in robot base", output_dir, "tcp_linear_acceleration_base.png")
    line_plot(data["time"], data["tcp_jerk"], tcp_labels, "m/s^3", "TCP linear jerk in robot base", output_dir, "tcp_linear_jerk_base.png")
    plot_tcp_3d(data["time"], data["tcp_position"], data["tcp_command"], output_dir)
    plot_contact_and_progress(data["time"], data["contact_force"], data["erase_progress"], data["remaining_points"], output_dir)
    plot_dashboard(data["time"], data, output_dir)
    print(f"Input: {input_path}")
    print(f"Samples: {len(data['time'])} at {record['sampling']['frequency_hz']} Hz")
    print(f"Wrote curves to: {output_dir}")


if __name__ == "__main__":
    main()
