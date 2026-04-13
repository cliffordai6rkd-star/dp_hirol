from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hirol_reader import HiROLEpisodeReader


def resolve_episode_dirs(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if (root / "data.json").is_file():
        return [root]
    episode_dirs = HiROLEpisodeReader.list_episode_dirs(root)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found under: {root}")
    return episode_dirs


def iter_ee_xyz_and_tool_data(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    用于 --data
    继续使用原来的 HiROLEpisodeReader 逻辑
    """
    reader = HiROLEpisodeReader(episode_dir)
    if reader.primary_stream is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    xyz_points: list[np.ndarray] = []
    tool_values: list[np.float32] = []

    for step in reader.iter_steps(load_images=False):
        role = step["primary_stream"] or reader.primary_stream
        ee_pose = step["ee_pose"].get(role)
        tool_value = step["tool_position"].get(role, np.float32(np.nan))

        if ee_pose is None:
            continue

        ee_pose = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose.shape[0] < 3:
            continue

        xyz = ee_pose[:3]
        if not np.all(np.isfinite(xyz)):
            continue

        xyz_points.append(xyz)
        tool_values.append(np.float32(tool_value))

    if not xyz_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.stack(xyz_points, axis=0), np.asarray(tool_values, dtype=np.float32)


def iter_ee_xyz_and_tool_infer(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    用于 --infer
    直接读取 data.json，不经过 HiROLEpisodeReader，
    只解析 ee_states.single.pose 和 tools.single.position
    """
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.json not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    steps = payload.get("data")
    if not isinstance(steps, list):
        raise ValueError(f"Invalid data field in {data_path}")

    xyz_points: list[np.ndarray] = []
    tool_values: list[np.float32] = []

    for step in steps:
        if not isinstance(step, dict):
            continue

        ee_states = step.get("ee_states") or {}
        tools = step.get("tools") or {}

        if not isinstance(ee_states, dict):
            ee_states = {}
        if not isinstance(tools, dict):
            tools = {}

        ee_single = ee_states.get("single") or {}
        tool_single = tools.get("single") or {}

        if not isinstance(ee_single, dict):
            ee_single = {}
        if not isinstance(tool_single, dict):
            tool_single = {}

        ee_pose = ee_single.get("pose")
        tool_value = tool_single.get("position", np.nan)


        if ee_pose is None:
            continue

        ee_pose = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose.ndim != 1 or ee_pose.shape[0] < 3:
            continue

        xyz = ee_pose[:3]
        if not np.all(np.isfinite(xyz)):
            continue

        xyz_points.append(xyz)

        try:
            tool_values.append(np.float32(tool_value))
        except (TypeError, ValueError):
            tool_values.append(np.float32(np.nan))

    if not xyz_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.stack(xyz_points, axis=0), np.asarray(tool_values, dtype=np.float32)


def set_equal_axes(ax, all_points: np.ndarray) -> None:
    if all_points.size == 0:
        return
    mins = np.nanmin(all_points, axis=0)
    maxs = np.nanmax(all_points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if not np.isfinite(radius) or radius == 0:
        radius = 0.1
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_group(
    ax,
    episode_dirs: Iterable[Path],
    color: str,
    linewidth: float ,
    label: str,
    tool_threshold: float ,
    reader_fn,
) -> tuple[list[np.ndarray], int]:
    all_xyz: list[np.ndarray] = []
    black_point_count = 0
    first_line = True

    for episode_dir in episode_dirs:
        try:
            xyz, tool = reader_fn(episode_dir)
        except Exception as exc:
            print(f"Skip episode {episode_dir}: {exc}")
            continue

        if xyz.shape[0] == 0:
            continue

        all_xyz.append(xyz)
        line_label = label if first_line else None
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=color,
            alpha=0.8,
            linewidth=linewidth,
            label=line_label,
        )
        first_line = False

        # 满足条件的部分，加粗显示
        zero_mask = np.isfinite(tool) & (tool < tool_threshold)

        # 按线段加粗，而不是按点块拆段
        # 只要一条线段的两个端点中有任意一个满足条件，就把这条线段加粗
        for i in range(len(xyz) - 1):
            if zero_mask[i] or zero_mask[i + 1]:
                seg_xyz = xyz[i:i + 2]
                ax.plot(
                    seg_xyz[:, 0],
                    seg_xyz[:, 1],
                    seg_xyz[:, 2],
                    color=color,   # 或 color=color
                    alpha=1.0,
                    linewidth=linewidth,
                )
                black_point_count += 1

    return all_xyz, black_point_count

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize HIROL ee_states xyz trajectories for two folders."
    )
    parser.add_argument("--data", required=True, help="HIROL dataset root or episode dir for group A.")
    parser.add_argument("--infer", required=True, help="Inference dataset root or episode dir for group B.")
    parser.add_argument(
        "--tool-threshold",
        type=float,
        default=85,
        help="Mark ee points in black when tools.<role>.position is smaller than this value.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("dp_output/Visualize/ee_states_xyz.png"),
        help="Optional output image path.",
    )
    parser.add_argument(
        "--title",
        default="HIROL EE States XYZ Trajectories",
        help="Plot title.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    episode_dirs_a = resolve_episode_dirs(args.data)
    episode_dirs_b = resolve_episode_dirs(args.infer)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xyz_a, black_a = plot_group(
        ax,
        episode_dirs_a,
        color="blue",
        linewidth=1.0,
        label="data",
        tool_threshold=args.tool_threshold,
        reader_fn=iter_ee_xyz_and_tool_data,
    )

    xyz_b, black_b = plot_group(
        ax,
        episode_dirs_b,
        color="black",
        linewidth=1.0,
        label="infer",
        tool_threshold=args.tool_threshold,
        reader_fn=iter_ee_xyz_and_tool_infer,
    )

    all_xyz_arrays = [arr for arr in [*xyz_a, *xyz_b] if arr.size > 0]
    if all_xyz_arrays:
        set_equal_axes(ax, np.concatenate(all_xyz_arrays, axis=0))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(args.title)
    ax.legend()

    print(f"data episodes: {len(episode_dirs_a)}, black points: {black_a}")
    print(f"infer episodes: {len(episode_dirs_b)}, black points: {black_b}")

    fig.tight_layout()
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=200)
        print(f"Saved figure to: {args.save_path}")

    if os.environ.get("DISPLAY"):
        plt.show()
    elif args.save_path is None:
        print("DISPLAY is not set; use --save-path to write the figure to disk.")


if __name__ == "__main__":
    main()