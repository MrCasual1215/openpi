"""
Convert raw UMI episodes to LeRobot format.

Expected raw directory layout:
    <data_dir>/
      episode*/
        camera/color/pikaDepthCamera/
          sync.txt
          *.jpg
        localization/pose/pika/
          sync.txt
          *.json
        gripper/encoder/pika/
          sync.txt
          *.json

This script uses the color camera sync file as the master timeline, aligns pose and
gripper readings by nearest timestamp, downsamples to a target FPS, converts orientation
to rotation vectors, and stores next-step absolute targets as actions.

Example usage:
    uv run examples/umi/convert_umi_data_to_lerobot.py \
        --data-dir /home/sunpeng/sp/pi/dataset/single/20260428 \
        --repo-id your_hf_username/umi_dataset \
        --task "do something"
"""

from __future__ import annotations

import bisect
import dataclasses
import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    data_dir: Path
    repo_id: str = "your_hf_username/umi_dataset"
    task: str = "do something"
    fps: int = 10
    max_time_delta_s: float = 0.05
    push_to_hub: bool = False
    image_writer_threads: int = 10
    image_writer_processes: int = 5


def _read_sync_file(sync_path: Path) -> list[Path]:
    if not sync_path.exists():
        raise FileNotFoundError(sync_path)

    result = []
    for line in sync_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        result.append(sync_path.parent / line)
    return result


def _timestamp_from_path(path: Path) -> float:
    return float(path.stem)


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _rpy_to_rotvec(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll/pitch/yaw to rotation vector using ZYX convention."""
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rot_mat = rot_z @ rot_y @ rot_x
    rot_vec, _ = cv2.Rodrigues(rot_mat)
    return rot_vec[:, 0].astype(np.float32)


def _quat_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    qx, qy, qz, qw = quat

    rot_mat = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    rot_vec, _ = cv2.Rodrigues(rot_mat)
    return rot_vec[:, 0].astype(np.float32)


def _orientation_to_rotvec(pose: dict) -> np.ndarray:
    if {"roll", "pitch", "yaw"}.issubset(pose):
        return _rpy_to_rotvec(pose["roll"], pose["pitch"], pose["yaw"])
    if {"qx", "qy", "qz", "qw"}.issubset(pose):
        return _quat_to_rotvec(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    raise ValueError(f"Unsupported pose orientation format. Keys: {sorted(pose)}")


def _nearest_path(query_ts: float, candidates: list[Path], candidate_ts: list[float], max_delta: float) -> Path | None:
    if not candidates:
        return None

    idx = bisect.bisect_left(candidate_ts, query_ts)
    candidate_indices = []
    if idx < len(candidate_ts):
        candidate_indices.append(idx)
    if idx > 0:
        candidate_indices.append(idx - 1)
    if not candidate_indices:
        return None

    best_idx = min(candidate_indices, key=lambda i: abs(candidate_ts[i] - query_ts))
    if abs(candidate_ts[best_idx] - query_ts) > max_delta:
        return None
    return candidates[best_idx]


def _sample_master_paths(paths: list[Path], fps: int) -> list[Path]:
    if not paths:
        return []

    step = 1.0 / fps
    sampled = [paths[0]]
    last_ts = _timestamp_from_path(paths[0])
    for path in paths[1:]:
        ts = _timestamp_from_path(path)
        if ts >= last_ts + step - 1e-6:
            sampled.append(path)
            last_ts = ts
    return sampled


def _load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_state(pose_path: Path, gripper_path: Path) -> np.ndarray:
    pose = _load_json(pose_path)
    gripper = _load_json(gripper_path)

    rotvec = _orientation_to_rotvec(pose)
    gripper_width = np.array([gripper["distance"]], dtype=np.float32)
    position = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
    return np.concatenate([position, rotvec, gripper_width], axis=0).astype(np.float32)


def _episode_dirs(data_dir: Path) -> list[Path]:
    def key_fn(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("episode")
        return (int(suffix) if suffix.isdigit() else 10**9, path.name)

    return sorted([path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("episode")], key=key_fn)


def _build_episode_samples(episode_dir: Path, fps: int, max_time_delta_s: float) -> list[dict]:
    color_dir = episode_dir / "camera" / "color" / "pikaDepthCamera"
    pose_dir = episode_dir / "localization" / "pose" / "pika"
    gripper_dir = episode_dir / "gripper" / "encoder" / "pika"

    required_syncs = [color_dir / "sync.txt", pose_dir / "sync.txt", gripper_dir / "sync.txt"]
    if not all(path.exists() for path in required_syncs):
        return []

    color_paths = _sample_master_paths(_read_sync_file(color_dir / "sync.txt"), fps)
    pose_paths = _read_sync_file(pose_dir / "sync.txt")
    gripper_paths = _read_sync_file(gripper_dir / "sync.txt")
    pose_timestamps = [_timestamp_from_path(path) for path in pose_paths]
    gripper_timestamps = [_timestamp_from_path(path) for path in gripper_paths]

    aligned = []
    for color_path in color_paths:
        color_ts = _timestamp_from_path(color_path)
        pose_path = _nearest_path(color_ts, pose_paths, pose_timestamps, max_time_delta_s)
        gripper_path = _nearest_path(color_ts, gripper_paths, gripper_timestamps, max_time_delta_s)
        if pose_path is None or gripper_path is None or not color_path.exists():
            continue
        aligned.append(
            {
                "timestamp": color_ts,
                "image_path": color_path,
                "state": _load_state(pose_path, gripper_path),
            }
        )

    if len(aligned) < 2:
        return []

    samples = []
    for current, nxt in zip(aligned[:-1], aligned[1:], strict=True):
        samples.append(
            {
                "image_path": current["image_path"],
                "ee_pose": current["state"][:6],
                "gripper_width": current["state"][6:7],
                # Store next-step absolute targets. The training pipeline converts the first 6 dims to deltas.
                "action": nxt["state"],
            }
        )
    return samples


def _create_dataset(repo_id: str, image_shape: tuple[int, int, int], fps: int, image_writer_threads: int, image_writer_processes: int) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="umi",
        fps=fps,
        features={
            "observation.wrist_image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.ee_pose": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["ee_pose"],
            },
            "observation.gripper_width": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_width"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def main(args: Args) -> None:
    episode_dirs = _episode_dirs(args.data_dir)
    print(f"Found {len(episode_dirs)} episode directories")

    dataset = None
    converted_episodes = 0
    skipped_episodes = 0

    for episode_dir in tqdm.tqdm(episode_dirs, desc="Converting UMI episodes"):
        samples = _build_episode_samples(episode_dir, args.fps, args.max_time_delta_s)
        if not samples:
            skipped_episodes += 1
            continue

        first_image = _load_rgb_image(samples[0]["image_path"])
        if dataset is None:
            dataset = _create_dataset(
                args.repo_id,
                first_image.shape,
                args.fps,
                args.image_writer_threads,
                args.image_writer_processes,
            )

        for sample_idx, sample in enumerate(samples):
            image = first_image if sample_idx == 0 else _load_rgb_image(sample["image_path"])
            dataset.add_frame(
                {
                    "observation.wrist_image": image,
                    "observation.ee_pose": sample["ee_pose"],
                    "observation.gripper_width": sample["gripper_width"],
                    "actions": sample["action"],
                    "task": args.task,
                }
            )
        dataset.save_episode()
        converted_episodes += 1

    if dataset is None:
        raise ValueError("No valid episodes were found. Check the raw directory layout and sync files.")

    print(f"Converted {converted_episodes} episodes, skipped {skipped_episodes} episodes.")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["umi", "single-arm", "wrist-camera"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
