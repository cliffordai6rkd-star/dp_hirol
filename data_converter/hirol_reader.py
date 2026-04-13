from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np


DEFAULT_CAMERAS: Sequence[str] = (
    "left_hand_color",
    "right_hand_color",
    "head_color",
    "left_hand_fisheye_color",
    "right_hand_fisheye_color",
)
ROLE_PRIORITY: Sequence[str] = ("single", "left", "right", "head")
DEFAULT_TIMESTAMP_PRIORITY: Sequence[tuple[str, Optional[str]]] = (
    ("joint_states", None),
    ("ee_states", None),
    ("tools", None),
    ("actions", "joint"),
    ("actions", "ee"),
    ("actions", "tool"),
)


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an RGB image as HWC uint8 numpy array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_bgr_image(path: Path) -> np.ndarray:
    """Load a BGR image for direct video encoding."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


def _unique_preserve_order(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _preferred_role(keys: Sequence[str], hint_text: str = "") -> Optional[str]:
    keys = _unique_preserve_order(keys)
    if not keys:
        return None
    if "single" in keys:
        return "single"

    hint = hint_text.lower()
    for role in ("left", "right", "head"):
        if role in keys and role in hint:
            return role
    for role in ROLE_PRIORITY:
        if role in keys:
            return role
    return keys[0]


def _nan_vector(length: int) -> np.ndarray:
    return np.full((length,), np.nan, dtype=np.float32)


def _as_float_vector(value, length: int) -> np.ndarray:
    if isinstance(value, list) and len(value) == length:
        return np.asarray(value, dtype=np.float32)
    return _nan_vector(length)


def _as_float_scalar(value) -> np.float32:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return np.float32(np.nan)
    if array.size != 1:
        return np.float32(np.nan)
    return np.float32(array[0])


def _as_float_scalar_array(value) -> np.ndarray:
    return np.asarray([_as_float_scalar(value)], dtype=np.float32)

def _action_gripper(value) -> np.float32:
    return _as_float_scalar(value)

class HiROLEpisodeReader:
    """Read one HIROL episode from `<episode_dir>/data.json` and media files."""

    def __init__(self, episode_dir: str | Path):
        self.episode_dir = Path(episode_dir)
        self.data_json_path = self.episode_dir / "data.json"
        if not self.data_json_path.is_file():
            raise FileNotFoundError(f"data.json not found: {self.data_json_path}")

        with self.data_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.info: Dict = payload.get("info") or {}
        self.text: Dict = payload.get("text") or {}
        self._records: List[Dict] = payload.get("data") or []
        if not isinstance(self._records, list):
            raise ValueError(f"Invalid data field in {self.data_json_path}: expected list")

        self._exists_cache: Dict[str, bool] = {}

        self.camera_keys: Sequence[str] = self._infer_cameras()
        self.joint_keys: Sequence[str] = self._infer_stream_keys("joint_states")
        self.ee_keys: Sequence[str] = self._infer_stream_keys("ee_states")
        self.tool_keys: Sequence[str] = self._infer_stream_keys("tools")
        self.action_keys: Sequence[str] = self._infer_stream_keys("actions")
        all_keys = list(self.action_keys) + list(self.joint_keys) + list(self.ee_keys) + list(self.tool_keys)
        hint_text = f"{self.episode_dir} {json.dumps(self.text, ensure_ascii=False)}"
        self.primary_stream: Optional[str] = _preferred_role(all_keys, hint_text=hint_text)

    @staticmethod
    def list_episode_dirs(dataset_root: str | Path) -> List[Path]:
        root = Path(dataset_root)
        return sorted([p for p in root.glob("episode_*") if p.is_dir()])

    @property
    def records(self) -> List[Dict]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def _infer_cameras(self) -> Sequence[str]:
        for record in self._records:
            colors = record.get("colors") or {}
            if isinstance(colors, dict) and colors:
                return tuple(colors.keys())
        return tuple(DEFAULT_CAMERAS)

    def _infer_stream_keys(self, field_name: str) -> Sequence[str]:
        for record in self._records:
            value = record.get(field_name) or {}
            if isinstance(value, dict) and value:
                return tuple(value.keys())
        return tuple()

    def _resolve_rel_path(self, rel_path: str) -> Path:
        return self.episode_dir / rel_path

    def _path_exists(self, rel_path: Optional[str]) -> bool:
        if not rel_path:
            return False
        cached = self._exists_cache.get(rel_path)
        if cached is not None:
            return cached
        exists = self._resolve_rel_path(rel_path).is_file()
        self._exists_cache[rel_path] = exists
        return exists

    def _select_role_value(self, mapping: Dict[str, np.ndarray | np.float32], default):
        if self.primary_stream and self.primary_stream in mapping:
            return mapping[self.primary_stream]
        if "single" in mapping:
            return mapping["single"]
        if mapping:
            first_key = next(iter(mapping))
            return mapping[first_key]
        return default

    def _select_role_key(self, mapping: Dict[str, object]) -> Optional[str]:
        if self.primary_stream and self.primary_stream in mapping:
            return self.primary_stream
        if "single" in mapping:
            return "single"
        if mapping:
            return next(iter(mapping))
        return None

    @staticmethod
    def _timestamp_value(entry: Dict) -> float:
        if not isinstance(entry, dict):
            return float("nan")
        try:
            return float(entry.get("time_stamp", np.nan))
        except (TypeError, ValueError):
            return float("nan")

    def extract_primary_timestamp(self, record: Dict) -> float:
        primary = self.primary_stream
        for field_name, nested_key in DEFAULT_TIMESTAMP_PRIORITY:
            mapping = record.get(field_name) or {}
            if primary and primary in mapping:
                entry = mapping.get(primary) or {}
                if nested_key is not None:
                    entry = entry.get(nested_key) or {}
                value = self._timestamp_value(entry)
                if np.isfinite(value):
                    return value

        for field_name, nested_key in DEFAULT_TIMESTAMP_PRIORITY:
            mapping = record.get(field_name) or {}
            for entry in mapping.values():
                item = entry or {}
                if nested_key is not None:
                    item = item.get(nested_key) or {}
                value = self._timestamp_value(item)
                if np.isfinite(value):
                    return value
        return float("nan")

    def missing_cameras(self, record: Dict) -> List[str]:
        missing: List[str] = []
        colors = record.get("colors") or {}
        for cam in self.camera_keys:
            entry = colors.get(cam)
            rel_path = entry.get("path") if isinstance(entry, dict) else None
            if not self._path_exists(rel_path):
                missing.append(cam)
        return missing

    def infer_image_shape(self) -> tuple[int, int, int]:
        for record in self._records:
            colors = record.get("colors") or {}
            for cam in self.camera_keys:
                entry = colors.get(cam)
                rel_path = entry.get("path") if isinstance(entry, dict) else None
                if self._path_exists(rel_path):
                    image = load_rgb_image(self._resolve_rel_path(rel_path))
                    return image.shape

        width = int((self.info.get("image") or {}).get("width") or 0)
        height = int((self.info.get("image") or {}).get("height") or 0)
        if width > 0 and height > 0:
            return (height, width, 3)

        raise RuntimeError(f"Cannot infer image shape from episode: {self.episode_dir}")

    def get_step(
        self,
        index: int,
        load_images: bool = False,
        fill_missing: str = "none",
        image_shape: Optional[tuple[int, int, int]] = None,
    ) -> Dict:
        """
        Read one step.

        fill_missing:
            - "none": keep missing images as None
            - "zeros": fill missing images with zero arrays
            - "error": raise FileNotFoundError if any image is missing
        """
        if fill_missing not in {"none", "zeros", "error"}:
            raise ValueError(f"Unsupported fill_missing={fill_missing}")

        if index < 0 or index >= len(self._records):
            raise IndexError(f"index out of range: {index}")

        record = self._records[index]
        step_idx = int(record.get("idx", index))

        colors = record.get("colors") or {}
        images: Dict[str, Optional[np.ndarray]] = {}
        image_paths: Dict[str, Optional[str]] = {}
        image_timestamps: Dict[str, float] = {}
        image_valid: Dict[str, bool] = {}

        if load_images and fill_missing == "zeros" and image_shape is None:
            image_shape = self.infer_image_shape()

        for cam in self.camera_keys:
            entry = colors.get(cam)
            rel_path = entry.get("path") if isinstance(entry, dict) else None
            ts = float(entry.get("time_stamp", np.nan)) if isinstance(entry, dict) else np.nan

            image_paths[cam] = rel_path
            image_timestamps[cam] = ts

            exists = self._path_exists(rel_path)
            image_valid[cam] = exists

            if not load_images:
                continue

            if exists:
                images[cam] = load_rgb_image(self._resolve_rel_path(rel_path))
            elif fill_missing == "error":
                raise FileNotFoundError(
                    f"Missing image for step {step_idx}, camera={cam}, path={rel_path}"
                )
            elif fill_missing == "zeros":
                if image_shape is None:
                    raise RuntimeError("image_shape is required when fill_missing='zeros'")
                images[cam] = np.zeros(image_shape, dtype=np.uint8)
            else:
                images[cam] = None

        joint_states = record.get("joint_states") or {}
        joint_position: Dict[str, np.ndarray] = {}
        joint_timestamps: Dict[str, float] = {}
        for role in self.joint_keys:
            state = joint_states.get(role) or {}
            joint_position[role] = _as_float_vector(state.get("position"), 7)
            joint_timestamps[role] = float(state.get("time_stamp", np.nan))

        ee_states = record.get("ee_states") or {}
        ee_pose: Dict[str, np.ndarray] = {}
        ee_timestamps: Dict[str, float] = {}
        for role in self.ee_keys:
            state = ee_states.get(role) or {}
            ee_pose[role] = _as_float_vector(state.get("pose"), 7)
            ee_timestamps[role] = float(state.get("time_stamp", np.nan))

        tools = record.get("tools") or {}
        tool_position: Dict[str, np.float32] = {}
        gripper_width: Dict[str, np.float32] = {}
        tool_timestamps: Dict[str, float] = {}
        for role in self.tool_keys:
            state = tools.get(role) or {}
            raw_pos = state.get("position", np.nan)
            tool_position[role] = _as_float_scalar(raw_pos)
            gripper_width[role] = _as_float_scalar(raw_pos)
            tool_timestamps[role] = float(state.get("time_stamp", np.nan))

        actions = record.get("actions") or {}
        action_joint_position: Dict[str, np.ndarray] = {}
        action_ee_pose: Dict[str, np.ndarray] = {}
        action_tool_position: Dict[str, np.float32] = {}
        action_gripper_width: Dict[str, np.float32] = {}
        action_joint_timestamps: Dict[str, float] = {}
        action_ee_timestamps: Dict[str, float] = {}
        action_tool_timestamps: Dict[str, float] = {}
        for role in self.action_keys:
            state = actions.get(role) or {}
            joint_state = state.get("joint") or {}
            ee_state = state.get("ee") or {}
            tool_state = state.get("tool") or {}

            action_joint_position[role] = _as_float_vector(joint_state.get("position"), 7)
            action_ee_pose[role] = _as_float_vector(ee_state.get("pose"), 7)

            raw_tool = tool_state.get("position", np.nan)
            action_tool_position[role] = _as_float_scalar(raw_tool)
            action_gripper_width[role] = _action_gripper(raw_tool)

            action_joint_timestamps[role] = float(joint_state.get("time_stamp", np.nan))
            action_ee_timestamps[role] = float(ee_state.get("time_stamp", np.nan))
            action_tool_timestamps[role] = float(tool_state.get("time_stamp", np.nan))

        primary_joint = self._select_role_value(joint_position, _nan_vector(7))
        primary_ee = self._select_role_value(ee_pose, _nan_vector(7))
        primary_gripper = self._select_role_value(gripper_width, np.float32(np.nan))
        primary_action_joint = self._select_role_value(action_joint_position, _nan_vector(7))
        primary_action_ee = self._select_role_value(action_ee_pose, _nan_vector(7))
        primary_action_gripper = self._select_role_value(action_gripper_width, np.float32(np.nan))

        state = np.concatenate(
            [primary_joint.astype(np.float32), np.asarray([primary_gripper], dtype=np.float32)],
            axis=0,
        )
        state_ee = np.concatenate(
            [
                primary_ee.astype(np.float32),
                primary_joint.astype(np.float32),
                np.asarray([primary_gripper], dtype=np.float32),
            ],
            axis=0,
        )
        action = np.concatenate(
            [primary_action_joint.astype(np.float32), np.asarray([primary_action_gripper], dtype=np.float32)],
            axis=0,
        )

        return {
            "idx": step_idx,
            "images": images if load_images else None,
            "image_paths": image_paths,
            "image_timestamps": image_timestamps,
            "image_valid": image_valid,
            "missing_images": [cam for cam, ok in image_valid.items() if not ok],
            "primary_stream": self.primary_stream,
            "joint_position": joint_position,
            "joint_timestamps": joint_timestamps,
            "ee_pose": ee_pose,
            "ee_timestamps": ee_timestamps,
            "tool_position": tool_position,
            "gripper_width": gripper_width,
            "tool_timestamps": tool_timestamps,
            "action_joint_position": action_joint_position,
            "action_ee_pose": action_ee_pose,
            "action_tool_position": action_tool_position,
            "action_gripper_width": action_gripper_width,
            "action_joint_timestamps": action_joint_timestamps,
            "action_ee_timestamps": action_ee_timestamps,
            "action_tool_timestamps": action_tool_timestamps,
            "state": state,
            "state_ee": state_ee,
            "action": action,
            "raw": record,
        }

    def iter_steps(
        self,
        load_images: bool = False,
        fill_missing: str = "none",
        image_shape: Optional[tuple[int, int, int]] = None,
    ) -> Iterator[Dict]:
        for i in range(len(self._records)):
            yield self.get_step(
                i,
                load_images=load_images,
                fill_missing=fill_missing,
                image_shape=image_shape,
            )

    def get_lerobot_frame(
        self,
        index: int,
        *,
        fallback_timestamp: float,
        episode_index: int,
        task_index: int,
        fill_missing: str = "none",
        image_shape: Optional[tuple[int, int, int]] = None,
        image_color_space: str = "rgb",
    ) -> Dict:
        if fill_missing not in {"none", "zeros", "error"}:
            raise ValueError(f"Unsupported fill_missing={fill_missing}")
        if index < 0 or index >= len(self._records):
            raise IndexError(f"index out of range: {index}")
        if self.primary_stream is None:
            raise RuntimeError(f"Episode {self.episode_dir} has no primary stream.")

        image_color_space = image_color_space.lower()
        if image_color_space not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported image_color_space={image_color_space!r}")

        record = self._records[index]
        if fill_missing == "zeros" and image_shape is None:
            image_shape = self.infer_image_shape()

        colors = record.get("colors") or {}
        frame: Dict[str, object] = {
            "episode_index": np.asarray([episode_index], dtype=np.int64),
            "task_index": np.asarray([task_index], dtype=np.int64),
        }

        image_loader = load_bgr_image if image_color_space == "bgr" else load_rgb_image
        for cam in self.camera_keys:
            entry = colors.get(cam)
            rel_path = entry.get("path") if isinstance(entry, dict) else None
            exists = self._path_exists(rel_path)
            frame[f"observation.images.{cam}.is_valid"] = np.asarray([exists], dtype=np.bool_)
            frame[f"observation.images.{cam}.timestamp"] = np.asarray(
                [float(entry.get("time_stamp", np.nan)) if isinstance(entry, dict) else np.nan],
                dtype=np.float32,
            )

            if exists:
                frame[f"observation.images.{cam}"] = image_loader(self._resolve_rel_path(rel_path))
            elif fill_missing == "error":
                raise FileNotFoundError(
                    f"Missing image for step {record.get('idx', index)}, camera={cam}, path={rel_path}"
                )
            elif fill_missing == "zeros":
                if image_shape is None:
                    raise RuntimeError("image_shape is required when fill_missing='zeros'")
                frame[f"observation.images.{cam}"] = np.zeros(image_shape, dtype=np.uint8)
            else:
                frame[f"observation.images.{cam}"] = None

        joint_states = record.get("joint_states") or {}
        joint_state = joint_states.get(self._select_role_key(joint_states) or "", {}) or {}
        joint_position = _as_float_vector(joint_state.get("position"), 7)

        ee_states = record.get("ee_states") or {}
        ee_state = ee_states.get(self._select_role_key(ee_states) or "", {}) or {}
        ee_pose = _as_float_vector(ee_state.get("pose"), 7)

        tools = record.get("tools") or {}
        tool_state = tools.get(self._select_role_key(tools) or "", {}) or {}
        gripper_width = _as_float_scalar_array(tool_state.get("position", np.nan))

        actions = record.get("actions") or {}
        action_state = actions.get(self._select_role_key(actions) or "", {}) or {}
        action_ee = _as_float_vector((action_state.get("ee") or {}).get("pose"), 7)
        action_joint = _as_float_vector((action_state.get("joint") or {}).get("position"), 7)
        action_gripper = _as_float_scalar_array(
            _action_gripper((action_state.get("tool") or {}).get("position", np.nan))
        )

        primary_timestamp = self.extract_primary_timestamp(record)
        if not np.isfinite(primary_timestamp):
            primary_timestamp = fallback_timestamp

        frame["timestamp"] = np.asarray([primary_timestamp], dtype=np.float32)
        frame["observation.state"] = np.concatenate([ee_pose, joint_position, gripper_width], axis=0).astype(np.float32)
        frame["observation.state.ee_pose"] = ee_pose.astype(np.float32, copy=False)
        frame["observation.state.joint_position"] = joint_position.astype(np.float32, copy=False)
        frame["observation.state.gripper_width"] = gripper_width
        frame["action"] = np.concatenate([action_ee, action_joint, action_gripper], axis=0).astype(np.float32)
        frame["action.ee_pose"] = action_ee.astype(np.float32, copy=False)
        frame["action.joint_position"] = action_joint.astype(np.float32, copy=False)
        frame["action.gripper_width"] = action_gripper
        return frame
