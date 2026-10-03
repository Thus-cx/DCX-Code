"""Utilities for keeping an exact link from a video clip to episode metadata.

The generated dataset stores a lightweight reference instead of duplicating the
large per-frame ``steps`` array in every clip record.  Frame intervals always
use Python/NumPy half-open semantics: ``[start_frame, end_frame_exclusive)``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "cothrc.clip_metadata_ref.v1"


class MetadataReferenceError(ValueError):
    """Raised when a clip metadata reference is malformed or inconsistent."""


def _as_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise MetadataReferenceError(f"{name} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MetadataReferenceError(f"{name} must be an integer: {value!r}") from exc
    if result != value:
        raise MetadataReferenceError(f"{name} must be an exact integer: {value!r}")
    return result


def _step_number(step: Mapping[str, Any], fallback: int) -> int:
    return _as_int("step", step.get("step", fallback))


def build_metadata_ref(
    *,
    metadata_file: os.PathLike[str] | str,
    episode_data: Mapping[str, Any],
    clip_id: str,
    task_index: int,
    task: Mapping[str, Any],
    start_frame: int,
    end_frame_exclusive: int,
    fps: float,
    robot_active_frame: Optional[int] = None,
    pick_frame: Optional[int] = None,
    arrival_frame: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and validate a portable metadata reference for one clip."""

    steps = episode_data.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise MetadataReferenceError("episode metadata must contain a steps array")

    start = _as_int("start_frame", start_frame)
    end = _as_int("end_frame_exclusive", end_frame_exclusive)
    total = len(steps)
    if start < 0 or end <= start or end > total:
        raise MetadataReferenceError(
            f"invalid half-open frame range [{start}, {end}) for {total} steps"
        )
    if fps <= 0:
        raise MetadataReferenceError(f"fps must be positive, got {fps!r}")

    source_path = os.path.normpath(os.fspath(metadata_file))
    recording = episode_data.get("recording", {})
    if not isinstance(recording, Mapping):
        recording = {}
    first_step = steps[start]
    last_step = steps[end - 1]
    if not isinstance(first_step, Mapping) or not isinstance(last_step, Mapping):
        raise MetadataReferenceError("each steps entry must be a JSON object")

    def optional_frame(name: str, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        frame = _as_int(name, value)
        if frame < 0 or frame >= total:
            raise MetadataReferenceError(
                f"{name}={frame} is outside episode frame range [0, {total})"
            )
        return frame

    ref: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": str(clip_id),
        "episode_id": str(episode_data.get("episode_id", "unknown")),
        "scene_id": episode_data.get("scene_id"),
        # source_file makes the dataset relocatable; source_path preserves provenance.
        "source_file": Path(source_path).name,
        "source_path": source_path,
        "boundary_semantics": "half_open_[start_frame,end_frame_exclusive)",
        "fps": float(fps),
        "source_total_frames": total,
        "source_frame_alignment": recording.get(
            "frame_metadata_alignment", "not_declared_legacy_recording"
        ),
        "start_frame": start,
        "end_frame_exclusive": end,
        "frame_count": end - start,
        "start_step": _step_number(first_step, start),
        "end_step_inclusive": _step_number(last_step, end - 1),
        # These are the exact video slicing timestamps used by metadata_process0302.py.
        "start_time_sec": start / float(fps),
        "end_time_sec_exclusive": end / float(fps),
        "task_index": _as_int("task_index", task_index),
        "task_start_step": _as_int("task.start_step", task["start_step"]),
        "task_end_step": _as_int("task.end_step", task["end_step"]),
        "robot_active_frame": optional_frame("robot_active_frame", robot_active_frame),
        "pick_frame": optional_frame("pick_frame", pick_frame),
        "arrival_frame": optional_frame("arrival_frame", arrival_frame),
    }
    validate_metadata_ref(ref)
    return ref


def validate_metadata_ref(ref: Mapping[str, Any]) -> None:
    """Validate schema and internal frame-count consistency."""

    if ref.get("schema_version") != SCHEMA_VERSION:
        raise MetadataReferenceError(
            f"unsupported schema_version={ref.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if ref.get("boundary_semantics") != "half_open_[start_frame,end_frame_exclusive)":
        raise MetadataReferenceError("unsupported or missing boundary_semantics")

    start = _as_int("start_frame", ref.get("start_frame"))
    end = _as_int("end_frame_exclusive", ref.get("end_frame_exclusive"))
    count = _as_int("frame_count", ref.get("frame_count"))
    total = _as_int("source_total_frames", ref.get("source_total_frames"))
    if start < 0 or end <= start or end > total or count != end - start:
        raise MetadataReferenceError(
            f"inconsistent frame range [{start}, {end}), count={count}, total={total}"
        )
    if not ref.get("source_file") and not ref.get("source_path"):
        raise MetadataReferenceError("metadata reference has no source file/path")


def resolve_metadata_path(
    ref: Mapping[str, Any],
    *,
    metadata_root: Optional[os.PathLike[str] | str] = None,
    project_root: Optional[os.PathLike[str] | str] = None,
) -> Path:
    """Resolve a reference after a dataset has been moved to another machine."""

    validate_metadata_ref(ref)
    candidates = []
    source_file = ref.get("source_file")
    source_path = ref.get("source_path")

    # An explicit metadata root is authoritative and therefore checked first.
    if metadata_root and source_file:
        candidates.append(Path(metadata_root) / str(source_file))
    if source_path:
        candidates.append(Path(str(source_path)))
        if project_root and not Path(str(source_path)).is_absolute():
            candidates.append(Path(project_root) / str(source_path))

    unique_candidates = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
        if candidate.is_file():
            return candidate.resolve()

    tried = ", ".join(str(path) for path in unique_candidates) or "<none>"
    raise FileNotFoundError(f"cannot resolve source metadata; tried: {tried}")


def load_referenced_steps(
    ref: Mapping[str, Any],
    *,
    metadata_root: Optional[os.PathLike[str] | str] = None,
    project_root: Optional[os.PathLike[str] | str] = None,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Load an episode JSON and return exactly the steps referenced by a clip."""

    path = resolve_metadata_path(
        ref, metadata_root=metadata_root, project_root=project_root
    )
    with path.open("r", encoding="utf-8") as file:
        episode = json.load(file)

    if str(episode.get("episode_id", "unknown")) != str(ref.get("episode_id")):
        raise MetadataReferenceError(
            f"episode mismatch: reference={ref.get('episode_id')!r}, "
            f"file={episode.get('episode_id')!r}"
        )
    steps = episode.get("steps")
    if not isinstance(steps, list):
        raise MetadataReferenceError(f"{path} does not contain a steps list")
    if len(steps) != _as_int("source_total_frames", ref["source_total_frames"]):
        raise MetadataReferenceError(
            f"source metadata changed: expected {ref['source_total_frames']} steps, "
            f"found {len(steps)}"
        )

    start = _as_int("start_frame", ref["start_frame"])
    end = _as_int("end_frame_exclusive", ref["end_frame_exclusive"])
    clip_steps = steps[start:end]
    if len(clip_steps) != _as_int("frame_count", ref["frame_count"]):
        raise MetadataReferenceError("loaded clip length does not match metadata reference")
    return episode, clip_steps
