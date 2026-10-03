#!/usr/bin/env python3
"""Backfill exact source-metadata references into existing clip indexes.

This script never reads or rewrites video frames. It reconstructs the same
slice plan used by metadata_process0302.py from the full episode metadata and
writes a new JSON index containing metadata_ref for every matched record.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

try:
    from .clip_metadata_reference import (
        MetadataReferenceError,
        build_metadata_ref,
        validate_metadata_ref,
    )
except ImportError:
    from clip_metadata_reference import (
        MetadataReferenceError,
        build_metadata_ref,
        validate_metadata_ref,
    )


DEFAULT_FPS = 10.0
DEFAULT_MIN_TASK_DURATION = 6.0
ClipKey = Tuple[str, int, str, str]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_with_metadata" + input_path.suffix)


def record_key(record: Mapping[str, Any]) -> ClipKey:
    required = ("episode_id", "task_index", "observation_type", "slice_param")
    missing = [name for name in required if name not in record]
    if missing:
        raise KeyError("missing key fields: " + ", ".join(missing))
    return (
        str(record["episode_id"]),
        int(record["task_index"]),
        str(record["observation_type"]),
        str(record["slice_param"]),
    )


def episode_fps(episode: Mapping[str, Any], fallback: float) -> float:
    recording = episode.get("recording")
    if isinstance(recording, Mapping):
        value = recording.get("fps")
        if value is not None:
            value = float(value)
            if value > 0:
                return value
    return fallback


def first_robot_active_frame(steps: Iterable[Mapping[str, Any]]) -> int:
    for index, step in enumerate(steps):
        if bool(step.get("robot_active", False)):
            return index
    return 0


def reconstruct_episode_refs(
    metadata_file: Path,
    episode: Mapping[str, Any],
    *,
    fallback_fps: float,
    min_task_duration: float,
) -> Dict[ClipKey, Dict[str, Any]]:
    """Reproduce the active slicing logic in metadata_process0302.py."""

    steps = episode.get("steps")
    subtasks = episode.get("subtasks")
    if not isinstance(steps, list):
        raise MetadataReferenceError(f"{metadata_file} has no steps list")
    if not isinstance(subtasks, list):
        raise MetadataReferenceError(f"{metadata_file} has no subtasks list")

    fps = episode_fps(episode, fallback_fps)
    total_frames = len(steps)
    robot_active_frame = first_robot_active_frame(steps)
    episode_id = str(episode.get("episode_id", metadata_file.stem))
    references: Dict[ClipKey, Dict[str, Any]] = {}

    for task_index, task in enumerate(subtasks):
        if not isinstance(task, Mapping):
            continue

        task_start = max(int(task["start_step"]), robot_active_frame)
        task_end = min(int(task["end_step"]), total_frames)
        if task_end - task_start < fps:
            continue

        pick_frame = None
        for frame in range(task_start, task_end):
            human = steps[frame].get("human_agent", {})
            if isinstance(human, Mapping) and bool(human.get("is_holding", False)):
                pick_frame = frame
                break
        if pick_frame is None:
            continue

        final_position = steps[task_end - 1]["human_agent"]["pos"]
        final_x = float(final_position[0])
        final_z = float(final_position[2])
        arrival_frame = None
        for frame in range(pick_frame, task_end):
            position = steps[frame]["human_agent"]["pos"]
            distance = math.hypot(
                float(position[0]) - final_x,
                float(position[2]) - final_z,
            )
            if distance < 2.0:
                arrival_frame = frame
                break
        if arrival_frame is None:
            arrival_frame = max(pick_frame + 1, task_end - 10)

        transport_frames = arrival_frame - pick_frame
        if transport_frames < min_task_duration * fps:
            continue

        length_30 = int(transport_frames * 0.3)
        length_60 = int(transport_frames * 0.6)
        length_90 = int(transport_frames * 0.9)
        length_15 = int(transport_frames * 0.15)

        pre_start = max(task_start, pick_frame - length_15)
        pre_end = min(pre_start + length_30, total_frames)
        post_end = task_end
        post_start = max(task_start, post_end - length_30)

        plans = (
            ("accumulative", "transit_30", pick_frame, pick_frame + length_30),
            ("accumulative", "transit_60", pick_frame, pick_frame + length_60),
            ("accumulative", "transit_90", pick_frame, pick_frame + length_90),
            ("fixed_len", "pre_transit", pre_start, pre_end),
            (
                "fixed_len",
                "transit_30_60",
                pick_frame + length_30,
                pick_frame + length_60,
            ),
            (
                "fixed_len",
                "transit_60_90",
                pick_frame + length_60,
                pick_frame + length_90,
            ),
            ("fixed_len", "post_transit", post_start, post_end),
        )

        for observation_type, slice_param, start_frame, end_frame in plans:
            key = (episode_id, task_index, observation_type, slice_param)
            if key in references:
                raise MetadataReferenceError(f"duplicate reconstructed key: {key}")
            clip_id = (
                f"{episode_id}_t{task_index}_{observation_type}_{slice_param}"
            )
            references[key] = build_metadata_ref(
                metadata_file=metadata_file,
                episode_data=episode,
                clip_id=clip_id,
                task_index=task_index,
                task=task,
                start_frame=int(start_frame),
                end_frame_exclusive=int(end_frame),
                fps=fps,
                robot_active_frame=robot_active_frame,
                pick_frame=pick_frame,
                arrival_frame=arrival_frame,
            )

    return references


def metadata_file_map(metadata_root: Path) -> Dict[str, Path]:
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"metadata directory not found: {metadata_root}")
    result: Dict[str, Path] = {}
    duplicates = []
    for path in sorted(metadata_root.glob("*.json")):
        if path.stem in result:
            duplicates.append(path.stem)
        result[path.stem] = path
    if duplicates:
        raise RuntimeError(
            "duplicate episode metadata filenames: " + ", ".join(duplicates[:10])
        )
    return result


def extract_records(document: Any) -> Tuple[list[MutableMapping[str, Any]], str | None]:
    if isinstance(document, list):
        if not all(isinstance(item, MutableMapping) for item in document):
            raise TypeError("index list must contain JSON objects")
        return document, None

    if isinstance(document, MutableMapping):
        for field in ("records", "data", "samples", "clips"):
            value = document.get(field)
            if isinstance(value, list) and all(
                isinstance(item, MutableMapping) for item in value
            ):
                return value, field
    raise TypeError(
        "input must be a JSON list or an object containing records/data/samples/clips"
    )


def duration_warning(
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
    tolerance_sec: float,
) -> str | None:
    if record.get("duration") is None:
        return None
    expected = float(reference["frame_count"]) / float(reference["fps"])
    actual = float(record["duration"])
    if abs(expected - actual) > tolerance_sec:
        return f"duration={actual}, reconstructed={expected}"
    return None


def backfill(
    *,
    input_path: Path,
    metadata_root: Path,
    output_path: Path,
    report_path: Path,
    fallback_fps: float,
    min_task_duration: float,
    duration_tolerance: float,
    replace_existing: bool,
    allow_unmatched: bool,
) -> Dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("refusing to overwrite input; choose a different --output")
    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; remove it after inspection "
            "or choose another --output"
        )

    document = read_json(input_path)
    document = copy.deepcopy(document)
    records, container_field = extract_records(document)
    files = metadata_file_map(metadata_root)

    episode_cache: Dict[str, Dict[ClipKey, Dict[str, Any]]] = {}
    stats = Counter()
    failures = []
    warnings = []

    for row_index, record in enumerate(records):
        existing = record.get("metadata_ref")
        if existing is not None and not replace_existing:
            try:
                validate_metadata_ref(existing)
                stats["already_valid"] += 1
                continue
            except (MetadataReferenceError, TypeError, KeyError) as error:
                failures.append(
                    {
                        "row_index": row_index,
                        "reason": "invalid_existing_metadata_ref",
                        "detail": str(error),
                    }
                )
                stats["invalid_existing"] += 1
                continue

        try:
            key = record_key(record)
        except (KeyError, TypeError, ValueError) as error:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "missing_or_invalid_clip_key",
                    "detail": str(error),
                }
            )
            stats["unmatched"] += 1
            continue

        episode_id = key[0]
        metadata_file = files.get(episode_id)
        if metadata_file is None:
            failures.append(
                {
                    "row_index": row_index,
                    "key": list(key),
                    "reason": "episode_metadata_not_found",
                    "expected_file": str(metadata_root / f"{episode_id}.json"),
                }
            )
            stats["unmatched"] += 1
            continue

        if episode_id not in episode_cache:
            try:
                episode = read_json(metadata_file)
                episode_cache[episode_id] = reconstruct_episode_refs(
                    metadata_file,
                    episode,
                    fallback_fps=fallback_fps,
                    min_task_duration=min_task_duration,
                )
                stats["episodes_loaded"] += 1
            except Exception as error:
                failures.append(
                    {
                        "row_index": row_index,
                        "key": list(key),
                        "reason": "episode_reconstruction_failed",
                        "detail": str(error),
                    }
                )
                episode_cache[episode_id] = {}
                stats["episode_errors"] += 1

        reference = episode_cache[episode_id].get(key)
        if reference is None:
            failures.append(
                {
                    "row_index": row_index,
                    "key": list(key),
                    "reason": "slice_key_not_reconstructed",
                }
            )
            stats["unmatched"] += 1
            continue

        record["metadata_ref"] = reference
        stats["backfilled"] += 1
        warning = duration_warning(record, reference, duration_tolerance)
        if warning is not None:
            warnings.append(
                {"row_index": row_index, "key": list(key), "detail": warning}
            )
            stats["duration_warnings"] += 1

    report = {
        "schema_version": "cothrc.metadata_backfill_report.v1",
        "input": str(input_path),
        "metadata_root": str(metadata_root),
        "output": str(output_path),
        "container_field": container_field,
        "total_records": len(records),
        "stats": dict(stats),
        "complete": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures[:200],
        "failure_list_truncated": len(failures) > 200,
        "warning_count": len(warnings),
        "warnings": warnings[:200],
        "warning_list_truncated": len(warnings) > 200,
    }
    write_json_atomic(report_path, report)

    if failures and not allow_unmatched:
        raise RuntimeError(
            f"{len(failures)} records could not be backfilled; "
            f"dataset output was not written. See {report_path}"
        )

    write_json_atomic(output_path, document)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add exact source metadata references to an existing clip index "
            "without reading, cutting, or rewriting any videos."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--min-task-duration",
        type=float,
        default=DEFAULT_MIN_TASK_DURATION,
    )
    parser.add_argument("--duration-tolerance", type=float, default=0.11)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace existing metadata_ref values instead of validating and keeping them",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="write a partial output even when some records cannot be matched",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input
    output_path = args.output or default_output_path(input_path)
    report_path = args.report or output_path.with_name(
        output_path.stem + "_report.json"
    )

    report = backfill(
        input_path=input_path,
        metadata_root=args.metadata_root,
        output_path=output_path,
        report_path=report_path,
        fallback_fps=args.fps,
        min_task_duration=args.min_task_duration,
        duration_tolerance=args.duration_tolerance,
        replace_existing=args.replace_existing,
        allow_unmatched=args.allow_unmatched,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "total_records": report["total_records"],
                "stats": report["stats"],
                "complete": report["complete"],
                "failure_count": report["failure_count"],
                "warning_count": report["warning_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
