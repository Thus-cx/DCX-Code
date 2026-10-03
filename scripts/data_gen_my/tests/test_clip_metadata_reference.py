import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clip_metadata_reference import (  # noqa: E402
    MetadataReferenceError,
    build_metadata_ref,
    load_referenced_steps,
)


class ClipMetadataReferenceTest(unittest.TestCase):
    def setUp(self):
        self.episode = {
            "episode_id": "ep0",
            "scene_id": "scene0",
            "subtasks": [{"start_step": 0, "end_step": 5}],
            "steps": [
                {"step": i, "time_sec": i / 10.0, "robot_active": i >= 1}
                for i in range(5)
            ],
        }

    def test_reference_round_trip_uses_half_open_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ep0.json"
            source.write_text(json.dumps(self.episode), encoding="utf-8")
            ref = build_metadata_ref(
                metadata_file="old_machine/metadata/ep0.json",
                episode_data=self.episode,
                clip_id="ep0_t0_transit_60",
                task_index=0,
                task=self.episode["subtasks"][0],
                start_frame=1,
                end_frame_exclusive=4,
                fps=10.0,
                robot_active_frame=1,
                pick_frame=2,
                arrival_frame=4,
            )

            _, steps = load_referenced_steps(ref, metadata_root=directory)

            self.assertEqual([step["step"] for step in steps], [1, 2, 3])
            self.assertEqual(ref["frame_count"], 3)
            self.assertEqual(ref["start_time_sec"], 0.1)
            self.assertEqual(ref["end_time_sec_exclusive"], 0.4)

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(MetadataReferenceError):
            build_metadata_ref(
                metadata_file="ep0.json",
                episode_data=self.episode,
                clip_id="bad",
                task_index=0,
                task=self.episode["subtasks"][0],
                start_frame=4,
                end_frame_exclusive=6,
                fps=10.0,
            )


if __name__ == "__main__":
    unittest.main()
