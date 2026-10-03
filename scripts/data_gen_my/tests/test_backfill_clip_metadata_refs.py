import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backfill_clip_metadata_refs import backfill  # noqa: E402


class BackfillClipMetadataRefsTest(unittest.TestCase):
    def test_existing_index_is_backfilled_without_video_files(self):
        steps = []
        for frame in range(100):
            progress = max(0.0, min(1.0, (frame - 10) / 89.0))
            steps.append(
                {
                    "step": frame,
                    "time_sec": frame / 10.0,
                    "robot_active": True,
                    "human_agent": {
                        "is_holding": frame >= 10,
                        "pos": [10.0 * progress, 0.0, 0.0],
                    },
                }
            )
        episode = {
            "episode_id": "ep0",
            "scene_id": "scene0",
            "subtasks": [
                {
                    "start_step": 0,
                    "end_step": 100,
                    "description_processed": "Move object",
                }
            ],
            "steps": steps,
        }
        index = [
            {
                "episode_id": "ep0",
                "task_index": 0,
                "observation_type": "accumulative",
                "slice_param": "transit_60",
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_root = root / "metadata"
            metadata_root.mkdir()
            (metadata_root / "ep0.json").write_text(
                json.dumps(episode), encoding="utf-8"
            )
            input_path = root / "benchmark_index.json"
            output_path = root / "benchmark_index_with_metadata.json"
            report_path = root / "report.json"
            input_path.write_text(json.dumps(index), encoding="utf-8")

            report = backfill(
                input_path=input_path,
                metadata_root=metadata_root,
                output_path=output_path,
                report_path=report_path,
                fallback_fps=10.0,
                min_task_duration=6.0,
                duration_tolerance=0.11,
                replace_existing=False,
                allow_unmatched=False,
            )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            reference = result[0]["metadata_ref"]
            self.assertTrue(report["complete"])
            self.assertEqual(report["stats"]["backfilled"], 1)
            self.assertEqual(reference["episode_id"], "ep0")
            self.assertEqual(
                reference["frame_count"],
                reference["end_frame_exclusive"] - reference["start_frame"],
            )


if __name__ == "__main__":
    unittest.main()
