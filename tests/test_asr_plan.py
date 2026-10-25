import tempfile
import unittest
from pathlib import Path

from mib_eval.asr import iter_item_dirs, missing_required, transcript_plan


class TranscriptPlanTest(unittest.TestCase):
    def test_full_duplex_dir_maps_to_scorer_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = Path(tmp) / "551"
            item.mkdir()
            for name in ("padded_input.wav", "output.wav", "input.wav", "combined.wav"):
                (item / name).write_bytes(b"")
            jobs = {job.output_name: job.source.name for job in transcript_plan(item)}
            self.assertEqual(
                jobs,
                {
                    "input.json": "padded_input.wav",
                    "output.json": "output.wav",
                    "unpadded_input.json": "input.wav",
                    "combined.json": "combined.wav",
                },
            )
            self.assertEqual(missing_required(transcript_plan(item)), [])

    def test_iter_finds_nested_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "gemini" / "gsm8k_full_duplex" / "12"
            item.mkdir(parents=True)
            (item / "padded_input.wav").write_bytes(b"")
            found = iter_item_dirs(root)
            self.assertEqual(found, [item])
            self.assertEqual(
                missing_required(transcript_plan(item)),
                ["output.wav"],
            )

    def test_turn_based_dir_uses_input_wav_as_user_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "gpt" / "gsm8k_full" / "7"
            item.mkdir(parents=True)
            (item / "input.wav").write_bytes(b"")
            (item / "output.wav").write_bytes(b"")
            self.assertEqual(iter_item_dirs(root), [item])
            jobs = {job.output_name: job.source.name for job in transcript_plan(item)}
            self.assertEqual(jobs, {"input.json": "input.wav", "output.json": "output.wav"})
            self.assertEqual(missing_required(transcript_plan(item)), [])


if __name__ == "__main__":
    unittest.main()
