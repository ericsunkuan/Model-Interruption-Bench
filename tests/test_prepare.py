import json
import tempfile
import unittest
from pathlib import Path

from mib_eval.prepare import (
    annotation_fields,
    find_solution_start,
    merge_utterances,
    prepare_items,
)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

GSM8K_ANN = [
    {
        "index": 0, "entry_type": "incorrect", "question": "How many?",
        "ground_truth_solution": "4*5+6=26", "model_output_solution_first_error_step": 2.0,
        "model_output_steps": "['Step 1: 4*5=20.', 'Step 2: 20+5=25.', 'Step 3: The answer is 25.']",
        "model_output_solution_first_error_reason": "Should add 6, not 5.",
        "rewritten_solution": '["Step 1"] Four times five is twenty. ["Step 2"] Twenty plus five is '
                              'twenty five. ["Step 3"] The answer is twenty five.',
    },
    {
        "index": 1, "entry_type": "correct", "question": "How many?",
        "ground_truth_solution": "4*5+6=26", "model_output_solution_first_error_step": None,
        "model_output_steps": "['Step 1: 4*5+6=26.']", "model_output_solution_first_error_reason": None,
        "rewritten_solution": '["Step 1"] Four times five plus six is twenty six.',
    },
]


def words(text, start, step=0.5):
    return [{"text": w, "timestamp": [start + i * step, start + i * step + 0.4]}
            for i, w in enumerate(text.split())]


class SolutionStartTest(unittest.TestCase):
    def test_exact_contracted_and_fuzzy_cues(self):
        exact = words("Here is the question. How many? Here is my answer. Four", 0)
        contracted = words("Here is the question. How many? Here's my answer. Four", 0)
        fuzzy = words("Here is the question. How many? Here is why answer. Four", 0)
        lowercase = words("question. All here is my answer: Four", 0)
        self.assertEqual(find_solution_start(exact), 10)
        self.assertEqual(find_solution_start(contracted), 9)
        self.assertEqual(find_solution_start(fuzzy), 10)
        self.assertEqual(find_solution_start(lowercase), 6)

    def test_question_cue_is_not_the_answer_cue(self):
        self.assertIsNone(find_solution_start(words("Here is the question. Four", 0)))

    def test_merge_utterances_splits_on_one_second_gaps(self):
        chunks = words("mm hmm", 10, step=0.3) + words("wait that is wrong", 15)
        self.assertEqual([u["text"] for u in merge_utterances(chunks)], ["mm hmm", "wait that is wrong"])


class AnnotationTest(unittest.TestCase):
    def test_gsm8k_alignment_is_spoken_error_step_only(self):
        ann = {row["index"]: row for row in GSM8K_ANN}
        fields = annotation_fields(ann, 0)
        self.assertTrue(fields["has_error"])
        self.assertEqual(fields["error_step_alignment_text"], "Twenty plus five is twenty five.")
        # The judge still sees the error step and the next one, as before.
        self.assertEqual(fields["ground_truth_error_step_text"], "Step 2: 20+5=25. Step 3: The answer is 25.")
        self.assertIsNone(annotation_fields(ann, 1)["ground_truth_error_step_text"])


class PrepareItemsTest(unittest.TestCase):
    def build(self, root: Path, item_id: int, user_text: str, model_chunks):
        d = root / str(item_id)
        d.mkdir()
        user = words(user_text, 0)
        (d / "input.json").write_text(json.dumps({"text": user_text, "chunks": user}))
        (d / "output.json").write_text(json.dumps({"text": "", "chunks": model_chunks}))

    def test_end_to_end_with_fake_judge(self):
        calls = []

        def judge(prompt):
            calls.append(prompt)
            if "## Text snippet" in prompt:
                return "7.5"
            return "(1)" if "\n\nmm" in prompt else "(2)"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann_path = root / "ann.json"
            ann_path.write_text(json.dumps(GSM8K_ANN))
            asr = root / "asr"
            asr.mkdir()
            solution = ("Here is the question. How many? Here is why answer. Four times five is twenty. "
                        "Twenty plus five is twenty five. The answer is twenty five.")
            # Model: a backchannel, then a real interruption, then a post-hoc reply.
            # User solution words start at 5.0 s and end at 12.9 s.
            model = words("mm", 7.0) + words("wait that should be plus six", 10.2) + words("so twenty six", 40.0)
            self.build(asr, 0, solution, model)
            self.build(asr, 1, "no cue in this clip at all", [])

            records = prepare_items(
                asr, ann_path, judge=judge,
                timestamp_template=(PROMPTS / "timestamp_mapping_evaluation_prompt.txt").read_text(),
                backchannel_template=(PROMPTS / "interruption_type.txt").read_text(),
                workers=2,
            )

        first, broken = records
        # "Twenty plus five is twenty five." is spoken 7.5 s to 10.4 s.
        self.assertEqual(first["ground_truth_error_step_timestamp"], 7.5)
        self.assertEqual(first["ground_truth_error_step_end_timestamp"], 10.4)
        self.assertTrue(first["has_any_non_bc"])
        self.assertEqual(first["first_non_bc_interruption_time"], 10.2)
        self.assertEqual(first["first_non_bc_interruption_text"], "wait that should be plus six")
        self.assertEqual(first["model_output_after_solution_ends_text"], "so twenty six")
        self.assertEqual([u["backchannel_label"] for u in first["mid_speech_utterances"]],
                         ["backchannel", "non_backchannel"])
        # The user text handed to the 4-way prompt stops at the real interruption.
        self.assertTrue(first["user_solution_before_model_speaks"].startswith("Four times five"))
        self.assertNotIn("answer is", first["user_solution_before_model_speaks"])
        self.assertNotIn("_input_data", first)
        # Unparseable item is kept, not dropped.
        self.assertIn("cue", broken["parse_error"])
        self.assertEqual(len(records), 2)
        # Timestamp prompt was asked about the spoken error step only.
        ts_prompts = [c for c in calls if "## Text snippet" in c]
        self.assertEqual(len(ts_prompts), 1)
        self.assertIn("Twenty plus five is twenty five.", ts_prompts[0])


class PostHocBackchannelTest(PrepareItemsTest):
    SOLUTION = ("Here is the question. How many? Here is my answer. Four times five is twenty. "
                "Twenty plus five is twenty five. The answer is twenty five.")

    def run_one(self, model_chunks):
        calls = []

        def judge(prompt):
            calls.append(prompt)
            if "## Text snippet" in prompt:
                return "9.5"
            return "(1)" if "\n\nmm" in prompt else "(2)"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann_path = root / "ann.json"
            ann_path.write_text(json.dumps(GSM8K_ANN))
            asr = root / "asr"
            asr.mkdir()
            self.build(asr, 0, self.SOLUTION, model_chunks)
            (record,) = prepare_items(
                asr, ann_path, judge=judge,
                timestamp_template=(PROMPTS / "timestamp_mapping_evaluation_prompt.txt").read_text(),
                backchannel_template=(PROMPTS / "interruption_type.txt").read_text(),
            )
        return record, calls

    def test_backchannel_only_reply_counts_as_silence(self):
        record, _ = self.run_one(words("mm", 7.0) + words("mm hmm", 40.0))
        self.assertFalse(record["has_any_non_bc"])
        self.assertEqual(record["model_output_after_solution_ends_text"], "")
        self.assertEqual(record["model_output_after_solution_ends_raw_text"], "mm hmm")

    def test_real_reply_is_kept(self):
        record, _ = self.run_one(words("mm hmm", 40.0) + words("you should add six", 43.0))
        self.assertEqual(record["model_output_after_solution_ends_text"], "mm hmm you should add six")

    def test_post_hoc_not_judged_after_mid_speech_interruption(self):
        record, calls = self.run_one(words("wait add six", 10.2) + words("mm hmm", 40.0))
        self.assertTrue(record["has_any_non_bc"])
        self.assertNotIn("backchannel_label", record["post_hoc_utterances"][0])


if __name__ == "__main__":
    unittest.main()
