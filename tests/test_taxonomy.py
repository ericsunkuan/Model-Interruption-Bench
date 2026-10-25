import unittest

from mib_eval.score import score_correct_item, score_error_item, score_streaming_item, summarize
from mib_eval.taxonomy import (
    assign_streaming_correct,
    assign_streaming_error,
    interruption_latency,
    parse_backchannel,
    parse_four_way,
    parse_streaming_validity,
    parse_timestamp,
    parse_turn_validity,
)


def error_label(**kw):
    base = dict(
        has_mid_speech_non_bc=True, first_non_bc_time=12.0, error_start=10.0, error_end=11.0,
        in_window_validity="valid", posthoc_text="", posthoc_validity=None,
    )
    base.update(kw)
    out = assign_streaming_error(**base)
    return out["result_type"], out["fn_sub_type"]


class TaxonomyTest(unittest.TestCase):
    def test_wrong_error_is_fn(self):
        self.assertEqual(error_label(in_window_validity="invalid", posthoc_text="ignored"), ("FN", "wrong_error"))

    def test_in_window_valid_is_tp(self):
        self.assertEqual(error_label(), ("TP", None))

    def test_too_early_ignores_content(self):
        self.assertEqual(error_label(first_non_bc_time=1.0), ("FN", "too_early"))

    def test_window_uses_error_end_plus_ten(self):
        self.assertEqual(error_label(first_non_bc_time=21.0), ("TP", None))
        self.assertEqual(error_label(first_non_bc_time=21.5), ("FN", "too_late"))

    def test_missing_inputs_are_unscored_not_guessed(self):
        # Backchannel judge never ran: not "missed".
        self.assertEqual(error_label(has_mid_speech_non_bc=None, first_non_bc_time=None),
                         (None, "needs_backchannel_judge"))
        # The model interrupted but t_s is unknown: not post-hoc.
        self.assertEqual(error_label(error_start=None), (None, "needs_error_timestamp"))
        # In window but no validity label: not wrong_error.
        self.assertEqual(error_label(in_window_validity=None), (None, "needs_validity_judge"))

    def test_parse_timestamp_rejects_minus_one(self):
        self.assertIsNone(parse_timestamp("-1"))
        self.assertIsNone(parse_timestamp(-1))
        self.assertIsNone(parse_timestamp("not found"))
        self.assertEqual(parse_timestamp("44.24"), 44.24)
        self.assertEqual(parse_timestamp(3), 3.0)

    def test_latency_follows_paper_definition(self):
        # t_s = 10, t_e = 11, t_u = 30
        self.assertEqual(interruption_latency(10.5, 10.0, 11.0, 30.0), 0.0)   # during the error step
        self.assertEqual(interruption_latency(14.0, 10.0, 11.0, 30.0), 3.0)   # after it, user speaking
        self.assertEqual(interruption_latency(25.0, 10.0, 11.0, 30.0), 14.0)  # too_late still counts
        self.assertIsNone(interruption_latency(9.0, 10.0, 11.0, 30.0))       # too early
        self.assertIsNone(interruption_latency(31.0, 10.0, 11.0, 30.0))      # after the user finished
        self.assertIsNone(interruption_latency(12.0, None, 11.0, 30.0))
        self.assertIsNone(interruption_latency(None, 10.0, 11.0, 30.0))

    def test_verdict_with_label_word_before_number(self):
        # Real GPT-5.2 output: the label word comes before the number.
        text = "(2) Invalid would apply if it named another step.\n\nFinal verdict: **Valid (1)**"
        self.assertEqual(parse_streaming_validity(text), "valid")
        self.assertEqual(parse_four_way("(C) is not right here.\nFinal verdict: **Unrelated (D)**"), "D")
        # No verdict line: the last label wins, not the first.
        self.assertEqual(parse_streaming_validity("(2) does not fit; the answer is (1)"), "valid")

    def test_parse_turn_validity_prefers_final_verdict(self):
        self.assertEqual(parse_turn_validity("Not (A) here. Final verdict: **Final verdict: (B) Invalid**"), "invalid")
        self.assertEqual(parse_turn_validity("(C)"), "unrelated")

    def test_parse_backchannel(self):
        self.assertEqual(parse_backchannel("(1)"), "backchannel")
        self.assertEqual(parse_backchannel("It takes the turn. (2)"), "non_backchannel")
        self.assertIsNone(parse_backchannel("unclear"))

    def test_posthoc_uses_validity_not_keywords(self):
        base = dict(
            has_mid_speech_non_bc=False,
            first_non_bc_time=None,
            error_start=10.0,
            error_end=11.0,
            in_window_validity=None,
            posthoc_text="The total should be fifty three minutes.",
        )
        corr = assign_streaming_error(**base, posthoc_validity="valid")
        other = assign_streaming_error(**base, posthoc_validity="unrelated")
        waiting = assign_streaming_error(**base, posthoc_validity=None)
        missed = assign_streaming_error(**{**base, "posthoc_text": "  "}, posthoc_validity=None)
        self.assertEqual(corr["fn_sub_type"], "ph_corr")
        self.assertEqual(other["fn_sub_type"], "ph_other")
        self.assertEqual(waiting["fn_sub_type"], "needs_posthoc_judge")
        self.assertIsNone(waiting["result_type"])
        self.assertEqual(missed["fn_sub_type"], "missed")

    def test_parse_label(self):
        self.assertEqual(parse_streaming_validity("Final verdict: (2) Invalid"), "invalid")

    def test_parse_label_prefers_final_verdict(self):
        text = "Valid would be (1) if it named an error. Final verdict: **(3)**"
        self.assertEqual(parse_streaming_validity(text), "unrelated")

    def test_parse_four_way(self):
        self.assertEqual(parse_four_way("Not (A) since it takes the turn. Final verdict: (C)"), "C")
        self.assertIsNone(parse_four_way("no label here"))


class CorrectAudioTest(unittest.TestCase):
    """Paper Table 8, FP half."""

    def label(self, **kw):
        base = dict(has_mid_speech_non_bc=True, four_way=None, posthoc_text="", posthoc_validity=None)
        base.update(kw)
        out = assign_streaming_correct(**base)
        return out["result_type"], out["fp_sub_type"]

    def test_mid_speech_four_way_labels(self):
        self.assertEqual(self.label(four_way="C"), ("FP", "false_claim"))
        self.assertEqual(self.label(four_way="D"), ("FP", "unrelated"))
        self.assertEqual(self.label(four_way="B"), ("FP", "affirm"))

    def test_a_then_posthoc(self):
        claim = self.label(four_way="A", posthoc_text="That should be 26.", posthoc_validity="invalid")
        affirm = self.label(four_way="A", posthoc_text="That's all correct.", posthoc_validity="valid")
        other = self.label(four_way="A", posthoc_text="Nice weather.", posthoc_validity="unrelated")
        silent = self.label(four_way="A", posthoc_text="  ")
        self.assertEqual(claim, ("FP", "ph_claim"))
        self.assertEqual(affirm, ("TN", None))
        self.assertEqual(other, ("TN", None))
        self.assertEqual(silent, ("TN", None))

    def test_no_non_bc_is_a_without_judge(self):
        self.assertEqual(self.label(has_mid_speech_non_bc=False), ("TN", None))

    def test_missing_labels_are_unscored(self):
        self.assertEqual(self.label(), (None, "needs_four_way_judge"))
        self.assertEqual(self.label(has_mid_speech_non_bc=None), (None, "needs_backchannel_judge"))
        self.assertEqual(
            self.label(has_mid_speech_non_bc=False, posthoc_text="That should be 26."),
            (None, "needs_posthoc_judge"),
        )

    def test_score_correct_item_calls_judges_in_order(self):
        calls = []

        def judge(prompt):
            calls.append(prompt)
            return "Final verdict: (A)" if len(calls) == 1 else "Final verdict: (2)"

        item = {
            "id": 1, "has_error": False, "question": "Q?",
            "user_solution_before_model_speaks": "First, four times five",
            "has_any_non_bc": True, "first_non_bc_interruption_text": "mm, okay",
            "model_output_after_solution_ends_text": "Actually the total should be 30.",
            "ground_truth_error_step_text": None, "model_output_solution_first_error_reason": None,
        }
        out = score_correct_item(
            item, validity_template="V {model_interruption_text} {error_step_text}",
            four_way_template="F {model_interruption_text}", judge=judge,
        )
        self.assertEqual(calls, ["F mm, okay", "V Actually the total should be 30. None"])
        self.assertEqual((out["result_type"], out["fp_sub_type"]), ("FP", "ph_claim"))
        self.assertNotIn("four_way_label", item)  # input is not modified


class ErrorItemScoringTest(unittest.TestCase):
    ITEM = {
        "id": 0, "has_error": True, "question": "Q?",
        "has_any_non_bc": True, "first_non_bc_interruption_time": 12.0,
        "first_non_bc_interruption_text": "Wait, that should be 53.",
        "ground_truth_error_step_timestamp": "10.0", "ground_truth_error_step_end_timestamp": 11.0,
        "ground_truth_error_step_text": "step two", "model_output_solution_first_error_reason": "r",
        "model_output_after_solution_ends_text": "",
    }

    def score(self, judge=None, **kw):
        return score_error_item({**self.ITEM, **kw},
                                validity_template="V {model_interruption_text}", judge=judge)

    def test_minus_one_timestamp_is_unscored(self):
        out = self.score(ground_truth_error_step_timestamp="-1", validity_judgement="(1)")
        self.assertEqual((out["result_type"], out["fn_sub_type"]), (None, "needs_error_timestamp"))

    def test_in_window_validity_is_judged_when_missing(self):
        calls = []
        out = self.score(judge=lambda p: calls.append(p) or "Final verdict: (1)")
        self.assertEqual(calls, ["V Wait, that should be 53."])
        self.assertEqual(out["result_type"], "TP")
        self.assertEqual(self.score()["fn_sub_type"], "needs_validity_judge")

    def test_latency_is_stored_per_item(self):
        out = self.score(validity_judgement="(1)", first_non_bc_interruption_time=14.0,
                         user_solution_end_time=30.0)
        self.assertEqual(out["latency_s"], 3.0)
        # Too early: undefined.
        self.assertIsNone(self.score(first_non_bc_interruption_time=5.0, user_solution_end_time=30.0)["latency_s"])

    def test_judge_failure_is_unscored_not_fatal(self):
        def judge(prompt):
            raise RuntimeError("Judge call failed after 5 attempts: boom")

        out = score_streaming_item({**self.ITEM}, prompt_template="V {model_interruption_text}",
                                   judge=judge)
        self.assertEqual((out["result_type"], out["fn_sub_type"]), (None, "judge_failed"))
        self.assertIn("boom", out["judge_error"])

    def test_parse_failed_items_stay_unscored(self):
        out = score_streaming_item({"id": 3, "has_error": True, "parse_error": "cue not found"},
                                   prompt_template=None, judge=None)
        self.assertEqual((out["result_type"], out["fn_sub_type"]), (None, "parse_failed"))


class SummaryTest(unittest.TestCase):
    def test_allow_unscored_uses_scored_denominators(self):
        items = [
            {"has_error": True, "result_type": "TP"},
            {"has_error": True, "result_type": "FN", "fn_sub_type": "missed"},
            {"has_error": True, "result_type": None, "fn_sub_type": "parse_failed"},
            {"has_error": False, "result_type": "TN"},
        ]
        self.assertIsNone(summarize(items)["metrics"])
        self.assertIsNone(summarize(items)["latency"])
        s = summarize(items, allow_unscored=True)
        self.assertEqual(s["metrics"]["TP%"], 50.0)
        self.assertEqual(s["unscored"], {"parse_failed": 1})

    def test_latency_mean_over_defined_items(self):
        items = [
            {"has_error": True, "result_type": "TP", "latency_s": 0.0},
            {"has_error": True, "result_type": "FN", "fn_sub_type": "too_late", "latency_s": 12.0},
            {"has_error": True, "result_type": "FN", "fn_sub_type": "too_early", "latency_s": None},
            {"has_error": False, "result_type": "TN"},
        ]
        self.assertEqual(summarize(items)["latency"], {"mean_s": 6.0, "n": 2})

    def test_metrics_and_fp_sub_types(self):
        items = (
            [{"has_error": True, "result_type": "TP"}]
            + [{"has_error": True, "result_type": "FN", "fn_sub_type": "missed"}] * 3
            + [{"has_error": False, "result_type": "TN"}] * 2
            + [{"has_error": False, "result_type": "FP", "fp_sub_type": "ph_claim"}] * 2
        )
        s = summarize(items)
        self.assertEqual(s["fp_sub_types"], {"ph_claim": 2})
        self.assertEqual(s["metrics"]["TP%"], 25.0)
        self.assertEqual(s["metrics"]["TN%"], 50.0)
        self.assertEqual(s["metrics"]["P%"], 33.33)
        self.assertEqual(s["metrics"]["R%"], 25.0)

    def test_unscored_blocks_metrics(self):
        items = [
            {"has_error": False, "result_type": None, "fp_sub_type": "needs_four_way_judge"},
            {"has_error": False},
        ]
        s = summarize(items)
        self.assertIsNone(s["metrics"])
        self.assertEqual(s["unscored"], {"needs_four_way_judge": 1, "missing_result_type": 1})


if __name__ == "__main__":
    unittest.main()
