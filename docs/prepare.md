# Prepare (`prepare-full-duplex`)

Turns the ASR transcripts of a full-duplex run into one JSON record per item: where the solution and the error step are, what the model said and when, and whether that speech was a backchannel. The labels themselves are assigned in [scoring](scoring.md).

```bash
python -m mib_eval prepare-full-duplex \
  --asr-root /path/to/asr/gpt/gsm8k_full_duplex \
  --annotations /path/to/mrgsm8k_annotations.json \
  --output /path/to/gsm8k_full_duplex.json
```

| Flag | Meaning |
|---|---|
| `--asr-root` | ASR output for one run: `<id>/input.json` and `<id>/output.json` |
| `--annotations` | GSM8K annotation file (below) |
| `--output` | New JSON file; an existing file is never overwritten |
| `--workers`, `--model`, `--cache` | Judge settings, see [judges](judges.md) |

## Steps

For each item:

1. **Annotation.** Reads the item's ground truth. `entry_type` says whether the solution is incorrect. For incorrect items it takes the first error step from `model_output_steps` and its spoken (colloquial) form from `rewritten_solution`, split on the `["Step N"]` markers.
2. **Solution span.** Finds the spoken cue "Here is my answer" in `input.json`. The solution starts at the next word and ends at the user's last word (`t_u`). Matching ignores case and punctuation, accepts "Here's", and tolerates one misrecognized word if "answer" is right.
3. **Model utterances.** Merges the words of `output.json` into utterances, splitting at pauses of 1 s or more. Utterances during the solution are mid-speech; utterances after `t_u` are post-hoc.
4. **Error step timing.** The [timestamp judge](judges.md#timestamp-mapping) picks the start of the spoken error step from the ASR words (`t_s`). `t_e` is then aligned deterministically by matching the error step's words forward from `t_s`; if fewer than 30% match, `t_e` is estimated at 0.4 s per word.
5. **Backchannels.** The [backchannel judge](judges.md#backchannel) labels the mid-speech utterances in order and stops at the first non-backchannel one: that is the interruption. If there is none, it also labels the post-hoc utterances; a reply made only of backchannels counts as no reply.

An item that cannot be read (missing transcript, cue not found, no annotation) is kept with a `parse_error` and is reported as unscored later. Judge failures are kept the same way in `judge_error`.

The command prints a short report: item counts, the ids with parse or judge errors, and the incorrect items whose error step could not be located (`t_s` unknown).

## Annotation file

A JSON list with one object per item, matched to the ASR directory by `index`:

| Field | Use |
|---|---|
| `index` | Item id |
| `entry_type` | `incorrect` or `correct` |
| `question`, `ground_truth_solution` | Judge context |
| `model_output_steps` | Written steps (Python list as a string) |
| `model_output_solution_first_error_step` | 1-based first error step |
| `model_output_solution_first_error_reason` | Why that step is wrong |
| `rewritten_solution` | The spoken solution, with `["Step N"]` markers |

## Output fields

| Field | Meaning |
|---|---|
| `id`, `has_error` | Item id; whether the solution contains an error |
| `question`, `ground_truth_solution`, `complete_user_text_intended_to_say` | Judge context |
| `ground_truth_error_step_text` | Error step and the following step, as given to the validity judge |
| `error_step_alignment_text` | Spoken error step alone, used for `t_s` and `t_e` |
| `model_output_solution_first_error_reason` | Error explanation |
| `solution_start_time`, `user_solution_end_time` | Solution start; `t_u` |
| `ground_truth_error_step_timestamp`, `ground_truth_error_step_end_timestamp` | `t_s`, `t_e` (null if not found) |
| `timestamp_mapping_judgement` | Raw timestamp judge answer |
| `mid_speech_utterances`, `post_hoc_utterances` | `{text, start, end}` with `backchannel_label` and raw `backchannel_judgement` where judged |
| `has_any_non_bc` | Whether the model interrupted mid-speech (null if unknown) |
| `first_non_bc_interruption_time`, `first_non_bc_interruption_text` | The interruption |
| `user_solution_before_model_speaks` | User speech before the interruption (whole solution if none) |
| `model_output_after_solution_ends_text` | Post-hoc reply; empty if the model was silent or only backchanneled (original text in `model_output_after_solution_ends_raw_text`) |
| `parse_error`, `judge_error` | Why the item could not be prepared |
