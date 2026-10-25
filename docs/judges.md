# Judges

Every LLM call goes through one client (`mib_eval/judge.py`), with one prompt per judge in `prompts/`.

| Judge | Prompt | Used by | Answer |
|---|---|---|---|
| [Timestamp mapping](#timestamp-mapping) | `timestamp_mapping_evaluation_prompt.txt` | prepare | Start time of the error step, or -1 |
| [Backchannel](#backchannel) | `interruption_type.txt` | prepare, per model utterance | (1) backchannel / (2) not a backchannel |
| [Validity](#validity) | `validity.txt` | score: in-window and post-hoc replies | (1) Valid / (2) Invalid / (3) Unrelated |
| [4-way](#4-way) | `four_way.txt` | score: correct audio | (A) / (B) / (C) / (D) |
| [Turn-based validity](#turn-based-validity) | `validity_last_response.txt` | score-turn-based | (A) Valid / (B) Invalid / (C) Unrelated |

## Model, decoding and caching

| Setting | Value |
|---|---|
| Model | `gpt-5.2-2025-12-11` (`--model` to change) |
| Decoding | One completion, seed 42. Reasoning models (gpt-5, o-series) get no temperature; other models get temperature 0 |
| Retries | Up to 5 attempts with backoff. Empty answers are retried. A call that still fails marks the item unscored |
| Cache | `.openai_cache/judge_cache.jsonl`, keyed by model and prompt (`--cache` to change). Reruns only call the API for new prompts |
| Parallelism | `--workers` items at a time (default 8) |

Labels are read from the last "Final verdict" line of the answer, in any of the judge's formats (`Final verdict: (2)`, `Final verdict: **Invalid (2)**`). Without such a line, the last label in the answer is used. An answer with no label leaves the item unscored.

## Timestamp mapping

Locates the error step in the user's speech.

| Input | Value |
|---|---|
| `{input_data}` | The full user transcript with word timestamps (`input.json`) |
| `{ground_truth_error_step_text}` | The spoken error step |

The judge matches the step to the transcript approximately (ASR errors allowed) and returns the start time of its first word, which becomes `t_s`. It returns -1 if the step is not in the transcript; `t_s` is then unknown. Only incorrect items are mapped. The end `t_e` is found without a judge (see [prepare](prepare.md#steps)).

## Backchannel

Decides whether a model utterance takes the turn.

| Input | Value |
|---|---|
| `{question}`, `{ground_truth_solution}` | The problem and its reference solution |
| `{user_solution_before_model_speaks}` | User speech before this utterance |
| `{model_interruption_text}` | The utterance |

(1) is a backchannel ("hmm", "uh-huh"): the model is listening. (2) means the model takes the turn. Mid-speech utterances are judged in order until the first (2). Post-hoc utterances are judged only when nothing mid-speech was a (2).

## Validity

Decides whether what the model said is a correct correction. Used for the in-window interruption and the post-hoc reply on incorrect audio, and for the post-hoc reply on correct audio.

| Input | Value |
|---|---|
| `{question}`, `{ground_truth_solution}` | The problem and its reference solution |
| `{model_interruption_text}` | The interruption or post-hoc reply |
| `{complete_user_text_intended_to_say}` | The user's whole spoken solution |
| `{error_step_text}` | Error step and the following step (`None` on correct audio) |
| `{error_reason}` | Why the step is wrong (`None` on correct audio) |

| Label | Incorrect audio | Correct audio |
|---|---|---|
| (1) Valid | Identifies the annotated error | Treats the solution as correct |
| (2) Invalid | Wrong issue or wrong reason | Claims an error |
| (3) Unrelated | Not about the solution | No stance on correctness |

## 4-way

Classifies the first mid-speech non-backchannel utterance on correct audio.

| Input | Value |
|---|---|
| `{question}` | The problem |
| `{user_solution_before_model_speaks}` | User speech before the utterance |
| `{model_interruption_text}` | The utterance |

| Label | Meaning | Scored as |
|---|---|---|
| (A) No interrupt or backchannel | The model did not take the turn | post-hoc check, then TN or FP `ph_claim` |
| (B) Interrupt-right | Affirms the user is correct | FP `affirm` |
| (C) Interrupt-wrong | Claims an error | FP `false_claim` |
| (D) Unrelated | Off-topic, a readiness statement, or solving the problem itself | FP `unrelated` |

## Turn-based validity

Judges the single response in the turn-based ablations.

| Input | Value |
|---|---|
| `{question}`, `{ground_truth_solution}` | The problem and its reference solution |
| `{user_solution_before_model_speaks}` | Everything the user said |
| `{model_interruption_text}` | The response |
| `{complete_user_text_intended_to_say}` | The user's whole intended solution |
| `{error_step_text}`, `{error_reason}` | As for validity (`None` on correct items) |

(A) Valid means the response identifies the error, or on a correct item acknowledges that there is none. (B) Invalid means a wrong diagnosis, a missed error, or a claimed error on a correct item. (C) Unrelated means it does not engage with correctness.
