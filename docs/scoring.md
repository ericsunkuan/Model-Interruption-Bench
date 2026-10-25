# Scoring (`score-full-duplex`)

Assigns each prepared item a paper label (TP, FN, TN, FP and a sub-type) and computes the metrics.

```bash
python -m mib_eval score-full-duplex \
  --results /path/to/gsm8k_full_duplex.json \
  --output /path/to/gsm8k_full_duplex_scored.json \
  --judge
```

| Flag | Meaning |
|---|---|
| `--results` | Output of [prepare](prepare.md) |
| `--output` | Scored items; must differ from `--results` |
| `--judge` | Call the judges for the labels below. Without it, items that need a judge are unscored |
| `--allow-unscored` | Report metrics over the scored items even if some items are unscored |
| `--workers`, `--model`, `--cache` | Judge settings, see [judges](judges.md) |

## Incorrect audio

The error window is `[t_s, t_e + 10 s]`. The first mid-speech non-backchannel utterance, at time `t`, decides the label:

| Label | Rule |
|---|---|
| TP | `t` in the window and the [validity judge](judges.md#validity) says Valid |
| FN `wrong_error` | `t` in the window, judged Invalid or Unrelated |
| FN `too_early` | `t < t_s` |
| FN `too_late` | `t > t_e + 10 s`, user still speaking |
| FN `ph_corr` | No interruption; the post-hoc reply is judged Valid |
| FN `ph_other` | No interruption; the post-hoc reply is judged Invalid or Unrelated |
| FN `missed` | No interruption and no post-hoc reply |

Timing failures are not split by content. The post-hoc reply goes to the same validity prompt as an in-window interruption.

## Correct audio

The first mid-speech non-backchannel utterance gets the [4-way judge](judges.md#4-way). If there was none, the label is (A) without a judge call, and the post-hoc reply is checked with the validity prompt (error fields set to `None`).

| 4-way label | Post-hoc validity | Label |
|---|---|---|
| (C) Interrupt-wrong | – | FP `false_claim` |
| (D) Unrelated | – | FP `unrelated` |
| (B) Interrupt-right | – | FP `affirm` |
| (A) No interrupt or backchannel | Invalid | FP `ph_claim` |
| (A) No interrupt or backchannel | Valid, Unrelated, or no reply | TN |

## Metrics

| Metric | Definition |
|---|---|
| TP%, FN% | Over incorrect items |
| TN%, FP% | Over correct items |
| P%, R%, F1% | Precision = TP / (TP + FP), recall = TP / (TP + FN) |
| Latency | Mean of `max(0, t − t_e)` over incorrect items whose interruption lands in `[t_s, t_u]`, with `n` = number of such items |

Latency is 0 for an interruption inside the error step and grows with how far past `t_e` it lands. Too-early interruptions and items without an interruption have no latency.

## Unscored items

An item whose label depends on something missing is reported under `unscored` and never counted as TP/FN/TN/FP.

| Reason | Meaning |
|---|---|
| `parse_failed` | Prepare could not read the item (see `parse_error`) |
| `judge_failed` | A judge call failed after its retries (see `judge_error`); rerun to retry |
| `needs_backchannel_judge` | Unknown whether the model interrupted |
| `needs_error_timestamp` | The model interrupted mid-speech but the error step could not be located |
| `needs_validity_judge` | In-window interruption without a validity label |
| `needs_four_way_judge`, `needs_posthoc_judge` | 4-way or post-hoc label missing |

If any item is unscored, the metrics are left empty and the command exits with status 1, unless `--allow-unscored` is given; then the rates are computed over the scored items only.

## Output

The command prints a summary:

```json
{
  "n": 463, "n_error": 232, "n_correct": 231, "n_unknown": 0,
  "counts": {"TP": 19, "FN": 207, "TN": 119, "FP": 106},
  "fn_sub_types": {"ph_corr": 107, "too_late": 30, "...": 0},
  "fp_sub_types": {"false_claim": 47, "...": 0},
  "unscored": {"parse_failed": 12},
  "metrics": {"TP%": 8.4, "FN%": 91.6, "TN%": 52.9, "FP%": 47.1, "P%": 15.2, "R%": 8.41, "F1%": 10.83},
  "latency": {"mean_s": 11.59, "n": 77}
}
```

Each item in `--output` keeps its prepare fields and adds `result_type`, `fn_sub_type`, `fp_sub_type`, `latency_s`, and the raw judge answers (`validity_judgement`, `posthoc_validity_judgement`, `four_way_judgement`, `four_way_label`).
