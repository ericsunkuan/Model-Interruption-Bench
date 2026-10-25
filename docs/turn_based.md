# Turn-based ablations (`score-turn-based`)

The ablations check whether a model can find the error at all when it does not have to interrupt: it hears the user audio, then gives one response. See [inference](inference.md#turn-based-ablations) for how the runs are made.

```bash
python -m mib_eval score-turn-based --setting full \
  --asr-root /path/to/asr/gpt/gsm8k_full \
  --annotations /path/to/mrgsm8k_annotations.json \
  --output /path/to/gsm8k_full_scored.json
```

| Flag | Meaning |
|---|---|
| `--setting` | `truncated` (Truncated ablation) or `full` (Full ablation) |
| `--asr-root` | ASR output of the run (`<id>/input.json`, `<id>/output.json`) |
| `--annotations` | GSM8K annotation file, as for [prepare](prepare.md#annotation-file) |
| `--output` | New JSON file; an existing file is never overwritten |
| `--allow-unscored` | Report metrics over the scored items even if some are unscored |
| `--workers`, `--model`, `--cache` | Judge settings, see [judges](judges.md) |

## How an item is scored

1. The transcripts are parsed as in [prepare](prepare.md#steps): solution span, model utterances.
2. The response is the model speech after the user's last word. Speech during the user's turn is kept as `model_speech_during_user_turn` but not judged.
3. An empty response is `no_response`. Otherwise the [turn-based validity judge](judges.md#turn-based-validity) labels it Valid, Invalid or Unrelated:
   - incorrect item: Valid if it identifies the error;
   - correct item: Valid if it acknowledges there is no error.

The Truncated ablation is defined on incorrect items only; correct items found in a Truncated run are skipped and counted under `excluded_correct_items`.

## Metrics

| Metric | Definition |
|---|---|
| TP% | Valid rate on incorrect items |
| TN% | Valid rate on correct items (Full ablation only) |

`no_response`, Invalid and Unrelated all count as not Valid. Items that cannot be read are unscored (`parse_failed`, `needs_validity_judge`, `unreadable_judgement`) with the same `--allow-unscored` behaviour as [scoring](scoring.md#unscored-items).

## Output

The summary gives the label counts for incorrect and correct items, the metrics and the unscored reasons. Each item in `--output` has `response_text`, `turn_label`, the raw `validity_judgement`, and `unscored_reason` when it could not be scored.
