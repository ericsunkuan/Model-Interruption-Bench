# ASR

`python -m mib_eval.asr` transcribes every item directory with word timestamps, using NVIDIA Parakeet-TDT 0.6B v2 (`nvidia/parakeet-tdt-0.6b-v2`). Install the dependencies from `requirements-asr.txt`; a GPU is used by default.

```bash
python -m mib_eval.asr \
  --inference-root /path/to/inference/gpt/gsm8k_full_duplex \
  --output-root /path/to/asr/gpt/gsm8k_full_duplex \
  --skip-existing
```

| Flag | Meaning |
|---|---|
| `--inference-root` | Searched recursively for item directories (any directory with `padded_input.wav` or `input.wav`) |
| `--output-root` | Transcripts are written to the same relative paths under this root |
| `--skip-existing` | Keep transcripts that already exist |
| `--dry-run` | Print the file mapping without loading the model |
| `--device` | `cuda` (default) or `cpu` |
| `--model` | Another NeMo ASR checkpoint |

## File mapping

| Audio | Transcript | Used by |
|---|---|---|
| `padded_input.wav` | `input.json` | prepare (user speech, on the model clock) |
| `output.wav` | `output.json` | prepare (model speech) |
| `input.wav` | `unpadded_input.json` | reference only |
| `combined.wav` | `combined.json` | reference only |

For turn-based directories (no `padded_input.wav`), `input.wav` is transcribed to `input.json` instead.

## Format

```json
{"text": "Here is my answer ...", "chunks": [{"text": "Here", "timestamp": [44.24, 44.48]}, ...]}
```

Times are in seconds. Items whose audio is missing or unreadable are reported at the end of the run; the later stages mark them as unscored.
