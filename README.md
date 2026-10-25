# Model Interruption Bench

Evaluation code for **Model Interruption Bench (MIB)**, a benchmark for real-time, content-aware error correction in full-duplex voice agents. A synthesized user speaks a step-by-step GSM8K solution that may contain an error, after asking to be corrected immediately. The agent should interrupt with the right correction while the user is still speaking, and stay quiet when the solution is correct.

This repository covers the GSM8K split: the full-duplex evaluation and the two turn-based ablations (Truncated and Full).

## Dataset

Download `mib_gsm8k.zip` and `mrgsm8k_annotations.json` from [Google Drive](https://drive.google.com/drive/folders/19WHSwhe59aSXcHpxX74P1sUGXI3JAyPj?usp=sharing):

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/19WHSwhe59aSXcHpxX74P1sUGXI3JAyPj
unzip mib_gsm8k.zip
```

```
mib_gsm8k/
  gsm8k_full_duplex/<id>/<id>.wav, answer_timestamp.json   600 items, full-duplex evaluation
  gsm8k_truncated/<id>/<id>.wav                            300 items, Truncated ablation
  gsm8k_full/<id>/<id>.wav                                 600 items, Full ablation
```

Each `<id>` is the `index` in `mrgsm8k_annotations.json`. Even ids are incorrect solutions, and `<id> + 1` is the corrected solution to the same question. The Truncated ablation has only the even ids. Audio is 24 kHz mono 16-bit WAV.

## Pipeline

| Stage | Command | Details |
|---|---|---|
| 1. Inference | Your agent runner (halt-on-response protocol) | [docs/inference.md](docs/inference.md) |
| 2. ASR | `python -m mib_eval.asr` | [docs/asr.md](docs/asr.md) |
| 3. Prepare | `python -m mib_eval prepare-full-duplex` | [docs/prepare.md](docs/prepare.md) |
| 4. Score | `python -m mib_eval score-full-duplex` | [docs/scoring.md](docs/scoring.md) |
| Ablations | `python -m mib_eval score-turn-based` | [docs/turn_based.md](docs/turn_based.md) |

All LLM judges (prompts, inputs, labels, decoding, caching) are described in [docs/judges.md](docs/judges.md).

## Install

ASR needs NeMo; the rest only needs the OpenAI client.

```bash
conda create -n mib-asr python=3.10 && conda activate mib-asr
pip install -r requirements-asr.txt

conda create -n mib python=3.10 && conda activate mib
pip install -r requirements.txt
export OPENAI_API_KEY=...
```

## Quick start

```bash
# 2. Word-level transcripts of the user and model audio
python -m mib_eval.asr \
  --inference-root /path/to/inference/gpt/gsm8k_full_duplex \
  --output-root /path/to/asr/gpt/gsm8k_full_duplex

# 3. Locate the solution and error step, run the backchannel judge
python -m mib_eval prepare-full-duplex \
  --asr-root /path/to/asr/gpt/gsm8k_full_duplex \
  --annotations /path/to/mrgsm8k_annotations.json \
  --output /path/to/gsm8k_full_duplex.json

# 4. Judge the interruptions and compute the metrics
python -m mib_eval score-full-duplex \
  --results /path/to/gsm8k_full_duplex.json \
  --output /path/to/gsm8k_full_duplex_scored.json \
  --judge

# Turn-based ablations (setting: truncated or full)
python -m mib_eval score-turn-based --setting full \
  --asr-root /path/to/asr/gpt/gsm8k_full \
  --annotations /path/to/mrgsm8k_annotations.json \
  --output /path/to/gsm8k_full_scored.json
```

`score-full-duplex` prints TP%, FN%, TN%, FP%, precision, recall, F1, mean latency, and the FN/FP sub-type counts, and writes every item with its label. Items that cannot be scored are reported separately and never guessed; see [docs/scoring.md](docs/scoring.md#unscored-items).

## Repository

| Path | Contents |
|---|---|
| `mib_eval/asr.py` | Parakeet-TDT word-timestamp ASR |
| `mib_eval/prepare.py` | Transcript parsing, error-step timing, backchannel judge |
| `mib_eval/score.py` | Full-duplex scorer and metrics |
| `mib_eval/taxonomy.py` | TP/FN/TN/FP rules, label parsers, latency |
| `mib_eval/window.py` | Error-step end alignment |
| `mib_eval/turn_based.py` | Truncated and Full ablation scorer |
| `mib_eval/judge.py` | Cached OpenAI judge client |
| `mib_eval/cli.py` | `python -m mib_eval` commands |
| `prompts/` | Judge prompts |
| `docs/` | Per-stage documentation |
| `tests/` | Unit tests (`python -m unittest discover -s tests`) |

## License

Code and data are released under [Apache-2.0](LICENSE). The data is derived from MR-GSM8K (Apache-2.0) and GSM8K (MIT); see [NOTICE](NOTICE).

## Citation

```bibtex
@inproceedings{kuan2026mib,
  title={Model Interruption Bench: Evaluating Real-Time Error Correction in Full-Duplex Voice Agents},
  author={Kuan, Shih-Yun Shan and Chiang, Cheng-Han and Lin, Guan-Ting and Lee, Hung-yi},
  year={2026}
}
```
