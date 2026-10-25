"""Transcribe halt-on-response wavs into the JSON layout the full-duplex scorer reads.

Each inference item directory is expected to contain:

    padded_input.wav       user audio on the conversation clock
    output.wav              model audio on that same clock
    input.wav               original user clip
    combined.wav            stereo mix of padded user audio and the model

The full-duplex scorer aligns the error step against ``<id>/input.json``. That file has to
be the padded user timeline, not the original clip. This script writes:

    padded_input.wav        -> input.json
    output.wav              -> output.json
    input.wav               -> unpadded_input.json
    combined.wav            -> combined.json

Turn-based runs never pause the user, so they have no padded_input.wav and
input.wav is already on the model clock. For those directories:

    input.wav               -> input.json

Word times are seconds from NVIDIA Parakeet-TDT 0.6B v2.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


@dataclass(frozen=True)
class TranscriptJob:
    source: Path
    output_name: str
    role: str


def transcript_plan(item_dir: Path) -> list[TranscriptJob]:
    """Map the wavs in one item directory onto scorer JSON names."""
    jobs: list[TranscriptJob] = []
    padded = item_dir / "padded_input.wav"
    if padded.is_file():
        jobs.append(TranscriptJob(padded, "input.json", "user_padded"))
    model = item_dir / "output.wav"
    if model.is_file():
        jobs.append(TranscriptJob(model, "output.json", "model"))
    original = item_dir / "input.wav"
    if original.is_file():
        if padded.is_file():
            jobs.append(TranscriptJob(original, "unpadded_input.json", "user_original"))
        else:
            jobs.insert(0, TranscriptJob(original, "input.json", "user_unpadded"))
    combined = item_dir / "combined.wav"
    if combined.is_file():
        jobs.append(TranscriptJob(combined, "combined.json", "combined"))
    return jobs


def iter_item_dirs(inference_root: Path) -> list[Path]:
    """Item directories are those that contain a user wav (padded or not)."""
    dirs = {
        path.parent
        for name in ("padded_input.wav", "input.wav")
        for path in inference_root.rglob(name)
        if path.is_file()
    }
    return sorted(dirs)


def missing_required(jobs: list[TranscriptJob]) -> list[str]:
    roles = {job.role for job in jobs}
    missing = []
    if not roles & {"user_padded", "user_unpadded"}:
        missing.append("padded_input.wav or input.wav")
    if "model" not in roles:
        missing.append("output.wav")
    return missing


def transcribe_audio_file(audio_path: Path, asr_model) -> dict:
    """Return ``{"text", "chunks"}`` with word start/end in seconds."""
    import soundfile as sf

    waveform, sample_rate = sf.read(audio_path)
    if getattr(waveform, "ndim", 1) > 1:
        waveform = waveform.mean(axis=1)

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
        sf.write(tmp_name, waveform, sample_rate)
        asr_outputs = asr_model.transcribe([tmp_name], timestamps=True)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)

    chunks = []
    words = []
    for word in asr_outputs[0].timestamp["word"]:
        token = word["word"]
        words.append(token)
        chunks.append({"text": token, "timestamp": [word["start"], word["end"]]})
    return {"text": " ".join(words), "chunks": chunks}


def load_asr_model(model_name: str, device: str):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    if device == "cpu":
        return model.cpu()
    return model.cuda()


def _write_transcript(job: TranscriptJob, dest_dir: Path, asr_model, skip_existing: bool) -> str:
    dest = dest_dir / job.output_name
    if skip_existing and dest.exists():
        return "skipped"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(transcribe_audio_file(job.source, asr_model), indent=2))
    return "wrote"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe MIB inference wavs to input.json and output.json."
    )
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    inference_root = args.inference_root.resolve()
    output_root = args.output_root.resolve()
    item_dirs = iter_item_dirs(inference_root)
    print(f"Inference root: {inference_root}")
    print(f"Output root: {output_root}")
    n_padded = sum(1 for d in item_dirs if (d / "padded_input.wav").is_file())
    print(f"Items: {len(item_dirs)} ({n_padded} with padded_input.wav, "
          f"{len(item_dirs) - n_padded} turn-based with input.wav only)")
    if not item_dirs:
        print("No item directories found. Expected <item>/padded_input.wav or <item>/input.wav.")
        return 1

    planned = 0
    incomplete = 0
    for item_dir in item_dirs:
        jobs = transcript_plan(item_dir)
        missing = missing_required(jobs)
        rel = item_dir.relative_to(inference_root)
        if missing:
            incomplete += 1
            print(f"  {rel}: missing {', '.join(missing)}")
        if args.dry_run:
            for job in jobs:
                print(f"  {job.source} -> {output_root / rel / job.output_name}")
        planned += len(jobs)
    if args.dry_run:
        print(f"Would write {planned} transcripts. Incomplete items: {incomplete}.")
        return 0

    asr_model = load_asr_model(args.model, args.device)
    wrote = skipped = errors = 0
    for item_dir in item_dirs:
        rel = item_dir.relative_to(inference_root)
        dest_dir = output_root / rel
        for job in transcript_plan(item_dir):
            try:
                status = _write_transcript(job, dest_dir, asr_model, args.skip_existing)
            except Exception as exc:
                errors += 1
                print(f"Error processing {job.source}: {exc}")
                continue
            if status == "skipped":
                skipped += 1
            else:
                wrote += 1
                print(f"Wrote {dest_dir / job.output_name}")
    print(f"Wrote {wrote}, skipped {skipped}, errors {errors}, incomplete items {incomplete}.")
    print(f"Transcripts saved to: {output_root}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
