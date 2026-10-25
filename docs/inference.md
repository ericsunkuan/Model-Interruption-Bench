# Inference

The agent runner is not part of this repository. This page describes the protocol and the output layout the rest of the pipeline expects.

## Full-duplex (halt-on-response)

Input: `mib_gsm8k/gsm8k_full_duplex/<id>/` (see [Dataset](../README.md#dataset)).

The user clip `<id>.wav` is streamed to the agent in real time. `answer_timestamp.json` gives the time at which the user finishes saying "Here is my answer.":

```json
{ "index": 0, "answer_start_timestamp": 35.17, "audio_file": "0.wav" }
```

- **Before `answer_start_timestamp`** (preamble and question), user audio keeps playing while the model speaks, so acknowledgements do not insert silence.
- **After it**, the user stream pauses whenever the model speaks. A frame is silent when the RMS of its decoded int16 samples is below 15. Once the model has been silent for 2 s, the user stream resumes.
- Every pause is written into `padded_input.wav` as silence, so the padded user audio and the model audio share one clock.
- After the user clip ends, the session stays open for 30 s so the model can reply. That reply is the post-hoc utterance.

## Turn-based ablations

Input: `mib_gsm8k/gsm8k_truncated/<id>/<id>.wav` or `mib_gsm8k/gsm8k_full/<id>/<id>.wav`. There is no timestamp file.

The model is kept from taking the floor until the user audio ends, then one response is recorded.

| Setting | User audio |
|---|---|
| Truncated ablation | Played only through the first error step (incorrect items only) |
| Full ablation | The whole solution (incorrect and correct items) |

Commercial models raise the server VAD silence duration to 3000 ms. Moshi, PersonaPlex and Freeze-Omni have no VAD setting, so their runs force the PAD token until the user's silence boundary and then decode normally.

## Output layout

One directory per item, named by the item id, e.g. `551/` for `551.wav`:

| File | Content |
|---|---|
| `padded_input.wav` | User clip with the inserted pauses. The clock for `t_s`, `t_e` and interruption times (full-duplex only) |
| `output.wav` | Model audio on the same clock, silent where the model did not speak |
| `input.wav` | Original user clip |
| `combined.wav` | Stereo mix of user and model audio (optional) |

Turn-based runs never pause the user, so they have no `padded_input.wav`; `input.wav` is already on the model clock.
