# Using the vidparse Transcription Backend

This is the operator guide for running video-use with the local `vidparse` transcription backend instead of ElevenLabs Scribe.

The switch is a single env var. The tradeoffs, gotchas, and recovery paths are documented here in one place so you can refer to it during soak runs or when onboarding a new machine.

## Overview

video-use has two interchangeable transcription backends:

| Backend | Engine | Requires | Cost model |
|---------|--------|----------|------------|
| `legacy` | ElevenLabs Scribe v1 API (remote) | `ELEVENLABS_API_KEY` | per-minute API billing |
| `vidparse` | Local Whisper (mlx-audio) + pyannote + YAMNet | `HF_TOKEN` (HF model downloads) + `vidparse[rich]` installed | local compute only |

Both produce the same on-disk file at `<edit_dir>/transcripts/<video_stem>.json` in the Scribe envelope shape (`{"words": [...]}` with per-word `{type, text, start, end, speaker_id}`). Downstream (`pack_transcripts.py`, `render.py`, the skill's EDL reasoning) doesn't know which backend ran.

## One-time setup

### 1. Install vidparse[rich] (already done if you `uv sync`ed after PR #1 merged)

```bash
cd /Users/vincent/Workspace/iamvincentong/video-use
uv sync
```

This resolves `vidparse[rich]` from the local editable path at `/Users/vincent/Workspace/iamvincentong/vidparse` and pulls `pyannote.audio`, `onnxruntime`, `torchaudio`, `soundfile`, `huggingface_hub`.

### 2. Accept pyannote model licenses (one-time, per HF account)

Visit each URL while logged in to your HF account and accept the terms:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 3. Store your HF token

```bash
huggingface-cli login   # interactive, pastes token into ~/.cache/huggingface/token
```

Once cached there, `vidparse` finds it automatically — no need to export `HF_TOKEN` in every shell. You can still override by exporting `HF_TOKEN=hf_...` if you want.

### Sanity check

```bash
cd /Users/vincent/Workspace/iamvincentong/video-use
uv run python tools/smoke_vidparse_backend.py
```

Expected output:
```
  saved: english_5s.json (1.2 KB)
OK: wrote .../english_5s.json, <N> words, shape matches Scribe envelope
```

If this works, your environment is ready.

## Using vidparse for a real edit

### Quick switch

```bash
export VIDEO_USE_TRANSCRIBER=vidparse
```

That's it — any subsequent `python helpers/transcribe.py ...`, `claude` session, or skill invocation uses the vidparse backend until you unset or change the var. Put it in `.env` for the folder if you want it sticky for a project.

### Manual transcription (if testing one file)

```bash
cd /Users/vincent/Workspace/iamvincentong/video-use
VIDEO_USE_TRANSCRIBER=vidparse uv run python helpers/transcribe.py /path/to/your/video.mp4
```

Output lands at `<video_parent>/edit/transcripts/<video_stem>.json`.

**Cache-skip gotcha:** if a transcript file already exists at that path (e.g. from a prior legacy run), the backend returns early without re-transcribing. To force a fresh vidparse run, delete the stale JSON first:

```bash
rm /path/to/your/edit/transcripts/<video_stem>.json
```

### Full pipeline via Claude Code

```bash
export VIDEO_USE_TRANSCRIBER=vidparse
cd /path/to/your/videos
claude
```

Then in the session, use your normal prompt (e.g. `edit these into a launch video`). The skill's pipeline — transcribe → pack → EDL → render → self-eval — runs end-to-end.

### Phrase-packing threshold (important)

`helpers/pack_transcripts.py`'s default silence threshold is `0.5s`, tuned for ElevenLabs' acoustic-boundary word timings. Whisper (vidparse) emits DTW-aligned timings that pack tighter — ~88% of word-to-word gaps are under 0.1s — so at 0.5s almost nothing splits, and phrases can run 80+ words (unusable for EDL editing).

**When using the `vidparse` backend, pass `--silence-threshold 0.15`:**

```bash
uv run python helpers/pack_transcripts.py --edit-dir <edit_dir> --silence-threshold 0.15
```

This brings phrase granularity close to legacy's (~8-10 words/phrase avg).

**Inside a Claude Code session:** the skill packs with the default threshold. If phrases come out coarse, ask Claude to re-pack with `--silence-threshold 0.15` and re-plan the EDL. A permanent fix (backend-aware pack default) is an open follow-up.

## Rollback

Any time:

```bash
export VIDEO_USE_TRANSCRIBER=legacy   # back to ElevenLabs Scribe
```

Or unset — the default on `main` after Task 7's PR merges is `vidparse`; before that, the default is `legacy`. No data migration either way: both backends write the same file shape at the same path.

## Troubleshooting

### `vidparse.VidparseAuthError` or `huggingface_hub.errors.GatedRepoError`

Your HF token isn't present or doesn't have accepted pyannote terms. Fix:
1. Ensure `~/.cache/huggingface/token` exists (run `huggingface-cli login`).
2. Accept the two pyannote model licenses (links above under "One-time setup").

### Lots of `None` speaker IDs in the transcript

pyannote didn't attribute those words to a confident speaker — common on short segments, audience noise, or overlapping speech. `render_markdown` in `pack_transcripts.py` handles `None` gracefully (omits the speaker tag), so this is cosmetic not catastrophic.

If it's pervasive (say, >20% of words):
- Try explicit `num_speakers` — pass it through to `transcribe_one` when you know the count.
- Verify audio is mono 16kHz (vidparse re-extracts internally, but clean source helps).
- Check for background music / noise; heavy beds confuse pyannote.

### Phrases are 50+ words long

You're packing with the default 0.5s threshold on vidparse output. Re-run with `--silence-threshold 0.15`.

### Cache-skip using stale ElevenLabs JSON

If you set `VIDEO_USE_TRANSCRIBER=vidparse` but get transcripts that look like ElevenLabs output (e.g. speaker IDs like `"speaker_0"` with finer-grained phrases than you'd expect), a prior legacy JSON is cached at `<edit_dir>/transcripts/<stem>.json`. Delete and re-run.

### ElevenLabs `detected_unusual_activity` (401)

Free tier flags rapid API calls or VPN traffic. Since `vidparse` is local, this doesn't affect the vidparse backend — only the legacy backend and the parity harness (which calls both). If you hit this on the harness, wait 15-30 min and retry, or switch to a paid ElevenLabs tier.

### First run is slow

vidparse lazy-downloads Whisper + pyannote + YAMNet models on first use (~1-2 GB total, cached under `~/.cache/huggingface/`). First transcribe may take a minute to warm up; subsequent runs are fast. Nothing's wrong.

## Parity validation

Before trusting the vidparse backend on a new machine or after a vidparse version bump, run the parity harness:

```bash
ELEVENLABS_API_KEY=... uv run python tools/parity_harness.py /path/to/fixture_dir
```

(`HF_TOKEN` can come from your cached login.)

Exit 0 = both backends agree within tolerance. See `tools/README.md` for tolerance gate details.

Fixtures: any directory with one or more `.wav` files at 16 kHz mono. The Plan B parity run used `tests/fixtures/adapters/english_5s.wav` + `tests/fixtures/stress/two_speakers_short.wav` from the vidparse repo.

## Task 7 soak checklist (for reviewers deciding to flip the default)

Before merging the "flip default to vidparse" PR, tick these by hand:

- [ ] Task 5 parity report: PASS (committed at vidparse `7611e9c`)
- [ ] At least one real production edit completed end-to-end with `VIDEO_USE_TRANSCRIBER=vidparse` (record project + date in the commit message)
- [ ] No unhandled exceptions from the vidparse backend during that run
- [ ] Packed-phrase output from the real run was visually reviewed (not just harness-asserted)
- [ ] Rendered output has no timing-drift complaints on reviewed cuts
