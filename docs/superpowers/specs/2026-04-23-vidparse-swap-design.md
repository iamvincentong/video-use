# video-use — Swap ElevenLabs for vidparse

## Purpose

Replace video-use's ElevenLabs Scribe HTTP integration with a direct Python call into `vidparse` (local, Apple Silicon, no cloud). Downstream consumers of the transcript JSON (`pack_transcripts.py`, `render.py`, `timeline_view.py`) are **not modified** — vidparse emits a byte-compatible Scribe-shape JSON via its new `scribe_json` path.

This is Spec B of a two-spec effort. Spec A (`vidparse/docs/superpowers/specs/2026-04-23-scribe-parity-upgrade-design.md`) delivers the vidparse capabilities this spec depends on: diarization, audio events, filler verbatim, `format_scribe` output.

## Preconditions

- Spec A is landed and tests green in vidparse.
- `pip install "vidparse[rich]"` succeeds from the target install location (local editable path or git URL).
- User has run `huggingface-cli login` and accepted the two pyannote model licenses (one-time setup; documented in video-use's SKILL.md).

## Scope

Mechanical swap. No new features in video-use. Invariant: the bytes written to `<edit_dir>/transcripts/<stem>.json` after the swap are shape-compatible with what Scribe wrote before — `pack_transcripts.py`, `render.py`, `timeline_view.py` continue to work without modification.

## Files changed

| File | Action |
|---|---|
| `helpers/transcribe.py` | Rewrite — delete ElevenLabs HTTP path, import vidparse |
| `helpers/transcribe_batch.py` | No change — still wraps `transcribe_one()` in ThreadPoolExecutor |
| `helpers/pack_transcripts.py` | No change |
| `helpers/render.py` | No change |
| `helpers/timeline_view.py` | No change |
| `.env.example` | Delete `ELEVENLABS_API_KEY=` line; if file becomes empty, delete the file |
| `pyproject.toml` | Add `vidparse[rich]` dependency; drop `requests` if unused elsewhere |
| `SKILL.md` | Remove ElevenLabs references; document vidparse setup |
| `README.md` | Same doc updates |

## Component detail

### 1. `helpers/transcribe.py` — rewritten

```python
"""Transcribe a single video via vidparse, emit Scribe-compatible JSON."""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path

from vidparse import parse


def extract_audio(video: Path, wav_out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", str(wav_out)],
        check=True, capture_output=True,
    )


def transcribe_one(
    video: Path,
    edit_dir: Path,
    language: str | None = None,
    num_speakers: int | None = None,
) -> Path:
    """Transcribe one video. Caches per-source in edit_dir/transcripts/<stem>.json."""
    out_json = edit_dir / "transcripts" / f"{video.stem}.json"
    if out_json.exists():
        return out_json

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / f"{video.stem}.wav"
        extract_audio(video, wav)
        result = parse(
            str(wav),
            model="whisper-turbo",
            language=language,
            rich=True,
            diarize=True,
            num_speakers=num_speakers,
            detect_events=True,
            filler_verbatim=True,
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.to_scribe_dict(), indent=2))
    return out_json
```

Notes:

- The `extract_audio` helper is kept explicit. vidparse would extract audio internally too, but video-use already owns the 16kHz-mono-PCM-s16le-WAV contract; passing it a pre-extracted WAV is cheap and keeps control visible.
- `result.to_scribe_dict()` is the in-memory form of vidparse's `format_scribe()`. Writing to a tempfile + reading back would work but round-trips uselessly through disk.
- The CLI entry (`if __name__ == "__main__"`) block is deleted — nothing calls transcribe.py directly; `transcribe_batch.py` is the only caller.

### 2. `helpers/transcribe_batch.py` — unchanged

Kept as-is. The ThreadPoolExecutor with 4 workers still calls `transcribe_one()` for each video. Because vidparse's pipeline holds pyannote + YAMNet models in-process, and because `parse()` internally loads models once per call, the practical parallelism picture is:

- Each worker thread loads its own model instances on first call → 4× model memory on a fully warm pool.
- On Apple Silicon with 32GB unified memory this is workable; on 16GB machines reduce `--workers` to 2.

If memory pressure becomes an issue, a follow-up optimization is to run all transcription in a single worker with an in-process model cache. Out of scope for this spec.

### 3. `.env.example`

Delete the single line `ELEVENLABS_API_KEY=`. If no other env vars remain, delete the file.

### 4. `pyproject.toml`

```toml
[project]
dependencies = [
    # existing:
    "librosa",
    "matplotlib",
    "pillow",
    "numpy",
    # new:
    "vidparse[rich] @ file:///Users/vincent/Workspace/iamvincentong/vidparse",
    # "requests" — REMOVE if no other file imports it; plan will grep to confirm
]
```

The git+ssh URL form is a reasonable alternative if we want this to be installable from elsewhere:

```toml
"vidparse[rich] @ git+ssh://git@github-iamvincentong/iamvincentong/vidparse.git@main",
```

Decision deferred to the implementation plan: pick based on whether we want the two repos to stay loosely coupled via git (preferred) or tightly coupled via local path (fast iteration during development).

### 5. `SKILL.md` — edits

- Remove all occurrences of "ElevenLabs" and "Scribe".
- Replace the transcription capability bullets with:
  > Transcribes with vidparse (local, Apple Silicon via MLX). Word-level timestamps, speaker diarization (pyannote), audio events (laughs / sighs / applause / coughs), filler verbatim (umm, uh). Same downstream guarantees as before.
- Replace setup steps:
  > **Setup:**
  > 1. `pip install "video-use"` (includes `vidparse[rich]` transitively).
  > 2. First-run: `huggingface-cli login`, paste a free HF token.
  > 3. Accept the pyannote licenses at https://huggingface.co/pyannote/speaker-diarization-3.1 and https://huggingface.co/pyannote/segmentation-3.0.
- Hard Rules section: §7 (word-level timestamps ±50–100ms), §8 (verbatim ASR), §9 (cache transcripts, never re-transcribe) are preserved as-is — vidparse meets all three.

### 6. `README.md` — edits

- Replace the "Powered by Scribe" / ElevenLabs mention with a "Powered by vidparse" line.
- Preserve all capability bullets (word timestamps, diarization, audio events, filler preservation) — vidparse delivers all of them.
- Remove any `ELEVENLABS_API_KEY` references from setup instructions.

## Migration verification

Before committing the swap, run this golden-set diff check on 2–3 videos from the current workflow:

1. Save current `transcripts/*.json` produced by ElevenLabs to a backup directory.
2. Delete `transcripts/` cache, re-run `transcribe_batch.py` under the new code.
3. Compare:
   - `pack_transcripts.py` output (`takes_packed.md`): phrase boundaries will shift slightly (different underlying model) — **expected**. Speaker count should match on 2-speaker content.
   - `render.py` subtitle output (`master.srt`): timing alignment should be within ±150ms of ElevenLabs output on spot-checked cut points.
   - `timeline_view.py` composites: visually QC at 3–5 cut boundaries. Look for hidden captions, audio pops, word-boundary cut failures.
4. Run the full render pipeline end-to-end on one video. Confirm `final.mp4` is visually and audibly correct.

Diffs outside these tolerances are a failure signal — investigate before merging.

## Testing

- No new test files required. The byte-compatibility contract means existing `pack_transcripts.py` / `render.py` / `timeline_view.py` behavior is the regression gate; they will fail or produce garbage if the Scribe JSON shape drifts.
- One optional smoke test: `tests/test_transcribe_integration.py` that asserts `transcribe_one()` on a 5-second 2-speaker fixture video produces a JSON whose `words[0]` contains keys `{"type", "text", "start", "end", "speaker_id"}` and whose `type` values are a subset of `{"word", "spacing", "audio_event"}`.

## Rollout sequence

1. Spec A (vidparse upgrade) lands, tests green.
2. Install `vidparse[rich]` locally.
3. Run the golden-set diff check on 2–3 representative videos.
4. Rewrite `helpers/transcribe.py`; update config files and docs.
5. Run full pipeline on one video end-to-end.
6. Commit as a single atomic change (one PR / one commit) so the rollback path is `git revert`.

## Known limitations carried forward

- Apple Silicon only. Cross-platform (Linux / Windows / Intel Mac) support is out of scope. If cross-platform is ever needed, the path is to re-introduce a pluggable backend — explicitly deferred.
- First-run UX friction: HF login + two license clicks. One-time per user, documented prominently in SKILL.md.
- End-to-end transcription time roughly 2× Scribe's wall-clock on the same hardware due to the pyannote pass. This is the price of local, unlimited, cloud-free operation.

## Non-goals

- No URL ingest exposed to video-use (vidparse's yt-dlp capability). If needed later, add a thin helper that calls `vidparse.parse(url, ...)` directly without the ffmpeg-extract step.
- No refactor of `transcribe_batch.py` or downstream helpers.
- No model auto-selection logic in video-use. Whisper-turbo is hardcoded. Non-English support is a follow-up.

## Deliverable checklist

- [ ] Rewrite `helpers/transcribe.py` (~40 lines)
- [ ] Update `pyproject.toml` (add vidparse[rich], drop requests if unused)
- [ ] Update `.env.example` (delete ELEVENLABS_API_KEY)
- [ ] Update `SKILL.md` (remove ElevenLabs refs, add vidparse setup)
- [ ] Update `README.md` (same)
- [ ] Golden-set diff check on 2–3 videos (manual)
- [ ] Full pipeline smoke test on 1 video (manual)
- [ ] (Optional) Add `tests/test_transcribe_integration.py`
