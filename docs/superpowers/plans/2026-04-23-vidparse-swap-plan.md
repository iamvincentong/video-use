# vidparse Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace video-use's ElevenLabs Scribe HTTP integration with a direct Python call into vidparse (local, Apple Silicon). Downstream consumers (`pack_transcripts.py`, `render.py`, `timeline_view.py`) are not modified — vidparse emits a byte-compatible Scribe-shape JSON.

**Architecture:** `helpers/transcribe.py` shrinks from ~175 lines to ~60. It now imports `vidparse.parse(...)`, calls `to_scribe_dict()`, writes the envelope to `transcripts/<stem>.json` (same path + shape as before). Caching, parallelism, and every other part of the pipeline stay identical.

**Tech Stack:** Python 3.10+, vidparse[rich] (local git/path install).

**Spec reference:** `docs/superpowers/specs/2026-04-23-vidparse-swap-design.md`

**Preconditions:**
- Plan A (`vidparse/docs/superpowers/plans/2026-04-23-scribe-parity-upgrade-plan.md`) is complete. `uv run pytest` green in vidparse.
- `huggingface-cli login` run once by the user; pyannote licenses accepted.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `vidparse[rich]`, drop `requests` |
| `helpers/transcribe.py` | Rewrite | ~60 lines; imports `vidparse.parse`, writes scribe envelope |
| `helpers/transcribe_batch.py` | No change | ThreadPoolExecutor still wraps `transcribe_one` |
| `helpers/pack_transcripts.py` | No change | Consumes the same `words[]` shape |
| `helpers/render.py` | No change | Same |
| `helpers/timeline_view.py` | No change | Same |
| `.env.example` | Delete | No secrets required anymore |
| `SKILL.md` | Modify | Remove Scribe mentions; document vidparse setup |
| `README.md` | Modify | Same |
| `tests/test_transcribe_integration.py` | Create (optional) | One smoke test asserting envelope shape |

---

## Task 1: Verify the `requests` dependency is no longer needed

**Files:**
- Inspect: entire `helpers/` directory

- [ ] **Step 1: Grep for `requests` usage**

Run: `grep -rn "import requests\|from requests" helpers/ skills/ poster.html pyproject.toml`
Expected output: Only `helpers/transcribe.py` uses `requests`. If anything else shows up, keep `requests` in `pyproject.toml` at Task 2 time.

- [ ] **Step 2: Record the finding**

Based on the grep output, write a one-line note for Task 2 about whether to remove `requests` from `pyproject.toml`.

No commit needed for this investigation step.

---

## Task 2: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Modify `pyproject.toml`**

Replace the `dependencies` list to add `vidparse[rich]` and remove `requests` if Task 1 confirmed it's unused.

Pick one of two installation modes:

**Option A — local editable path** (fast iteration while developing):

```toml
dependencies = [
    "librosa",
    "matplotlib",
    "pillow",
    "numpy",
    "vidparse[rich] @ file:///Users/vincent/Workspace/iamvincentong/vidparse",
]
```

**Option B — git URL** (portable across machines/CI):

```toml
dependencies = [
    "librosa",
    "matplotlib",
    "pillow",
    "numpy",
    "vidparse[rich] @ git+ssh://git@github-iamvincentong/iamvincentong/vidparse.git@main",
]
```

Choose A for now (fastest path during the swap); switch to B when publishing.

- [ ] **Step 2: Install locally and verify import**

Run:

```bash
uv pip install -e .
uv run python -c "import vidparse; print(vidparse.__file__)"
uv run python -c "from vidparse import parse, TranscriptResult; print('ok')"
```

Expected: `ok` printed, no `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: swap elevenlabs path for vidparse[rich]"
```

---

## Task 3: Delete `.env.example`

**Files:**
- Delete: `.env.example`

- [ ] **Step 1: Confirm the file only contained the ElevenLabs key**

Run: `cat .env.example`
Expected: a single line `ELEVENLABS_API_KEY=` and nothing else.

- [ ] **Step 2: Remove the file**

Run: `trash .env.example`

(Do not use `rm -rf`; `trash` moves it to macOS Trash — recoverable per CLAUDE.md house rule.)

- [ ] **Step 3: Commit**

```bash
git add -A .env.example
git commit -m "config: remove .env.example (no secrets required after vidparse swap)"
```

---

## Task 4: Rewrite `helpers/transcribe.py`

**Files:**
- Modify: `helpers/transcribe.py` (full rewrite)
- Test (optional): `tests/test_transcribe_integration.py`

- [ ] **Step 1: Write the failing smoke test (optional but recommended)**

Create `tests/test_transcribe_integration.py`:

```python
"""Smoke test: transcribe_one() produces a Scribe-shape JSON.

Requires vidparse[rich] installed and HF auth set up. Marked `slow`
so it only runs when explicitly selected.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.slow


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    # Use any committed short sample. Fallback: skip if no fixture.
    candidates = list(Path("tests/fixtures").glob("*.mp4")) if Path("tests/fixtures").exists() else []
    if not candidates:
        pytest.skip("No tests/fixtures/*.mp4 available")
    src = candidates[0]
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    return dst


def test_transcribe_one_produces_scribe_envelope(sample_video, tmp_path):
    from helpers.transcribe import transcribe_one

    edit_dir = tmp_path / "edit"
    out_path = transcribe_one(sample_video, edit_dir)
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    assert "words" in payload
    assert isinstance(payload["words"], list)
    assert len(payload["words"]) > 0

    first = payload["words"][0]
    assert set(first.keys()) == {"type", "text", "start", "end", "speaker_id"}
    assert first["type"] in {"word", "spacing", "audio_event"}
```

If `tests/` / `tests/fixtures/` doesn't exist, you can skip writing this test — the golden-set diff check at Task 6 is the authoritative regression gate.

- [ ] **Step 2: Replace `helpers/transcribe.py` entirely**

Write:

```python
"""Transcribe a video via vidparse, emit Scribe-compatible JSON.

Extracts mono 16 kHz audio via ffmpeg, calls `vidparse.parse(...)` with
diarization + audio-event detection + verbatim-filler prompt enabled,
writes the Scribe-shape envelope to <edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from vidparse import parse


def extract_audio(video_path: Path, dest: Path) -> None:
    """Extract mono 16 kHz PCM WAV from a video via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_one(
    video: Path,
    edit_dir: Path,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to Scribe-compatible transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  transcribing {video.stem}.wav ({size_mb:.1f} MB)", flush=True)

        result = parse(
            str(audio),
            model="whisper-turbo",
            language=language,
            rich=True,
            diarize=True,
            num_speakers=num_speakers,
            detect_events=True,
            filler_verbatim=True,
        )

    envelope = result.to_scribe_dict()
    out_path.write_text(json.dumps(envelope, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        words_count = len(envelope.get("words", []))
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {words_count}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with vidparse")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        language=args.language,
        num_speakers=args.num_speakers,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify `transcribe_batch.py` still wires up**

Read `helpers/transcribe_batch.py`. Confirm its import is `from helpers.transcribe import transcribe_one` (or equivalent relative) and that it calls `transcribe_one(video=..., edit_dir=..., api_key=..., language=..., num_speakers=...)`. **The `api_key` parameter is gone.**

Run: `grep -n "api_key" helpers/transcribe_batch.py`

For every line returned, update the call site to drop `api_key=...`. Do not change the file structurally — only remove the api_key passthrough.

After changes:

```bash
grep -n "api_key\|ELEVENLABS" helpers/transcribe_batch.py
```

Expected: no matches.

- [ ] **Step 4: Syntax check both files**

Run:

```bash
python -m py_compile helpers/transcribe.py helpers/transcribe_batch.py
```

Expected: exit 0, no output.

- [ ] **Step 5: Commit**

```bash
git add helpers/transcribe.py helpers/transcribe_batch.py
if [ -f tests/test_transcribe_integration.py ]; then git add tests/test_transcribe_integration.py; fi
git commit -m "helpers: rewrite transcribe on top of vidparse

Drops ElevenLabs HTTP call and .env-based auth. Uses vidparse.parse()
with diarize + detect_events + filler_verbatim flags, persists the
result via to_scribe_dict() in the same <stem>.json path consumed by
pack_transcripts/render/timeline_view."
```

---

## Task 5: Update `SKILL.md`

**Files:**
- Modify: `SKILL.md`

- [ ] **Step 1: Read the current SKILL.md**

Run: `wc -l SKILL.md && head -120 SKILL.md`
Expected: identify sections that reference "ElevenLabs", "Scribe", or `ELEVENLABS_API_KEY`.

- [ ] **Step 2: Find all references**

Run: `grep -n -i "elevenlabs\|scribe" SKILL.md`

For each hit, decide whether to replace with vidparse phrasing or remove. General replacements:
- "ElevenLabs Scribe" → "vidparse"
- "Scribe (ElevenLabs)" → "vidparse"
- "ELEVENLABS_API_KEY" setup step → vidparse install + HF login block

- [ ] **Step 3: Apply replacements**

Use targeted `Edit` calls rather than a full rewrite. For each grep hit, make a focused edit. Sample replacement for the Setup section (exact text will vary — use Read first to see current state):

Replacement for a setup block like:

> Add `ELEVENLABS_API_KEY=<your key>` to a `.env` file at the project root.

becomes:

> Install vidparse with rich extras: `uv pip install -e .` (bundles `vidparse[rich]`).
> First-run setup: `huggingface-cli login`, then accept the pyannote licenses at
> <https://huggingface.co/pyannote/speaker-diarization-3.1> and
> <https://huggingface.co/pyannote/segmentation-3.0>.

Capability bullets — preserve the substance. Example replacement for a line like:

> Word-level timestamps, diarization, and audio events via ElevenLabs Scribe.

becomes:

> Word-level timestamps, diarization (pyannote), audio events (laughs / sighs / applause / coughs), filler verbatim — all via local vidparse.

- [ ] **Step 4: Verify no ElevenLabs references remain**

Run: `grep -n -i "elevenlabs\|scribe\|xi-api-key\|ELEVENLABS_API_KEY" SKILL.md`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add SKILL.md
git commit -m "docs(SKILL): replace ElevenLabs references with vidparse"
```

---

## Task 6: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find all references**

Run: `grep -n -i "elevenlabs\|scribe" README.md`

- [ ] **Step 2: Apply replacements**

Same approach as Task 5 — targeted `Edit` calls. Replace mentions of ElevenLabs / Scribe with vidparse. Preserve every capability bullet point.

If there's a "Getting Started" block that mentions setting `ELEVENLABS_API_KEY`, replace it with the vidparse install + HF login sequence from Task 5 Step 3.

- [ ] **Step 3: Verify no references remain**

Run: `grep -n -i "elevenlabs\|scribe" README.md`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(README): replace ElevenLabs references with vidparse"
```

---

## Task 7: Golden-set diff verification (manual)

**Files:**
- N/A (manual step, no code changes)

This is the safety net before declaring the swap complete. Do it before merging.

- [ ] **Step 1: Preserve existing ElevenLabs-generated transcripts as a baseline**

Pick 2–3 videos from your current workflow that already have `transcripts/*.json` produced by the ElevenLabs path. Copy the `transcripts/` directory elsewhere:

```bash
cp -r /path/to/edit/transcripts /path/to/edit/transcripts.elevenlabs.baseline
```

- [ ] **Step 2: Regenerate via the new path**

Delete the original `transcripts/` cache and re-run:

```bash
trash /path/to/edit/transcripts
python helpers/transcribe_batch.py <videos_dir> --edit-dir /path/to/edit --workers 2
```

(Use `--workers 2` on 16 GB Macs; 4 on 32 GB+.)

- [ ] **Step 3: Diff the downstream artifacts**

For each video, regenerate the downstream views and compare:

```bash
python helpers/pack_transcripts.py --edit-dir /path/to/edit
diff /path/to/edit/takes_packed.md /path/to/edit.old/takes_packed.md | head -200
```

Expected: phrase boundaries shift slightly (different underlying model). Accept. Red flags:
- Completely different word content (not a fidelity shift — a bug).
- Missing speaker labels (`S0`/`S1`) on content that had them before.
- Missing `(laughs)` / `(applause)` markers at moments the audio clearly has them.

- [ ] **Step 4: End-to-end smoke**

Run the full edit pipeline on one video — render a preview, eyeball the output. Confirm:
- Cuts land on word boundaries (no mid-word chopping).
- Subtitles align within ±150 ms of audio.
- No hidden captions or audio pops at cut boundaries (use `timeline_view` for spot QC).

If any of these fail, stop and triage before merging the swap.

- [ ] **Step 5: Remove the baseline backup when satisfied**

```bash
trash /path/to/edit/transcripts.elevenlabs.baseline
```

---

## Task 8: Final sweep

**Files:**
- N/A (verification only)

- [ ] **Step 1: Confirm no ElevenLabs bits remain anywhere**

Run:

```bash
grep -rni "elevenlabs\|ELEVENLABS_API_KEY\|xi-api-key\|scribe_v1" . --exclude-dir=.git --exclude-dir=.omc --exclude-dir=docs/superpowers
```

Expected: no matches. The `docs/superpowers/` directory is excluded because the spec docs legitimately reference ElevenLabs as the thing we migrated away from.

- [ ] **Step 2: Verify .env.example is gone**

Run: `ls -la .env.example 2>&1`
Expected: `No such file or directory`.

- [ ] **Step 3: Verify imports**

Run:

```bash
python -c "from helpers.transcribe import transcribe_one; print('import ok')"
```

Expected: `import ok`.

- [ ] **Step 4: No commit needed; swap is verified**

---

## Self-Review

After all tasks complete, check against spec (`docs/superpowers/specs/2026-04-23-vidparse-swap-design.md`):

**Spec coverage:**
- [ ] §Preconditions (Plan A done, HF login) — acknowledged as external prerequisites
- [ ] §Scope (byte-compat invariant) — preserved: `pack_transcripts.py`, `render.py`, `timeline_view.py` untouched
- [ ] §Files changed table — Tasks 2, 3, 4, 5, 6 cover every row
- [ ] §Component detail 1 (`transcribe.py` rewrite) — Task 4
- [ ] §Component detail 2 (`transcribe_batch.py` unchanged) — verified in Task 4 Step 3
- [ ] §Component detail 3 (`.env.example`) — Task 3
- [ ] §Component detail 4 (`pyproject.toml`) — Task 2
- [ ] §Component detail 5 (`SKILL.md`) — Task 5
- [ ] §Component detail 6 (`README.md`) — Task 6
- [ ] §Migration verification — Task 7 golden-set diff
- [ ] §Testing — Task 4 Step 1 optional smoke test; Task 7 downstream regression
- [ ] §Rollout sequence — the plan order follows the spec's 6-step sequence
- [ ] §Deliverable checklist — each box has a Task

**Placeholder scan:** No "TBD" / "TODO" / "implement later". Every step has a concrete command or file change.

**Type consistency:** `transcribe_one(video, edit_dir, language=, num_speakers=, verbose=)` — the `api_key` parameter is removed in Task 4 and Task 4 Step 3 updates the sole caller.

**Scope check:** Single-repo, 8 tasks (one manual verification), atomic commits, no creep into vidparse changes.
