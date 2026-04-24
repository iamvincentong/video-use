# video-use/tools

Operator scripts for the Plan B transcription backend swap.

## `parity_harness.py`

Runs both legacy (ElevenLabs Scribe) and vidparse backends on a fixture dir,
diffs word count, speaker count, and packed-phrase count. Produces a parity
report at `<fixture_dir>/parity_report.md`.

Usage:

```bash
ELEVENLABS_API_KEY=sk_... HF_TOKEN=hf_... \
    uv run python tools/parity_harness.py /path/to/fixture_dir
```

`<fixture_dir>` must contain one or more `.wav` files. 16 kHz mono is strongly
preferred (matches what `transcribe_one()` expects).

Exit codes:
- `0`: all tolerance gates passed
- `1`: one or more gates failed (report shows which)
- `2`: usage / argument error

Tolerance gates are hardcoded near the top of the script:
- word-count ratio (vidparse / legacy) in [0.85, 1.15] — may need to rebase after first real-fixture run; see `TOLERANCES` comments
- speaker count: delta <= 1 (clustering-divergence noise between ElevenLabs and pyannote is expected)
- packed-phrase count: within 15% of legacy phrase count, floor 1 for tiny fixtures
- audio-event count: informational only (legacy emits 0; positive vidparse count just confirms YAMNet fired)

## `smoke_vidparse_backend.py`

One-off smoke test for the vidparse backend. Runs `transcribe_one` with
`VIDEO_USE_TRANSCRIBER=vidparse` on the shared 5-second English fixture and
asserts the output shape matches the Scribe envelope.

Usage:

```bash
HF_TOKEN=hf_... uv run python tools/smoke_vidparse_backend.py
```

## Rollback

Flip the env var:

```bash
export VIDEO_USE_TRANSCRIBER=legacy
```

Or unset it entirely (default is legacy). No data migration needed — both
backends write the same file at the same path.
