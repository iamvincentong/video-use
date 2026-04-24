#!/usr/bin/env python3
"""Plan B parity harness: runs legacy + vidparse backends on a fixture dir, diffs.

Usage:
    ELEVENLABS_API_KEY=xxx HF_TOKEN=yyy python tools/parity_harness.py <fixture_dir>

Where <fixture_dir> contains one or more .wav files (16 kHz mono preferred).

Output: writes a parity report to <fixture_dir>/parity_report.md and exits 0
if all tolerance gates pass, non-zero otherwise.

Tolerance gates (hardcoded; derived from Plan A's ElevenLabs byte-compat run):
    - word count ratio (vidparse / legacy) in [0.85, 1.15]
    - speaker count: delta <= 1
    - packed-phrase count: within 15% of legacy count (floor 1 for tiny fixtures)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
VIDEO_USE = THIS.parent.parent
sys.path.insert(0, str(VIDEO_USE / "helpers"))

from pack_transcripts import pack_one_file  # noqa: E402
from transcribe import transcribe_one  # noqa: E402


TOLERANCES = {
    # Word-count ratio (vidparse / legacy). Widened from [0.90, 1.10] after
    # review: stress test saw 0.98 on n=1; real fixtures will vary with noise,
    # accent, overlap. First real-fixture run may justify a rebase with evidence.
    "word_count_ratio_min": 0.85,
    "word_count_ratio_max": 1.15,
    # Speaker count: +/-1 allowed. ElevenLabs and pyannote use different clustering
    # and will occasionally differ by 1 on brief/quiet speakers. Larger delta is
    # a real regression.
    "speaker_count_delta_max": 1,
    # Packed-phrase delta as a fraction of legacy phrase count — fixture-size
    # aware. 15% matches the word-count band. Floor of 1 prevents false-fail on
    # tiny fixtures (e.g. 5-phrase clip where 15% rounds to 0).
    "packed_phrase_delta_ratio_max": 0.15,
    "packed_phrase_delta_floor": 1,
}


def run_backend(backend: str, audio: Path, edit_dir: Path) -> Path:
    """Run one backend on one audio, return the written JSON path.

    Note: mutates process env (VIDEO_USE_TRANSCRIBER) and does not restore.
    Harness-internal only — if you reuse this helper from a long-lived
    caller, wrap in try/finally with env restore.
    """
    os.environ["VIDEO_USE_TRANSCRIBER"] = backend
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    transcribe_one(audio, edit_dir, api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
    out = edit_dir / "transcripts" / (audio.stem + ".json")
    if not out.exists():
        raise RuntimeError(f"{backend} did not produce {out}")
    return out


def envelope_stats(json_path: Path) -> dict:
    """Extract parity-relevant stats from a scribe envelope."""
    data = json.loads(json_path.read_text())
    words = [w for w in data.get("words", []) if w.get("type") == "word"]
    events = [w for w in data.get("words", []) if w.get("type") == "audio_event"]
    speakers = sorted({w.get("speaker_id") for w in words if w.get("speaker_id")})
    return {
        "word_count": len(words),
        # Informational only — legacy ElevenLabs emits 0 audio_event entries; a
        # positive vidparse count just confirms YAMNet fired. Diff is expected.
        "audio_event_count": len(events),
        "speaker_count": len(speakers),
        "speakers": speakers,
    }


def packed_phrases(json_path: Path) -> list[str]:
    """Run pack_one_file on this transcript, return the packed phrase texts.

    pack_one_file(json_path, silence_threshold) returns
    (stem, duration, phrases) where phrases is a list of dicts with a 'text'
    key. We call it in-process with the default silence threshold (0.5s).
    """
    _stem, _duration, phrases = pack_one_file(json_path, silence_threshold=0.5)
    return [p["text"] for p in phrases if p.get("text", "").strip()]


def compare_one(audio: Path, tmp: Path) -> dict:
    """Run both backends on one audio; return a delta dict."""
    legacy_dir = tmp / f"{audio.stem}_legacy"
    vidparse_dir = tmp / f"{audio.stem}_vidparse"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    vidparse_dir.mkdir(parents=True, exist_ok=True)

    legacy_json = run_backend("legacy", audio, legacy_dir)
    vidparse_json = run_backend("vidparse", audio, vidparse_dir)

    legacy_stats = envelope_stats(legacy_json)
    vidparse_stats = envelope_stats(vidparse_json)

    legacy_phrases = packed_phrases(legacy_json)
    vidparse_phrases = packed_phrases(vidparse_json)

    ratio = (
        vidparse_stats["word_count"] / legacy_stats["word_count"]
        if legacy_stats["word_count"] else float("inf")
    )

    return {
        "fixture": audio.name,
        "legacy": legacy_stats,
        "vidparse": vidparse_stats,
        "word_count_ratio": round(ratio, 3),
        "packed_phrase_delta": abs(len(vidparse_phrases) - len(legacy_phrases)),
        "legacy_packed": len(legacy_phrases),
        "vidparse_packed": len(vidparse_phrases),
    }


def _phrase_delta_limit(legacy_packed: int) -> int:
    """Fixture-size-aware packed-phrase delta limit."""
    scaled = int(legacy_packed * TOLERANCES["packed_phrase_delta_ratio_max"])
    return max(TOLERANCES["packed_phrase_delta_floor"], scaled)


def evaluate(results: list[dict]) -> tuple[bool, list[str]]:
    """Apply tolerance gates; return (passed, list of failure reasons)."""
    failures = []
    for r in results:
        if not (TOLERANCES["word_count_ratio_min"] <= r["word_count_ratio"] <= TOLERANCES["word_count_ratio_max"]):
            failures.append(
                f"{r['fixture']}: word-count ratio {r['word_count_ratio']} outside "
                f"[{TOLERANCES['word_count_ratio_min']}, {TOLERANCES['word_count_ratio_max']}]"
            )
        speaker_delta = abs(r["legacy"]["speaker_count"] - r["vidparse"]["speaker_count"])
        if speaker_delta > TOLERANCES["speaker_count_delta_max"]:
            failures.append(
                f"{r['fixture']}: speaker-count delta {speaker_delta} > "
                f"{TOLERANCES['speaker_count_delta_max']} "
                f"(legacy={r['legacy']['speaker_count']}, vidparse={r['vidparse']['speaker_count']})"
            )
        phrase_limit = _phrase_delta_limit(r["legacy_packed"])
        if r["packed_phrase_delta"] > phrase_limit:
            failures.append(
                f"{r['fixture']}: packed-phrase delta {r['packed_phrase_delta']} > "
                f"{phrase_limit} "
                f"({int(TOLERANCES['packed_phrase_delta_ratio_max'] * 100)}% of legacy={r['legacy_packed']}, "
                f"floor={TOLERANCES['packed_phrase_delta_floor']}; vidparse={r['vidparse_packed']})"
            )
    return (not failures, failures)


def format_report(results: list[dict], passed: bool, failures: list[str]) -> str:
    """Produce a Markdown parity report."""
    lines = ["# Plan B Parity Report", ""]
    lines.append(f"**Verdict:** {'PASS' if passed else 'FAIL'}")
    lines.append(f"**Fixtures:** {len(results)}")
    lines.append("")
    lines.append("## Per-fixture deltas")
    lines.append("")
    lines.append("| Fixture | Legacy words | Vidparse words | Ratio | Legacy spk | Vidparse spk | Packed delta | Legacy ev | Vidparse ev |")
    lines.append("|---------|--------------|----------------|-------|------------|--------------|--------------|-----------|-------------|")
    for r in results:
        lines.append(
            f"| {r['fixture']} | {r['legacy']['word_count']} | {r['vidparse']['word_count']} | "
            f"{r['word_count_ratio']} | {r['legacy']['speaker_count']} | {r['vidparse']['speaker_count']} | "
            f"{r['packed_phrase_delta']} | {r['legacy']['audio_event_count']} | {r['vidparse']['audio_event_count']} |"
        )
    lines.append("")
    lines.append("_`Legacy ev` / `Vidparse ev` are informational — legacy (ElevenLabs) emits 0, so a positive vidparse count just confirms YAMNet fired. Not a gate._")
    lines.append("")
    lines.append("## Tolerance gates")
    lines.append("")
    for k, v in TOLERANCES.items():
        lines.append(f"- `{k}`: {v}")
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    fixture_dir = Path(sys.argv[1]).resolve()
    if not fixture_dir.is_dir():
        print(f"not a directory: {fixture_dir}", file=sys.stderr)
        return 2

    # Preconditions: both backends will run, so both env vars must be set. Bail
    # early with a clean message — otherwise legacy backend fails mid-run with a
    # confusing HTTP 401.
    missing_env = [v for v in ("ELEVENLABS_API_KEY", "HF_TOKEN") if not os.environ.get(v)]
    if missing_env:
        print(
            f"precondition failure: missing env vars {missing_env}. "
            f"Legacy needs ELEVENLABS_API_KEY; vidparse needs HF_TOKEN.",
            file=sys.stderr,
        )
        return 2

    audios = sorted(fixture_dir.glob("*.wav"))
    if not audios:
        print(f"no .wav files in {fixture_dir}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        results = [compare_one(a, tmp) for a in audios]

    passed, failures = evaluate(results)
    report = format_report(results, passed, failures)
    report_path = fixture_dir / "parity_report.md"
    report_path.write_text(report)
    print(report)
    print(f"\n[wrote {report_path}]")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
