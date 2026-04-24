"""Smoke test: does _transcribe_one_vidparse write the expected file shape?

Run manually. Kept long-term as a quick verification of the vidparse backend.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["VIDEO_USE_TRANSCRIBER"] = "vidparse"
sys.path.insert(0, "/Users/vincent/Workspace/iamvincentong/video-use/helpers")
from transcribe import transcribe_one  # noqa: E402

audio = Path("/Users/vincent/Workspace/iamvincentong/vidparse-worktrees/feat-scribe-parity/tests/fixtures/adapters/english_5s.wav")
assert audio.exists(), f"fixture missing: {audio}"

with tempfile.TemporaryDirectory() as tmp:
    edit_dir = Path(tmp) / "edit"
    (edit_dir / "transcripts").mkdir(parents=True)
    transcribe_one(audio, edit_dir, api_key="unused-by-vidparse-backend")
    out = edit_dir / "transcripts" / (audio.stem + ".json")
    assert out.exists(), f"vidparse backend didn't write: {out}. contents: {list((edit_dir / 'transcripts').iterdir())}"
    data = json.loads(out.read_text())
    assert "words" in data, f"no 'words' key: {list(data.keys())}"
    assert data["words"], "empty words list"
    first = data["words"][0]
    expected_keys = {"end", "speaker_id", "start", "text", "type"}
    assert expected_keys.issubset(first.keys()), f"first-word keys: {sorted(first.keys())} missing from {sorted(expected_keys)}"
    print(f"OK: wrote {out}, {len(data['words'])} words, shape matches Scribe envelope")
