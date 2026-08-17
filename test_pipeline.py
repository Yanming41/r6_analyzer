"""
Unit tests for the pipeline components (no GPU required).
Tests parsing, merging, and output logic with synthetic data.

Run after activating the venv:
    python test_pipeline.py
"""
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "✓"
FAIL = "✗"


# ── individual tests ──────────────────────────────────────────────────────────

def test_event_regex():
    from model_inference import _EVENT_RE, _mmss_to_seconds

    samples = [
        ("[00:08] - KILL - Headshot on Rook",         "KILL",        8),
        ("[01:32] - DEATH - Kapkan trap",              "DEATH",       92),
        ("[02:47] - MULTI_KILL - Double at window",    "MULTI_KILL",  167),
        ("[04:55] - ROUND_END - Attack wins",          "ROUND_END",   295),
        ("00:15 – CLUTCH – 1v3 clutch plant",         "CLUTCH",      15),
    ]

    for line, expected_type, expected_secs in samples:
        m = _EVENT_RE.search(line)
        assert m, f"No match for: {line!r}"
        ts, etype, _ = m.groups()
        assert etype.upper() == expected_type, f"Expected {expected_type}, got {etype}"
        secs = _mmss_to_seconds(ts)
        assert secs == expected_secs, f"Expected {expected_secs}s, got {secs}s for {ts!r}"

    # Should NOT match garbage
    assert not _EVENT_RE.search("[XX:YY] - FRAG - whatever"), "False positive on bad type"

    logger.info(f"{PASS} Event regex + timestamp conversion")


def test_no_events():
    from model_inference import _EVENT_RE

    assert not list(_EVENT_RE.finditer("NO_EVENTS_DETECTED"))
    logger.info(f"{PASS} NO_EVENTS_DETECTED produces empty list")


def test_timestamp_merger_offset():
    from timestamp_merger import TimestampMerger

    merger = TimestampMerger()
    events = [
        {"timestamp": 10, "time_str": "00:10", "event_type": "KILL",       "description": "test"},
        {"timestamp": 55, "time_str": "00:55", "event_type": "ROUND_END",  "description": "win"},
    ]
    adjusted = merger.apply_offset(events, 300)

    assert adjusted[0]["timestamp"] == 310, f"Expected 310, got {adjusted[0]['timestamp']}"
    assert adjusted[0]["time_str"] == "05:10", f"Bad time_str: {adjusted[0]['time_str']}"
    assert adjusted[1]["timestamp"] == 355
    assert adjusted[1]["time_str"] == "05:55"
    logger.info(f"{PASS} TimestampMerger offset application")


def test_deduplication():
    from timestamp_merger import TimestampMerger

    merger = TimestampMerger()
    events = [
        {"timestamp": 100.0, "time_str": "01:40", "event_type": "KILL", "description": "a"},
        {"timestamp": 101.5, "time_str": "01:41", "event_type": "KILL", "description": "b"},  # dup
        {"timestamp": 105.0, "time_str": "01:45", "event_type": "KILL", "description": "c"},
        {"timestamp": 200.0, "time_str": "03:20", "event_type": "DEATH","description": "d"},
    ]
    merged = merger.merge_and_sort(events)
    # 101.5 is within 3s of 100.0, should be removed
    assert len(merged) == 3, f"Expected 3 events after dedup, got {len(merged)}"
    assert merged[0]["timestamp"] == 100.0
    logger.info(f"{PASS} Near-duplicate removal")


def test_result_output():
    from result_output import ResultOutput

    events = [
        {"timestamp": 8.0,  "time_str": "00:08", "event_type": "KILL",       "description": "Headshot on Rook"},
        {"timestamp": 295.0,"time_str": "04:55", "event_type": "ROUND_END",  "description": "Attack wins"},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out = ResultOutput(Path(tmp))
        json_path = out.save_json(events, "test_video", "test_video.mp4")
        txt_path  = out.save_txt(events,  "test_video", "test_video.mp4")

        assert json_path.exists()
        assert txt_path.exists()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["total_events"] == 2
        assert data["event_counts"]["KILL"] == 1
        assert data["events"][0]["time_str"] == "00:08"

        txt = txt_path.read_text(encoding="utf-8")
        assert "[00:08]" in txt
        assert "ROUND_END" in txt

    logger.info(f"{PASS} ResultOutput JSON + TXT generation")


def test_ffmpeg_available():
    ok = shutil.which("ffmpeg") and shutil.which("ffprobe")
    if ok:
        logger.info(f"{PASS} FFmpeg and FFprobe found in PATH")
    else:
        logger.warning(
            f"{FAIL} FFmpeg not found – video segmentation will fail.\n"
            "        Install: winget install --id Gyan.FFmpeg"
        )
    return bool(ok)


def test_cuda_available():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"{PASS} CUDA available: {name}  ({vram:.1f} GB)")
            return True
        else:
            logger.warning(f"{FAIL} CUDA not available – check drivers and PyTorch wheel")
            return False
    except ImportError:
        logger.warning(f"{FAIL} torch not installed")
        return False


def test_transformers_version():
    try:
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if ver >= (4, 45):
            logger.info(f"{PASS} transformers {transformers.__version__} (≥4.45 required for LlavaOnevision)")
        else:
            logger.warning(
                f"{FAIL} transformers {transformers.__version__} is too old.\n"
                "        pip install --upgrade transformers>=4.45"
            )
    except ImportError:
        logger.warning(f"{FAIL} transformers not installed")


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 56)
    logger.info("  R6 Analyzer – Pipeline Component Tests")
    logger.info("=" * 56)

    failures = 0
    for fn in [
        test_event_regex,
        test_no_events,
        test_timestamp_merger_offset,
        test_deduplication,
        test_result_output,
    ]:
        try:
            fn()
        except AssertionError as exc:
            logger.error(f"{FAIL} {fn.__name__}: {exc}")
            failures += 1
        except Exception as exc:
            logger.error(f"{FAIL} {fn.__name__} raised: {exc}", exc_info=True)
            failures += 1

    # System checks (non-fatal)
    test_ffmpeg_available()
    test_cuda_available()
    test_transformers_version()

    logger.info("=" * 56)
    if failures == 0:
        logger.info("  All unit tests passed!")
    else:
        logger.error(f"  {failures} test(s) failed.")

    logger.info("")
    logger.info("  Next steps:")
    logger.info("  1. python create_test_video.py        # generate a synthetic R6 clip")
    logger.info("  2. python analyze.py --video test_clip.mp4 --segment_length 30")
    logger.info("=" * 56)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
