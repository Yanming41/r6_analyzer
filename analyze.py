"""
R6 Siege Video Analyzer – CLI entry point.

Usage:
    python analyze.py --video input.mp4 [options]

Options:
    --segment_length INT   Segment length in seconds (default 300)
    --fps FLOAT            Frames per second to sample (default 1.0)
    --max_frames INT       Max frames per segment sent to model (default 64)
    --output_dir PATH      Output directory (default ./output)
    --model STR            HuggingFace model ID (default lmms-lab/LLaVA-Video-7B-Qwen2)
    --quantize             Enable 4-bit quantization (saves ~9 GB VRAM)
    --keep_segments        Do not delete temporary segment files after analysis
    --verbose              Enable DEBUG logging
"""
import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from model_inference import ModelInference
from result_output import ResultOutput
from timestamp_merger import TimestampMerger
from video_processor import VideoProcessor


# ── logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool, log_dir: Path) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s"
    datefmt = "%H:%M:%S"

    log_file = log_dir / f"r6_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rainbow Six Siege gameplay video event analyzer (LLaVA-Video)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Input video file path")
    p.add_argument("--segment_length", type=int, default=300, metavar="SEC",
                   help="Segment duration in seconds")
    p.add_argument("--fps", type=float, default=1.0, metavar="FPS",
                   help="Frame sampling rate sent to the model")
    p.add_argument("--max_frames", type=int, default=64, metavar="N",
                   help="Maximum frames per segment (reduce if OOM)")
    p.add_argument("--output_dir", default="./output", metavar="DIR",
                   help="Where to write results")
    p.add_argument("--model", default="llava-hf/LLaVA-NeXT-Video-7B-hf",
                   help="HuggingFace model ID (default: llava-hf/LLaVA-NeXT-Video-7B-hf). "
                        "For lmms-lab/LLaVA-Video-7B-Qwen2, install the custom package first: "
                        "pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git")
    p.add_argument("--quantize", action="store_true",
                   help="4-bit quantization (~4 GB VRAM instead of ~14 GB)")
    p.add_argument("--keep_segments", action="store_true",
                   help="Keep temporary segment files after analysis")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # ── validate input ────────────────────────────────────────────────────────
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(args.verbose, output_dir)
    logger = logging.getLogger("analyze")

    logger.info("=" * 64)
    logger.info("  R6 Siege Video Analyzer – Starting")
    logger.info(f"  Video        : {video_path}")
    logger.info(f"  Model        : {args.model}")
    logger.info(f"  Segment len  : {args.segment_length}s")
    logger.info(f"  FPS / frames : {args.fps} / {args.max_frames}")
    logger.info(f"  Quantize     : {args.quantize}")
    logger.info(f"  Output       : {output_dir}")
    logger.info("=" * 64)

    # ── Step 1 – segment video ────────────────────────────────────────────────
    logger.info("[Step 1/4] Segmenting video with FFmpeg…")
    seg_dir = output_dir / "segments"
    try:
        proc = VideoProcessor(segment_length=args.segment_length)
        segments = proc.segment_video(video_path, seg_dir)
    except Exception as exc:
        logger.error(f"Video segmentation failed: {exc}")
        return 1

    logger.info(f"  → {len(segments)} segment(s) created")

    # ── Step 2 – load model ───────────────────────────────────────────────────
    logger.info("[Step 2/4] Loading LLaVA-Video model…")
    try:
        inference = ModelInference(
            model_id=args.model,
            fps=args.fps,
            max_frames=args.max_frames,
            quantize=args.quantize,
        )
    except Exception as exc:
        logger.error(f"Model loading failed: {exc}")
        return 1

    # ── Step 3 – analyse segments ─────────────────────────────────────────────
    logger.info("[Step 3/4] Analysing segments…")
    merger = TimestampMerger()
    all_events = []
    failed = 0

    for seg in segments:
        logger.info(
            f"  Segment {seg['index'] + 1}/{len(segments)}: "
            f"{seg['path'].name}  offset={seg['offset']:.0f}s"
        )
        local_events = inference.analyze_segment(seg["path"])

        if local_events is None:
            logger.warning(f"  → Segment {seg['index']} analysis failed, skipping")
            failed += 1
            continue

        global_events = merger.apply_offset(local_events, seg["offset"])
        all_events.extend(global_events)
        logger.info(f"  → {len(local_events)} event(s) detected")

    if failed:
        logger.warning(f"  {failed} segment(s) could not be analysed")

    # ── Step 4 – merge & output ────────────────────────────────────────────────
    logger.info("[Step 4/4] Merging results and writing output…")
    final_events = merger.merge_and_sort(all_events)
    logger.info(f"  Total events: {len(final_events)}")

    output = ResultOutput(output_dir)
    stem = video_path.stem
    json_path = output.save_json(final_events, stem, str(video_path))
    txt_path = output.save_txt(final_events, stem, str(video_path))

    # ── clean up segment files ────────────────────────────────────────────────
    if not args.keep_segments and seg_dir.exists():
        shutil.rmtree(seg_dir, ignore_errors=True)
        logger.debug("Temporary segment files removed")

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"  ANALYSIS COMPLETE: {video_path.name}")
    print("=" * 64)
    output.print_summary(final_events)
    print()
    print(f"  JSON → {json_path}")
    print(f"  TXT  → {txt_path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
