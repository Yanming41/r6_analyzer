"""
Video preprocessing: split long recording into fixed-length segments via FFmpeg,
recording each segment's time offset for later global timestamp reconstruction.
"""
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def _check_ffmpeg() -> str:
    """Return path to ffmpeg binary or raise a helpful error."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "FFmpeg / FFprobe not found in PATH.\n"
            "  Install options:\n"
            "    winget install --id Gyan.FFmpeg\n"
            "    or download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH"
        )
    return ffmpeg


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    _check_ffmpeg()
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


class VideoProcessor:
    """Splits a video into equal-length segments and records their time offsets."""

    def __init__(self, segment_length: int = 300):
        """
        Args:
            segment_length: Each segment's target duration in seconds (default 300 = 5 min).
        """
        self.segment_length = segment_length
        _check_ffmpeg()

    def segment_video(self, video_path: Path, output_dir: Path) -> List[Dict]:
        """
        Split video_path into segments of self.segment_length seconds.

        Returns:
            List of dicts, each with:
                path     – Path to segment file
                offset   – Start time of segment in the source video (seconds)
                duration – Actual duration of segment (seconds)
                index    – 0-based segment index
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        total_duration = get_video_duration(video_path)
        logger.info(
            f"Video: {video_path.name}  |  "
            f"Duration: {total_duration:.1f}s ({total_duration / 60:.1f} min)"
        )

        segments: List[Dict] = []
        offset = 0.0
        idx = 0

        while offset < total_duration:
            seg_dur = min(self.segment_length, total_duration - offset)
            out_path = output_dir / f"segment_{idx:04d}.mp4"

            if not out_path.exists():
                logger.info(
                    f"  Cutting segment {idx:04d}: "
                    f"offset={offset:.1f}s  duration={seg_dur:.1f}s  -> {out_path.name}"
                )
                self._cut_segment(video_path, out_path, offset, seg_dur)
            else:
                logger.debug(f"  Segment {out_path.name} already exists, skipping cut.")

            segments.append(
                {
                    "path": out_path,
                    "offset": offset,
                    "duration": seg_dur,
                    "index": idx,
                }
            )
            offset += self.segment_length
            idx += 1

        logger.info(f"Total segments: {len(segments)}")
        return segments

    # ── private ──────────────────────────────────────────────────────────────

    def _cut_segment(
        self, src: Path, dst: Path, offset: float, duration: float
    ) -> None:
        """Run FFmpeg to cut a segment. Uses stream-copy for speed."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{offset:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            # Stream-copy is fast; avoids re-encoding game footage
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for segment {dst.name}:\n{result.stderr[-800:]}"
            )
