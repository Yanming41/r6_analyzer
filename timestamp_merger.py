"""
Timestamp merging: convert per-segment local timestamps to global video timestamps
and remove near-duplicates that straddle segment boundaries.
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_DEDUP_WINDOW_SECS = 3.0  # events of the same type within this window are merged


def _seconds_to_mmss(secs: float) -> str:
    return f"{int(secs // 60):02d}:{int(secs % 60):02d}"


class TimestampMerger:
    """Adds segment offsets and deduplicates the merged event list."""

    def apply_offset(self, events: List[Dict], offset: float) -> List[Dict]:
        """
        Shift every event's timestamp by `offset` seconds (the segment's start
        time in the source video) to produce global timestamps.
        """
        adjusted = []
        for ev in events:
            global_secs = ev["timestamp"] + offset
            adjusted.append(
                {
                    **ev,
                    "timestamp": global_secs,
                    "time_str": _seconds_to_mmss(global_secs),
                    "local_timestamp": ev["timestamp"],
                    "segment_offset": offset,
                }
            )
        return adjusted

    def merge_and_sort(self, all_events: List[Dict]) -> List[Dict]:
        """
        Sort all events globally and remove near-duplicates:
        if two events share the same type and are within _DEDUP_WINDOW_SECS,
        keep only the first occurrence.
        """
        if not all_events:
            return []

        sorted_events = sorted(all_events, key=lambda e: e["timestamp"])

        deduped: List[Dict] = []
        for ev in sorted_events:
            dup = any(
                ev["event_type"] == prev["event_type"]
                and abs(ev["timestamp"] - prev["timestamp"]) < _DEDUP_WINDOW_SECS
                for prev in deduped[-10:]
            )
            if not dup:
                deduped.append(ev)

        removed = len(sorted_events) - len(deduped)
        if removed:
            logger.info(f"Removed {removed} near-duplicate events (within {_DEDUP_WINDOW_SECS}s window)")

        return deduped
