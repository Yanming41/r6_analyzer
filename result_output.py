"""
Output formatting: save analysis results as structured JSON and human-readable TXT.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class ResultOutput:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── public ────────────────────────────────────────────────────────────────

    def save_json(
        self, events: List[Dict], video_name: str, video_path: str
    ) -> Path:
        payload = {
            "video": video_path,
            "analysis_date": datetime.now().isoformat(timespec="seconds"),
            "total_events": len(events),
            "event_counts": self._count(events),
            "events": [self._clean(e) for e in events],
        }
        path = self.output_dir / f"{video_name}_events.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"JSON saved: {path}")
        return path

    def save_txt(
        self, events: List[Dict], video_name: str, video_path: str
    ) -> Path:
        lines = [
            "Rainbow Six Siege – Event Analysis Report",
            f"Video   : {video_path}",
            f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Events  : {len(events)}",
            "",
            "=" * 64,
            "  TIMELINE",
            "=" * 64,
        ]

        for ev in events:
            lines.append(f"  [{ev['time_str']}]  {ev['event_type']:<12}  {ev['description']}")

        lines += [
            "",
            "=" * 64,
            "  EVENT COUNTS",
            "=" * 64,
        ]
        for etype, count in sorted(self._count(events).items(), key=lambda x: -x[1]):
            lines.append(f"  {etype:<15} {count:>4}")

        path = self.output_dir / f"{video_name}_events.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"TXT saved : {path}")
        return path

    def print_summary(self, events: List[Dict]) -> None:
        if not events:
            print("  (no events detected)")
            return

        for ev in events:
            print(f"  [{ev['time_str']}]  {ev['event_type']:<12}  {ev['description']}")

        counts = self._count(events)
        print(f"\n  Counts: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _count(events: List[Dict]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for ev in events:
            result[ev["event_type"]] = result.get(ev["event_type"], 0) + 1
        return result

    @staticmethod
    def _clean(ev: Dict) -> Dict:
        """Remove internal fields before JSON serialisation."""
        keep = {"timestamp", "time_str", "event_type", "description"}
        return {k: v for k, v in ev.items() if k in keep}
