from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.core.storage import atomic_write_json, ensure_dir, read_json
from app.models.schemas import MarkerSyncRequest, MarkerSyncResponse


class MarkerService:
    def __init__(self) -> None:
        ensure_dir(settings.markers_dir)

    def sync(self, payload: MarkerSyncRequest) -> MarkerSyncResponse:
        day_dir = ensure_dir(settings.markers_dir / f"day-{payload.day}")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        snapshot_path = day_dir / f"markers.{timestamp}.json"
        latest_path = day_dir / "latest.json"

        document = {
            "day": payload.day,
            "source_url": str(payload.source_url) if payload.source_url else None,
            "source_video_path": payload.source_video_path,
            "full_refresh": payload.full_refresh,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "markers": [item.model_dump(mode="json") for item in payload.markers],
        }

        atomic_write_json(snapshot_path, document)
        atomic_write_json(latest_path, document)

        return MarkerSyncResponse(
            day=payload.day,
            marker_count=len(payload.markers),
            latest_path=str(latest_path),
            snapshot_path=str(snapshot_path),
        )

    def latest_markers_path(self, day: int) -> Path:
        return settings.markers_dir / f"day-{day}" / "latest.json"

    def latest_markers_document(self, day: int) -> dict:
        path = self.latest_markers_path(day)
        if not path.exists():
            return {}
        payload = read_json(path, default={})
        return payload if isinstance(payload, dict) else {}

    def estimate_draft_inputs(self, day: int, surah_number: int, ayah_start: int, ayah_end: int) -> dict:
        doc = self.latest_markers_document(day)
        markers = doc.get("markers", []) if isinstance(doc.get("markers"), list) else []
        if not markers:
            raise RuntimeError(f"No synced markers found for day {day}.")

        def _num(value: object, fallback: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        in_range = [
            marker
            for marker in markers
            if int(marker.get("surah_number", -1) or -1) == surah_number
            and ayah_start <= int(marker.get("ayah", -1) or -1) <= ayah_end
        ]
        if not in_range:
            raise RuntimeError(
                f"No markers found for day {day}, surah {surah_number}, ayah {ayah_start}-{ayah_end}."
            )

        in_range.sort(key=lambda item: _num(item.get("time")))
        first = in_range[0]
        last = in_range[-1]
        source_id = str(first.get("source_id", "")).strip() or None

        scoped = markers
        if source_id:
            scoped = [m for m in markers if str(m.get("source_id", "")).strip() == source_id]
            if not scoped:
                scoped = markers
        scoped = sorted(scoped, key=lambda item: _num(item.get("time")))

        first_time = _num(first.get("start_time"), _num(first.get("time")))
        last_time = _num(last.get("time"))
        last_end_time = _num(last.get("end_time"), last_time)
        next_marker = next((m for m in scoped if _num(m.get("time")) > last_time), None)

        clip_start = max(0.0, first_time - 4.0)
        if next_marker:
            next_time = _num(next_marker.get("time"))
            clip_end = max(last_end_time + 2.0, min(next_time - 0.35, last_time + 50.0))
        else:
            clip_end = max(last_end_time + 2.0, last_time + 14.0)

        duration = max(12.0, min(180.0, clip_end - clip_start))

        reciter_votes: dict[str, int] = {}
        for marker in in_range:
            reciter = str(marker.get("reciter", "")).strip()
            if not reciter:
                continue
            reciter_votes[reciter] = reciter_votes.get(reciter, 0) + 1
        sheikh = max(reciter_votes.items(), key=lambda kv: kv[1])[0] if reciter_votes else None
        if sheikh in {"Hasan", "Samir"}:
            sheikh = f"Sheikh {sheikh}"

        source_url = doc.get("source_url")
        source_video_path = doc.get("source_video_path")
        if isinstance(source_url, str) and source_url and not source_url.lower().startswith(("http://", "https://")):
            source_video_path = source_video_path or source_url
            source_url = None

        return {
            "estimated_clip_start": round(clip_start, 2),
            "estimated_duration": round(duration, 2),
            "estimated_sheikh": sheikh,
            "source_url": source_url if isinstance(source_url, str) else None,
            "source_video_path": source_video_path if isinstance(source_video_path, str) else None,
            "marker_count_in_range": len(in_range),
        }

    def list_available_days(self) -> list[dict]:
        rows: list[dict] = []
        if not settings.markers_dir.exists():
            return rows

        for entry in settings.markers_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("day-"):
                continue
            try:
                day = int(entry.name.split("-", 1)[1])
            except (ValueError, IndexError):
                continue

            latest = entry / "latest.json"
            payload = read_json(latest, default={}) if latest.exists() else {}
            markers = payload.get("markers", []) if isinstance(payload, dict) else []
            marker_count = len(markers) if isinstance(markers, list) else 0
            synced_at = payload.get("synced_at") if isinstance(payload, dict) else None

            summary_latest = settings.summaries_dir / f"day-{day}" / "latest.json"
            summary_payload = read_json(summary_latest, default={}) if summary_latest.exists() else {}
            summary_title = summary_payload.get("title") if isinstance(summary_payload, dict) else None

            rows.append(
                {
                    "day": day,
                    "marker_count": marker_count,
                    "synced_at": synced_at,
                    "has_summary": bool(summary_title),
                    "summary_title": summary_title,
                }
            )

        rows.sort(key=lambda row: row["day"])
        return rows

    def day_index(self, day: int) -> dict:
        doc = self.latest_markers_document(day)
        markers = doc.get("markers", []) if isinstance(doc.get("markers"), list) else []
        if not markers:
            return {"day": day, "surahs": []}

        surah_map: dict[int, dict] = {}
        for marker in markers:
            try:
                surah_number = int(marker.get("surah_number", 0) or 0)
                ayah = int(marker.get("ayah", 0) or 0)
            except (TypeError, ValueError):
                continue
            if surah_number <= 0 or ayah <= 0:
                continue
            item = surah_map.get(
                surah_number,
                {
                    "surah_number": surah_number,
                    "surah_name": str(marker.get("surah", "")).strip() or f"Surah {surah_number}",
                    "ayahs": set(),
                },
            )
            item["ayahs"].add(ayah)
            surah_map[surah_number] = item

        surahs: list[dict] = []
        for surah_number, item in sorted(surah_map.items(), key=lambda kv: kv[0]):
            ayahs_sorted = sorted(item["ayahs"])
            surahs.append(
                {
                    "surah_number": surah_number,
                    "surah_name": item["surah_name"],
                    "ayahs": ayahs_sorted,
                    "ayah_min": ayahs_sorted[0],
                    "ayah_max": ayahs_sorted[-1],
                }
            )

        return {"day": day, "surahs": surahs}
