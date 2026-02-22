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
