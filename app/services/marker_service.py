from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.core.storage import atomic_write_json, ensure_dir
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
