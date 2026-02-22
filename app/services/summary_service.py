from datetime import datetime, timezone

from app.core.config import settings
from app.core.storage import atomic_write_json, ensure_dir
from app.models.schemas import SummarySyncRequest, SummarySyncResponse


class SummaryService:
    def __init__(self) -> None:
        ensure_dir(settings.summaries_dir)

    def sync(self, payload: SummarySyncRequest) -> SummarySyncResponse:
        paths: list[str] = []
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        for item in payload.summaries:
            day_dir = ensure_dir(settings.summaries_dir / f"day-{item.day}")
            snapshot_path = day_dir / f"summary.{timestamp}.json"
            latest_path = day_dir / "latest.json"
            document = {
                "day": item.day,
                "title": item.title,
                "summary": item.summary,
                "themes": item.themes,
                "synced_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(snapshot_path, document)
            atomic_write_json(latest_path, document)
            paths.append(str(latest_path))

        return SummarySyncResponse(count=len(payload.summaries), paths=paths)
