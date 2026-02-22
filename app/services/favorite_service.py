from __future__ import annotations

from datetime import datetime, timezone

from app.core.config import settings
from app.core.storage import atomic_write_json, ensure_dir, read_json
from app.models.schemas import FavoriteSyncRequest, FavoriteSyncResponse, QueueBuildRequest, QueueBuildResponse


class FavoriteService:
    def __init__(self) -> None:
        ensure_dir(settings.favorites_dir)
        ensure_dir(settings.queue_dir)

    def sync(self, payload: FavoriteSyncRequest) -> FavoriteSyncResponse:
        path = settings.favorites_dir / "latest.json"
        rows = [item.model_dump(mode="json") for item in payload.items]
        doc = {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "full_refresh": payload.full_refresh,
            "count": len(rows),
            "items": rows,
        }
        atomic_write_json(path, doc)
        return FavoriteSyncResponse(count=len(rows), path=str(path))

    def latest(self) -> dict:
        path = settings.favorites_dir / "latest.json"
        payload = read_json(path, default={})
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _parse_ayah_ref(ayah_ref: str) -> tuple[int, int, int] | None:
        text = str(ayah_ref or "").strip()
        if not text or ":" not in text:
            return None
        surah_part, ayah_part = text.split(":", 1)
        try:
            surah_number = int(surah_part)
        except ValueError:
            return None
        if "-" in ayah_part:
            left, right = ayah_part.split("-", 1)
            try:
                start = int(left)
                end = int(right)
            except ValueError:
                return None
        else:
            try:
                start = int(ayah_part)
            except ValueError:
                return None
            end = start
        if surah_number < 1 or surah_number > 114 or start < 1 or end < start:
            return None
        return surah_number, start, end

    def build_queue(self, payload: QueueBuildRequest) -> QueueBuildResponse:
        doc = self.latest()
        items = doc.get("items", []) if isinstance(doc, dict) else []
        if not isinstance(items, list):
            items = []

        day_filter = {int(day) for day in payload.days if int(day) > 0} if payload.days else set()
        theme_filter = set(payload.include_theme_types) if payload.include_theme_types else set()

        queue_items: list[dict] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            day = int(item.get("day", 0) or 0)
            if day_filter and day not in day_filter:
                continue
            theme_type = str(item.get("theme_type", "")).strip()
            if theme_filter and theme_type not in theme_filter:
                continue
            parsed = self._parse_ayah_ref(str(item.get("ayah_ref", "")))
            if not parsed:
                continue
            surah_number, ayah_start, ayah_end = parsed
            queue_items.append(
                {
                    "id": f"q-{idx:03d}",
                    "status": "pending",
                    "day": day,
                    "surah_number": surah_number,
                    "ayah_start": ayah_start,
                    "ayah_end": ayah_end,
                    "theme_type": theme_type,
                    "short_title": str(item.get("short_title", "")).strip(),
                    "ayah_ref": str(item.get("ayah_ref", "")).strip(),
                    "summary": item.get("summary"),
                }
            )

        out = settings.queue_dir / "current.json"
        queue_doc = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(queue_items),
            "items": queue_items,
        }
        atomic_write_json(out, queue_doc)
        return QueueBuildResponse(count=len(queue_items), path=str(out))

    def current_queue(self) -> dict:
        path = settings.queue_dir / "current.json"
        payload = read_json(path, default={})
        return payload if isinstance(payload, dict) else {}
