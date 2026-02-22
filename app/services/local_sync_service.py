from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.models.schemas import (
    LocalBundleImportResponse,
    MarkerSyncRequest,
    FavoriteSyncRequest,
    SummarySyncRequest,
)
from app.services.favorite_service import FavoriteService
from app.services.marker_service import MarkerService
from app.services.summary_service import SummaryService


class LocalSyncService:
    def __init__(self) -> None:
        self.markers = MarkerService()
        self.summaries = SummaryService()
        self.favorites = FavoriteService()

    def import_bundle(self, bundle_path: str) -> LocalBundleImportResponse:
        path = Path(bundle_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Bundle not found: {path}")

        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        days = payload.get("days", []) if isinstance(payload, dict) else []
        summaries = payload.get("summaries", []) if isinstance(payload, dict) else []
        favorites = payload.get("favorites", []) if isinstance(payload, dict) else []

        marker_days_synced = 0
        marker_rows_synced = 0
        for item in days:
            try:
                req = MarkerSyncRequest(**item)
            except ValidationError as exc:
                raise RuntimeError(f"Invalid day payload in bundle: {exc}") from exc
            resp = self.markers.sync(req)
            marker_days_synced += 1
            marker_rows_synced += resp.marker_count

        summary_days_synced = 0
        if summaries:
            try:
                req = SummarySyncRequest(summaries=summaries)
            except ValidationError as exc:
                raise RuntimeError(f"Invalid summary payload in bundle: {exc}") from exc
            resp = self.summaries.sync(req)
            summary_days_synced = resp.count

        if favorites:
            try:
                favorites_req = FavoriteSyncRequest(items=favorites, full_refresh=True)
                self.favorites.sync(favorites_req)
            except ValidationError as exc:
                raise RuntimeError(f"Invalid favorites payload in bundle: {exc}") from exc

        return LocalBundleImportResponse(
            bundle_path=str(path),
            marker_days_synced=marker_days_synced,
            marker_rows_synced=marker_rows_synced,
            summary_days_synced=summary_days_synced,
        )
