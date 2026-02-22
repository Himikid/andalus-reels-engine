import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.core.storage import atomic_write_json, ensure_dir, read_json
from app.models.schemas import DraftGenerateRequest, DraftMetadata, DraftResponse, DraftStatus


class DraftService:
    def __init__(self) -> None:
        ensure_dir(settings.drafts_dir)
        ensure_dir(settings.drafts_dir / "index")

    def compute_request_hash(self, req: DraftGenerateRequest) -> str:
        if req.segments:
            parts: list[str] = []
            for seg in req.segments:
                parts.append(
                    ":".join(
                        [
                            str(seg.day),
                            str(seg.surah_number),
                            str(seg.ayah_start),
                            str(seg.ayah_end),
                            str(round(float(seg.clip_start or 0.0), 3)),
                            str(round(float(seg.duration or 0.0), 3)),
                        ]
                    )
                )
            canonical = "|".join(parts)
        else:
            canonical = "|".join(
                [
                    str(req.day),
                    str(req.surah_number),
                    str(req.ayah_start),
                    str(req.ayah_end),
                    str(round(float(req.clip_start or 0.0), 3)),
                    str(round(float(req.duration or 0.0), 3)),
                ]
            )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def create_or_get_by_hash(self, req: DraftGenerateRequest) -> tuple[DraftMetadata, bool]:
        request_hash = self.compute_request_hash(req)
        hash_index_path = settings.drafts_dir / "index" / "request_hash.json"
        index = read_json(hash_index_path, default={}) or {}
        existing_id = index.get(request_hash)
        if existing_id:
            metadata = self.load_metadata(existing_id)
            if metadata:
                return metadata, False

        draft_id = str(uuid4())
        draft_dir = ensure_dir(settings.drafts_dir / draft_id)
        now = datetime.now(timezone.utc)
        metadata = DraftMetadata(
            draft_id=draft_id,
            request_hash=request_hash,
            status=DraftStatus.created,
            created_at=now,
            updated_at=now,
            step="created",
            request=req.model_dump(mode="json"),
            artifacts={
                "metadata": str(draft_dir / "metadata.json"),
                "ayah_timing_map": str(draft_dir / "ayah_timing_map.json"),
                "ayah_timing_map_edited": str(draft_dir / "ayah_timing_map_edited.json"),
                "draft_video": str(draft_dir / "draft.mp4"),
                "audio": str(draft_dir / "audio.wav"),
                "subtitle_map": str(draft_dir / "subtitle_map.json"),
                "subtitle_map_edited": str(draft_dir / "subtitle_map_edited.json"),
                "final_video": str(draft_dir / "final.mp4"),
            },
        )
        self.save_metadata(metadata)
        index[request_hash] = draft_id
        atomic_write_json(hash_index_path, index)
        return metadata, True

    def metadata_path(self, draft_id: str) -> Path:
        return settings.drafts_dir / draft_id / "metadata.json"

    def draft_dir(self, draft_id: str) -> Path:
        return settings.drafts_dir / draft_id

    def save_metadata(self, metadata: DraftMetadata) -> None:
        metadata.updated_at = datetime.now(timezone.utc)
        atomic_write_json(self.metadata_path(metadata.draft_id), metadata.model_dump(mode="json"))

    def load_metadata(self, draft_id: str) -> DraftMetadata | None:
        payload = read_json(self.metadata_path(draft_id), default=None)
        if not payload:
            return None
        return DraftMetadata.model_validate(payload)

    def update_status(self, draft_id: str, status: DraftStatus, step: str, error: str | None = None) -> DraftMetadata:
        metadata = self.load_metadata(draft_id)
        if not metadata:
            raise RuntimeError("Draft not found")
        metadata.status = status
        metadata.step = step
        metadata.error = error
        self.save_metadata(metadata)
        return metadata

    def response_for(self, metadata: DraftMetadata, base_url: str, running: bool = False) -> DraftResponse:
        urls = {
            key: f"{base_url}/draft/{metadata.draft_id}/artifact/{key}"
            for key in [
                "ayah_timing_map",
                "ayah_timing_map_edited",
                "draft_video",
                "audio",
                "subtitle_map",
                "subtitle_map_edited",
                "final_video",
            ]
        }
        return DraftResponse(
            draft_id=metadata.draft_id,
            status=metadata.status,
            metadata=metadata,
            artifact_urls=urls,
            running=running,
        )

    def list_metadata(self) -> list[DraftMetadata]:
        items: list[DraftMetadata] = []
        for entry in settings.drafts_dir.iterdir():
            if not entry.is_dir() or entry.name == "index":
                continue
            metadata = self.load_metadata(entry.name)
            if metadata:
                items.append(metadata)
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items

    def active_metadata(self) -> DraftMetadata | None:
        active_statuses = {
            DraftStatus.created,
            DraftStatus.ready_for_timing,
            DraftStatus.generating,
            DraftStatus.ready_for_edit,
            DraftStatus.rendering_final,
        }
        for item in self.list_metadata():
            if item.status in active_statuses:
                return item
        return None

    def _hash_index_path(self) -> Path:
        return settings.drafts_dir / "index" / "request_hash.json"

    def _remove_from_hash_index(self, draft_id: str) -> None:
        path = self._hash_index_path()
        index = read_json(path, default={}) or {}
        if not isinstance(index, dict):
            return
        removed = False
        for key, value in list(index.items()):
            if value == draft_id:
                index.pop(key, None)
                removed = True
        if removed:
            atomic_write_json(path, index)

    def delete_draft(self, draft_id: str) -> bool:
        path = self.draft_dir(draft_id)
        if not path.exists():
            return False
        self._remove_from_hash_index(draft_id)
        shutil.rmtree(path)
        return True

    def delete_completed_drafts(self) -> int:
        count = 0
        for metadata in self.list_metadata():
            if metadata.status != DraftStatus.completed:
                continue
            if self.delete_draft(metadata.draft_id):
                count += 1
        return count
