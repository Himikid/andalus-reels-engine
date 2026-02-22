from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.core.worker import worker
from app.models.schemas import DraftGenerateRequest, DraftResponse, DraftStatus, RenderFinalRequest, SubtitleUpdateRequest
from app.services.draft_service import DraftService
from app.services.reel_pipeline import ReelPipeline
from app.services.subtitle_service import SubtitleService

router = APIRouter(tags=["draft"])
drafts = DraftService()
pipeline = ReelPipeline()
subtitles = SubtitleService()


@router.post("/draft/generate", response_model=DraftResponse)
def draft_generate(payload: DraftGenerateRequest, request: Request) -> DraftResponse:
    metadata, created = drafts.create_or_get_by_hash(payload)
    base_url = str(request.base_url).rstrip("/")

    if created:
        job_id = f"generate:{metadata.draft_id}"

        def run() -> None:
            try:
                pipeline.generate_draft(metadata.draft_id)
            except Exception as exc:  # noqa: BLE001
                drafts.update_status(metadata.draft_id, DraftStatus.failed, "failed", error=str(exc))

        worker.submit(job_id, run)

    running = worker.is_running(f"generate:{metadata.draft_id}") or worker.is_running(f"final:{metadata.draft_id}")
    fresh = drafts.load_metadata(metadata.draft_id) or metadata
    return drafts.response_for(fresh, base_url=base_url, running=running)


@router.post("/draft/update-subtitles", response_model=DraftResponse)
def update_subtitles(payload: SubtitleUpdateRequest, request: Request) -> DraftResponse:
    metadata = drafts.load_metadata(payload.draft_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Draft not found")

    draft_dir = drafts.draft_dir(payload.draft_id)
    subtitles.save_edited_map(draft_dir, payload.chunks)
    updated = drafts.update_status(payload.draft_id, metadata.status.ready_for_edit, "subtitles_updated")
    return drafts.response_for(updated, base_url=str(request.base_url).rstrip("/"), running=False)


@router.post("/render/final", response_model=DraftResponse)
def render_final(payload: RenderFinalRequest, request: Request) -> DraftResponse:
    metadata = drafts.load_metadata(payload.draft_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Draft not found")

    job_id = f"final:{payload.draft_id}"

    def run() -> None:
        try:
            pipeline.render_final(payload.draft_id)
        except Exception as exc:  # noqa: BLE001
            drafts.update_status(payload.draft_id, DraftStatus.failed, "failed", error=str(exc))

    worker.submit(job_id, run)
    fresh = drafts.load_metadata(payload.draft_id) or metadata
    return drafts.response_for(fresh, base_url=str(request.base_url).rstrip("/"), running=True)


@router.get("/draft/{draft_id}", response_model=DraftResponse)
def get_draft(draft_id: str, request: Request) -> DraftResponse:
    metadata = drafts.load_metadata(draft_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Draft not found")

    running = worker.is_running(f"generate:{draft_id}") or worker.is_running(f"final:{draft_id}")
    return drafts.response_for(metadata, base_url=str(request.base_url).rstrip("/"), running=running)


@router.get("/drafts", response_model=list[DraftResponse])
def list_drafts(request: Request) -> list[DraftResponse]:
    base_url = str(request.base_url).rstrip("/")
    rows: list[DraftResponse] = []
    for metadata in drafts.list_metadata():
        running = worker.is_running(f"generate:{metadata.draft_id}") or worker.is_running(f"final:{metadata.draft_id}")
        rows.append(drafts.response_for(metadata, base_url=base_url, running=running))
    return rows


@router.get("/draft/{draft_id}/artifact/{artifact_name}")
def stream_artifact(draft_id: str, artifact_name: str) -> FileResponse:
    metadata = drafts.load_metadata(draft_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Draft not found")

    allowed = {
        "draft_video": "draft.mp4",
        "audio": "audio.wav",
        "subtitle_map": "subtitle_map.json",
        "subtitle_map_edited": "subtitle_map_edited.json",
        "final_video": "final.mp4",
    }
    name = allowed.get(artifact_name)
    if not name:
        raise HTTPException(status_code=404, detail="Unknown artifact")

    path = (drafts.draft_dir(draft_id) / name).resolve()
    base = drafts.draft_dir(draft_id).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc

    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not ready")

    media_type = "application/octet-stream"
    if path.suffix == ".mp4":
        media_type = "video/mp4"
    if path.suffix == ".wav":
        media_type = "audio/wav"
    if path.suffix == ".json":
        media_type = "application/json"

    return FileResponse(path, media_type=media_type, filename=path.name)
