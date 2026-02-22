from fastapi import APIRouter

from app.models.schemas import MarkerSyncRequest, MarkerSyncResponse
from app.services.marker_service import MarkerService

router = APIRouter(tags=["markers"])
service = MarkerService()


@router.post("/markers/sync", response_model=MarkerSyncResponse)
def sync_markers(payload: MarkerSyncRequest) -> MarkerSyncResponse:
    return service.sync(payload)


@router.get("/markers/available")
def available_markers() -> list[dict]:
    return service.list_available_days()


@router.get("/markers/day/{day}/index")
def day_marker_index(day: int) -> dict:
    return service.day_index(day)
