from fastapi import APIRouter

from app.models.schemas import MarkerSyncRequest, MarkerSyncResponse
from app.services.marker_service import MarkerService

router = APIRouter(tags=["markers"])
service = MarkerService()


@router.post("/markers/sync", response_model=MarkerSyncResponse)
def sync_markers(payload: MarkerSyncRequest) -> MarkerSyncResponse:
    return service.sync(payload)
