from fastapi import APIRouter

from app.models.schemas import (
    FavoriteSyncRequest,
    FavoriteSyncResponse,
    QueueBuildRequest,
    QueueBuildResponse,
)
from app.services.favorite_service import FavoriteService

router = APIRouter(tags=["favorites"])
service = FavoriteService()


@router.post("/favorites/sync", response_model=FavoriteSyncResponse)
def sync_favorites(payload: FavoriteSyncRequest) -> FavoriteSyncResponse:
    return service.sync(payload)


@router.get("/favorites")
def get_favorites() -> dict:
    return service.latest()


@router.post("/queue/from-favorites", response_model=QueueBuildResponse)
def build_queue_from_favorites(payload: QueueBuildRequest) -> QueueBuildResponse:
    return service.build_queue(payload)


@router.get("/queue/current")
def get_current_queue() -> dict:
    return service.current_queue()
