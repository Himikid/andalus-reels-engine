from fastapi import APIRouter, HTTPException

from app.models.schemas import LocalBundleImportRequest, LocalBundleImportResponse
from app.services.local_sync_service import LocalSyncService

router = APIRouter(tags=["local-sync"])
service = LocalSyncService()


@router.post("/sync/import-local", response_model=LocalBundleImportResponse)
def import_local_bundle(payload: LocalBundleImportRequest) -> LocalBundleImportResponse:
    try:
        return service.import_bundle(payload.bundle_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
