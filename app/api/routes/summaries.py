from fastapi import APIRouter

from app.models.schemas import SummarySyncRequest, SummarySyncResponse
from app.services.summary_service import SummaryService

router = APIRouter(tags=["summaries"])
service = SummaryService()


@router.post("/summaries/sync", response_model=SummarySyncResponse)
def sync_summaries(payload: SummarySyncRequest) -> SummarySyncResponse:
    return service.sync(payload)
