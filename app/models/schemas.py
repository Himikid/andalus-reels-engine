from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class MarkerItem(BaseModel):
    time: float = Field(ge=0)
    surah: str
    ayah: int = Field(ge=1)
    surah_number: int = Field(ge=1, le=114)
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    confidence: float | None = None
    quality: Literal["high", "ambiguous", "inferred", "manual"] | None = None
    reciter: str | None = None
    arabic_text: str | None = None
    english_text: str | None = None
    source_id: str | None = None
    source_url: str | None = None


class MarkerSyncRequest(BaseModel):
    day: int = Field(ge=1, le=30)
    source_url: str | None = None
    source_video_path: str | None = None
    full_refresh: bool = True
    markers: list[MarkerItem]


class MarkerSyncResponse(BaseModel):
    day: int
    marker_count: int
    latest_path: str
    snapshot_path: str


class DaySummaryItem(BaseModel):
    day: int = Field(ge=1, le=30)
    title: str
    summary: str
    themes: list[str]


class SummarySyncRequest(BaseModel):
    summaries: list[DaySummaryItem]


class SummarySyncResponse(BaseModel):
    count: int
    paths: list[str]


class DraftStatus(str, Enum):
    created = "created"
    generating = "generating"
    ready_for_edit = "ready_for_edit"
    rendering_final = "rendering_final"
    completed = "completed"
    failed = "failed"


class DraftGenerateRequest(BaseModel):
    day: int = Field(ge=1, le=30)
    surah_number: int = Field(ge=1, le=114)
    ayah_start: int = Field(ge=1)
    ayah_end: int = Field(ge=1)
    clip_start: float = Field(ge=0)
    duration: float = Field(gt=0)
    sheikh: str | None = None
    youtube_url: HttpUrl | None = None
    source_video_path: str | None = None
    style: str = "fit"
    variants: str = "clean"
    align_subtitles: bool = True


class SubtitleChunk(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class SubtitleUpdateRequest(BaseModel):
    draft_id: str
    chunks: list[SubtitleChunk]


class RenderFinalRequest(BaseModel):
    draft_id: str


class DraftMetadata(BaseModel):
    draft_id: str
    request_hash: str
    status: DraftStatus
    created_at: datetime
    updated_at: datetime
    step: str | None = None
    error: str | None = None
    request: dict
    artifacts: dict[str, str]


class DraftResponse(BaseModel):
    draft_id: str
    status: DraftStatus
    metadata: DraftMetadata
    artifact_urls: dict[str, str]
    running: bool
