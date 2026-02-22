from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


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
    day: int | None = Field(default=None, ge=1, le=30)
    surah_number: int | None = Field(default=None, ge=1, le=114)
    ayah_start: int | None = Field(default=None, ge=1)
    ayah_end: int | None = Field(default=None, ge=1)
    clip_start: float | None = Field(default=None, ge=0)
    duration: float | None = Field(default=None, gt=0)
    sheikh: str | None = None
    youtube_url: HttpUrl | None = None
    style: str = "fit"
    variants: str = "clean"
    align_subtitles: bool = True
    transition_seconds: float = Field(default=0.45, ge=0, le=2.0)
    segments: list["DraftSegmentInput"] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "DraftGenerateRequest":
        if self.segments:
            return self
        if None in (self.day, self.surah_number, self.ayah_start, self.ayah_end):
            raise ValueError("Provide either segments[] or top-level day/surah_number/ayah_start/ayah_end.")
        return self


class DraftSegmentInput(BaseModel):
    segment_id: str | None = None
    day: int = Field(ge=1, le=30)
    surah_number: int = Field(ge=1, le=114)
    ayah_start: int = Field(ge=1)
    ayah_end: int = Field(ge=1)
    clip_start: float | None = Field(default=None, ge=0)
    duration: float | None = Field(default=None, gt=0)
    sheikh: str | None = None
    youtube_url: HttpUrl | None = None
    source_id: str | None = None
    marker_count_in_range: int | None = None


class ResolvedDraftSegment(BaseModel):
    segment_id: str
    day: int
    surah_number: int
    ayah_start: int
    ayah_end: int
    clip_start: float
    duration: float
    sheikh: str | None = None
    youtube_url: str | None = None
    source_id: str | None = None
    marker_count_in_range: int


class DraftEstimateResponse(BaseModel):
    day: int
    surah_number: int
    ayah_start: int
    ayah_end: int
    estimated_clip_start: float
    estimated_duration: float
    estimated_sheikh: str | None = None
    source_url: str | None = None
    source_id: str | None = None
    marker_count_in_range: int


class SubtitleChunk(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    day: int | None = None
    surah_number: int | None = None
    ayah: int | None = None
    source_id: str | None = None
    verified: bool | None = None


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
