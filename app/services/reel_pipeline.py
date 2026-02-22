from __future__ import annotations

from pathlib import Path

from app.core.storage import draft_lock, read_json
from app.models.schemas import DraftStatus
from app.services.audio_service import AudioService
from app.services.draft_service import DraftService
from app.services.marker_service import MarkerService
from app.services.render_service import RenderService
from app.services.subtitle_service import SubtitleService


class ReelPipeline:
    def __init__(self) -> None:
        self.drafts = DraftService()
        self.markers = MarkerService()
        self.audio = AudioService()
        self.subtitles = SubtitleService()
        self.render = RenderService()

    def _resolved_segments(self, request_payload: dict) -> list[dict]:
        segments = request_payload.get("segments")
        if isinstance(segments, list) and segments:
            return segments
        return [
            {
                "segment_id": "seg-1",
                "day": request_payload["day"],
                "surah_number": request_payload["surah_number"],
                "ayah_start": request_payload["ayah_start"],
                "ayah_end": request_payload["ayah_end"],
                "clip_start": request_payload["clip_start"],
                "duration": request_payload["duration"],
                "sheikh": request_payload.get("sheikh"),
                "youtube_url": request_payload.get("youtube_url"),
                "source_id": request_payload.get("source_id"),
                "marker_count_in_range": request_payload.get("marker_count_in_range", 0),
            }
        ]

    def generate_draft(self, draft_id: str) -> None:
        metadata = self.drafts.load_metadata(draft_id)
        if not metadata:
            raise RuntimeError("Draft not found")

        draft_dir = self.drafts.draft_dir(draft_id)
        with draft_lock(draft_dir):
            req = metadata.request
            transition_seconds = float(req.get("transition_seconds", 0.45) or 0.45)
            segments = self._resolved_segments(req)

            self.drafts.update_status(draft_id, DraftStatus.generating, "resolving_sources")

            preview_segments: list[Path] = []
            segment_data: list[dict] = []
            for idx, seg in enumerate(segments, start=1):
                seg_id = seg.get("segment_id") or f"seg-{idx}"
                self.drafts.update_status(draft_id, DraftStatus.generating, f"processing_{seg_id}")

                source_video = self.audio.resolve_video_source(seg.get("youtube_url"))

                preview_file = draft_dir / f"preview-{seg_id}.mp4"
                self.render.generate_preview_segment(
                    source_video,
                    preview_file,
                    float(seg["clip_start"]),
                    float(seg["duration"]),
                )
                preview_segments.append(preview_file)

                marker_doc = read_json(self.markers.latest_markers_path(int(seg["day"])), default={"markers": []})
                marker_list = marker_doc.get("markers", []) if isinstance(marker_doc, dict) else []
                segment_data.append({**seg, "markers": marker_list})

            self.drafts.update_status(draft_id, DraftStatus.generating, "joining_preview")
            self.render.join_with_transitions(preview_segments, draft_dir / "draft.mp4", transition_seconds=transition_seconds)

            self.drafts.update_status(draft_id, DraftStatus.generating, "building_subtitle_map")
            self.subtitles.build_initial_map_for_segments(
                draft_dir=draft_dir,
                segments=segment_data,
                transition_seconds=transition_seconds,
            )

            self.drafts.update_status(draft_id, DraftStatus.generating, "extracting_audio")
            self.audio.extract_full_audio(draft_dir / "draft.mp4", draft_dir / "audio.wav")

            self.drafts.update_status(draft_id, DraftStatus.ready_for_edit, "ready_for_edit")

    def render_final(self, draft_id: str) -> None:
        metadata = self.drafts.load_metadata(draft_id)
        if not metadata:
            raise RuntimeError("Draft not found")

        draft_dir = self.drafts.draft_dir(draft_id)
        with draft_lock(draft_dir):
            req = metadata.request
            transition_seconds = float(req.get("transition_seconds", 0.45) or 0.45)
            segments = self._resolved_segments(req)

            self.drafts.update_status(draft_id, DraftStatus.rendering_final, "rendering_final_segments")
            final_segments: list[Path] = []
            for idx, seg in enumerate(segments, start=1):
                seg_id = seg.get("segment_id") or f"seg-{idx}"
                source_video = self.audio.resolve_video_source(seg.get("youtube_url"))
                final_file = draft_dir / f"final-{seg_id}.mp4"
                self.render.generate_final_segment(
                    source_video,
                    final_file,
                    float(seg["clip_start"]),
                    float(seg["duration"]),
                )
                final_segments.append(final_file)

            self.drafts.update_status(draft_id, DraftStatus.rendering_final, "joining_final")
            self.render.join_with_transitions(final_segments, draft_dir / "final.mp4", transition_seconds=transition_seconds)

            subtitle_map = draft_dir / "subtitle_map_edited.json"
            if not subtitle_map.exists():
                subtitle_map = draft_dir / "subtitle_map.json"
            subtitle_payload = read_json(subtitle_map, default={"chunks": []})
            chunks = subtitle_payload.get("chunks", []) if isinstance(subtitle_payload, dict) else []
            if isinstance(chunks, list):
                self.subtitles.save_verified_ayah_memory(draft_id=draft_id, chunks=[c for c in chunks if isinstance(c, dict)])

            self.drafts.update_status(draft_id, DraftStatus.completed, "completed")
