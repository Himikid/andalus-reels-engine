from __future__ import annotations

from pathlib import Path

from app.core.storage import atomic_write_json, draft_lock, read_json
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

    def initialize_draft(self, draft_id: str) -> None:
        metadata = self.drafts.load_metadata(draft_id)
        if not metadata:
            raise RuntimeError("Draft not found")
        draft_dir = self.drafts.draft_dir(draft_id)
        with draft_lock(draft_dir):
            req = metadata.request
            segments = self._resolved_segments(req)
            segment_data: list[dict] = []
            for seg in segments:
                marker_doc = read_json(self.markers.latest_markers_path(int(seg["day"])), default={"markers": []})
                marker_list = marker_doc.get("markers", []) if isinstance(marker_doc, dict) else []
                segment_data.append({**seg, "markers": marker_list})
            self.subtitles.build_ayah_timing_map(draft_dir=draft_dir, segments=segment_data)
            self.drafts.update_status(draft_id, DraftStatus.ready_for_timing, "ready_for_timing")

    def _segments_with_timing_overrides(self, draft_dir: Path, segments: list[dict]) -> tuple[list[dict], list[dict]]:
        timing_payload = self.subtitles.load_ayah_timing_map(draft_dir, prefer_edited=True)
        timing_rows = timing_payload.get("chunks", []) if isinstance(timing_payload, dict) else []
        if not isinstance(timing_rows, list):
            timing_rows = []
        rows_by_segment: dict[str, list[dict]] = {}
        for row in timing_rows:
            if not isinstance(row, dict):
                continue
            seg_id = str(row.get("segment_id", "")).strip()
            if not seg_id:
                continue
            rows_by_segment.setdefault(seg_id, []).append(row)

        resolved: list[dict] = []
        for idx, seg in enumerate(segments, start=1):
            seg_id = str(seg.get("segment_id") or f"seg-{idx}")
            rows = rows_by_segment.get(seg_id, [])
            if rows:
                starts = [float(r.get("start", 0.0) or 0.0) for r in rows]
                ends = [float(r.get("end", 0.0) or 0.0) for r in rows]
                clip_start = max(0.0, min(starts) - 2.0)
                clip_end = max(ends) + 2.0
                duration = max(12.0, clip_end - clip_start)
                resolved.append({**seg, "clip_start": clip_start, "duration": duration, "segment_id": seg_id})
            else:
                resolved.append({**seg, "segment_id": seg_id})
        return resolved, timing_rows

    def _segment_subtitle_map(
        self,
        *,
        draft_dir: Path,
        segment_id: str,
        global_subtitle_map: dict,
    ) -> Path:
        chunks = global_subtitle_map.get("chunks", []) if isinstance(global_subtitle_map, dict) else []
        segment_runs = global_subtitle_map.get("segments", []) if isinstance(global_subtitle_map, dict) else []
        run = next((row for row in segment_runs if str(row.get("segment_id", "")) == segment_id), None)
        if not isinstance(run, dict):
            out = draft_dir / f"subtitle-map-{segment_id}.json"
            atomic_write_json(out, {"chunks": []})
            return out

        global_start = float(run.get("global_start", 0.0) or 0.0)
        global_end = float(run.get("global_end", global_start) or global_start)
        local_chunks: list[dict] = []
        for row in chunks if isinstance(chunks, list) else []:
            if not isinstance(row, dict):
                continue
            start = float(row.get("start", 0.0) or 0.0)
            end = float(row.get("end", start) or start)
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            if end < global_start or start > global_end:
                continue
            local_start = max(0.0, start - global_start)
            local_end = max(local_start + 0.2, min(global_end, end) - global_start)
            local_chunks.append(
                {
                    "start": round(local_start, 2),
                    "end": round(local_end, 2),
                    "text": text,
                }
            )

        local_chunks.sort(key=lambda row: (row["start"], row["end"]))
        out = draft_dir / f"subtitle-map-{segment_id}.json"
        atomic_write_json(out, {"chunks": local_chunks})
        return out

    def generate_draft(self, draft_id: str) -> None:
        metadata = self.drafts.load_metadata(draft_id)
        if not metadata:
            raise RuntimeError("Draft not found")

        draft_dir = self.drafts.draft_dir(draft_id)
        with draft_lock(draft_dir):
            req = metadata.request
            transition_seconds = float(req.get("transition_seconds", 0.45) or 0.45)
            segments, timing_rows = self._segments_with_timing_overrides(draft_dir, self._resolved_segments(req))

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
                ayah_timing_rows=timing_rows,
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
            subtitle_map = draft_dir / "subtitle_map_edited.json"
            if not subtitle_map.exists():
                subtitle_map = draft_dir / "subtitle_map.json"
            subtitle_payload = read_json(subtitle_map, default={"chunks": [], "segments": []})

            self.drafts.update_status(draft_id, DraftStatus.rendering_final, "rendering_final_segments")
            final_segments: list[Path] = []
            for idx, seg in enumerate(segments, start=1):
                seg_id = seg.get("segment_id") or f"seg-{idx}"
                source_video = self.audio.resolve_video_source(seg.get("youtube_url"))
                final_file = draft_dir / f"final-{seg_id}.mp4"
                segment_map_path = self._segment_subtitle_map(
                    draft_dir=draft_dir,
                    segment_id=str(seg_id),
                    global_subtitle_map=subtitle_payload,
                )
                made = self.render.try_generate_with_make_reel(
                    day=int(seg["day"]),
                    surah_number=int(seg["surah_number"]),
                    ayah_start=int(seg["ayah_start"]),
                    ayah_end=int(seg["ayah_end"]),
                    clip_start=float(seg["clip_start"]),
                    duration=float(seg["duration"]),
                    sheikh=seg.get("sheikh"),
                    youtube_url=seg.get("youtube_url"),
                    source_video_path=source_video,
                    draft_video_out=final_file,
                    subtitle_map_out=draft_dir / f"subtitle-map-final-{seg_id}.json",
                    align_subtitles=False,
                    subtitle_map_file=segment_map_path,
                    style="fit",
                    variants="clean",
                )
                if not made:
                    self.render.generate_final_segment(
                        source_video,
                        final_file,
                        float(seg["clip_start"]),
                        float(seg["duration"]),
                    )
                final_segments.append(final_file)

            self.drafts.update_status(draft_id, DraftStatus.rendering_final, "joining_final")
            self.render.join_with_transitions(final_segments, draft_dir / "final.mp4", transition_seconds=transition_seconds)

            chunks = subtitle_payload.get("chunks", []) if isinstance(subtitle_payload, dict) else []
            if isinstance(chunks, list):
                self.subtitles.save_verified_ayah_memory(draft_id=draft_id, chunks=[c for c in chunks if isinstance(c, dict)])

            self.drafts.update_status(draft_id, DraftStatus.completed, "completed")
