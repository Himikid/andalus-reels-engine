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

    def generate_draft(self, draft_id: str) -> None:
        metadata = self.drafts.load_metadata(draft_id)
        if not metadata:
            raise RuntimeError("Draft not found")

        draft_dir = self.drafts.draft_dir(draft_id)
        with draft_lock(draft_dir):
            req = metadata.request
            self.drafts.update_status(draft_id, DraftStatus.generating, "resolving_source")
            source_video = self.audio.resolve_video_source(req.get("youtube_url"), req.get("source_video_path"))

            self.drafts.update_status(draft_id, DraftStatus.generating, "extracting_audio")
            self.audio.extract_audio_clip(
                source_video,
                draft_dir / "audio.wav",
                float(req["clip_start"]),
                float(req["duration"]),
            )

            self.drafts.update_status(draft_id, DraftStatus.generating, "rendering_preview")
            generated = self.render.try_generate_with_make_reel(
                day=int(req["day"]),
                surah_number=int(req["surah_number"]),
                ayah_start=int(req["ayah_start"]),
                ayah_end=int(req["ayah_end"]),
                clip_start=float(req["clip_start"]),
                duration=float(req["duration"]),
                sheikh=req.get("sheikh"),
                youtube_url=req.get("youtube_url"),
                source_video_path=source_video,
                draft_video_out=draft_dir / "draft.mp4",
                subtitle_map_out=draft_dir / "subtitle_map.json",
                align_subtitles=bool(req.get("align_subtitles", True)),
            )
            if not generated:
                self.render.generate_preview(
                    source_video,
                    draft_dir / "draft.mp4",
                    float(req["clip_start"]),
                    float(req["duration"]),
                )
                self.drafts.update_status(draft_id, DraftStatus.generating, "building_subtitle_map")
                marker_doc = read_json(self.markers.latest_markers_path(int(req["day"])), default={"markers": []})
                self.subtitles.build_initial_map(
                    draft_dir=draft_dir,
                    clip_duration=float(req["duration"]),
                    markers=marker_doc.get("markers", []),
                    surah_number=int(req["surah_number"]),
                    ayah_start=int(req["ayah_start"]),
                    ayah_end=int(req["ayah_end"]),
                )

            self.drafts.update_status(draft_id, DraftStatus.ready_for_edit, "ready_for_edit")

    def render_final(self, draft_id: str) -> None:
        metadata = self.drafts.load_metadata(draft_id)
        if not metadata:
            raise RuntimeError("Draft not found")

        draft_dir = self.drafts.draft_dir(draft_id)
        with draft_lock(draft_dir):
            req = metadata.request
            self.drafts.update_status(draft_id, DraftStatus.rendering_final, "rendering_final")
            source_video = self.audio.resolve_video_source(req.get("youtube_url"), req.get("source_video_path"))
            subtitle_map = draft_dir / "subtitle_map_edited.json"
            if not subtitle_map.exists():
                subtitle_map = draft_dir / "subtitle_map.json"
            self.render.render_final_with_subtitles(
                source_video,
                subtitle_map,
                draft_dir / "final.mp4",
                float(req["clip_start"]),
                float(req["duration"]),
            )
            self.drafts.update_status(draft_id, DraftStatus.completed, "completed")
