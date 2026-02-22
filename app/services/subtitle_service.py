from pathlib import Path

from app.core.storage import atomic_write_json, read_json
from app.models.schemas import SubtitleChunk


class SubtitleService:
    def build_initial_map(
        self,
        draft_dir: Path,
        clip_duration: float,
        markers: list[dict],
        surah_number: int,
        ayah_start: int,
        ayah_end: int,
    ) -> Path:
        chunks: list[dict] = []
        target = [
            marker
            for marker in markers
            if int(marker.get("surah_number", -1)) == surah_number and ayah_start <= int(marker.get("ayah", -1)) <= ayah_end
        ]
        target.sort(key=lambda item: float(item.get("time", 0)))

        if not target:
            chunks = [{"start": 0.0, "end": round(float(clip_duration), 2), "text": ""}]
        else:
            for i, marker in enumerate(target):
                start = max(0.0, float(marker.get("time", 0)))
                end = float(target[i + 1].get("time", clip_duration)) if i + 1 < len(target) else float(clip_duration)
                chunks.append(
                    {
                        "start": round(start, 2),
                        "end": round(max(start, min(end, float(clip_duration))), 2),
                        "text": marker.get("english_text") or marker.get("arabic_text") or f"{marker.get('surah', '')} {marker.get('ayah', '')}",
                    }
                )

        payload = {"chunks": chunks}
        out = draft_dir / "subtitle_map.json"
        atomic_write_json(out, payload)
        return out

    def save_edited_map(self, draft_dir: Path, chunks: list[SubtitleChunk]) -> Path:
        normalized = []
        previous_end = 0.0
        for chunk in chunks:
            if chunk.end < chunk.start:
                raise RuntimeError("Subtitle chunk has end < start")
            if chunk.start < previous_end:
                raise RuntimeError("Subtitle chunks overlap")
            previous_end = chunk.end
            normalized.append(chunk.model_dump(mode="json"))

        out = draft_dir / "subtitle_map_edited.json"
        atomic_write_json(out, {"chunks": normalized})
        return out

    def load_subtitle_map(self, draft_dir: Path, prefer_edited: bool = True) -> dict:
        if prefer_edited:
            edited = draft_dir / "subtitle_map_edited.json"
            if edited.exists():
                return read_json(edited, default={"chunks": []})
        return read_json(draft_dir / "subtitle_map.json", default={"chunks": []})
