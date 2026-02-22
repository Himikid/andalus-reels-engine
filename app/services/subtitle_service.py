from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.core.storage import atomic_write_json, ensure_dir, read_json
from app.models.schemas import AyahTimingChunk, SubtitleChunk


class SubtitleService:
    def __init__(self) -> None:
        self.memory_dir = ensure_dir(settings.data_dir / "subtitles")
        self.memory_path = self.memory_dir / "verified_ayahs.json"

    def _verified_memory(self) -> dict:
        payload = read_json(self.memory_path, default={"entries": {}})
        return payload if isinstance(payload, dict) else {"entries": {}}

    def _save_verified_memory(self, payload: dict) -> None:
        atomic_write_json(self.memory_path, payload)

    def build_initial_map_for_segments(
        self,
        *,
        draft_dir: Path,
        segments: list[dict],
        transition_seconds: float,
        ayah_timing_rows: list[dict] | None = None,
    ) -> Path:
        verified = self._verified_memory().get("entries", {})
        timing_rows = ayah_timing_rows if isinstance(ayah_timing_rows, list) else []
        timing_by_key: dict[str, dict] = {}
        for row in timing_rows:
            if not isinstance(row, dict):
                continue
            seg_id = str(row.get("segment_id", "")).strip()
            day = int(row.get("day", 0) or 0)
            surah_number = int(row.get("surah_number", 0) or 0)
            ayah = int(row.get("ayah", 0) or 0)
            if not seg_id or day <= 0 or surah_number <= 0 or ayah <= 0:
                continue
            timing_by_key[f"{seg_id}:{day}:{surah_number}:{ayah}"] = row
        chunks: list[dict] = []
        segment_runs: list[dict] = []

        cursor = 0.0
        for index, segment in enumerate(segments):
            seg_id = str(segment["segment_id"])
            day = int(segment["day"])
            surah_number = int(segment["surah_number"])
            ayah_start = int(segment["ayah_start"])
            ayah_end = int(segment["ayah_end"])
            clip_start = float(segment["clip_start"])
            duration = float(segment["duration"])
            source_id = segment.get("source_id")
            markers = segment.get("markers", []) if isinstance(segment.get("markers"), list) else []

            target = [
                marker
                for marker in markers
                if int(marker.get("surah_number", -1) or -1) == surah_number
                and ayah_start <= int(marker.get("ayah", -1) or -1) <= ayah_end
            ]
            if source_id:
                target = [m for m in target if str(m.get("source_id", "")).strip() == str(source_id).strip()]
            target.sort(key=lambda item: float(item.get("time", 0) or 0.0))

            if not target:
                chunks.append(
                    {
                        "start": round(cursor, 2),
                        "end": round(cursor + duration, 2),
                        "text": "",
                        "day": day,
                        "surah_number": surah_number,
                        "ayah": ayah_start,
                        "source_id": source_id,
                        "verified": False,
                    }
                )
            else:
                for i, marker in enumerate(target):
                    ayah = int(marker.get("ayah", 0) or 0)
                    marker_time = float(marker.get("time", 0) or 0.0)
                    next_time = float(target[i + 1].get("time", marker_time + 6.0) or (marker_time + 6.0)) if i + 1 < len(target) else (marker_time + 6.0)
                    key = f"{seg_id}:{day}:{surah_number}:{ayah}"
                    timing = timing_by_key.get(key, {})
                    source_start = float(timing.get("start", marker_time) or marker_time)
                    source_end = float(timing.get("end", next_time) or next_time)
                    local_start = max(0.0, source_start - clip_start)
                    local_end = min(duration, max(local_start, source_end - clip_start))
                    global_start = cursor + local_start
                    global_end = cursor + local_end

                    memory_key = f"{day}:{surah_number}:{ayah}:{source_id or ''}"
                    remembered = verified.get(memory_key, {}) if isinstance(verified, dict) else {}
                    default_text = marker.get("english_text") or marker.get("arabic_text") or f"{marker.get('surah', '')} {ayah}"
                    text = remembered.get("text") or default_text

                    chunks.append(
                        {
                            "start": round(global_start, 2),
                            "end": round(max(global_start + 0.2, global_end), 2),
                            "text": text,
                            "day": day,
                            "surah_number": surah_number,
                            "ayah": ayah,
                            "source_id": source_id,
                            "verified": bool(remembered),
                        }
                    )

            segment_runs.append(
                {
                    "segment_id": seg_id,
                    "index": index,
                    "day": day,
                    "surah_number": surah_number,
                    "ayah_start": ayah_start,
                    "ayah_end": ayah_end,
                    "global_start": round(cursor, 2),
                    "global_end": round(cursor + duration, 2),
                    "duration": duration,
                    "source_id": source_id,
                }
            )

            cursor += duration
            if index < len(segments) - 1:
                cursor += transition_seconds

        payload = {
            "chunks": chunks,
            "segments": segment_runs,
            "transition_seconds": transition_seconds,
        }
        out = draft_dir / "subtitle_map.json"
        atomic_write_json(out, payload)
        return out

    def build_ayah_timing_map(self, *, draft_dir: Path, segments: list[dict]) -> Path:
        chunks: list[dict] = []
        for segment in segments:
            seg_id = str(segment["segment_id"])
            day = int(segment["day"])
            surah_number = int(segment["surah_number"])
            ayah_start = int(segment["ayah_start"])
            ayah_end = int(segment["ayah_end"])
            clip_start = float(segment["clip_start"])
            duration = float(segment["duration"])
            source_id = segment.get("source_id")
            markers = segment.get("markers", []) if isinstance(segment.get("markers"), list) else []
            target = [
                marker
                for marker in markers
                if int(marker.get("surah_number", -1) or -1) == surah_number
                and ayah_start <= int(marker.get("ayah", -1) or -1) <= ayah_end
            ]
            if source_id:
                target = [m for m in target if str(m.get("source_id", "")).strip() == str(source_id).strip()]
            target.sort(key=lambda item: float(item.get("time", 0) or 0.0))

            marker_by_ayah: dict[int, dict] = {}
            for marker in target:
                ayah = int(marker.get("ayah", 0) or 0)
                if ayah <= 0 or ayah in marker_by_ayah:
                    continue
                marker_by_ayah[ayah] = marker

            fallback_cursor = clip_start
            for ayah in range(ayah_start, ayah_end + 1):
                marker = marker_by_ayah.get(ayah)
                if marker:
                    start = float(marker.get("start_time", marker.get("time", fallback_cursor)) or fallback_cursor)
                    end = float(marker.get("end_time", marker.get("time", start + 4.0)) or (start + 4.0))
                    fallback_cursor = max(fallback_cursor, end)
                else:
                    start = fallback_cursor
                    end = start + 5.0
                    fallback_cursor = end
                chunks.append(
                    {
                        "segment_id": seg_id,
                        "day": day,
                        "surah_number": surah_number,
                        "ayah": ayah,
                        "start": round(max(0.0, start), 2),
                        "end": round(max(start + 0.2, end), 2),
                        "source_id": source_id,
                    }
                )

        payload = {"chunks": chunks}
        out = draft_dir / "ayah_timing_map.json"
        atomic_write_json(out, payload)
        return out

    def save_ayah_timing_map(self, draft_dir: Path, chunks: list[AyahTimingChunk]) -> Path:
        normalized: list[dict] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.end < chunk.start:
                raise RuntimeError("Ayah timing chunk has end < start")
            key = f"{chunk.segment_id}:{chunk.day}:{chunk.surah_number}:{chunk.ayah}"
            if key in seen:
                raise RuntimeError(f"Duplicate ayah timing row: {key}")
            seen.add(key)
            normalized.append(chunk.model_dump(mode="json"))
        out = draft_dir / "ayah_timing_map_edited.json"
        atomic_write_json(out, {"chunks": normalized, "edited_at": datetime.now(timezone.utc).isoformat()})
        return out

    def load_ayah_timing_map(self, draft_dir: Path, prefer_edited: bool = True) -> dict:
        if prefer_edited:
            edited = draft_dir / "ayah_timing_map_edited.json"
            if edited.exists():
                return read_json(edited, default={"chunks": []})
        return read_json(draft_dir / "ayah_timing_map.json", default={"chunks": []})

    def save_edited_map(self, draft_dir: Path, chunks: list[SubtitleChunk]) -> Path:
        normalized: list[dict] = []
        previous_end = 0.0
        for chunk in chunks:
            if chunk.end < chunk.start:
                raise RuntimeError("Subtitle chunk has end < start")
            if chunk.start < previous_end:
                raise RuntimeError("Subtitle chunks overlap")
            previous_end = chunk.end
            normalized.append(chunk.model_dump(mode="json"))

        base_map = read_json(draft_dir / "subtitle_map.json", default={})
        payload = {
            "chunks": normalized,
            "segments": base_map.get("segments", []) if isinstance(base_map, dict) else [],
            "transition_seconds": base_map.get("transition_seconds", 0.45) if isinstance(base_map, dict) else 0.45,
            "edited_at": datetime.now(timezone.utc).isoformat(),
        }
        out = draft_dir / "subtitle_map_edited.json"
        atomic_write_json(out, payload)
        return out

    def load_subtitle_map(self, draft_dir: Path, prefer_edited: bool = True) -> dict:
        if prefer_edited:
            edited = draft_dir / "subtitle_map_edited.json"
            if edited.exists():
                return read_json(edited, default={"chunks": []})
        return read_json(draft_dir / "subtitle_map.json", default={"chunks": []})

    def save_verified_ayah_memory(self, *, draft_id: str, chunks: list[dict]) -> None:
        payload = self._verified_memory()
        entries = payload.get("entries", {}) if isinstance(payload.get("entries"), dict) else {}

        for chunk in chunks:
            day = chunk.get("day")
            surah = chunk.get("surah_number")
            ayah = chunk.get("ayah")
            if not (isinstance(day, int) and isinstance(surah, int) and isinstance(ayah, int)):
                continue
            source_id = str(chunk.get("source_id", "") or "")
            key = f"{day}:{surah}:{ayah}:{source_id}"
            entries[key] = {
                "day": day,
                "surah_number": surah,
                "ayah": ayah,
                "source_id": source_id or None,
                "text": chunk.get("text", ""),
                "verified": True,
                "draft_id": draft_id,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }

        payload["entries"] = entries
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_verified_memory(payload)

    def list_verified(self, day: int | None = None, surah_number: int | None = None) -> list[dict]:
        entries = self._verified_memory().get("entries", {})
        rows = [value for value in entries.values() if isinstance(value, dict)]
        if day is not None:
            rows = [row for row in rows if int(row.get("day", -1)) == day]
        if surah_number is not None:
            rows = [row for row in rows if int(row.get("surah_number", -1)) == surah_number]
        rows.sort(key=lambda row: (int(row.get("day", 0)), int(row.get("surah_number", 0)), int(row.get("ayah", 0))))
        return rows
