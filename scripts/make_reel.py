#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import textwrap
from pathlib import Path

from rapidfuzz import fuzz

SUBTITLE_OVERLAY_HEIGHT = 456
SUBTITLE_FADE_IN_SECONDS = 0.18
SUBTITLE_FADE_OUT_SECONDS = 0.20
ARABIC_STOPWORDS = {
    "و",
    "ف",
    "ثم",
    "ان",
    "إن",
    "أن",
    "في",
    "من",
    "على",
    "الى",
    "إلى",
    "لا",
    "ما",
    "هو",
    "هي",
    "هم",
    "هذا",
    "هذه",
    "ذلك",
    "يا",
    "قد",
    "كان",
    "كانت",
    "كل",
    "عن",
    "او",
    "أو",
    "بل",
    "لن",
    "لم",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def grade_from_score(score: float) -> str:
    if score >= 0.88:
        return "A"
    if score >= 0.76:
        return "B"
    if score >= 0.62:
        return "C"
    if score >= 0.48:
        return "D"
    return "E"


def require_binary(name: str) -> None:
    result = subprocess.run(["which", name], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Missing required binary: {name}")


def parse_timestamp(value: str) -> float:
    raw = value.strip()
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        return float(raw)
    parts = raw.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise SystemExit(f"Invalid timestamp format: {value}")


def _quality_rank(value: object) -> int:
    quality = str(value or "").strip().lower()
    if quality == "high":
        return 3
    if quality == "ambiguous":
        return 2
    if quality == "inferred":
        return 1
    return 0


def _is_http_source(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def load_day_payload(day: int) -> tuple[dict, str]:
    primary = Path(f"public/data/day-{day}.json")
    v2 = Path(f"public/data/day-{day}-v2.json")
    if v2.exists():
        return json.loads(v2.read_text(encoding="utf-8")), str(v2)
    if primary.exists():
        return json.loads(primary.read_text(encoding="utf-8")), str(primary)

    part_files = sorted(Path("public/data").glob(f"day-{day}-part-*.json"))
    if not part_files:
        raise SystemExit(f"Could not find day JSON: {primary}, {v2}, or day-{day}-part-*.json")

    merged_markers: list[dict] = []
    sources: list[str] = []
    for part_file in part_files:
        payload = json.loads(part_file.read_text(encoding="utf-8"))
        source = str(payload.get("source", "")).strip()
        if source:
            sources.append(source)
        markers = payload.get("markers", [])
        if isinstance(markers, list):
            for marker in markers:
                if not isinstance(marker, dict):
                    continue
                m = dict(marker)
                if source:
                    m.setdefault("_source_url", source)
                merged_markers.append(m)

    deduped: list[dict] = []
    seen_keys: set[tuple[int, int, int, int, int, str]] = set()
    for marker in merged_markers:
        key = (
            int(marker.get("surah_number", 0) or 0),
            int(marker.get("ayah", 0) or 0),
            int(marker.get("time", 0) or 0),
            int(marker.get("start_time", marker.get("time", 0)) or 0),
            int(marker.get("end_time", marker.get("start_time", marker.get("time", 0))) or 0),
            str(marker.get("_source_url", "")).strip(),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(marker)

    merged = {
        "day": day,
        "source": sources[0] if sources else "",
        "sources": sorted(set(sources)),
        "markers": deduped,
        "meta": {"merged_from_parts": [str(p) for p in part_files]},
    }
    return merged, ", ".join(str(p) for p in part_files)


def _extract_surahs(payload: dict) -> list[dict]:
    if isinstance(payload.get("surahs"), list):
        return payload.get("surahs", [])
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("surahs"), list):
        return data.get("surahs", [])
    return []


def load_quran_maps(
    english_file: Path | None = None,
    fallback_english_file: Path | None = None,
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], str], dict[int, str]]:
    arabic_file = Path("data/quran/quran_arabic.json")
    english_primary = english_file or Path("data/quran/quran_abdelhaleem_en.json")
    english_fallback = fallback_english_file or Path("data/quran/quran_asad_en.json")

    arabic_map: dict[tuple[int, int], str] = {}
    english_map: dict[tuple[int, int], str] = {}
    surah_names: dict[int, str] = {}
    if not arabic_file.exists():
        return arabic_map, english_map, surah_names

    arabic_payload = json.loads(arabic_file.read_text(encoding="utf-8"))
    english_payload: dict = {}
    if english_primary.exists():
        english_payload = json.loads(english_primary.read_text(encoding="utf-8"))
    elif english_fallback.exists():
        english_payload = json.loads(english_fallback.read_text(encoding="utf-8"))

    for surah in _extract_surahs(arabic_payload):
        surah_num = int(surah.get("number", 0) or 0)
        surah_name = str(surah.get("name", "")).strip()
        if surah_num:
            surah_names[surah_num] = surah_name
        for ayah in surah.get("ayahs", []):
            ay_num = int(ayah.get("number", 0) or 0)
            if surah_num and ay_num:
                arabic_map[(surah_num, ay_num)] = str(ayah.get("text", "")).strip()

    for surah in _extract_surahs(english_payload):
        surah_num = int(surah.get("number", 0) or 0)
        for ayah in surah.get("ayahs", []):
            ay_num = int(ayah.get("number", 0) or 0)
            if surah_num and ay_num:
                english_map[(surah_num, ay_num)] = str(ayah.get("text", "")).strip()

    return arabic_map, english_map, surah_names


def collect_ayah_range_text(
    day_payload: dict,
    surah_number: int,
    ayah_start: int,
    ayah_end: int,
    english_file: Path | None = None,
    fallback_english_file: Path | None = None,
    prefer_marker_english: bool = False,
) -> dict:
    if ayah_end < ayah_start:
        ayah_end = ayah_start

    markers = day_payload.get("markers", [])
    marker_map: dict[int, dict] = {}
    source_votes: dict[str, int] = {}

    def _prefer_marker(candidate: dict, existing: dict | None) -> bool:
        if existing is None:
            return True
        candidate_quality = _quality_rank(candidate.get("quality"))
        existing_quality = _quality_rank(existing.get("quality"))
        if candidate_quality != existing_quality:
            return candidate_quality > existing_quality
        candidate_conf = float(candidate.get("confidence", 0.0) or 0.0)
        existing_conf = float(existing.get("confidence", 0.0) or 0.0)
        if candidate_conf != existing_conf:
            return candidate_conf > existing_conf
        cand_t = float(candidate.get("start_time", candidate.get("time", 1e9)) or 1e9)
        exist_t = float(existing.get("start_time", existing.get("time", 1e9)) or 1e9)
        return cand_t < exist_t

    if isinstance(markers, list):
        for marker in markers:
            if not isinstance(marker, dict):
                continue
            if int(marker.get("surah_number", 0) or 0) != surah_number:
                continue
            ayah = int(marker.get("ayah", 0) or 0)
            if ayah_start <= ayah <= ayah_end and _prefer_marker(marker, marker_map.get(ayah)):
                marker_map[ayah] = marker

    arabic_map, english_map, surah_names = load_quran_maps(
        english_file=english_file,
        fallback_english_file=fallback_english_file,
    )
    arabic_parts: list[str] = []
    english_parts: list[str] = []
    surah_name = ""
    selected_source = ""

    for ayah in range(ayah_start, ayah_end + 1):
        marker = marker_map.get(ayah)
        if marker:
            if not surah_name:
                surah_name = str(marker.get("surah", "")).strip()
            ar_text = str(marker.get("arabic_text", "")).strip()
            marker_en_text = str(marker.get("english_text", "")).strip()
            marker_source = str(marker.get("_source_url", "")).strip()
            if marker_source:
                source_votes[marker_source] = source_votes.get(marker_source, 0) + 1
        else:
            ar_text = ""
            marker_en_text = ""

        if not surah_name:
            surah_name = surah_names.get(surah_number, "")
        if not ar_text:
            ar_text = arabic_map.get((surah_number, ayah), "")
        corpus_en_text = english_map.get((surah_number, ayah), "")
        if prefer_marker_english:
            en_text = marker_en_text or corpus_en_text
        else:
            en_text = corpus_en_text or marker_en_text

        if ar_text:
            arabic_parts.append(ar_text)
        if en_text:
            english_parts.append(en_text)

    if not surah_name:
        surah_name = f"Surah {surah_number}"
    if source_votes:
        selected_source = sorted(source_votes.items(), key=lambda i: (-i[1], i[0]))[0][0]
    elif _is_http_source(day_payload.get("source", "")):
        selected_source = str(day_payload.get("source", "")).strip()

    return {
        "surah_name": surah_name,
        "english_text": " ".join(part for part in english_parts if part).strip(),
        "arabic_text": " ".join(part for part in arabic_parts if part).strip(),
        "has_marker": any(ayah in marker_map for ayah in range(ayah_start, ayah_end + 1)),
        "source_url": selected_source,
    }


def build_ayah_caption_chunks_from_markers(
    day_payload: dict,
    surah_number: int,
    ayah_start: int,
    ayah_end: int,
    clip_start: float,
    clip_end: float,
    english_file: Path | None = None,
    fallback_english_file: Path | None = None,
    prefer_marker_english: bool = False,
    source_url: str | None = None,
    split_long_ayahs: bool = False,
) -> list[tuple[str, float, float]]:
    if ayah_end < ayah_start:
        ayah_end = ayah_start
    duration = max(1.0, clip_end - clip_start)
    markers = day_payload.get("markers", [])
    if not isinstance(markers, list):
        return []

    marker_map: dict[int, dict] = {}
    source_filter = str(source_url or "").strip()
    for marker in markers:
        if not isinstance(marker, dict):
            continue
        marker_source = str(marker.get("_source_url", "")).strip()
        if source_filter and marker_source and marker_source != source_filter:
            continue
        if int(marker.get("surah_number", 0) or 0) != surah_number:
            continue
        ayah = int(marker.get("ayah", 0) or 0)
        if ayah_start <= ayah <= ayah_end and ayah not in marker_map:
            marker_map[ayah] = marker

    _arabic_map, english_map, _surah_names = load_quran_maps(
        english_file=english_file,
        fallback_english_file=fallback_english_file,
    )

    def _marker_time(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    known_starts: list[tuple[int, float]] = []
    start_points: dict[int, float] = {}
    end_points: dict[int, float] = {}
    for ayah in range(ayah_start, ayah_end + 1):
        marker = marker_map.get(ayah, {})
        start_abs = _marker_time(marker.get("start_time", marker.get("time")))
        end_abs = _marker_time(marker.get("end_time", marker.get("start_time", marker.get("time"))))
        if start_abs is not None:
            rel_start = start_abs - clip_start
            start_points[ayah] = rel_start
            known_starts.append((ayah, rel_start))
        if end_abs is not None:
            end_points[ayah] = end_abs - clip_start

    known_starts.sort(key=lambda item: item[0])

    def _interpolate_start(target_ayah: int, index_in_range: int, count: int) -> float:
        if not known_starts:
            span = max(0.8, duration - 0.30)
            return 0.12 + (span * (index_in_range / max(1, count)))
        if len(known_starts) == 1:
            anchor_ayah, anchor_time = known_starts[0]
            return anchor_time + ((target_ayah - anchor_ayah) * 7.0)
        if target_ayah <= known_starts[0][0]:
            l_ayah, l_time = known_starts[0]
            r_ayah, r_time = known_starts[1]
            slope = (r_time - l_time) / max(1, r_ayah - l_ayah)
            return l_time + ((target_ayah - l_ayah) * slope)
        if target_ayah >= known_starts[-1][0]:
            l_ayah, l_time = known_starts[-2]
            r_ayah, r_time = known_starts[-1]
            slope = (r_time - l_time) / max(1, r_ayah - l_ayah)
            return r_time + ((target_ayah - r_ayah) * slope)
        for (l_ayah, l_time), (r_ayah, r_time) in zip(known_starts, known_starts[1:]):
            if l_ayah <= target_ayah <= r_ayah:
                if r_ayah == l_ayah:
                    return l_time
                ratio = (target_ayah - l_ayah) / (r_ayah - l_ayah)
                return l_time + ((r_time - l_time) * ratio)
        return known_starts[-1][1]

    rows: list[dict] = []
    ayah_count = ayah_end - ayah_start + 1
    for i, ayah in enumerate(range(ayah_start, ayah_end + 1)):
        marker = marker_map.get(ayah, {})
        marker_en = str(marker.get("english_text", "")).strip()
        corpus_en = str(english_map.get((surah_number, ayah), "")).strip()
        text = marker_en or corpus_en if prefer_marker_english else corpus_en or marker_en
        if not text:
            text = f"Surah {surah_number}:{ayah}"
        start_rel = start_points.get(ayah)
        if start_rel is None:
            start_rel = _interpolate_start(ayah, i, ayah_count)
        end_rel = end_points.get(ayah)
        rows.append({"ayah": ayah, "text": text, "start": start_rel, "end": end_rel})

    rows.sort(key=lambda row: (row["start"], row["ayah"]))
    for i, row in enumerate(rows):
        next_start = rows[i + 1]["start"] if i + 1 < len(rows) else (duration - 0.12)
        end_rel = row["end"]
        if end_rel is None:
            end_rel = next_start - 0.06
        end_rel = min(end_rel, next_start - 0.02)
        if i == len(rows) - 1:
            end_rel = max(end_rel, duration - 0.18)
        start_rel = clamp(float(row["start"]), 0.08, duration - 0.25)
        end_rel = clamp(float(end_rel), start_rel + 0.45, duration - 0.05)
        row["start"] = round(start_rel, 2)
        row["end"] = round(end_rel, 2)

    chunks: list[tuple[str, float, float]] = []
    for row in rows:
        if row["end"] <= 0.10 or row["start"] >= duration - 0.05:
            continue
        text = str(row["text"]).strip()
        start = float(row["start"])
        end = float(row["end"])
        span = max(0.6, end - start)
        words = [word for word in text.split() if word.strip()]
        if not words:
            continue

        desired_chunks = 1
        if split_long_ayahs and len(words) > 18 and span >= 6.0:
            desired_chunks = max(2, int(round(len(words) / 16)))
            if span < 10.0:
                desired_chunks = min(desired_chunks, 2)
            elif span < 16.0:
                desired_chunks = min(desired_chunks, 3)
            desired_chunks = min(5, max(2, desired_chunks))

        if desired_chunks == 1:
            chunks.append((text, start, end))
            continue

        for index in range(desired_chunks):
            w_start = int(round(index * len(words) / desired_chunks))
            w_end = int(round((index + 1) * len(words) / desired_chunks))
            if w_end <= w_start:
                continue
            part_text = " ".join(words[w_start:w_end]).strip()
            if not part_text:
                continue
            part_start = start + (span * (index / desired_chunks))
            part_end = start + (span * ((index + 1) / desired_chunks))
            part_end = max(part_start + 0.35, part_end)
            chunks.append((part_text, part_start, min(end, part_end)))
    return chunks


def choose_source_video(args: argparse.Namespace, work_dir: Path, default_youtube_url: str | None = None) -> Path:
    if args.video_file:
        path = Path(args.video_file)
        if not path.exists():
            raise SystemExit(f"Video file not found: {path}")
        return path

    youtube_url = str(args.youtube_url or default_youtube_url or "").strip()
    if not youtube_url:
        raise SystemExit("Provide either --video-file or --youtube-url")

    require_binary("yt-dlp")
    out_dir = work_dir / "source"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "input.mp4"
    if output_file.exists():
        return output_file

    out_tmpl = out_dir / "input.%(ext)s"
    command = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_tmpl),
        youtube_url,
    ]
    subprocess.run(command, check=True)

    candidates = sorted(out_dir.glob("input.*"))
    if not candidates:
        raise SystemExit("yt-dlp completed but no source video was produced.")
    return candidates[-1]


def parse_variants(value: str) -> list[str]:
    allowed = {"clean", "focus", "context"}
    raw = value.strip().lower()
    if raw == "all":
        return ["clean", "focus", "context"]
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("No variants selected. Use clean,focus,context or all.")
    invalid = [p for p in parts if p not in allowed]
    if invalid:
        raise SystemExit(f"Invalid variant(s): {', '.join(invalid)}. Allowed: clean, focus, context.")
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return deduped


def normalize_arabic_token(value: str) -> str:
    try:
        from ai_pipeline.quran import normalize_arabic

        return normalize_arabic(value, strict=False)
    except Exception:
        value = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", value)
        value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": ""}))
        value = re.sub(r"[^\u0621-\u063A\u0641-\u064A\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value


def transcribe_clip_words(
    source_video: Path,
    start_seconds: float,
    end_seconds: float,
    work_dir: Path,
    model_size: str,
) -> list[dict]:
    cache_path = work_dir / f"clip-transcript-{int(start_seconds)}-{int(end_seconds)}-{model_size}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        words = payload.get("words", [])
        if isinstance(words, list):
            return [w for w in words if isinstance(w, dict)]

    audio_path = work_dir / f"clip-{int(start_seconds)}-{int(end_seconds)}.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_seconds}",
            "-to",
            f"{end_seconds}",
            "-i",
            str(source_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise SystemExit("faster-whisper is required for subtitle alignment.") from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language="ar", word_timestamps=True, vad_filter=True)
    words: list[dict] = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            text = str(word.word or "").strip()
            if not text:
                continue
            confidence = float(getattr(word, "probability", 0.0) or 0.0)
            words.append({"text": text, "start": float(word.start), "end": float(word.end), "confidence": confidence})

    cache_path.write_text(json.dumps({"words": words}, ensure_ascii=False, indent=2), encoding="utf-8")
    return words


def tokenize_transcript_words(words: list[dict]) -> list[dict]:
    tokens: list[dict] = []
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        confidence = float(word.get("confidence", 1.0) if "confidence" in word else 1.0)
        normalized = normalize_arabic_token(str(word.get("text", "")))
        if not normalized:
            continue
        for piece in normalized.split():
            if piece:
                tokens.append({"token": piece, "start": start, "end": end, "confidence": confidence})
    return tokens


def _token_similarity(canonical: str, transcript: str) -> float:
    token_set = float(fuzz.token_set_ratio(canonical, transcript))
    partial = float(fuzz.partial_ratio(canonical, transcript))
    ratio = float(fuzz.ratio(canonical, transcript))
    score = (0.50 * token_set) + (0.30 * partial) + (0.20 * ratio)
    if canonical in ARABIC_STOPWORDS or transcript in ARABIC_STOPWORDS:
        score *= 0.78
    if len(canonical) <= 2 or len(transcript) <= 2:
        score *= 0.88
    return score


def align_arabic_tokens(
    canonical_tokens: list[str],
    transcript_tokens: list[dict],
    min_word_confidence: float = 0.38,
    band_radius: int = 22,
    min_match_score: float = 70.0,
) -> list[tuple[int, float, float, float]]:
    if not canonical_tokens or not transcript_tokens:
        return []

    filtered = [
        row
        for row in transcript_tokens
        if float(row.get("confidence", 1.0) if "confidence" in row else 1.0) >= min_word_confidence
    ]
    if not filtered:
        return []

    n = len(canonical_tokens)
    m = len(filtered)
    if n == 0 or m == 0:
        return []

    gap_c = -18.0
    gap_t = -10.0
    dp: dict[tuple[int, int], float] = {(0, 0): 0.0}
    back: dict[tuple[int, int], tuple[int, int, str]] = {}

    def in_band(i: int, j: int) -> bool:
        expected = (i * m) / max(1, n)
        return abs(j - expected) <= band_radius

    for i in range(0, n + 1):
        for j in range(0, m + 1):
            if i == 0 and j == 0:
                continue
            if not in_band(i, j):
                continue
            candidates: list[tuple[float, tuple[int, int, str]]] = []
            if i > 0 and in_band(i - 1, j) and (i - 1, j) in dp:
                candidates.append((dp[(i - 1, j)] + gap_c, (i - 1, j, "up")))
            if j > 0 and in_band(i, j - 1) and (i, j - 1) in dp:
                candidates.append((dp[(i, j - 1)] + gap_t, (i, j - 1, "left")))
            if i > 0 and j > 0 and in_band(i - 1, j - 1) and (i - 1, j - 1) in dp:
                sim = _token_similarity(canonical_tokens[i - 1], str(filtered[j - 1]["token"]))
                candidates.append((dp[(i - 1, j - 1)] + sim, (i - 1, j - 1, "diag")))
            if not candidates:
                continue
            best_val, best_prev = max(candidates, key=lambda item: item[0])
            dp[(i, j)] = best_val
            back[(i, j)] = best_prev

    end_state: tuple[int, int] | None = None
    end_score = float("-inf")
    for j in range(0, m + 1):
        state = (n, j)
        if state in dp and dp[state] > end_score:
            end_score = dp[state]
            end_state = state
    if end_state is None:
        return []

    pairs: list[tuple[int, int, float]] = []
    i, j = end_state
    while (i, j) != (0, 0):
        prev = back.get((i, j))
        if prev is None:
            break
        pi, pj, move = prev
        if move == "diag" and i > 0 and j > 0:
            score = _token_similarity(canonical_tokens[i - 1], str(filtered[j - 1]["token"]))
            pairs.append((i - 1, j - 1, score))
        i, j = pi, pj
    pairs.reverse()

    aligned: list[tuple[int, float, float, float]] = []
    for c_idx, t_idx, score in pairs:
        if score < min_match_score:
            continue
        row = filtered[t_idx]
        aligned.append((c_idx, float(row["start"]), float(row["end"]), float(score)))
    return aligned


def interpolate_time(points: list[tuple[int, float]], index: int) -> float:
    if not points:
        return 0.0
    if index <= points[0][0]:
        return points[0][1]
    if index >= points[-1][0]:
        return points[-1][1]
    for left, right in zip(points, points[1:]):
        l_idx, l_time = left
        r_idx, r_time = right
        if l_idx <= index <= r_idx:
            if r_idx == l_idx:
                return l_time
            ratio = (index - l_idx) / (r_idx - l_idx)
            return l_time + ((r_time - l_time) * ratio)
    return points[-1][1]


def apply_alignment_to_chunks(
    chunks: list[tuple[str, float, float]],
    canonical_tokens: list[str],
    aligned_tokens: list[tuple[int, float, float, float]],
    duration: float,
    subtitle_advance: float = 0.40,
) -> tuple[list[tuple[str, float, float]], list[float], dict]:
    if not chunks:
        return [], [], {
            "used_alignment": False,
            "reason": "no_chunks",
            "canonical_tokens": len(canonical_tokens),
            "aligned_tokens": len(aligned_tokens),
            "coverage": 0.0,
            "avg_score": 0.0,
        }

    defaults = [0.62 for _ in chunks]
    if not canonical_tokens or not aligned_tokens:
        return chunks, defaults, {
            "used_alignment": False,
            "reason": "missing_tokens",
            "canonical_tokens": len(canonical_tokens),
            "aligned_tokens": len(aligned_tokens),
            "coverage": 0.0,
            "avg_score": 0.0,
        }

    avg_score = sum(item[3] for item in aligned_tokens) / max(1, len(aligned_tokens))
    coverage = len(aligned_tokens) / max(1, len(canonical_tokens))
    token_max = max(0, len(canonical_tokens) - 1)
    word_weights = [max(1, len(text.split())) for text, _s, _e in chunks]
    total_weight = max(1, sum(word_weights))

    token_ranges: list[tuple[int, int]] = []
    running = 0
    for index in range(len(chunks)):
        start_ratio = running / total_weight
        running += word_weights[index]
        end_ratio = running / total_weight
        start_idx = int(round(start_ratio * token_max))
        end_idx = max(start_idx, int(round(end_ratio * token_max)))
        token_ranges.append((start_idx, end_idx))

    chunk_confidences: list[float] = []
    for start_idx, end_idx in token_ranges:
        local_scores = [score for idx, _s, _e, score in aligned_tokens if start_idx <= idx <= end_idx]
        token_span = max(1, end_idx - start_idx + 1)
        local_density = len(local_scores) / token_span
        if local_scores:
            score_component = (sum(local_scores) / len(local_scores)) / 100.0
            confidence = clamp(0.34 + (0.43 * score_component) + (0.34 * local_density), 0.38, 0.99)
        else:
            confidence = clamp((coverage * 0.48) + ((avg_score / 100.0) * 0.20), 0.28, 0.67)
        chunk_confidences.append(round(confidence, 3))

    min_hits = 3 if len(canonical_tokens) <= 40 else 5
    min_coverage = 0.24 if len(canonical_tokens) <= 40 else 0.30
    can_realign = len(aligned_tokens) >= min_hits and avg_score >= 70 and coverage >= min_coverage
    if not can_realign:
        return chunks, chunk_confidences, {
            "used_alignment": False,
            "reason": "low_quality_alignment",
            "canonical_tokens": len(canonical_tokens),
            "aligned_tokens": len(aligned_tokens),
            "required_aligned_tokens": min_hits,
            "coverage": round(coverage, 3),
            "required_coverage": round(min_coverage, 3),
            "avg_score": round(avg_score, 2),
        }

    anchor_points = sorted((idx, start) for idx, start, _e, _score in aligned_tokens)
    advance = clamp(subtitle_advance, 0.0, 1.2)

    realigned: list[tuple[str, float, float]] = []
    total_shift = 0.0
    for idx, (text, old_start, old_end) in enumerate(chunks):
        start_idx, end_idx = token_ranges[idx]
        local_hits = [(s, e) for token_idx, s, e, _score in aligned_tokens if start_idx <= token_idx <= end_idx]
        if local_hits:
            first_hit = min(hit[0] for hit in local_hits)
            last_hit = max(hit[1] for hit in local_hits)
            # Conservative subtitle handoff: avoid jumping ahead of recitation.
            aligned_start = clamp(first_hit + 0.08, 0.12, duration - 0.35)
            aligned_end = clamp(last_hit + 0.30, aligned_start + 0.58, duration - 0.10)
        else:
            base_start = interpolate_time(anchor_points, start_idx)
            base_end = interpolate_time(anchor_points, max(start_idx + 1, end_idx))
            aligned_start = clamp(base_start - advance, 0.12, duration - 0.35)
            aligned_end = clamp(base_end - (advance * 0.35), aligned_start + 0.58, duration - 0.10)

        old_start_f = float(old_start)
        old_end_f = float(old_end)
        confidence = chunk_confidences[idx]
        blend = clamp((confidence - 0.42) / 0.45, 0.20, 0.95)

        proposed_start = (old_start_f * (1.0 - blend)) + (aligned_start * blend)
        proposed_end = (old_end_f * (1.0 - blend)) + (aligned_end * blend)

        max_shift = 2.6 if confidence >= 0.76 else (1.9 if confidence >= 0.62 else 1.25)
        start_t = clamp(proposed_start, old_start_f - max_shift, old_start_f + max_shift)
        end_t = clamp(proposed_end, old_end_f - (max_shift * 1.15), old_end_f + (max_shift * 1.15))
        start_t = clamp(start_t, 0.08, duration - 0.30)
        end_t = clamp(end_t, start_t + 0.58, duration - 0.08)
        total_shift += abs(start_t - old_start_f)

        if realigned:
            prev_text, prev_start, prev_end = realigned[-1]
            start_t = max(prev_start + 0.30, min(start_t, prev_end - 0.14))
            tightened_prev_end = clamp(start_t + 0.14, prev_start + 0.58, prev_end)
            realigned[-1] = (prev_text, prev_start, round(tightened_prev_end, 2))

        realigned.append((text, round(start_t, 2), round(end_t, 2)))

    for idx in range(len(realigned) - 1):
        text, start_t, end_t = realigned[idx]
        next_start = realigned[idx + 1][1]
        adjusted_end = min(end_t, next_start + 0.12)
        adjusted_end = max(start_t + 0.58, adjusted_end)
        realigned[idx] = (text, round(start_t, 2), round(adjusted_end, 2))

    return realigned, chunk_confidences, {
        "used_alignment": True,
        "reason": "ok",
        "canonical_tokens": len(canonical_tokens),
        "aligned_tokens": len(aligned_tokens),
        "coverage": round(coverage, 3),
        "avg_score": round(avg_score, 2),
        "mean_start_shift_seconds": round(total_shift / max(1, len(chunks)), 3),
    }


def resolve_output_targets(output_path: Path, style: str, variants: list[str]) -> list[tuple[str, str, Path]]:
    styles = ["fit", "fill"] if style == "both" else [style]
    targets: list[tuple[str, str, Path]] = []
    for style_name in styles:
        for variant in variants:
            file_path = output_path.with_name(f"{output_path.stem}-{style_name}-{variant}{output_path.suffix}")
            targets.append((style_name, variant, file_path))
    return targets


def load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as exc:
        raise SystemExit("Pillow is required.") from exc
    return Image, ImageDraw, ImageFont


def load_font(image_font, size: int, font_file: str | None):
    candidates: list[str] = []
    if font_file:
        candidates.append(font_file)
    candidates.extend([
        "/System/Library/Fonts/Supplemental/GillSans.ttc",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ])
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return image_font.truetype(str(path), size=size)
            except Exception:
                continue
    return image_font.load_default()


def draw_centered(draw, text: str, font, y: int, width: int, fill):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    x = max(16, (width - text_w) // 2)
    draw.text((x, y), text, fill=fill, font=font)


def create_top_overlay(
    path: Path,
    width: int,
    variant: str,
    surah_name: str,
    surah_number: int,
    ayah_start: int,
    ayah_end: int,
    sheikh: str,
    day: int,
    font_file: str | None,
) -> None:
    image, image_draw, image_font = load_pillow()
    height = 220
    canvas = image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = image_draw.Draw(canvas)

    font_main = load_font(image_font, 62, font_file)
    font_sub = load_font(image_font, 36, font_file)

    ayah_ref = f"{surah_number:02d}:{ayah_start}" if ayah_start == ayah_end else f"{surah_number:02d}:{ayah_start}-{ayah_end}"
    top_line = f"{surah_name} {ayah_ref}"
    draw_centered(draw, top_line, font_main, 46, width, (255, 255, 255, 246))
    if variant == "context":
        sub_line = f"Ramadan Day {day} • {sheikh}"
    else:
        sub_line = sheikh
    draw_centered(draw, sub_line, font_sub, 132, width, (235, 245, 248, 214))
    canvas.save(path, "PNG")


def build_caption_chunks(text: str, duration: float, variant: str) -> list[tuple[str, float, float]]:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return []
    total = len(words)
    start_pad = 0.28
    end_pad = 0.18
    clip_end = max(0.9, duration - end_pad)
    segment_span = max(0.7, clip_end - start_pad)

    if total <= 12:
        return [(" ".join(words), start_pad, clip_end)]

    chunk_size = 11 if variant == "clean" else 10
    step = max(6, chunk_size - 2)

    # Prefer punctuation boundaries to reduce subtitle semantic drift.
    sentence_like = [s.strip() for s in re.split(r"(?<=[\\.!\\?;:])\\s+", text) if s.strip()]
    text_chunks: list[str] = []
    if len(sentence_like) > 1:
        buffer_words: list[str] = []
        for sentence in sentence_like:
            sentence_words = sentence.split()
            if not sentence_words:
                continue
            if buffer_words and len(buffer_words) + len(sentence_words) > (chunk_size + 4):
                text_chunks.append(" ".join(buffer_words))
                buffer_words = list(sentence_words)
            else:
                buffer_words.extend(sentence_words)
        if buffer_words:
            text_chunks.append(" ".join(buffer_words))

        # Guard against too many tiny chunks: collapse by stepping.
        if len(text_chunks) >= 8:
            compact: list[str] = []
            i = 0
            while i < len(text_chunks):
                merged = text_chunks[i]
                if i + 1 < len(text_chunks) and len(merged.split()) < 7:
                    merged = f"{merged} {text_chunks[i + 1]}".strip()
                    i += 1
                compact.append(merged)
                i += 1
            text_chunks = compact

    if not text_chunks:
        start_idx = 0
        while start_idx < total:
            end_idx = min(total, start_idx + chunk_size)
            text_chunks.append(" ".join(words[start_idx:end_idx]))
            if end_idx >= total:
                break
            start_idx += step

    n = len(text_chunks)
    if n == 1:
        return [(text_chunks[0], start_pad, clip_end)]

    chunks: list[tuple[str, float, float]] = []
    for idx, chunk in enumerate(text_chunks):
        base_start = start_pad + (segment_span * (idx / n))
        base_end = start_pad + (segment_span * ((idx + 1) / n))
        start_t = base_start - (0.12 if idx > 0 else 0.0)
        end_t = base_end + (0.08 if idx < n - 1 else 0.0)
        if idx == n - 1:
            end_t = clip_end
        start_t = max(start_pad, start_t)
        end_t = min(clip_end, max(start_t + 0.62, end_t))
        chunks.append((chunk, round(start_t, 2), round(end_t, 2)))
    return chunks


def load_subtitle_map(path: Path, duration: float) -> list[tuple[str, float, float]]:
    if not path.exists():
        raise SystemExit(f"Subtitle map not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("chunks", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid subtitle map format in: {path}")

    chunks: list[tuple[str, float, float]] = []
    clip_end = max(0.9, duration - 0.08)
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(row.get("start", 0.0))
            end = float(row.get("end", start + 0.8))
        except (TypeError, ValueError):
            continue
        start = clamp(start, 0.05, clip_end - 0.35)
        end = clamp(end, start + 0.35, clip_end)
        chunks.append((text, round(start, 2), round(end, 2)))

    if not chunks:
        raise SystemExit(f"Subtitle map has no valid chunks: {path}")
    chunks.sort(key=lambda row: (row[1], row[2]))
    return chunks


def write_subtitle_map(
    path: Path,
    *,
    day: int,
    surah_number: int,
    ayah_start: int,
    ayah_end: int,
    start_seconds: float,
    duration: float,
    source: str,
    chunks: list[tuple[str, float, float]],
    render: dict | None = None,
) -> None:
    payload = {
        "version": 1,
        "day": day,
        "surah_number": surah_number,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
        "clip_start_seconds": round(start_seconds, 3),
        "duration_seconds": round(duration, 3),
        "source": source,
        "chunks": [
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
            }
            for text, start, end in chunks
        ],
    }
    if render:
        payload["render"] = render
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def hold_caption_chunks_until_next(
    chunks: list[tuple[str, float, float]],
    duration: float,
    chunk_confidences: list[float] | None = None,
) -> list[tuple[str, float, float]]:
    if not chunks:
        return []

    clip_end = max(0.5, duration - 0.08)
    ordered = sorted(chunks, key=lambda row: (row[1], row[2]))
    confidences = list(chunk_confidences or [])
    if len(confidences) < len(ordered):
        confidences.extend([0.62] * (len(ordered) - len(confidences)))
    elif len(confidences) > len(ordered):
        confidences = confidences[: len(ordered)]

    cleaned: list[dict] = []
    for index, (text, start, end) in enumerate(ordered):
        s = max(0.05, float(start))
        e = max(s + 0.30, float(end))
        cleaned.append({"text": str(text), "start": s, "end": e, "confidence": float(confidences[index])})

    for index in range(len(cleaned) - 1):
        next_row = cleaned[index + 1]
        next_confidence = float(next_row["confidence"])
        if next_confidence >= 0.62:
            continue
        delay = min(1.8, (0.62 - next_confidence) * 4.2)
        delayed_start = float(next_row["start"]) + delay
        latest_start = float(next_row["end"]) - 0.36
        delayed_start = min(delayed_start, latest_start)
        delayed_start = max(float(cleaned[index]["start"]) + 0.38, delayed_start)
        next_row["start"] = delayed_start
        next_row["end"] = max(delayed_start + 0.45, float(next_row["end"]))

    held: list[tuple[str, float, float]] = []
    for index, row in enumerate(cleaned):
        text = str(row["text"])
        start = float(row["start"])
        end = float(row["end"])
        if held:
            prev_start, prev_end = held[-1][1], held[-1][2]
            start = max(start, prev_end + 0.06, prev_start + 0.40)
            end = max(end, start + 0.30)
        if index < len(cleaned) - 1:
            next_start = max(start + 0.30, float(cleaned[index + 1]["start"]))
            end = max(end, next_start)
        else:
            end = max(end, clip_end)
        end = min(end, clip_end)
        end = max(start + 0.30, end)
        held.append((text, round(start, 2), round(end, 2)))
    return held


def create_caption_overlay(path: Path, width: int, text: str, variant: str, font_file: str | None) -> None:
    image, image_draw, image_font = load_pillow()
    height = SUBTITLE_OVERLAY_HEIGHT
    canvas = image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = image_draw.Draw(canvas)

    box_color = {
        "clean": (21, 67, 82, 226),
        "focus": (16, 59, 72, 234),
        "context": (23, 70, 84, 224),
    }.get(variant, (21, 67, 82, 226))
    draw.rounded_rectangle((44, 20, width - 44, height - 20), radius=32, fill=box_color)

    # Auto-fit long translations so we don't clip the second half of long ayat.
    # Prefer full text visibility over large font size.
    horizontal_padding = 88
    vertical_padding = 40
    content_w = max(200, width - horizontal_padding)
    content_h = max(120, height - vertical_padding)

    chosen_caption = text.strip()
    chosen_font = load_font(image_font, 34, font_file)
    chosen_spacing = 9
    fit_found = False
    for font_size in [36, 34, 32, 30, 28, 26]:
        test_font = load_font(image_font, font_size, font_file)
        # Wider wrap when font shrinks; keep all lines (no hard truncation).
        wrap_width = max(26, int((content_w / max(10, font_size)) * 1.95))
        wrapped_lines = textwrap.wrap(text, width=wrap_width)
        if not wrapped_lines:
            wrapped_lines = [text.strip()]
        test_caption = "\n".join(wrapped_lines)
        spacing = 8 if font_size <= 34 else 10
        left, top, right, bottom = draw.multiline_textbbox((0, 0), test_caption, font=test_font, spacing=spacing, align="center")
        text_w = right - left
        text_h = bottom - top
        if text_w <= content_w and text_h <= content_h:
            chosen_caption = test_caption
            chosen_font = test_font
            chosen_spacing = spacing
            fit_found = True
            break

    if not fit_found:
        # Force-paginate for very long captions so we always keep full text visible.
        fallback_font_size = 28
        fallback_font = load_font(image_font, fallback_font_size, font_file)
        wrap_width = max(24, int((content_w / max(10, fallback_font_size)) * 1.85))
        wrapped_lines = textwrap.wrap(text, width=wrap_width) or [text.strip()]
        chosen_caption = "\n".join(wrapped_lines)
        chosen_font = fallback_font
        chosen_spacing = 8

    _, _, _, total_h = draw.multiline_textbbox((0, 0), chosen_caption, font=chosen_font, spacing=chosen_spacing, align="center")
    draw.multiline_text(
        (width // 2, max(28, (height - total_h) // 2 - 2)),
        chosen_caption,
        fill=(255, 255, 255, 250),
        font=chosen_font,
        spacing=chosen_spacing,
        align="center",
        anchor="ma",
    )
    canvas.save(path, "PNG")


def build_filter_complex(
    style: str,
    width_i: int,
    height_i: int,
    bg_dim: float,
    bg_blur: int,
    caption_windows: list[tuple[float, float]],
) -> str:
    bg_dim = clamp(bg_dim, 0.0, 0.9)
    bg_blur = max(0, bg_blur)

    top_band_y = 0
    if style == "fill":
        chain = [f"[0:v]scale={width_i}:{height_i}:force_original_aspect_ratio=increase,crop={width_i}:{height_i}[b0]"]
        subtitle_y = max(220, min(height_i - SUBTITLE_OVERLAY_HEIGHT - 36, int(height_i * 0.62)))
    else:
        foreground_h = int(round(width_i * 9 / 16))
        transition_top = (height_i - foreground_h) // 2
        transition_bottom = transition_top + foreground_h
        top_band_y = max(14, transition_top - 236)
        chain = [
            f"[0:v]scale={width_i}:{height_i}:force_original_aspect_ratio=increase,crop={width_i}:{height_i},boxblur={bg_blur}:2,drawbox=x=0:y=0:w=iw:h=ih:color=black@{bg_dim:.2f}:t=fill[bg]",
            f"[0:v]scale={width_i}:{height_i}:force_original_aspect_ratio=decrease[fg]",
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[b0]",
        ]
        subtitle_y = max(220, min(height_i - SUBTITLE_OVERLAY_HEIGHT - 36, transition_bottom + 16))

    chain.append(f"[b0]drawbox=x=0:y={top_band_y}:w={width_i}:h=230:color=0x184b59@0.66:t=fill[b1]")
    chain.append(f"[b1][1:v]overlay=0:{top_band_y + 8}[b2]")

    prev = "b2"
    fade_in = SUBTITLE_FADE_IN_SECONDS
    fade_out = SUBTITLE_FADE_OUT_SECONDS
    for idx, (start, end) in enumerate(caption_windows):
        input_index = idx + 2
        next_label = f"b{idx + 3}"
        caption_label = f"cap{idx}"
        fade_out_start = max(start + 0.02, end - fade_out)
        chain.append(
            f"[{input_index}:v]format=rgba,"
            f"fade=t=in:st={start:.2f}:d={fade_in:.2f}:alpha=1,"
            f"fade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}:alpha=1"
            f"[{caption_label}]"
        )
        chain.append(f"[{prev}][{caption_label}]overlay=0:{subtitle_y}[{next_label}]")
        prev = next_label
    chain.append(f"[{prev}]format=yuv420p[v]")
    return ";".join(chain)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create vertical reel clips from Taraweeh ayah highlights.")
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--surah-number", type=int, required=True)
    parser.add_argument("--ayah", type=int, required=True)
    parser.add_argument("--ayah-end", type=int)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--duration", type=float, default=22.0)
    parser.add_argument("--sheikh", type=str, required=True)
    parser.add_argument("--video-file", type=str)
    parser.add_argument("--youtube-url", type=str)
    parser.add_argument("--size", type=str, default="1080x1920")
    parser.add_argument("--style", type=str, choices=["fit", "fill", "both"], default="fit")
    parser.add_argument("--variants", type=str, default="clean,focus,context")
    parser.add_argument("--bg-dim", type=float, default=0.40)
    parser.add_argument("--bg-blur", type=int, default=24)
    parser.add_argument("--audio-fade-out", type=float, default=0.28)
    parser.add_argument("--align-subtitles", action="store_true")
    parser.add_argument("--subtitle-model", type=str, default="medium")
    parser.add_argument("--subtitle-advance", type=float, default=0.0)
    parser.add_argument("--no-marker-caption-timing", action="store_true")
    parser.add_argument("--disable-subtitles", action="store_true")
    parser.add_argument("--subtitle-replacement-text", type=str, default="")
    parser.add_argument("--english-corpus", type=str, default="data/quran/quran_abdelhaleem_en.json")
    parser.add_argument("--fallback-english-corpus", type=str, default="data/quran/quran_asad_en.json")
    parser.add_argument("--prefer-marker-english", action="store_true")
    parser.add_argument("--font-file", type=str)
    parser.add_argument("--subtitle-map-file", type=str, help="Load manual subtitle map JSON (chunks with start/end/text).")
    parser.add_argument("--subtitle-map-output", type=str, help="Where to write generated subtitle map JSON.")
    parser.add_argument("--output", type=str)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    require_binary("ffmpeg")
    payload, day_source_path = load_day_payload(args.day)

    ayah_end = int(args.ayah_end) if args.ayah_end is not None else int(args.ayah)
    ayah_start = int(args.ayah)
    if ayah_end < ayah_start:
        ayah_end = ayah_start

    ayah_content = collect_ayah_range_text(
        payload,
        args.surah_number,
        ayah_start,
        ayah_end,
        english_file=Path(args.english_corpus),
        fallback_english_file=Path(args.fallback_english_corpus),
        prefer_marker_english=bool(args.prefer_marker_english),
    )
    english_text = str(ayah_content.get("english_text", "")).strip()
    if not english_text:
        english_text = f"Surah {args.surah_number}, Ayah {ayah_start}" if ayah_start == ayah_end else f"Surah {args.surah_number}, Ayat {ayah_start}-{ayah_end}"
    arabic_text = str(ayah_content.get("arabic_text", "")).strip()

    start_seconds = parse_timestamp(args.start)
    duration = max(5.0, float(args.duration))
    end_seconds = start_seconds + duration

    range_slug = f"a{ayah_start}" if ayah_start == ayah_end else f"a{ayah_start}-{ayah_end}"
    work_dir = Path("output/reels") / f"day-{args.day}" / f"s{args.surah_number}-{range_slug}"
    work_dir.mkdir(parents=True, exist_ok=True)

    resolved_source_url = str(args.youtube_url or ayah_content.get("source_url") or payload.get("source") or "").strip()
    if not resolved_source_url:
        sources = payload.get("sources", [])
        if isinstance(sources, list) and sources:
            resolved_source_url = str(sources[0]).strip()

    source_video = choose_source_video(args, work_dir, default_youtube_url=resolved_source_url)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output/reels") / f"day-{args.day}" / (f"surah-{args.surah_number}-ayah-{ayah_start}.mp4" if ayah_start == ayah_end else f"surah-{args.surah_number}-ayah-{ayah_start}-{ayah_end}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = args.size.split("x")
    width_i = int(width)
    height_i = int(height)

    variants = parse_variants(args.variants)
    surah_name = str(ayah_content.get("surah_name") or f"Surah {args.surah_number}").strip()
    targets = resolve_output_targets(output_path, args.style, variants)
    rendered: list[Path] = []

    transcript_tokens: list[dict] = []
    canonical_tokens: list[str] = []
    aligned_tokens: list[tuple[int, float, float, float]] = []
    subtitle_qa_rows: list[dict] = []

    replacement_text = str(args.subtitle_replacement_text or "").strip()
    subtitles_disabled = bool(args.disable_subtitles)
    manual_subtitle_chunks: list[tuple[str, float, float]] = []
    if args.subtitle_map_file:
        manual_subtitle_chunks = load_subtitle_map(Path(args.subtitle_map_file), duration=duration)
    alignment_active = bool(args.align_subtitles and arabic_text and not subtitles_disabled)
    if alignment_active and not manual_subtitle_chunks:
        clip_words = transcribe_clip_words(
            source_video=source_video,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            work_dir=work_dir,
            model_size=args.subtitle_model,
        )
        transcript_tokens = tokenize_transcript_words(clip_words)
        canonical_tokens = [token for token in normalize_arabic_token(arabic_text).split() if token]
        aligned_tokens = align_arabic_tokens(canonical_tokens, transcript_tokens)

    subtitle_map_written = False
    subtitle_map_path_value: Path | None = None
    for style_name, variant_name, target_path in targets:
        top_overlay = work_dir / f"overlay-top-{variant_name}.png"
        create_top_overlay(
            path=top_overlay,
            width=width_i,
            variant=variant_name,
            surah_name=surah_name,
            surah_number=args.surah_number,
            ayah_start=ayah_start,
            ayah_end=ayah_end,
            sheikh=args.sheikh.strip(),
            day=args.day,
            font_file=args.font_file,
        )

        if subtitles_disabled:
            if replacement_text:
                clip_end = max(0.9, duration - 0.18)
                caption_chunks = [(replacement_text, 0.28, clip_end)]
            else:
                caption_chunks = []
            caption_confidences = [0.92 for _ in caption_chunks]
            alignment_summary = {
                "used_alignment": False,
                "reason": "subtitles_disabled",
                "canonical_tokens": len(canonical_tokens),
                "aligned_tokens": len(aligned_tokens),
                "coverage": 0.0,
                "avg_score": 0.0,
            }
            timing_source = "replacement_text" if caption_chunks else "disabled"
        else:
            caption_chunks = []
            timing_source = "fallback"
            marker_timing_used = False
            caption_confidences: list[float] = []
            alignment_summary = {
                "used_alignment": False,
                "reason": "not_requested_or_unavailable",
                "canonical_tokens": len(canonical_tokens),
                "aligned_tokens": len(aligned_tokens),
                "coverage": 0.0,
                "avg_score": 0.0,
            }
            if manual_subtitle_chunks:
                caption_chunks = list(manual_subtitle_chunks)
                timing_source = "manual-map"
                caption_confidences = [0.96 for _ in caption_chunks]
                alignment_summary = {
                    "used_alignment": False,
                    "reason": "manual_subtitle_map",
                    "canonical_tokens": len(canonical_tokens),
                    "aligned_tokens": len(aligned_tokens),
                    "coverage": 1.0,
                    "avg_score": 100.0,
                }
            elif not args.no_marker_caption_timing:
                caption_chunks = build_ayah_caption_chunks_from_markers(
                    day_payload=payload,
                    surah_number=args.surah_number,
                    ayah_start=ayah_start,
                    ayah_end=ayah_end,
                    clip_start=start_seconds,
                    clip_end=end_seconds,
                    english_file=Path(args.english_corpus),
                    fallback_english_file=Path(args.fallback_english_corpus),
                    prefer_marker_english=bool(args.prefer_marker_english),
                    source_url=resolved_source_url or None,
                    split_long_ayahs=True,
                )
                if caption_chunks:
                    marker_timing_used = True
                    timing_source = "marker"
                    caption_confidences = [0.78 for _ in caption_chunks]
            if not caption_chunks:
                caption_chunks = build_caption_chunks(english_text, duration, variant_name)
                caption_confidences = [0.78 if marker_timing_used else 0.62 for _ in caption_chunks]
            if alignment_active and caption_chunks and not manual_subtitle_chunks:
                caption_chunks, aligned_confidences, alignment_summary = apply_alignment_to_chunks(
                    chunks=caption_chunks,
                    canonical_tokens=canonical_tokens,
                    aligned_tokens=aligned_tokens,
                    duration=duration,
                    subtitle_advance=args.subtitle_advance,
                )
                if aligned_confidences:
                    caption_confidences = aligned_confidences

            # Accuracy-first safety: when confidence is weak on fallback timing, avoid rapid chunk flips.
            # Do not collapse marker-timed chunks; they are the primary recitation progression signal.
            avg_conf_preview = sum(caption_confidences) / max(1, len(caption_confidences))
            if (not marker_timing_used) and (not manual_subtitle_chunks) and avg_conf_preview < 0.50 and len(caption_chunks) > 1:
                merged_text = " ".join(part for part, _s, _e in caption_chunks if part).strip()
                if merged_text:
                    caption_chunks = [(merged_text, 0.28, max(0.9, duration - 0.18))]
                    caption_confidences = [max(0.40, avg_conf_preview)]
                    timing_source = f"{timing_source}-stabilized"

            caption_chunks = hold_caption_chunks_until_next(caption_chunks, duration, chunk_confidences=caption_confidences)

        if not subtitle_map_written:
            if args.subtitle_map_output:
                subtitle_map_path = Path(args.subtitle_map_output)
            else:
                subtitle_map_path = target_path.parent / f"{target_path.stem}.subtitle-map.json"
            write_subtitle_map(
                subtitle_map_path,
                day=args.day,
                surah_number=args.surah_number,
                ayah_start=ayah_start,
                ayah_end=ayah_end,
                start_seconds=start_seconds,
                duration=duration,
                source=timing_source,
                chunks=caption_chunks,
                render={
                    "day": args.day,
                    "surah_number": args.surah_number,
                    "ayah_start": ayah_start,
                    "ayah_end": ayah_end,
                    "start": args.start,
                    "duration": duration,
                    "sheikh": args.sheikh.strip(),
                    "video_file": str(source_video),
                    "style": style_name,
                    "variants": variant_name,
                    "subtitle_model": args.subtitle_model,
                    "output": str(target_path),
                },
            )
            subtitle_map_written = True
            subtitle_map_path_value = subtitle_map_path

        avg_conf = sum(caption_confidences) / max(1, len(caption_confidences))
        low_conf_count = len([value for value in caption_confidences if value < 0.62])
        alignment_cov = float((alignment_summary or {}).get("coverage", 0.0) or 0.0)
        qa_score = clamp((avg_conf * 0.72) + (alignment_cov * 0.28), 0.0, 1.0)
        subtitle_qa_rows.append(
            {
                "variant": variant_name,
                "style": style_name,
                "timing_source": timing_source,
                "chunk_count": len(caption_chunks),
                "avg_chunk_confidence": round(avg_conf, 3),
                "low_confidence_chunks": low_conf_count,
                "qa_score": round(qa_score, 3),
                "qa_grade": grade_from_score(qa_score),
                "alignment": alignment_summary,
                "chunks": [
                    {
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "confidence": round(caption_confidences[index], 3) if index < len(caption_confidences) else 0.0,
                        "text_preview": chunk_text[:120],
                    }
                    for index, (chunk_text, start, end) in enumerate(caption_chunks)
                ],
            }
        )

        caption_paths: list[Path] = []
        caption_windows: list[tuple[float, float]] = []
        for index, (chunk_text, start, end) in enumerate(caption_chunks):
            caption_overlay = work_dir / f"overlay-caption-{variant_name}-{index + 1:02d}.png"
            create_caption_overlay(
                path=caption_overlay,
                width=width_i,
                text=chunk_text,
                variant=variant_name,
                font_file=args.font_file,
            )
            caption_paths.append(caption_overlay)
            caption_windows.append((start, end))

        filter_complex = build_filter_complex(
            style=style_name,
            width_i=width_i,
            height_i=height_i,
            bg_dim=args.bg_dim,
            bg_blur=args.bg_blur,
            caption_windows=caption_windows,
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_seconds),
            "-to",
            str(end_seconds),
            "-i",
            str(source_video),
            "-loop",
            "1",
            "-i",
            str(top_overlay),
        ]
        for path in caption_paths:
            command.extend(["-loop", "1", "-i", str(path)])
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-shortest",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "19",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
            ]
        )

        fade_out = max(0.0, float(args.audio_fade_out))
        if fade_out > 0:
            fade_start = max(0.0, duration - fade_out)
            command.extend(["-af", f"afade=t=out:st={fade_start:.2f}:d={fade_out:.2f}"])

        command.extend(["-movflags", "+faststart", str(target_path)])
        subprocess.run(command, check=True)
        rendered.append(target_path)

    qa_report_path = work_dir / "subtitle-qa.json"
    qa_report = {
        "day": args.day,
        "surah_number": args.surah_number,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
        "clip_start_seconds": round(start_seconds, 2),
        "clip_end_seconds": round(end_seconds, 2),
        "duration_seconds": round(duration, 2),
        "alignment_requested": bool(args.align_subtitles),
        "alignment_active": alignment_active,
        "subtitle_model": args.subtitle_model if alignment_active else None,
        "canonical_tokens": len(canonical_tokens),
        "transcript_tokens": len(transcript_tokens),
        "aligned_tokens": len(aligned_tokens),
        "rendered_files": [str(path) for path in rendered],
        "overall_qa_score": round(
            clamp(
                sum(float(row.get("qa_score", 0.0)) for row in subtitle_qa_rows) / max(1, len(subtitle_qa_rows)),
                0.0,
                1.0,
            ),
            3,
        ),
        "overall_qa_grade": grade_from_score(
            clamp(
                sum(float(row.get("qa_score", 0.0)) for row in subtitle_qa_rows) / max(1, len(subtitle_qa_rows)),
                0.0,
                1.0,
            )
        ),
        "subtitle_map_file": str(subtitle_map_path_value) if subtitle_map_path_value else None,
        "variants": subtitle_qa_rows,
    }
    qa_report_path.write_text(json.dumps(qa_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Source: {source_video}")
    print(f"Day JSON source: {day_source_path}")
    if resolved_source_url:
        print(f"Source URL: {resolved_source_url}")
    if ayah_start == ayah_end:
        print(f"Ayah range: surah {args.surah_number}, ayah {ayah_start}")
    else:
        print(f"Ayah range: surah {args.surah_number}, ayat {ayah_start}-{ayah_end}")
    if alignment_active:
        print(
            "Subtitle alignment:"
            f" canonical_tokens={len(canonical_tokens)}"
            f" transcript_tokens={len(transcript_tokens)}"
            f" aligned_tokens={len(aligned_tokens)}"
        )
    print(f"Subtitle QA: {qa_report_path}")
    if subtitle_map_path_value:
        print(f"Subtitle map: {subtitle_map_path_value}")
    for path in rendered:
        print(f"Clip: {path}")
    print(f"Overlay assets: {work_dir}")


if __name__ == "__main__":
    main()
