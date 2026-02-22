from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from rapidfuzz import fuzz

from .types import Marker, TranscriptSegment, TranscriptWord

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_PUNCT = re.compile(r"[^\u0621-\u063A\u0641-\u064A\s]")
MULTI_SPACE = re.compile(r"\s+")
ARABIC_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ـ": "",
    }
)
ASAD_API_URL = "https://api.alquran.cloud/v1/quran/en.asad"
STRICT_NORMALIZATION = os.getenv("STRICT_NORMALIZATION", "false").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
MUQATTAAT_SPOKEN_FORMS: dict[str, list[str]] = {
    "الم": ["الف لام ميم"],
    "المر": ["الف لام ميم را"],
    "الر": ["الف لام را"],
    "كهيعص": ["كاف ها يا عين صاد"],
    "طه": ["طا ها"],
    "طسم": ["طا سين ميم"],
    "طس": ["طا سين"],
    "يس": ["يا سين"],
    "ص": ["صاد"],
    "حم": ["حا ميم"],
    "عسق": ["عين سين قاف"],
    "ق": ["قاف"],
    "ن": ["نون"],
}
MUQATTAAT_COMPACT_FORMS = set(MUQATTAAT_SPOKEN_FORMS.keys())
FATIHA_HINTS = [
    "الحمد لله رب العالمين",
    "الرحمن الرحيم",
    "مالك يوم الدين",
    "اياك نعبد واياك نستعين",
    "اهدنا الصراط المستقيم",
    "صراط الذين انعمت عليهم غير المغضوب عليهم ولا الضالين",
]

NON_RECITATION_HINTS = [
    "الله اكبر",
    "سمع الله لمن حمده",
    "ربنا ولك الحمد",
    "ربنا لك الحمد",
    "سبحان ربي العظيم",
    "سبحان ربي الاعلى",
    "السلام عليكم ورحمة الله",
]
FATIHA_HINTS_NORM: list[str] = []
NON_RECITATION_HINTS_NORM: list[str] = []
ARABIC_ANCHOR_STOPWORDS = {
    "و",
    "ف",
    "ثم",
    "لا",
    "ما",
    "من",
    "في",
    "على",
    "الى",
    "إلى",
    "ب",
    "الذي",
    "الذين",
    "هذا",
    "ذلك",
}


@dataclass
class AyahEntry:
    surah_number: int
    surah: str
    ayah: int
    text: str
    normalized: str
    match_forms: list[str]


@dataclass
class WordWindow:
    normalized_text: str
    start_time: float
    end_time: float
    word_indices: list[int]


@dataclass
class CandidateEvidence:
    adjusted_score: float
    score: float
    overlap: float
    penalty: float
    source: str
    normalized_text: str
    start_time: float
    end_time: float
    word_indices: list[int]
    segment_start_index: int | None = None
    segment_end_index: int | None = None


@dataclass
class TokenAlignmentResult:
    start_time: int
    end_time: int
    matched_token_indices: list[list[int]]
    alignment_score: float


def is_excluded_surah(surah: str) -> bool:
    normalized = surah.casefold().replace("-", "").replace(" ", "")
    return "fatiha" in normalized or "faatiha" in normalized or "فاتحة" in surah


def normalize_arabic(text: str, strict: bool | None = None) -> str:
    if strict is None:
        strict = STRICT_NORMALIZATION

    text = ARABIC_DIACRITICS.sub("", text)
    if not strict:
        text = text.translate(ARABIC_CHAR_MAP)
        text = text.replace("ـ", "")
    text = ARABIC_PUNCT.sub(" ", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    if not strict and text:
        tokens = text.split()
        collapsed: list[str] = []
        for token in tokens:
            if not collapsed or token != collapsed[-1]:
                collapsed.append(token)
        text = " ".join(collapsed)
    return text


def _word_text(word: TranscriptWord | dict) -> str:
    if isinstance(word, dict):
        return str(word.get("text", "")).strip()
    return str(getattr(word, "text", "")).strip()


def _word_start(word: TranscriptWord | dict, fallback: float) -> float:
    if isinstance(word, dict):
        return float(word.get("start", fallback))
    return float(getattr(word, "start", fallback))


def _word_end(word: TranscriptWord | dict, fallback: float) -> float:
    if isinstance(word, dict):
        return float(word.get("end", fallback))
    return float(getattr(word, "end", fallback))


def generate_word_windows(
    segment_words: list[TranscriptWord | dict],
    min_window: int = 4,
    max_window: int = 8,
):
    normalized_words: list[tuple[int, str, float, float]] = []
    for original_index, word in enumerate(segment_words):
        text = normalize_arabic(_word_text(word), strict=False)
        if not text:
            continue
        start = _word_start(word, fallback=0.0)
        end = _word_end(word, fallback=start)
        normalized_words.append((original_index, text, start, end))

    if min_window <= 0 or max_window <= 0:
        return
    if not normalized_words:
        return

    max_window = min(max_window, len(normalized_words))
    min_window = min(min_window, max_window)

    for window_size in range(min_window, max_window + 1):
        for left in range(0, len(normalized_words) - window_size + 1):
            chunk = normalized_words[left : left + window_size]
            normalized_text = normalize_arabic(" ".join(item[1] for item in chunk), strict=False)
            if not normalized_text:
                continue
            yield WordWindow(
                normalized_text=normalized_text,
                start_time=chunk[0][2],
                end_time=chunk[-1][3],
                word_indices=[item[0] for item in chunk],
            )


def _window_penalty(window_size: int) -> float:
    if window_size >= 8:
        return 0.0
    return max(0.0, float(8 - window_size) * 0.35)


def _token_similarity(token_a: str, token_b: str) -> float:
    if not token_a or not token_b:
        return 0.0
    partial = float(fuzz.partial_ratio(token_a, token_b))
    return partial / 100.0


def _align_tokens(transcript_tokens: list[str], canonical_tokens: list[str]) -> tuple[list[list[int]], float, float]:
    if not transcript_tokens or not canonical_tokens:
        return [], 0.0, 0.0

    m = len(transcript_tokens)
    n = len(canonical_tokens)
    gap_penalty = -0.45
    mismatch_penalty = -0.55
    match_threshold = 0.62

    scores = [[0.0] * (n + 1) for _ in range(m + 1)]
    moves = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        scores[i][0] = i * gap_penalty
        moves[i][0] = "up"
    for j in range(1, n + 1):
        scores[0][j] = j * gap_penalty
        moves[0][j] = "left"

    for i in range(1, m + 1):
        token_a = transcript_tokens[i - 1]
        for j in range(1, n + 1):
            token_b = canonical_tokens[j - 1]
            similarity = _token_similarity(token_a, token_b)
            diag_gain = similarity if similarity >= match_threshold else mismatch_penalty
            diag = scores[i - 1][j - 1] + diag_gain
            up = scores[i - 1][j] + gap_penalty
            left = scores[i][j - 1] + gap_penalty
            best = max(diag, up, left)
            scores[i][j] = best
            if best == diag:
                moves[i][j] = "diag"
            elif best == up:
                moves[i][j] = "up"
            else:
                moves[i][j] = "left"

    i = m
    j = n
    pairs: list[tuple[int, int, float]] = []
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "diag" and i > 0 and j > 0:
            similarity = _token_similarity(transcript_tokens[i - 1], canonical_tokens[j - 1])
            if similarity >= match_threshold:
                pairs.append((i - 1, j - 1, similarity))
            i -= 1
            j -= 1
        elif move == "up" and i > 0:
            i -= 1
        elif j > 0:
            j -= 1
        else:
            break

    if not pairs:
        return [], 0.0, 0.0

    pairs.reverse()
    mapping = [[left, right] for left, right, _ in pairs]
    avg_similarity = sum(score for _, _, score in pairs) / len(pairs)
    coverage = len(pairs) / max(1, min(len(transcript_tokens), len(canonical_tokens)))
    return mapping, avg_similarity, coverage


def _tokenize_transcript_words(
    words: list[TranscriptWord],
    selected_indices: list[int] | None = None,
) -> tuple[list[str], list[float], list[float]]:
    if not words:
        return [], [], []

    index_filter = set(selected_indices) if selected_indices else None
    tokens: list[str] = []
    starts: list[float] = []
    ends: list[float] = []

    for word_index, word in enumerate(words):
        if index_filter is not None and word_index not in index_filter:
            continue
        normalized = normalize_arabic(str(getattr(word, "text", "")), strict=False)
        if not normalized:
            continue
        pieces = [piece for piece in normalized.split() if piece]
        if not pieces:
            continue
        for piece in pieces:
            tokens.append(piece)
            starts.append(float(getattr(word, "start", 0.0)))
            ends.append(float(getattr(word, "end", float(getattr(word, "start", 0.0)))))

    return tokens, starts, ends


def _collect_words_from_segments(
    transcript_segments: list[TranscriptSegment],
    start_index: int,
    end_index: int,
    max_gap_seconds: float = 2.5,
) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    if start_index < 0 or end_index < start_index:
        return words

    previous_end: float | None = None
    max_index = min(end_index, len(transcript_segments) - 1)
    for index in range(start_index, max_index + 1):
        segment = transcript_segments[index]
        if previous_end is not None and float(segment.start) - previous_end > max_gap_seconds:
            break
        words.extend(list(getattr(segment, "words", None) or []))
        previous_end = float(segment.end)

    return words


def _align_entry_to_words(
    words: list[TranscriptWord],
    entry: AyahEntry,
    selected_word_indices: list[int] | None = None,
) -> TokenAlignmentResult | None:
    transcript_tokens, starts, ends = _tokenize_transcript_words(words, selected_indices=selected_word_indices)
    if len(transcript_tokens) < 2:
        return None

    best_alignment: TokenAlignmentResult | None = None
    best_score = -1.0

    for form in entry.match_forms:
        canonical_tokens = [token for token in form.split() if token]
        if len(canonical_tokens) < 2:
            continue
        mapping, avg_similarity, coverage = _align_tokens(transcript_tokens, canonical_tokens)
        if not mapping:
            continue

        alignment_score = (0.6 * coverage) + (0.4 * avg_similarity)
        if coverage < 0.2 or avg_similarity < 0.6:
            continue

        first_token = mapping[0][0]
        last_token = mapping[-1][0]
        start_time = int(round(starts[first_token]))
        end_time = int(round(ends[last_token]))
        if end_time < start_time:
            end_time = start_time

        result = TokenAlignmentResult(
            start_time=start_time,
            end_time=end_time,
            matched_token_indices=mapping,
            alignment_score=round(alignment_score, 3),
        )
        if alignment_score > best_score:
            best_alignment = result
            best_score = alignment_score

    return best_alignment


def _align_entry_to_segment_words(
    segment: TranscriptSegment,
    entry: AyahEntry,
    selected_word_indices: list[int] | None = None,
) -> TokenAlignmentResult | None:
    words = list(getattr(segment, "words", None) or [])
    return _align_entry_to_words(words=words, entry=entry, selected_word_indices=selected_word_indices)


def _resolve_marker_times(
    segment: TranscriptSegment,
    entry: AyahEntry,
    evidence: CandidateEvidence,
    transcript_segments: list[TranscriptSegment],
    segment_index: int,
) -> tuple[int, int, list[list[int]] | None]:
    default_start = int(round(_estimate_marker_onset_time(segment, entry)))
    default_end = int(round(max(default_start, evidence.end_time)))

    alignment: TokenAlignmentResult | None = None
    if (
        evidence.segment_start_index is not None
        and evidence.segment_end_index is not None
        and evidence.segment_end_index > evidence.segment_start_index
    ):
        merged_words = _collect_words_from_segments(
            transcript_segments=transcript_segments,
            start_index=evidence.segment_start_index,
            end_index=evidence.segment_end_index,
        )
        alignment = _align_entry_to_words(words=merged_words, entry=entry)

    if alignment is None:
        local_indices = evidence.word_indices if evidence.word_indices and evidence.segment_end_index == segment_index else None
        alignment = _align_entry_to_segment_words(
            segment=segment,
            entry=entry,
            selected_word_indices=local_indices,
        )

    if alignment is not None:
        return alignment.start_time, alignment.end_time, alignment.matched_token_indices

    return default_start, default_end, None


def _candidate_confidence(score: float, rival_score: float, overlap: float) -> float:
    margin = max(0.0, score - max(0.0, rival_score))
    return (0.55 * (score / 100.0)) + (0.25 * min(1.0, margin / 20.0)) + (0.2 * overlap)


def _best_rival_score(
    entry_candidates: dict[int, CandidateEvidence],
    index: int,
    excluded: set[int] | None = None,
) -> float:
    rival = -1.0
    for candidate_index, evidence in entry_candidates.items():
        if candidate_index == index:
            continue
        if excluded and candidate_index in excluded:
            continue
        if evidence.adjusted_score > rival:
            rival = evidence.adjusted_score
    return max(0.0, rival)


def _word_windows_overlap_ambiguously(left: CandidateEvidence, right: CandidateEvidence) -> bool:
    if not left.word_indices or not right.word_indices:
        return True

    left_set = set(left.word_indices)
    right_set = set(right.word_indices)
    overlap = len(left_set & right_set)
    if overlap <= 0:
        return False

    smaller = max(1, min(len(left_set), len(right_set)))
    return (overlap / smaller) > 0.35


def _build_match_forms(ayah_number: int, normalized_text: str) -> list[str]:
    forms = [normalized_text]
    compact = normalized_text.replace(" ", "")

    # Muqatta'at are often recited as spoken letter names rather than compact letters.
    if ayah_number == 1 and compact in MUQATTAAT_SPOKEN_FORMS:
        for variant in MUQATTAAT_SPOKEN_FORMS[compact]:
            normalized_variant = normalize_arabic(variant, strict=False)
            if normalized_variant and normalized_variant not in forms:
                forms.append(normalized_variant)
    return forms


def _is_muqattaat_entry(entry: AyahEntry) -> bool:
    if entry.ayah != 1:
        return False
    compact = entry.normalized.replace(" ", "")
    return compact in MUQATTAAT_COMPACT_FORMS


def _has_muqattaat_phrase_match(normalized_text: str, entry: AyahEntry) -> bool:
    if not normalized_text:
        return False
    text_tokens = [token for token in normalized_text.split() if token]
    if not text_tokens:
        return False

    for form in entry.match_forms:
        form_tokens = [token for token in form.split() if token]
        if not form_tokens:
            continue

        # Direct phrase/token sequence match is the strongest signal for muqatta'at.
        n = len(form_tokens)
        if n <= len(text_tokens):
            for idx in range(0, len(text_tokens) - n + 1):
                if text_tokens[idx : idx + n] == form_tokens:
                    return True

        # Fallback: very high fuzzy similarity for short phrase transcripts.
        if len(text_tokens) <= 14 and float(fuzz.token_set_ratio(normalized_text, form)) >= 95.0:
            return True

    return False


def _score_segment_against_entry(normalized_segment: str, entry: AyahEntry) -> tuple[float, float]:
    top_score = -1.0
    top_overlap = 0.0
    for candidate in entry.match_forms:
        token_set = float(fuzz.token_set_ratio(normalized_segment, candidate))
        partial = float(fuzz.partial_ratio(normalized_segment, candidate))
        score = (0.75 * token_set) + (0.25 * partial)
        if score > top_score:
            top_score = score
            top_overlap = _token_overlap(normalized_segment, candidate)
    return top_score, top_overlap


def _anchor_tokens_for_form(form: str) -> list[str]:
    tokens = [token for token in form.split() if token]
    if not tokens:
        return []

    strong = [token for token in tokens if len(token) >= 4 and token not in ARABIC_ANCHOR_STOPWORDS]
    if strong:
        return strong

    medium = [token for token in tokens if len(token) >= 3]
    if medium:
        return medium

    return tokens


def _estimate_marker_onset_time(segment: TranscriptSegment, entry: AyahEntry) -> int:
    words = getattr(segment, "words", None) or []
    if not words:
        return int(round(segment.start))

    best_time: float | None = None
    best_score = -1.0

    for form in entry.match_forms:
        anchors = _anchor_tokens_for_form(form)
        if not anchors:
            continue

        for word in words:
            normalized_word = normalize_arabic(str(getattr(word, "text", "")), strict=False)
            if not normalized_word:
                continue

            local_score = max(
                max(float(fuzz.ratio(normalized_word, anchor)), float(fuzz.partial_ratio(normalized_word, anchor)))
                for anchor in anchors
            )
            if local_score >= 80:
                word_start = float(getattr(word, "start", segment.start))
                if best_time is None or word_start < best_time:
                    best_time = word_start
                if local_score > best_score:
                    best_score = local_score

    return int(round(best_time if best_time is not None else segment.start))


def _is_fatiha_like_segment(normalized_segment: str, min_score: int = 90) -> bool:
    global FATIHA_HINTS_NORM
    if len(normalized_segment) < 10:
        return False
    if len(normalized_segment) > 80:
        return False

    if not FATIHA_HINTS_NORM:
        FATIHA_HINTS_NORM = [normalize_arabic(text, strict=False) for text in FATIHA_HINTS]

    scores = [float(fuzz.partial_ratio(normalized_segment, phrase)) for phrase in FATIHA_HINTS_NORM]
    if not scores:
        return False

    medium_hits = sum(score >= (min_score - 6) for score in scores)
    long_hit = any(len(phrase) >= 18 and score >= (min_score - 2) for phrase, score in zip(FATIHA_HINTS_NORM, scores))
    return long_hit or medium_hits >= 2


def _is_non_recitation_segment(normalized_segment: str, min_score: int = 95) -> bool:
    global NON_RECITATION_HINTS_NORM
    if len(normalized_segment) < 4:
        return False
    if len(normalized_segment) > 48:
        return False
    tokens = [token for token in normalized_segment.split() if token]
    if len(tokens) > 6:
        return False

    if not NON_RECITATION_HINTS_NORM:
        NON_RECITATION_HINTS_NORM = [normalize_arabic(text, strict=False) for text in NON_RECITATION_HINTS]

    best_score = 0.0
    best_overlap = 0.0
    for phrase in NON_RECITATION_HINTS_NORM:
        if normalized_segment == phrase:
            return True
        if normalized_segment in phrase or phrase in normalized_segment:
            if _token_overlap(normalized_segment, phrase) >= 0.7:
                return True
        score = float(fuzz.partial_ratio(normalized_segment, phrase))
        overlap = _token_overlap(normalized_segment, phrase)
        if score > best_score:
            best_score = score
            best_overlap = overlap
    return best_score >= float(min_score) and best_overlap >= 0.65


def clean_transcript_for_matching(
    transcript_segments: list[TranscriptSegment],
    min_arabic_chars: int = 2,
) -> list[TranscriptSegment]:
    cleaned: list[TranscriptSegment] = []
    for segment in transcript_segments:
        text = str(segment.text or "").strip()
        if not text:
            continue

        normalized = normalize_arabic(text, strict=False)
        if len(normalized) < 2:
            continue
        if _is_fatiha_like_segment(normalized):
            continue
        if _is_non_recitation_segment(normalized):
            continue

        arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        latin_chars = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
        if arabic_chars < min_arabic_chars:
            continue
        if latin_chars > 0 and arabic_chars < (latin_chars * 2):
            continue

        cleaned.append(segment)
    return cleaned


def detect_reset_markers_from_transcript(transcript_segments: list[TranscriptSegment]) -> list[float]:
    strict_phrases = [normalize_arabic(text, strict=False) for text in NON_RECITATION_HINTS]

    def is_strict_reset_phrase(normalized: str) -> bool:
        tokens = [token for token in normalized.split() if token]
        if not tokens or len(tokens) > 6:
            return False
        for phrase in strict_phrases:
            if normalized == phrase:
                return True
            phrase_tokens = [token for token in phrase.split() if token]
            if abs(len(tokens) - len(phrase_tokens)) > 1:
                continue
            ratio = float(fuzz.ratio(normalized, phrase))
            overlap = _token_overlap(normalized, phrase)
            if ratio >= 97.0 and overlap >= 0.85:
                return True
        return False

    reset_points: list[float] = []
    for segment in transcript_segments:
        normalized = normalize_arabic(str(segment.text or ""), strict=False)
        if not normalized:
            continue
        if _is_fatiha_like_segment(normalized) or is_strict_reset_phrase(normalized):
            reset_points.append(float(segment.start))
    return sorted(dict.fromkeys(reset_points))


def _reset_points_between(reset_times: list[float], left: int, right: int, margin: int = 3) -> list[int]:
    if not reset_times:
        return []
    lo = min(left, right) + margin
    hi = max(left, right) - margin
    if hi <= lo:
        return []
    return sorted(int(round(item)) for item in reset_times if lo <= item <= hi)


def _recover_missing_gap_with_search(
    left: Marker,
    right: Marker,
    entry_lookup: dict[tuple[str, int], AyahEntry],
    transcript_segments: list[TranscriptSegment],
    min_gap_seconds: int,
    min_score: int,
    min_overlap: float,
    min_confidence: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
    require_weak_support_for_inferred: bool,
    search_floor_time: int | None = None,
    exhaustive_ahead_search: bool = False,
) -> list[Marker]:
    recovered: list[Marker] = []
    window_start_limit = left.time + min_gap_seconds
    window_end_limit = right.time - min_gap_seconds
    if window_end_limit <= window_start_limit:
        return recovered

    missing_ayahs = list(range(left.ayah + 1, right.ayah))
    if not missing_ayahs:
        return recovered

    cursor = window_start_limit
    span = max(1, right.time - left.time)
    for ayah in missing_ayahs:
        entry = entry_lookup.get((left.surah, ayah))
        if entry is None:
            continue
        ratio = (ayah - left.ayah) / max(1, right.ayah - left.ayah)
        expected = int(round(left.time + (span * ratio)))
        half = max(20, int(round(span / max(6, len(missing_ayahs)))))
        if exhaustive_ahead_search:
            search_start = max(cursor, int(search_floor_time or window_start_limit))
            search_end = window_end_limit
        else:
            search_start = max(cursor, expected - half)
            if search_floor_time is not None:
                search_start = max(search_start, int(search_floor_time))
            search_end = min(window_end_limit, expected + (half * 2))
        if search_end <= search_start:
            continue

        best = _find_best_ayah_timestamp(
            transcript_segments=transcript_segments,
            entry=entry,
            window_start=search_start,
            window_end=search_end,
            expected_time=max(search_start, min(search_end, expected)),
            min_score=max(68 if exhaustive_ahead_search else 72, min_score - (8 if exhaustive_ahead_search else 4)),
            min_overlap=max(0.06 if exhaustive_ahead_search else 0.10, min_overlap - 0.08),
            min_confidence=max(0.50 if exhaustive_ahead_search else 0.56, min_confidence - 0.10),
            ambiguous_min_score=max(64 if exhaustive_ahead_search else 68, ambiguous_min_score - 6),
            ambiguous_min_confidence=max(0.46 if exhaustive_ahead_search else 0.50, ambiguous_min_confidence - 0.02),
        )
        if best is None:
            continue

        matched_time, matched_end, quality, confidence = best
        matched_time = max(search_start, min(search_end, matched_time))
        matched_end = max(matched_time, min(search_end, matched_end))
        if require_weak_support_for_inferred and quality == "ambiguous":
            if not _has_weak_local_support(
                transcript_segments=transcript_segments,
                entry=entry,
                window_start=max(window_start_limit, matched_time - 14),
                window_end=min(window_end_limit, matched_time + 14),
                min_score=max(62, ambiguous_min_score - 8),
                min_overlap=max(0.08, min_overlap - 0.06),
            ):
                continue

        recovered.append(
            Marker(
                time=matched_time,
                start_time=matched_time,
                end_time=matched_end,
                surah=left.surah,
                surah_number=left.surah_number,
                ayah=ayah,
                juz=get_juz_for_ayah(left.surah_number or 1, ayah),
                quality=quality,
                confidence=confidence,
            )
        )
        cursor = max(cursor + 1, matched_time + 2)
        if cursor >= window_end_limit:
            break

    return recovered


def _token_overlap(query: str, reference: str) -> float:
    query_tokens = {token for token in query.split() if token and token not in ARABIC_ANCHOR_STOPWORDS}
    reference_tokens = {token for token in reference.split() if token and token not in ARABIC_ANCHOR_STOPWORDS}
    if not query_tokens or not reference_tokens:
        return 0.0

    shared = len(query_tokens & reference_tokens)
    return shared / max(1, len(reference_tokens))


def _contains_fatiha_reset_between(reset_times: list[float], left: int, right: int, margin: int = 3) -> bool:
    if not reset_times:
        return False
    lo = min(left, right) + margin
    hi = max(left, right) - margin
    if hi <= lo:
        return False
    return any(lo <= timestamp <= hi for timestamp in reset_times)


def _has_low_data_gap(
    transcript_segments: list[TranscriptSegment],
    start_time: int,
    end_time: int,
    max_silence_seconds: int = 20,
    min_density: float = 0.07,
) -> bool:
    if end_time <= start_time:
        return False

    start = float(start_time)
    end = float(end_time)
    span = end - start
    if span <= 0:
        return False

    relevant: list[tuple[float, float]] = []
    for segment in transcript_segments:
        seg_start = float(segment.start)
        seg_end = float(segment.end)
        if seg_end < start or seg_start > end:
            continue
        normalized = normalize_arabic(segment.text, strict=False)
        if len(normalized) < 3:
            continue
        relevant.append((max(start, seg_start), min(end, seg_end)))

    if not relevant:
        return True

    relevant.sort(key=lambda item: item[0])
    max_gap = 0.0
    prev = start
    covered = 0.0
    for left, right in relevant:
        if left > prev:
            max_gap = max(max_gap, left - prev)
        covered += max(0.0, right - left)
        prev = max(prev, right)
    if end > prev:
        max_gap = max(max_gap, end - prev)

    density = covered / span
    return max_gap > float(max_silence_seconds) or density < min_density


def _has_weak_local_support(
    transcript_segments: list[TranscriptSegment],
    entry: AyahEntry | None,
    window_start: int,
    window_end: int,
    min_score: int = 62,
    min_overlap: float = 0.08,
) -> bool:
    if entry is None or window_end <= window_start:
        return False
    is_muqattaat = _is_muqattaat_entry(entry)

    top_score = -1.0
    top_overlap = 0.0
    for segment in transcript_segments:
        if segment.end < window_start or segment.start > window_end:
            continue
        normalized = normalize_arabic(segment.text, strict=False)
        if len(normalized) < 3:
            continue
        if is_muqattaat and not _has_muqattaat_phrase_match(normalized, entry):
            continue
        score, overlap = _score_segment_against_entry(normalized, entry)
        if score > top_score:
            top_score = score
            top_overlap = overlap

    return top_score >= float(min_score) and top_overlap >= min_overlap


def _defer_inferred_time_after_fatiha(
    inferred_time: int,
    fatiha_reset_times: list[float],
    hold_seconds: int = 26,
) -> int:
    if not fatiha_reset_times:
        return inferred_time

    adjusted = inferred_time
    for reset_time in fatiha_reset_times:
        reset = int(round(reset_time))
        # If marker lands around restart/Fatiha zone, hold until likely Fatiha completion.
        if reset - 8 <= adjusted <= reset + hold_seconds:
            adjusted = max(adjusted, reset + hold_seconds)
    return adjusted


def load_corpus(corpus_path: Path) -> list[AyahEntry]:
    if not corpus_path.exists():
        return []

    with corpus_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    surahs = payload.get("surahs", [])
    entries: list[AyahEntry] = []

    for surah in surahs:
        surah_number = int(surah.get("number", 0))
        surah_name = str(surah.get("name", "Unknown Surah"))
        for ayah in surah.get("ayahs", []):
            ayah_number = int(ayah.get("number", 0))
            ayah_text = str(ayah.get("text", "")).strip()
            normalized = normalize_arabic(ayah_text, strict=False)
            if not normalized:
                continue

            entries.append(
                AyahEntry(
                    surah_number=surah_number,
                    surah=surah_name,
                    ayah=ayah_number,
                    text=ayah_text,
                    normalized=normalized,
                    match_forms=_build_match_forms(ayah_number, normalized),
                )
            )

    return entries


def _parse_translation_payload(payload: dict) -> tuple[dict[tuple[int, int], str], dict]:
    root = payload.get("data", payload)
    surahs = root.get("surahs", [])
    transformed: dict = {"surahs": []}
    lookup: dict[tuple[int, int], str] = {}

    for surah in surahs:
        surah_number = int(surah.get("number", 0))
        if surah_number <= 0:
            continue

        ayahs_out: list[dict] = []
        for ayah in surah.get("ayahs", []):
            ayah_number = int(ayah.get("numberInSurah", ayah.get("number", 0)))
            text = str(ayah.get("text", "")).strip()
            if ayah_number <= 0 or not text:
                continue

            ayahs_out.append({"number": ayah_number, "text": text})
            lookup[(surah_number, ayah_number)] = text

        transformed["surahs"].append({"number": surah_number, "ayahs": ayahs_out})

    return lookup, transformed


def load_asad_translation(asad_path: Path) -> dict[tuple[int, int], str]:
    if asad_path.exists():
        with asad_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        lookup, _ = _parse_translation_payload(payload)
        if lookup:
            return lookup

    try:
        response = requests.get(ASAD_API_URL, timeout=45)
        response.raise_for_status()
        payload = response.json()
        lookup, transformed = _parse_translation_payload(payload)
        if lookup:
            asad_path.parent.mkdir(parents=True, exist_ok=True)
            with asad_path.open("w", encoding="utf-8") as handle:
                json.dump(transformed, handle, ensure_ascii=False, indent=2)
        return lookup
    except requests.RequestException:
        return {}


def enrich_marker_texts(
    markers: list[Marker],
    corpus_entries: list[AyahEntry],
    asad_lookup: dict[tuple[int, int], str],
) -> list[Marker]:
    if not markers:
        return markers

    arabic_lookup: dict[tuple[int, int], str] = {}
    for entry in corpus_entries:
        arabic_lookup[(entry.surah_number, entry.ayah)] = entry.text

    for marker in markers:
        if marker.surah_number is None:
            continue
        key = (marker.surah_number, marker.ayah)
        marker.arabic_text = arabic_lookup.get(key)
        marker.english_text = asad_lookup.get(key)

    return markers


JUZ_STARTS: list[tuple[int, int, int]] = [
    (1, 1, 1),
    (2, 2, 142),
    (3, 2, 253),
    (4, 3, 93),
    (5, 4, 24),
    (6, 4, 148),
    (7, 5, 82),
    (8, 6, 111),
    (9, 7, 88),
    (10, 8, 41),
    (11, 9, 93),
    (12, 11, 6),
    (13, 12, 53),
    (14, 15, 1),
    (15, 17, 1),
    (16, 18, 75),
    (17, 21, 1),
    (18, 23, 1),
    (19, 25, 21),
    (20, 27, 56),
    (21, 29, 46),
    (22, 33, 31),
    (23, 36, 28),
    (24, 39, 32),
    (25, 41, 47),
    (26, 46, 1),
    (27, 51, 31),
    (28, 58, 1),
    (29, 67, 1),
    (30, 78, 1),
]


def get_juz_for_ayah(surah_number: int, ayah_number: int) -> int:
    for juz, start_surah, start_ayah in reversed(JUZ_STARTS):
        if surah_number > start_surah:
            return juz
        if surah_number == start_surah and ayah_number >= start_ayah:
            return juz
    return 1


def _find_best_ayah_timestamp(
    transcript_segments: list[TranscriptSegment],
    entry: AyahEntry,
    window_start: int,
    window_end: int,
    expected_time: int,
    min_score: int,
    min_overlap: float,
    min_confidence: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
) -> tuple[int, int, str, float] | None:
    if window_end <= window_start:
        return None

    top_segment: TranscriptSegment | None = None
    top_segment_index = -1
    top_evidence: CandidateEvidence | None = None
    top_score = -1.0
    second_score = -1.0
    top_overlap = 0.0
    is_muqattaat = _is_muqattaat_entry(entry)

    for seg_index, segment in enumerate(transcript_segments):
        if segment.end < window_start or segment.start > window_end:
            continue

        normalized_segment = normalize_arabic(segment.text, strict=False)
        if len(normalized_segment) < 10:
            continue
        if is_muqattaat and not _has_muqattaat_phrase_match(normalized_segment, entry):
            continue

        best_candidate: CandidateEvidence | None = None
        score, overlap = _score_segment_against_entry(normalized_segment, entry)
        if _has_anchor_token_hit(entry, normalized_segment):
            best_candidate = CandidateEvidence(
                adjusted_score=score,
                score=score,
                overlap=overlap,
                penalty=0.0,
                source="segment",
                normalized_text=normalized_segment,
                start_time=float(segment.start),
                end_time=float(segment.end),
                word_indices=[],
                segment_start_index=seg_index,
                segment_end_index=seg_index,
            )

        # Cross-segment full-ayah matching: merge nearby transcript chunks so long ayahs
        # can be aligned against the canonical ayah text, not just a short local segment.
        combined_text = normalized_segment
        previous_end = float(segment.end)
        for offset in range(1, 6):
            next_idx = seg_index + offset
            if next_idx >= len(transcript_segments):
                break
            next_segment = transcript_segments[next_idx]
            if float(next_segment.start) - previous_end > 2.6:
                break
            if float(next_segment.start) > window_end:
                break
            next_normalized = normalize_arabic(next_segment.text, strict=False)
            if len(next_normalized) < 2:
                break

            combined_text = f"{combined_text} {next_normalized}".strip()
            if is_muqattaat and not _has_muqattaat_phrase_match(combined_text, entry):
                previous_end = float(next_segment.end)
                continue
            penalty = float(offset) * 0.45
            merged_score, merged_overlap = _score_segment_against_entry(combined_text, entry)
            has_anchor = _has_anchor_token_hit(entry, combined_text)
            adjusted = merged_score - penalty
            if not has_anchor and adjusted < float(max(64, min_score - 8)):
                previous_end = float(next_segment.end)
                continue
            if has_anchor:
                adjusted += 2.0
            if best_candidate is None or adjusted > best_candidate.adjusted_score:
                best_candidate = CandidateEvidence(
                    adjusted_score=adjusted,
                    score=merged_score,
                    overlap=merged_overlap,
                    penalty=penalty,
                    source="segment",
                    normalized_text=combined_text,
                    start_time=float(segment.start),
                    end_time=float(next_segment.end),
                    word_indices=[],
                    segment_start_index=seg_index,
                    segment_end_index=next_idx,
                )
            previous_end = float(next_segment.end)

        segment_windows = list(generate_word_windows(list(getattr(segment, "words", None) or []), min_window=4, max_window=8))
        for window in segment_windows:
            if is_muqattaat and not _has_muqattaat_phrase_match(window.normalized_text, entry):
                continue
            penalty = _window_penalty(len(window.word_indices))
            window_score, window_overlap = _score_segment_against_entry(window.normalized_text, entry)
            has_anchor = _has_anchor_token_hit(entry, window.normalized_text)
            adjusted = window_score - penalty
            if not has_anchor and adjusted < float(max(64, min_score - 8)):
                continue
            if has_anchor:
                adjusted += 2.0
            if best_candidate is None or adjusted > best_candidate.adjusted_score:
                best_candidate = CandidateEvidence(
                    adjusted_score=adjusted,
                    score=window_score,
                    overlap=window_overlap,
                    penalty=penalty,
                    source="window",
                    normalized_text=window.normalized_text,
                    start_time=window.start_time,
                    end_time=window.end_time,
                    word_indices=list(window.word_indices),
                    segment_start_index=seg_index,
                    segment_end_index=seg_index,
                )

        if best_candidate is None:
            continue

        if best_candidate.adjusted_score > top_score:
            second_score = top_score
            top_score = best_candidate.adjusted_score
            top_segment = segment
            top_segment_index = seg_index
            top_overlap = best_candidate.overlap
            top_evidence = best_candidate
        elif best_candidate.adjusted_score > second_score:
            second_score = best_candidate.adjusted_score

    if top_segment is None or top_evidence is None or top_score < ambiguous_min_score:
        return None

    top_time, top_end, _ = _resolve_marker_times(
        segment=top_segment,
        entry=entry,
        evidence=top_evidence,
        transcript_segments=transcript_segments,
        segment_index=top_segment_index,
    )
    margin = max(0.0, top_score - max(0.0, second_score))
    window_span = max(1, window_end - window_start)
    proximity = 1.0 - min(1.0, abs(top_time - expected_time) / window_span)
    confidence = (
        (0.5 * (top_score / 100.0))
        + (0.2 * min(1.0, margin / 20.0))
        + (0.2 * top_overlap)
        + (0.1 * proximity)
    )

    is_high = top_score >= min_score and top_overlap >= min_overlap and confidence >= min_confidence
    is_ambiguous = top_score >= ambiguous_min_score and confidence >= ambiguous_min_confidence
    if not is_high and not is_ambiguous:
        return None

    return top_time, max(top_time, top_end), ("high" if is_high else "ambiguous"), round(confidence, 3)


def _try_wide_reground_before_infer(
    transcript_segments: list[TranscriptSegment],
    entry: AyahEntry,
    window_start: int,
    window_end: int,
    expected_time: int,
    min_score: int,
    min_overlap: float,
    min_confidence: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
) -> tuple[int, int, str, float] | None:
    if window_end <= window_start:
        return None
    return _find_best_ayah_timestamp(
        transcript_segments=transcript_segments,
        entry=entry,
        window_start=window_start,
        window_end=window_end,
        expected_time=expected_time,
        min_score=max(60, min_score - 14),
        min_overlap=max(0.04, min_overlap - 0.12),
        min_confidence=max(0.48, min_confidence - 0.14),
        ambiguous_min_score=max(58, ambiguous_min_score - 12),
        ambiguous_min_confidence=max(0.42, ambiguous_min_confidence - 0.10),
    )


def _quality_rank(quality: str | None) -> int:
    if quality == "manual":
        return 4
    if quality == "high":
        return 3
    if quality == "ambiguous":
        return 2
    if quality == "inferred":
        return 1
    return 0


def _is_anchor_quality(quality: str | None) -> bool:
    return quality in {"high", "ambiguous", "manual"}


def _is_strong_anchor_marker(marker: Marker) -> bool:
    if marker.quality not in {"high", "manual"}:
        return False
    return float(marker.confidence or 0.0) >= 0.70


def _estimate_step_seconds(surah_markers: dict[int, Marker], default_step: int = 8) -> int:
    pairs = sorted(surah_markers.items(), key=lambda item: item[0])
    deltas: list[float] = []
    for (left_ayah, left_marker), (right_ayah, right_marker) in zip(pairs, pairs[1:]):
        ayah_gap = right_ayah - left_ayah
        time_gap = right_marker.time - left_marker.time
        if ayah_gap <= 0 or time_gap <= 0:
            continue
        deltas.append(time_gap / ayah_gap)

    if not deltas:
        return default_step

    deltas.sort()
    median = deltas[len(deltas) // 2]
    return int(max(4, min(20, round(median))))


def _estimated_ayah_duration_seconds(entry: AyahEntry | None) -> int:
    if entry is None:
        return 8
    token_count = len([token for token in entry.normalized.split() if token])
    # Rough pacing proxy from ayah length. Used as a floor for next-ayah onset.
    return max(6, min(95, int(round(token_count * 0.48))))


def _fill_surah_coverage_markers(
    existing_markers: list[Marker],
    entry_lookup: dict[tuple[str, int], AyahEntry],
    transcript_segments: list[TranscriptSegment] | None = None,
    fatiha_reset_times: list[float] | None = None,
    weak_support_score: int = 62,
    weak_support_overlap: float = 0.08,
    enforce_weak_support: bool = True,
    max_bridge_gap_ayahs: int = 60,
    max_bridge_gap_seconds: int = 2400,
    max_one_sided_extrapolation_ayahs: int = 5,
    min_bridge_step_seconds: float = 4.0,
    max_bridge_step_seconds: float = 28.0,
) -> list[Marker]:
    surah_map: dict[str, dict[int, Marker]] = {}
    for marker in sorted(existing_markers, key=lambda m: m.time):
        ayah_map = surah_map.setdefault(marker.surah, {})
        existing = ayah_map.get(marker.ayah)
        if existing is None:
            ayah_map[marker.ayah] = marker
            continue
        if _quality_rank(marker.quality) > _quality_rank(existing.quality):
            ayah_map[marker.ayah] = marker
        elif _quality_rank(marker.quality) == _quality_rank(existing.quality) and marker.time < existing.time:
            ayah_map[marker.ayah] = marker

    coverage_inferred: list[Marker] = []

    for surah, ayah_map in surah_map.items():
        if not ayah_map:
            continue

        known_ayahs = sorted(ayah_map.keys())
        min_ayah = known_ayahs[0]
        max_ayah = known_ayahs[-1]
        step_seconds = _estimate_step_seconds(ayah_map)

        for ayah in range(min_ayah, max_ayah + 1):
            if ayah in ayah_map:
                continue

            left_ayahs = [a for a in known_ayahs if a < ayah]
            right_ayahs = [a for a in known_ayahs if a > ayah]
            left_ayah = left_ayahs[-1] if left_ayahs else None
            right_ayah = right_ayahs[0] if right_ayahs else None
            left_anchor_ayahs = [a for a in left_ayahs if _is_anchor_quality(ayah_map[a].quality)]
            right_anchor_ayahs = [a for a in right_ayahs if _is_anchor_quality(ayah_map[a].quality)]
            left_anchor_ayah = left_anchor_ayahs[-1] if left_anchor_ayahs else None
            right_anchor_ayah = right_anchor_ayahs[0] if right_anchor_ayahs else None

            inferred_time: int
            left_marker: Marker | None = None
            right_marker: Marker | None = None
            if left_anchor_ayah is not None and right_anchor_ayah is not None:
                left_ayah = left_anchor_ayah
                right_ayah = right_anchor_ayah
                left_marker = ayah_map[left_ayah]
                right_marker = ayah_map[right_ayah]
                span_ayahs = right_ayah - left_ayah
                span_secs = right_marker.time - left_marker.time
                if span_ayahs <= 0:
                    continue
                if span_ayahs > max_bridge_gap_ayahs:
                    continue
                if span_secs <= 0 or span_secs > max_bridge_gap_seconds:
                    continue
                if fatiha_reset_times and _contains_fatiha_reset_between(fatiha_reset_times, left_marker.time, right_marker.time):
                    continue
                bridge_step = span_secs / float(span_ayahs)
                if bridge_step < float(min_bridge_step_seconds):
                    continue
                if bridge_step > float(max_bridge_step_seconds):
                    continue
                if not (_is_anchor_quality(left_marker.quality) and _is_anchor_quality(right_marker.quality)):
                    continue
                ratio = (ayah - left_ayah) / max(1, span_ayahs)
                if span_secs > 0:
                    inferred_time = int(round(left_marker.time + (span_secs * ratio)))
                else:
                    inferred_time = left_marker.time + ((ayah - left_ayah) * step_seconds)
            elif left_anchor_ayah is not None:
                left_ayah = left_anchor_ayah
                left_marker = ayah_map[left_ayah]
                if not _is_anchor_quality(left_marker.quality):
                    continue
                if (ayah - left_ayah) > max_one_sided_extrapolation_ayahs:
                    continue
                inferred_time = left_marker.time + ((ayah - left_ayah) * step_seconds)
                if fatiha_reset_times and _contains_fatiha_reset_between(fatiha_reset_times, left_marker.time, inferred_time):
                    continue
            elif right_anchor_ayah is not None:
                right_ayah = right_anchor_ayah
                right_marker = ayah_map[right_ayah]
                if not _is_anchor_quality(right_marker.quality):
                    continue
                if (right_ayah - ayah) > max_one_sided_extrapolation_ayahs:
                    continue
                inferred_time = max(0, right_marker.time - ((right_ayah - ayah) * step_seconds))
                if fatiha_reset_times and _contains_fatiha_reset_between(fatiha_reset_times, inferred_time, right_marker.time):
                    continue
            else:
                continue

            if fatiha_reset_times:
                inferred_time = _defer_inferred_time_after_fatiha(
                    inferred_time=inferred_time,
                    fatiha_reset_times=fatiha_reset_times,
                )
                if left_marker is not None:
                    inferred_time = max(left_marker.time + 1, inferred_time)
                if right_marker is not None:
                    inferred_time = min(right_marker.time - 1, inferred_time)

            entry = entry_lookup.get((surah, ayah))
            left_reciter = ayah_map[left_ayah].reciter if left_ayah is not None else None
            right_reciter = ayah_map[right_ayah].reciter if right_ayah is not None else None
            reciter = left_reciter or right_reciter
            surah_number = entry.surah_number if entry else ayah_map[known_ayahs[0]].surah_number
            if enforce_weak_support and transcript_segments is not None:
                window_start = max(0, inferred_time - 16)
                window_end = inferred_time + 16
                if not _has_weak_local_support(
                    transcript_segments=transcript_segments,
                    entry=entry,
                    window_start=window_start,
                    window_end=window_end,
                    min_score=weak_support_score,
                    min_overlap=weak_support_overlap,
                ):
                    continue

            coverage_inferred.append(
                Marker(
                    time=max(0, inferred_time),
                    start_time=max(0, inferred_time),
                    end_time=max(0, inferred_time),
                    surah=surah,
                    surah_number=surah_number,
                    ayah=ayah,
                    juz=get_juz_for_ayah(surah_number or 1, ayah),
                    quality="inferred",
                    reciter=reciter,
                    confidence=0.56,
                )
            )

    return coverage_inferred


def _dedupe_by_local_time_window(markers: list[Marker], window_seconds: int = 90) -> list[Marker]:
    if not markers:
        return []

    sorted_markers = sorted(markers, key=lambda marker: marker.time)
    deduped: list[Marker] = []

    for marker in sorted_markers:
        merged = False
        for index in range(len(deduped) - 1, -1, -1):
            candidate = deduped[index]
            if marker.time - candidate.time > window_seconds:
                break
            if marker.surah == candidate.surah and marker.ayah == candidate.ayah:
                if _quality_rank(marker.quality) > _quality_rank(candidate.quality):
                    deduped[index] = marker
                elif _quality_rank(marker.quality) == _quality_rank(candidate.quality) and marker.time < candidate.time:
                    deduped[index] = marker
                merged = True
                break
        if not merged:
            deduped.append(marker)

    return deduped


def _apply_overlap_conflict_resolution(markers: list[Marker]) -> list[Marker]:
    if len(markers) < 2:
        return markers

    def confidence(marker: Marker) -> float:
        return float(marker.confidence or 0.0)

    def is_strong(marker: Marker) -> bool:
        return marker.quality in {"high", "manual"} or confidence(marker) >= 0.72

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))
    resolved: list[Marker] = []
    for idx, marker in enumerate(ordered):
        if not resolved:
            resolved.append(marker)
            continue

        previous = resolved[-1]
        same_surah = previous.surah == marker.surah
        forward_next = marker.ayah == previous.ayah + 1
        overlap = (marker.start_time or marker.time) < (previous.end_time or previous.time)

        if same_surah and forward_next and overlap:
            prev_conf = confidence(previous)
            curr_conf = confidence(marker)
            # Keep current ayah active unless the next one has clearly stronger evidence.
            if not is_strong(marker) and curr_conf <= prev_conf + 0.02:
                delayed_start = (previous.end_time or previous.time) + 1
                next_start: int | None = None
                if idx + 1 < len(ordered):
                    next_marker = ordered[idx + 1]
                    if next_marker.surah == marker.surah and next_marker.ayah >= marker.ayah:
                        next_start = next_marker.start_time or next_marker.time

                # If there is space, delay instead of dropping.
                if next_start is None or delayed_start < next_start:
                    marker.start_time = delayed_start
                    marker.time = delayed_start
                    marker.end_time = max(marker.end_time or delayed_start, delayed_start)
                else:
                    # Only drop when confidence is truly weak; otherwise keep as a point marker.
                    if curr_conf < 0.45 and marker.quality == "inferred":
                        continue
                    marker.start_time = max(delayed_start, next_start)
                    marker.time = marker.start_time
                    marker.end_time = marker.start_time
            if (marker.start_time or marker.time) < (previous.end_time or previous.time):
                previous.end_time = max(previous.start_time or previous.time, (marker.start_time or marker.time) - 1)

        resolved.append(marker)

    return resolved


def _refine_weak_boundary_markers(
    markers: list[Marker],
    transcript_segments: list[TranscriptSegment],
    entry_lookup: dict[tuple[str, int], AyahEntry],
    fatiha_reset_times: list[float] | None,
    min_score: int,
    min_overlap: float,
    min_confidence: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
) -> list[Marker]:
    if len(markers) < 3:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))

    for idx in range(1, len(ordered) - 1):
        marker = ordered[idx]
        previous = ordered[idx - 1]
        next_marker = ordered[idx + 1]

        if marker.surah != previous.surah or marker.surah != next_marker.surah:
            continue
        if marker.ayah != previous.ayah + 1:
            continue
        if next_marker.ayah <= marker.ayah:
            continue
        if marker.quality not in {"ambiguous", "inferred"}:
            continue

        marker_conf = float(marker.confidence or 0.0)
        if marker_conf >= 0.66 and marker.quality == "ambiguous":
            continue

        prev_end = int(previous.end_time or previous.time)
        marker_start = int(marker.start_time or marker.time)
        # Focus on likely boundary slips where next ayah was placed immediately after previous ayah ends.
        if marker_start - prev_end > 3:
            continue

        gap_to_next = int(next_marker.time) - marker_start
        if gap_to_next < 25:
            continue

        entry = entry_lookup.get((marker.surah, marker.ayah))
        if entry is None:
            continue

        window_start = max(prev_end + 6, marker_start + 8)
        window_end = min(int(next_marker.time) - 6, marker_start + 220)
        if window_end <= window_start:
            continue

        best = _find_best_ayah_timestamp(
            transcript_segments=transcript_segments,
            entry=entry,
            window_start=window_start,
            window_end=window_end,
            expected_time=max(window_start, marker_start + 20),
            min_score=min_score,
            min_overlap=min_overlap,
            min_confidence=min_confidence,
            ambiguous_min_score=ambiguous_min_score,
            ambiguous_min_confidence=ambiguous_min_confidence,
        )
        if best is None:
            continue

        matched_time, matched_end, matched_quality, matched_confidence = best
        if matched_time <= marker_start + 5:
            continue
        if matched_confidence < max(marker_conf + 0.08, 0.60):
            continue

        bounded_time = max(window_start, min(window_end, matched_time))
        bounded_end = max(bounded_time, min(window_end, matched_end))
        if fatiha_reset_times:
            bounded_time = _defer_inferred_time_after_fatiha(
                inferred_time=bounded_time,
                fatiha_reset_times=fatiha_reset_times,
            )
            bounded_time = max(window_start, min(window_end, bounded_time))
            bounded_end = max(bounded_time, min(window_end, bounded_end))
        marker.time = bounded_time
        marker.start_time = bounded_time
        marker.end_time = bounded_end
        marker.quality = matched_quality
        marker.confidence = round(matched_confidence, 3)

    return ordered


def _refine_inferred_markers_with_local_search(
    markers: list[Marker],
    transcript_segments: list[TranscriptSegment],
    entry_lookup: dict[tuple[str, int], AyahEntry],
    fatiha_reset_times: list[float] | None,
    min_score: int,
    min_overlap: float,
    min_confidence: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
) -> list[Marker]:
    if len(markers) < 3:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))

    for idx, marker in enumerate(ordered):
        if marker.quality != "inferred":
            continue

        entry = entry_lookup.get((marker.surah, marker.ayah))
        if entry is None:
            continue

        prev_marker: Marker | None = None
        next_marker: Marker | None = None
        for left in range(idx - 1, -1, -1):
            if ordered[left].surah == marker.surah:
                prev_marker = ordered[left]
                break
        for right in range(idx + 1, len(ordered)):
            if ordered[right].surah == marker.surah:
                next_marker = ordered[right]
                break

        current_time = int(marker.start_time or marker.time)
        if prev_marker is not None:
            window_start = int(prev_marker.end_time or prev_marker.time) + 2
        else:
            window_start = max(0, current_time - 90)
        if next_marker is not None:
            window_end = int(next_marker.start_time or next_marker.time) - 2
        else:
            window_end = current_time + 90
        if window_end <= window_start:
            continue

        # Keep search local, but wide enough to recover if interpolation was poor.
        span_limit_start = max(0, current_time - 120)
        span_limit_end = current_time + 120
        window_start = max(window_start, span_limit_start)
        window_end = min(window_end, span_limit_end)
        if window_end <= window_start:
            continue

        best = _find_best_ayah_timestamp(
            transcript_segments=transcript_segments,
            entry=entry,
            window_start=window_start,
            window_end=window_end,
            expected_time=current_time,
            min_score=max(74, min_score - 6),
            min_overlap=max(0.06, min_overlap - 0.08),
            min_confidence=max(0.56, min_confidence - 0.08),
            ambiguous_min_score=max(66, ambiguous_min_score - 4),
            ambiguous_min_confidence=max(0.48, ambiguous_min_confidence - 0.04),
        )
        if best is None:
            continue

        matched_time, matched_end, matched_quality, matched_confidence = best
        if matched_confidence < 0.56:
            continue

        bounded_time = max(window_start, min(window_end, matched_time))
        bounded_end = max(bounded_time, min(window_end, matched_end))
        if fatiha_reset_times:
            bounded_time = _defer_inferred_time_after_fatiha(
                inferred_time=bounded_time,
                fatiha_reset_times=fatiha_reset_times,
            )
            bounded_time = max(window_start, min(window_end, bounded_time))
            bounded_end = max(bounded_time, min(window_end, bounded_end))
        if abs(bounded_time - current_time) < 2 and matched_quality == "ambiguous":
            marker.confidence = max(float(marker.confidence or 0.56), round(matched_confidence, 3))
            continue

        marker.time = bounded_time
        marker.start_time = bounded_time
        marker.end_time = max(bounded_end, marker.end_time or bounded_end)
        marker.quality = matched_quality
        marker.confidence = round(matched_confidence, 3)

    return ordered


def _redistribute_dense_weak_runs(markers: list[Marker]) -> list[Marker]:
    if len(markers) < 4:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))

    def is_weak(marker: Marker) -> bool:
        quality = marker.quality or ""
        confidence = float(marker.confidence or 0.0)
        if quality == "inferred":
            return True
        return quality == "ambiguous" and confidence <= 0.62

    idx = 1
    while idx < len(ordered) - 1:
        marker = ordered[idx]
        if not is_weak(marker):
            idx += 1
            continue

        run_start = idx
        run_end = idx
        while run_end + 1 < len(ordered):
            current = ordered[run_end]
            nxt = ordered[run_end + 1]
            current_time = int(current.start_time or current.time)
            next_time = int(nxt.start_time or nxt.time)
            if (
                nxt.surah == current.surah
                and nxt.ayah == current.ayah + 1
                and is_weak(nxt)
                and (next_time - current_time) <= 25
            ):
                run_end += 1
            else:
                break

        run = ordered[run_start : run_end + 1]
        if len(run) < 4:
            idx = run_end + 1
            continue

        run_span = int((run[-1].start_time or run[-1].time) - (run[0].start_time or run[0].time))
        if run_span > max(8, len(run)):
            idx = run_end + 1
            continue

        left = ordered[run_start - 1]
        right: Marker | None = None
        for right_idx in range(run_end + 1, len(ordered)):
            candidate = ordered[right_idx]
            if candidate.surah == run[0].surah and candidate.ayah > run[-1].ayah:
                right = candidate
                break
        if right is None:
            idx = run_end + 1
            continue

        left_bound = int(left.end_time or left.time) + 1
        right_bound = int(right.start_time or right.time) - 1
        available = right_bound - left_bound
        if available < len(run) * 3:
            idx = run_end + 1
            continue

        step = available / float(len(run) + 1)
        for offset, weak_marker in enumerate(run, start=1):
            target = int(round(left_bound + (step * offset)))
            if target <= left_bound:
                target = left_bound + offset
            weak_marker.time = target
            weak_marker.start_time = target
            weak_marker.end_time = target
            if weak_marker.quality == "inferred":
                weak_marker.confidence = max(float(weak_marker.confidence or 0.56), 0.58)

        idx = run_end + 1

    return ordered


def _extend_point_markers_to_next(markers: list[Marker], max_extension_seconds: int = 90) -> list[Marker]:
    if len(markers) < 2:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))
    for idx, marker in enumerate(ordered[:-1]):
        start = int(marker.start_time or marker.time)
        end = int(marker.end_time or marker.time)
        if end > start:
            continue
        if marker.quality not in {"inferred", "ambiguous"}:
            continue

        next_marker: Marker | None = None
        for look_ahead in range(idx + 1, len(ordered)):
            candidate = ordered[look_ahead]
            if candidate.surah != marker.surah:
                continue
            if int(candidate.ayah) <= int(marker.ayah):
                continue
            next_marker = candidate
            break
        if next_marker is None:
            continue

        next_start = int(next_marker.start_time or next_marker.time)
        if next_start <= start:
            continue

        extension = min(max_extension_seconds, max(0, next_start - start - 1))
        if extension <= 0:
            continue
        marker.end_time = start + extension

    return ordered


def _stabilize_weak_marker_durations(markers: list[Marker]) -> list[Marker]:
    if len(markers) < 2:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))

    def nearest_step_seconds(index: int, fallback: float = 14.0) -> float:
        center = ordered[index]
        # Use nearby same-surah anchors to estimate ayah pace.
        best: float | None = None
        for left_idx in range(index - 1, -1, -1):
            left = ordered[left_idx]
            if left.surah != center.surah:
                continue
            ayah_gap = center.ayah - left.ayah
            time_gap = (center.start_time or center.time) - (left.start_time or left.time)
            if ayah_gap > 0 and time_gap > 0:
                best = time_gap / ayah_gap
                break
        for right_idx in range(index + 1, len(ordered)):
            right = ordered[right_idx]
            if right.surah != center.surah:
                continue
            ayah_gap = right.ayah - center.ayah
            time_gap = (right.start_time or right.time) - (center.start_time or center.time)
            if ayah_gap > 0 and time_gap > 0:
                right_step = time_gap / ayah_gap
                if best is None:
                    best = right_step
                else:
                    best = (best + right_step) / 2.0
                break
        if best is None:
            return fallback
        return max(6.0, min(26.0, best))

    for idx, marker in enumerate(ordered[:-1]):
        if marker.quality not in {"inferred", "ambiguous"}:
            continue
        start = int(marker.start_time or marker.time)
        next_marker: Marker | None = None
        for look_ahead in range(idx + 1, len(ordered)):
            candidate = ordered[look_ahead]
            if candidate.surah != marker.surah:
                continue
            if int(candidate.ayah) <= int(marker.ayah):
                continue
            next_marker = candidate
            break
        if next_marker is None:
            continue

        next_start = int(next_marker.start_time or next_marker.time)
        available = next_start - start - 1
        if available <= 0:
            continue

        step = nearest_step_seconds(idx)
        hold_ratio = 0.78 if marker.quality == "inferred" else 0.62
        desired = int(round(step * hold_ratio))
        if marker.quality == "inferred":
            desired = max(12, desired)
        else:
            desired = max(8, desired)
        desired = min(desired, 40, available)

        current_end = int(marker.end_time or start)
        target_end = max(current_end, start + desired)
        target_end = min(target_end, next_start - 1)
        marker.end_time = max(start, target_end)

        # Prevent weak->weak markers from being too tightly packed when there is room to spread.
        if next_marker.quality in {"inferred", "ambiguous"}:
            min_gap = 14 if marker.quality == "inferred" else 10
            current_gap = next_start - start
            if current_gap < min_gap:
                ceiling = next_start
                for look_ahead in range(idx + 2, len(ordered)):
                    candidate = ordered[look_ahead]
                    if candidate.surah != marker.surah:
                        continue
                    if int(candidate.ayah) <= int(next_marker.ayah):
                        continue
                    ceiling = int(candidate.start_time or candidate.time) - 1
                    break
                target_next_start = min(ceiling, start + min_gap)
                if target_next_start > next_start:
                    next_marker.time = target_next_start
                    next_marker.start_time = target_next_start
                    next_marker.end_time = max(int(next_marker.end_time or target_next_start), target_next_start)

    return ordered


def _prune_unrealistic_progression(markers: list[Marker]) -> list[Marker]:
    if len(markers) < 3:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))
    kept: list[Marker] = []
    last_index_by_surah: dict[str, int] = {}

    def rank(marker: Marker) -> int:
        return _quality_rank(marker.quality)

    for marker in ordered:
        surah = marker.surah
        last_idx = last_index_by_surah.get(surah)
        if last_idx is None:
            kept.append(marker)
            last_index_by_surah[surah] = len(kept) - 1
            continue

        previous = kept[last_idx]
        prev_time = int(previous.start_time or previous.time)
        curr_time = int(marker.start_time or marker.time)
        dt = curr_time - prev_time
        da = int(marker.ayah) - int(previous.ayah)

        if da <= 0:
            # Do not allow backward/same ayah repeats in the same-surah timeline here.
            # Repeat handling should already have extended end_time of existing ayah.
            continue

        # If two far-apart ayahs land on the same second, keep only the stronger one.
        if dt <= 1 and da > 1:
            if rank(marker) > rank(previous):
                kept[last_idx] = marker
            continue

        # Pace guard: prevent unrealistic surah leaps over very short time.
        # Allow roughly one ayah every ~3 seconds with a small buffer.
        allowed_jump = max(3, int(max(0, dt) / 3) + 2)
        if da > allowed_jump and rank(marker) < 4:
            continue

        kept.append(marker)
        last_index_by_surah[surah] = len(kept) - 1

    return kept


def _enforce_surah_transition_order(
    markers: list[Marker],
    surah_totals: dict[str, int],
    min_gap_seconds: int = 6,
) -> list[Marker]:
    if len(markers) < 3:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))
    by_surah_number: dict[int, list[Marker]] = {}
    for marker in ordered:
        if marker.surah_number is None:
            continue
        by_surah_number.setdefault(int(marker.surah_number), []).append(marker)

    surah_numbers = sorted(by_surah_number.keys())
    for number in surah_numbers:
        previous = by_surah_number.get(number)
        upcoming = by_surah_number.get(number + 1)
        if not previous or not upcoming:
            continue

        previous_surah_name = previous[0].surah
        previous_total = surah_totals.get(previous_surah_name)
        if not previous_total:
            continue

        previous_final = next((item for item in previous if int(item.ayah) == int(previous_total)), None)
        if previous_final is None:
            continue

        previous_final_time = int(previous_final.start_time or previous_final.time)
        boundary_floor = previous_final_time + max(1, int(min_gap_seconds))

        upcoming_sorted = sorted(upcoming, key=lambda item: (item.time, item.ayah))
        first_upcoming_time = int(upcoming_sorted[0].start_time or upcoming_sorted[0].time)
        if first_upcoming_time >= boundary_floor:
            continue

        # Shift only early ayat near the transition so we preserve downstream timing.
        for marker in upcoming_sorted:
            marker_time = int(marker.start_time or marker.time)
            if marker_time >= boundary_floor:
                continue
            if int(marker.ayah) > 6:
                continue
            adjusted = boundary_floor + max(0, int(marker.ayah) - 1)
            marker.time = adjusted
            marker.start_time = adjusted
            marker.end_time = max(adjusted, int(marker.end_time or adjusted))

    return ordered


def _enforce_long_ayah_inferred_floor(
    markers: list[Marker],
    entry_lookup: dict[tuple[str, int], AyahEntry],
) -> list[Marker]:
    if len(markers) < 3:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))
    entry_by_number: dict[tuple[int, int], AyahEntry] = {}
    for entry in entry_lookup.values():
        entry_by_number[(int(entry.surah_number), int(entry.ayah))] = entry
    by_surah: dict[str, list[Marker]] = {}
    for marker in ordered:
        by_surah.setdefault(marker.surah, []).append(marker)

    for surah_markers in by_surah.values():
        surah_markers.sort(key=lambda item: (int(item.ayah), int(item.time)))
        for prev, curr in zip(surah_markers, surah_markers[1:]):
            if int(curr.ayah) != int(prev.ayah) + 1:
                continue
            if curr.quality != "inferred":
                continue
            if prev.quality not in {"high", "ambiguous", "manual"}:
                continue

            if prev.surah_number is None:
                continue
            prev_entry = entry_by_number.get((int(prev.surah_number), int(prev.ayah)))
            if prev_entry is None:
                continue

            token_count = len([token for token in prev_entry.normalized.split() if token])
            if token_count < 45:
                continue

            # For long ayat, require meaningful recitation span before allowing an inferred next ayah.
            min_hold = max(18, min(180, int(round(token_count * 0.80))))
            prev_start = int(prev.start_time or prev.time)
            required_start = prev_start + min_hold
            curr_start = int(curr.start_time or curr.time)
            if curr_start >= required_start:
                continue

            shift_to = required_start
            curr.time = shift_to
            curr.start_time = shift_to
            curr.end_time = max(shift_to, int(curr.end_time or shift_to))

    return sorted(ordered, key=lambda item: (item.time, item.surah_number or 0, item.ayah))


def _enforce_sequential_ayah_order(markers: list[Marker]) -> list[Marker]:
    if len(markers) < 3:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))
    by_surah: dict[str, list[Marker]] = {}
    for marker in ordered:
        by_surah.setdefault(marker.surah, []).append(marker)

    for surah_markers in by_surah.values():
        surah_markers.sort(key=lambda item: (int(item.ayah), int(item.time)))
        prev_time: int | None = None
        prev_ayah: int | None = None
        for marker in surah_markers:
            ayah = int(marker.ayah)
            current = int(marker.start_time or marker.time)
            if prev_time is None or prev_ayah is None:
                prev_time = current
                prev_ayah = ayah
                continue
            if ayah <= prev_ayah:
                continue
            min_gap = 1 if marker.quality in {"high", "manual"} else 2
            required = prev_time + min_gap
            if current < required:
                marker.time = required
                marker.start_time = required
                marker.end_time = max(required, int(marker.end_time or required))
                current = required
            prev_time = current
            prev_ayah = ayah

    return sorted(ordered, key=lambda item: (item.time, item.surah_number or 0, item.ayah))


def _quran_first_refine_weak_markers(
    markers: list[Marker],
    transcript_segments: list[TranscriptSegment],
    entry_lookup: dict[tuple[str, int], AyahEntry],
    min_score: int,
    min_overlap: float,
    min_confidence: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
) -> list[Marker]:
    if len(markers) < 3:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))

    def is_weak(marker: Marker) -> bool:
        return marker.quality in {"inferred", "ambiguous"}

    def is_anchor(marker: Marker) -> bool:
        return marker.quality in {"high", "manual"}

    for idx, marker in enumerate(ordered):
        if not is_weak(marker):
            continue

        entry = entry_lookup.get((marker.surah, marker.ayah))
        if entry is None:
            continue

        prev_any: Marker | None = None
        next_any: Marker | None = None
        prev_anchor: Marker | None = None
        next_anchor: Marker | None = None

        for left in range(idx - 1, -1, -1):
            candidate = ordered[left]
            if candidate.surah != marker.surah:
                continue
            if int(candidate.ayah) >= int(marker.ayah):
                continue
            prev_any = candidate
            if is_anchor(candidate):
                prev_anchor = candidate
                break
        for right in range(idx + 1, len(ordered)):
            candidate = ordered[right]
            if candidate.surah != marker.surah:
                continue
            if int(candidate.ayah) <= int(marker.ayah):
                continue
            if next_any is None:
                next_any = candidate
            if is_anchor(candidate):
                next_anchor = candidate
                break

        left_bound = int(prev_any.end_time or prev_any.time) + 1 if prev_any is not None else 0
        right_bound = int(next_any.start_time or next_any.time) - 1 if next_any is not None else int(marker.start_time or marker.time) + 180

        if prev_anchor is not None:
            left_bound = max(left_bound, int(prev_anchor.end_time or prev_anchor.time) + 1)
        if next_anchor is not None:
            right_bound = min(right_bound, int(next_anchor.start_time or next_anchor.time) - 1)

        if right_bound <= left_bound:
            continue

        current_start = int(marker.start_time or marker.time)
        expected = max(left_bound, min(right_bound, current_start))

        best = _find_best_ayah_timestamp(
            transcript_segments=transcript_segments,
            entry=entry,
            window_start=left_bound,
            window_end=right_bound,
            expected_time=expected,
            min_score=max(72, min_score - 8),
            min_overlap=max(0.05, min_overlap - 0.10),
            min_confidence=max(0.54, min_confidence - 0.10),
            ambiguous_min_score=max(64, ambiguous_min_score - 6),
            ambiguous_min_confidence=max(0.46, ambiguous_min_confidence - 0.06),
        )
        if best is None:
            continue

        matched_start, matched_end, matched_quality, matched_confidence = best
        bounded_start = max(left_bound, min(right_bound, matched_start))
        bounded_end = max(bounded_start, min(right_bound, matched_end))

        # Respect existing strong neighbors and avoid overlap with adjacent ayahs.
        if prev_any is not None and bounded_start <= int(prev_any.end_time or prev_any.time):
            continue
        if next_any is not None and bounded_end >= int(next_any.start_time or next_any.time):
            bounded_end = max(bounded_start, int(next_any.start_time or next_any.time) - 1)
            if bounded_end < bounded_start:
                continue

        marker.time = bounded_start
        marker.start_time = bounded_start
        marker.end_time = bounded_end
        marker.quality = matched_quality
        marker.confidence = round(max(float(marker.confidence or 0.0), matched_confidence), 3)

    return ordered


def _delay_weak_markers_after_resets(
    markers: list[Marker],
    fatiha_reset_times: list[float],
    transcript_segments: list[TranscriptSegment] | None = None,
    hold_seconds: int = 34,
) -> list[Marker]:
    if len(markers) < 2 or not fatiha_reset_times:
        return markers

    ordered = sorted(markers, key=lambda item: (item.time, item.surah_number or 0, item.ayah))
    reset_points = sorted(int(round(item)) for item in fatiha_reset_times)

    def is_weak(marker: Marker) -> bool:
        return marker.quality in {"inferred", "ambiguous"}

    def has_local_speech(second: int, window: int = 18) -> bool:
        if transcript_segments is None:
            return True
        lo = float(max(0, second - window))
        hi = float(second + window)
        for segment in transcript_segments:
            seg_start = float(segment.start)
            seg_end = float(segment.end)
            if seg_end < lo or seg_start > hi:
                continue
            normalized = normalize_arabic(segment.text, strict=False)
            if len(normalized) >= 6:
                return True
        return False

    for idx, marker in enumerate(ordered):
        if not is_weak(marker):
            continue

        marker_start = int(marker.start_time or marker.time)
        recent_reset: int | None = None
        next_reset: int | None = None
        for reset in reset_points:
            if reset <= marker_start:
                recent_reset = reset
            else:
                next_reset = reset
                break
        # If weak marker sits in a low-speech zone right before an upcoming reset,
        # prefer deferring to post-reset even when there was an earlier reset.
        if next_reset is not None and (next_reset - marker_start) <= 120 and not has_local_speech(marker_start):
            min_start = next_reset + hold_seconds
        elif recent_reset is None:
            continue
        else:
            min_start = recent_reset + hold_seconds
            if marker_start >= min_start:
                continue

        next_same_surah_start: int | None = None
        for look_ahead in range(idx + 1, len(ordered)):
            candidate = ordered[look_ahead]
            if candidate.surah != marker.surah:
                continue
            if int(candidate.ayah) <= int(marker.ayah):
                continue
            next_same_surah_start = int(candidate.start_time or candidate.time)
            break

        target = min_start
        if next_same_surah_start is not None:
            target = min(target, next_same_surah_start - 1)
        if target <= marker_start:
            continue

        marker.time = target
        marker.start_time = target
        marker.end_time = max(int(marker.end_time or target), target)

    return ordered


def _build_transition_tail_markers(
    previous: Marker,
    next_entry: AyahEntry,
    transition_time: int,
    surah_totals: dict[str, int],
    entry_lookup: dict[tuple[str, int], AyahEntry],
    transcript_segments: list[TranscriptSegment],
    fatiha_reset_times: list[float],
    min_gap_seconds: int,
    min_score: int,
    min_overlap: float,
    min_confidence: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
    require_weak_support_for_inferred: bool,
) -> list[Marker]:
    if previous.surah == next_entry.surah:
        return []
    if previous.surah_number is None or next_entry.surah_number != (previous.surah_number + 1):
        return []

    surah_total = surah_totals.get(previous.surah, previous.ayah)
    if surah_total <= previous.ayah:
        return []

    # Focus on near-tail transitions only. If the gap is too large, this is likely a genuine mismatch.
    tail_missing = surah_total - previous.ayah
    if tail_missing <= 0 or tail_missing > 12:
        return []

    window_start = int(previous.end_time or previous.time) + max(4, min_gap_seconds)
    window_end = int(transition_time) - max(4, min_gap_seconds)
    if window_end <= window_start:
        return []

    missing_ayahs = list(range(previous.ayah + 1, surah_total + 1))
    span = max(1, window_end - window_start)
    step = span / float(len(missing_ayahs) + 1)
    relaxed_min_score = max(66, min_score - 10)
    relaxed_min_overlap = max(0.06, min_overlap - 0.09)
    relaxed_min_conf = max(0.54, min_confidence - 0.10)
    relaxed_ambig_score = max(62, ambiguous_min_score - 10)
    relaxed_ambig_conf = max(0.45, ambiguous_min_confidence - 0.08)

    inferred: list[Marker] = []
    floor = window_start
    def _next_ayah_delay_seconds(entry: AyahEntry | None) -> int:
        if entry is None:
            return 8
        token_count = len([token for token in entry.normalized.split() if token])
        # Long ayat should reserve more time before searching for the next ayah start.
        # 2:282 and similar passages otherwise cause premature ayah shifts.
        return max(6, min(70, int(round(token_count * 0.45))))
    for offset, ayah in enumerate(missing_ayahs, start=1):
        entry = entry_lookup.get((previous.surah, ayah))
        if entry is None:
            continue

        expected = int(round(window_start + (step * offset)))
        window_half = max(10, int(round(step * 0.9)))
        local_start = max(floor, expected - window_half)
        local_end = min(window_end, expected + window_half)
        if local_end <= local_start:
            continue

        best = _find_best_ayah_timestamp(
            transcript_segments=transcript_segments,
            entry=entry,
            window_start=local_start,
            window_end=local_end,
            expected_time=expected,
            min_score=relaxed_min_score,
            min_overlap=relaxed_min_overlap,
            min_confidence=relaxed_min_conf,
            ambiguous_min_score=relaxed_ambig_score,
            ambiguous_min_confidence=relaxed_ambig_conf,
        )

        if best is None:
            # Always extend search forward across the remaining transition span before inferring.
            # This avoids false "missing" when the current ayah is long and the next ayah starts later.
            forward_start = max(floor, local_start + max(2, int(round(step * 0.35))))
            forward_end = window_end
            if forward_end > forward_start:
                best = _find_best_ayah_timestamp(
                    transcript_segments=transcript_segments,
                    entry=entry,
                    window_start=forward_start,
                    window_end=forward_end,
                    expected_time=max(forward_start, expected),
                    min_score=max(64, relaxed_min_score - 2),
                    min_overlap=max(0.05, relaxed_min_overlap - 0.02),
                    min_confidence=max(0.52, relaxed_min_conf - 0.02),
                    ambiguous_min_score=max(60, relaxed_ambig_score - 2),
                    ambiguous_min_confidence=max(0.44, relaxed_ambig_conf - 0.02),
                )

        if best is not None:
            matched_start, matched_end, quality, confidence = best
            marker_start = max(floor, min(window_end, int(matched_start)))
            marker_end = max(marker_start, min(window_end, int(matched_end)))
            marker = Marker(
                time=marker_start,
                start_time=marker_start,
                end_time=marker_end,
                surah=previous.surah,
                surah_number=previous.surah_number,
                ayah=ayah,
                juz=get_juz_for_ayah(previous.surah_number or 1, ayah),
                quality=quality,
                confidence=round(confidence, 3),
            )
        else:
            inferred_time = int(round(window_start + (step * offset)))
            inferred_time = _defer_inferred_time_after_fatiha(
                inferred_time=inferred_time,
                fatiha_reset_times=fatiha_reset_times,
            )
            inferred_time = max(floor, min(window_end, inferred_time))

            wide_best: tuple[int, int, str, float] | None = None
            if entry is not None:
                wide_start = max(floor, inferred_time)
                wide_end = window_end
                if wide_end > wide_start:
                    wide_best = _try_wide_reground_before_infer(
                        transcript_segments=transcript_segments,
                        entry=entry,
                        window_start=wide_start,
                        window_end=wide_end,
                        expected_time=max(wide_start, expected),
                        min_score=min_score,
                        min_overlap=min_overlap,
                        min_confidence=min_confidence,
                        ambiguous_min_score=ambiguous_min_score,
                        ambiguous_min_confidence=ambiguous_min_confidence,
                    )

            if wide_best is not None:
                matched_start, matched_end, quality, confidence = wide_best
                marker_start = max(floor, min(window_end, int(matched_start)))
                marker_end = max(marker_start, min(window_end, int(matched_end)))
                marker = Marker(
                    time=marker_start,
                    start_time=marker_start,
                    end_time=marker_end,
                    surah=previous.surah,
                    surah_number=previous.surah_number,
                    ayah=ayah,
                    juz=get_juz_for_ayah(previous.surah_number or 1, ayah),
                    quality=quality,
                    confidence=round(confidence, 3),
                )
                inferred.append(marker)
                delay_floor = int(marker.start_time or marker.time) + _next_ayah_delay_seconds(entry)
                floor = max(int(marker.end_time or marker.time) + 1, delay_floor)
                if floor >= window_end:
                    break
                continue

            if require_weak_support_for_inferred:
                # Never infer through obvious low-data gaps (ruku/fatiha/non-recitation windows).
                if _has_low_data_gap(
                    transcript_segments=transcript_segments,
                    start_time=local_start,
                    end_time=local_end,
                    max_silence_seconds=16,
                    min_density=0.14,
                ):
                    continue
                if not _has_weak_local_support(
                    transcript_segments=transcript_segments,
                    entry=entry,
                    window_start=local_start,
                    window_end=local_end,
                    min_score=max(58, relaxed_ambig_score - 6),
                    min_overlap=max(0.04, relaxed_min_overlap - 0.03),
                ):
                    continue

            marker = Marker(
                time=inferred_time,
                start_time=inferred_time,
                end_time=inferred_time,
                surah=previous.surah,
                surah_number=previous.surah_number,
                ayah=ayah,
                juz=get_juz_for_ayah(previous.surah_number or 1, ayah),
                quality="inferred",
                confidence=0.56,
            )

        inferred.append(marker)
        delay_floor = int(marker.start_time or marker.time) + _next_ayah_delay_seconds(entry)
        floor = max(int(marker.end_time or marker.time) + 1, delay_floor)
        if floor >= window_end:
            break

    return inferred


def _is_valid_forward_transition(
    previous: Marker | None,
    entry: AyahEntry,
    marker_time: int,
    surah_totals: dict[str, int],
    min_gap_seconds: int,
    max_forward_jump_ayahs: int,
) -> bool:
    if previous is None:
        return True
    if marker_time - previous.time < min_gap_seconds:
        return False

    if previous.surah != entry.surah:
        previous_total = surah_totals.get(previous.surah, previous.ayah)
        near_end_of_previous = previous.ayah >= max(1, previous_total - 5)
        next_surah = (
            previous.surah_number is not None
            and entry.surah_number is not None
            and entry.surah_number == previous.surah_number + 1
        )
        if not (near_end_of_previous and next_surah):
            return False

    if previous.surah == entry.surah:
        ayah_delta = entry.ayah - previous.ayah
        allowed_jump = max_forward_jump_ayahs
        if ayah_delta < 0:
            return False
        if ayah_delta > allowed_jump:
            return False

    return True


def _should_replace_existing(existing: Marker, candidate: Marker) -> bool:
    existing_rank = _quality_rank(existing.quality)
    candidate_rank = _quality_rank(candidate.quality)
    if candidate_rank > existing_rank:
        return True
    if candidate_rank < existing_rank:
        return False

    existing_conf = float(existing.confidence or 0.0)
    candidate_conf = float(candidate.confidence or 0.0)
    if candidate_conf > existing_conf + 0.03:
        return True

    if candidate.time < existing.time:
        return True
    return False


def _candidate_is_valid(
    candidate: CandidateEvidence,
    rival_score: float,
    local_min_score: int,
    local_min_overlap: float,
    threshold: float,
    ambiguous_min_score: int,
    ambiguous_min_confidence: float,
) -> tuple[bool, str | None, float]:
    confidence = _candidate_confidence(candidate.adjusted_score, rival_score, candidate.overlap)
    ambiguous_min_overlap = max(0.1, local_min_overlap * 0.6)
    is_high = (
        candidate.adjusted_score >= local_min_score
        and candidate.overlap >= local_min_overlap
        and confidence >= threshold
    )
    is_ambiguous = (
        candidate.adjusted_score >= ambiguous_min_score
        and candidate.overlap >= ambiguous_min_overlap
        and confidence >= ambiguous_min_confidence
    )
    if not is_high and not is_ambiguous:
        return False, None, round(confidence, 3)
    return True, ("high" if is_high else "ambiguous"), round(confidence, 3)


def _has_anchor_token_hit(
    entry: AyahEntry,
    normalized_text: str,
    min_anchor_len: int = 4,
    min_similarity: float = 85.0,
) -> bool:
    if not normalized_text:
        return False

    anchors: list[str] = []
    for form in entry.match_forms:
        anchors.extend(_anchor_tokens_for_form(form))

    if not anchors:
        return False

    # If a phrase only has short anchor tokens (e.g. muqatta'at like "الف لام ميم"),
    # relax min length while requiring slightly higher similarity.
    effective_min_anchor_len = min_anchor_len
    if not any(len(anchor) >= min_anchor_len for anchor in anchors):
        effective_min_anchor_len = 3

    tokens = [token for token in normalized_text.split() if len(token) >= 2]
    if not tokens:
        return False

    for anchor in anchors:
        if len(anchor) < effective_min_anchor_len:
            continue
        for token in tokens:
            required_similarity = min_similarity + 4.0 if len(anchor) <= 3 or len(token) <= 3 else min_similarity
            similarity = max(float(fuzz.ratio(anchor, token)), float(fuzz.partial_ratio(anchor, token)))
            if similarity >= required_similarity:
                return True
    return False


def match_quran_markers(
    transcript_segments: list[TranscriptSegment],
    corpus_entries: list[AyahEntry],
    min_score: int = 78,
    min_gap_seconds: int = 8,
    min_overlap: float = 0.18,
    min_confidence: float = 0.62,
    search_window: int = 25,
    recovery_after_seconds: int = 420,
    recovery_rewind_ayat: int = 40,
    recovery_window_multiplier: float = 3.0,
    ambiguous_min_score: int = 74,
    ambiguous_min_confidence: float = 0.5,
    max_infer_gap_ayahs: int = 8,
    max_infer_gap_seconds: int = 720,
    max_leading_infer_ayahs: int = 3,
    allow_unverified_leading_infer: bool = True,
    duplicate_ayah_window_seconds: int = 120,
    max_forward_jump_ayahs: int = 2,
    require_weak_support_for_inferred: bool = True,
    forced_start_index: int | None = None,
    repeat_lookback_ayahs: int = 1,
    repeat_min_score: int = 90,
    repeat_min_overlap: float = 0.25,
    repeat_min_confidence: float = 0.80,
    repeat_max_gap_seconds: int = 10,
    max_recovery_jump_ayahs: int = 12,
    min_infer_step_seconds: float = 4.0,
    max_infer_step_seconds: float = 28.0,
    non_recitation_hold_seconds: int = 16,
    long_break_reacquire_seconds: int = 180,
    precomputed_reset_times: list[float] | None = None,
    reanchor_points: list[tuple[int, int, int]] | None = None,
) -> list[Marker]:
    if not transcript_segments or not corpus_entries:
        return []

    surah_totals: dict[str, int] = {}
    for item in corpus_entries:
        current = surah_totals.get(item.surah, 0)
        if item.ayah > current:
            surah_totals[item.surah] = item.ayah

    entry_lookup: dict[tuple[str, int], AyahEntry] = {(entry.surah, entry.ayah): entry for entry in corpus_entries}
    markers: list[Marker] = []
    marker_positions: dict[tuple[str, int], int] = {}
    if forced_start_index is not None and 0 <= forced_start_index < len(corpus_entries):
        last_matched_index = forced_start_index - 1
    else:
        last_matched_index = -1

    corpus_index_by_surah_ayah: dict[tuple[int, int], int] = {}
    for idx, entry in enumerate(corpus_entries):
        corpus_index_by_surah_ayah[(int(entry.surah_number), int(entry.ayah))] = idx
    reanchor_schedule: list[tuple[float, int]] = []
    for item in reanchor_points or []:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        try:
            at_time = float(item[0])
            surah_number = int(item[1])
            ayah = int(item[2])
        except (TypeError, ValueError):
            continue
        mapped_index = corpus_index_by_surah_ayah.get((surah_number, ayah))
        if mapped_index is None:
            continue
        reanchor_schedule.append((at_time, mapped_index))
    reanchor_schedule.sort(key=lambda item: item[0])
    reanchor_cursor = 0

    last_marker_time = -1
    stale_segments = 0
    fatiha_reset_times: list[float] = sorted(precomputed_reset_times or [])
    awaiting_reacquire = False
    pause_reacquire_until: float | None = None
    previous_segment_end: float | None = None
    reacquire_lock_ayahs_remaining = 0

    for segment_index, segment in enumerate(transcript_segments):
        segment_start = float(segment.start)
        segment_end = float(segment.end)
        while reanchor_cursor < len(reanchor_schedule) and segment_start >= reanchor_schedule[reanchor_cursor][0]:
            mapped_index = reanchor_schedule[reanchor_cursor][1]
            last_matched_index = mapped_index - 1
            awaiting_reacquire = True
            pause_reacquire_until = None
            reacquire_lock_ayahs_remaining = max(reacquire_lock_ayahs_remaining, 8)
            reanchor_cursor += 1
        if (
            previous_segment_end is not None
            and (segment_start - previous_segment_end) >= float(max(30, long_break_reacquire_seconds))
        ):
            # After long breaks (talks/pauses), force strict re-acquire to avoid jumping ahead.
            awaiting_reacquire = True
            pause_reacquire_until = None
            reacquire_lock_ayahs_remaining = max(reacquire_lock_ayahs_remaining, 8)
        previous_segment_end = segment_end

        normalized_segment = normalize_arabic(segment.text, strict=False)
        if _is_fatiha_like_segment(normalized_segment):
            fatiha_reset_times.append(float(segment.start))
            awaiting_reacquire = True
            reacquire_lock_ayahs_remaining = max(reacquire_lock_ayahs_remaining, 8)
            continue
        if _is_non_recitation_segment(normalized_segment):
            fatiha_reset_times.append(float(segment.start))
            pause_reacquire_until = float(segment.end) + float(max(8, non_recitation_hold_seconds))
            awaiting_reacquire = True
            reacquire_lock_ayahs_remaining = max(reacquire_lock_ayahs_remaining, 8)
            continue
        if pause_reacquire_until is not None and float(segment.start) <= pause_reacquire_until:
            continue
        segment_variants: list[tuple[str, float, int, float]] = [(normalized_segment, 0.0, segment_index, float(segment.end))]
        combined_text = normalized_segment
        previous_end = float(segment.end)
        for offset in range(1, 7):
            next_idx = segment_index + offset
            if next_idx >= len(transcript_segments):
                break
            next_segment = transcript_segments[next_idx]
            if float(next_segment.start) - previous_end > 2.5:
                break
            next_normalized = normalize_arabic(next_segment.text, strict=False)
            if len(next_normalized) < 2:
                break
            combined_text = f"{combined_text} {next_normalized}".strip()
            segment_variants.append((combined_text, float(offset) * 1.1, next_idx, float(next_segment.end)))
            previous_end = float(next_segment.end)

        if max(len(text) for text, _, _, _ in segment_variants) < 14:
            continue

        segment_words = list(getattr(segment, "words", None) or [])
        word_windows = list(generate_word_windows(segment_words, min_window=4, max_window=8))

        def evaluate_index(index: int) -> CandidateEvidence | None:
            if index < 0 or index >= len(corpus_entries):
                return None
            entry = corpus_entries[index]
            if is_excluded_surah(entry.surah):
                return None
            is_muqattaat = _is_muqattaat_entry(entry)

            best: CandidateEvidence | None = None
            for variant_text, penalty, variant_end_index, variant_end_time in segment_variants:
                if is_muqattaat and not _has_muqattaat_phrase_match(variant_text, entry):
                    continue
                score, overlap = _score_segment_against_entry(variant_text, entry)
                has_anchor = _has_anchor_token_hit(entry, variant_text)
                adjusted = score - penalty
                if not has_anchor and adjusted < float(max(64, min_score - 6)):
                    continue
                if has_anchor:
                    adjusted += 2.0
                if best is None or adjusted > best.adjusted_score:
                    best = CandidateEvidence(
                        adjusted_score=adjusted,
                        score=score,
                        overlap=overlap,
                        penalty=penalty,
                        source="segment",
                        normalized_text=variant_text,
                        start_time=float(segment.start),
                        end_time=variant_end_time,
                        word_indices=[],
                        segment_start_index=segment_index,
                        segment_end_index=variant_end_index,
                    )

            for window in word_windows:
                if is_muqattaat and not _has_muqattaat_phrase_match(window.normalized_text, entry):
                    continue
                penalty = _window_penalty(len(window.word_indices))
                score, overlap = _score_segment_against_entry(window.normalized_text, entry)
                has_anchor = _has_anchor_token_hit(entry, window.normalized_text)
                adjusted = score - penalty
                if not has_anchor and adjusted < float(max(64, min_score - 6)):
                    continue
                if has_anchor:
                    adjusted += 2.0
                if best is None or adjusted > best.adjusted_score:
                    best = CandidateEvidence(
                        adjusted_score=adjusted,
                        score=score,
                        overlap=overlap,
                        penalty=penalty,
                        source="window",
                        normalized_text=window.normalized_text,
                        start_time=window.start_time,
                        end_time=window.end_time,
                        word_indices=list(window.word_indices),
                        segment_start_index=segment_index,
                        segment_end_index=segment_index,
                    )
            return best

        selected_index: int | None = None
        selected_candidate: CandidateEvidence | None = None
        selected_quality: str | None = None
        selected_confidence = 0.0
        selected_from_recovery = False
        repeat_detected = False
        repeat_detected_score = -1.0

        # Small rewind exception: if imam repeats one of the most recent ayat,
        # keep the forward pointer where it is and just extend that ayah's end_time.
        if (
            last_matched_index >= 0
            and markers
            and repeat_lookback_ayahs > 0
            and not awaiting_reacquire
        ):
            lookback = max(1, min(8, int(repeat_lookback_ayahs)))
            rewind_indices = [
                index
                for index in range(last_matched_index, last_matched_index - lookback, -1)
                if 0 <= index < len(corpus_entries)
            ]
            rewind_candidates: dict[int, CandidateEvidence] = {}
            for index in rewind_indices:
                candidate = evaluate_index(index)
                if candidate is not None:
                    rewind_candidates[index] = candidate

            if rewind_candidates:
                forward_probe_scores: list[float] = []
                for index in [last_matched_index + 1, last_matched_index + 2]:
                    candidate = evaluate_index(index)
                    if candidate is not None:
                        forward_probe_scores.append(candidate.adjusted_score)
                forward_best_score = max(forward_probe_scores, default=-1.0)

                repeat_best_index: int | None = None
                repeat_best_candidate: CandidateEvidence | None = None
                repeat_best_confidence = 0.0

                for index, candidate in rewind_candidates.items():
                    rival = _best_rival_score(rewind_candidates, index=index)
                    valid, quality, confidence = _candidate_is_valid(
                        candidate=candidate,
                        rival_score=rival,
                        local_min_score=max(min_score + 4, repeat_min_score),
                        local_min_overlap=max(min_overlap + 0.04, repeat_min_overlap),
                        threshold=max(min_confidence + 0.10, repeat_min_confidence),
                        ambiguous_min_score=99,
                        ambiguous_min_confidence=0.99,
                    )
                    if not valid or quality != "high":
                        continue
                    if candidate.adjusted_score < repeat_min_score:
                        continue
                    if candidate.overlap < repeat_min_overlap:
                        continue
                    if confidence < repeat_min_confidence:
                        continue
                    if candidate.adjusted_score < forward_best_score + 1.0:
                        continue
                    if (
                        repeat_best_candidate is None
                        or confidence > repeat_best_confidence
                        or (
                            confidence == repeat_best_confidence
                            and candidate.adjusted_score > repeat_best_candidate.adjusted_score
                        )
                    ):
                        repeat_best_index = index
                        repeat_best_candidate = candidate
                        repeat_best_confidence = confidence

                if repeat_best_index is not None and repeat_best_candidate is not None:
                    repeat_entry = corpus_entries[repeat_best_index]
                    repeat_key = (repeat_entry.surah, repeat_entry.ayah)
                    existing_index = marker_positions.get(repeat_key)
                    if existing_index is not None and 0 <= existing_index < len(markers):
                        existing_marker = markers[existing_index]
                        segment_start = int(round(float(segment.start)))
                        marker_start = int(existing_marker.start_time or existing_marker.time)
                        marker_end = int(existing_marker.end_time or marker_start)
                        repeat_gap_limit = max(8, repeat_max_gap_seconds)
                        if awaiting_reacquire:
                            # After long breaks, repeated carry-over of the previous ayah is common.
                            # Allow extending the prior ayah across a larger gap while reacquiring.
                            repeat_gap_limit = max(repeat_gap_limit, 900)
                        if segment_start - marker_end <= repeat_gap_limit:
                            _, repeat_end, _ = _resolve_marker_times(
                                segment=segment,
                                entry=repeat_entry,
                                evidence=repeat_best_candidate,
                                transcript_segments=transcript_segments,
                                segment_index=segment_index,
                            )
                            proposed_end = max(
                                int(existing_marker.end_time or existing_marker.time),
                                int(repeat_end),
                                int(round(repeat_best_candidate.end_time)),
                            )

                            next_same_surah_start: int | None = None
                            for candidate_marker in markers:
                                if candidate_marker.surah != existing_marker.surah:
                                    continue
                                if int(candidate_marker.ayah) <= int(existing_marker.ayah):
                                    continue
                                candidate_start = int(candidate_marker.start_time or candidate_marker.time)
                                if candidate_start <= marker_start:
                                    continue
                                if next_same_surah_start is None or candidate_start < next_same_surah_start:
                                    next_same_surah_start = candidate_start

                            if next_same_surah_start is not None:
                                proposed_end = min(proposed_end, next_same_surah_start - 1)

                            extended = False
                            if proposed_end >= marker_start and proposed_end > int(existing_marker.end_time or marker_start):
                                existing_marker.end_time = proposed_end
                                extended = True
                            if extended:
                                repeat_detected = True
                                repeat_detected_score = max(repeat_detected_score, float(repeat_best_candidate.adjusted_score))
                                last_marker_time = max(last_marker_time, int(existing_marker.end_time or marker_start))

        if last_matched_index < 0:
            acquire_start = forced_start_index if forced_start_index is not None else 0
            acquire_end = min(len(corpus_entries), acquire_start + 40)
            candidates: dict[int, CandidateEvidence] = {}
            top_score = -1.0
            top_index = -1
            second_score = -1.0

            for index in range(acquire_start, acquire_end):
                candidate = evaluate_index(index)
                if candidate is None:
                    continue
                candidates[index] = candidate
                if candidate.adjusted_score > top_score:
                    second_score = top_score
                    top_score = candidate.adjusted_score
                    top_index = index
                elif candidate.adjusted_score > second_score:
                    second_score = candidate.adjusted_score

            if top_index >= 0:
                top_candidate = candidates[top_index]
                valid, quality, confidence = _candidate_is_valid(
                    candidate=top_candidate,
                    rival_score=second_score,
                    local_min_score=min_score,
                    local_min_overlap=min_overlap,
                    threshold=max(0.7, min_confidence),
                    ambiguous_min_score=ambiguous_min_score,
                    ambiguous_min_confidence=ambiguous_min_confidence,
                )
                if valid and quality is not None:
                    selected_index = top_index
                    selected_candidate = top_candidate
                    selected_quality = quality
                    selected_confidence = confidence
        else:
            expected = last_matched_index + 1
            normal_candidates: dict[int, CandidateEvidence] = {}
            for index in range(expected - 1, expected + 3):
                candidate = evaluate_index(index)
                if candidate is not None:
                    normal_candidates[index] = candidate

            for index in range(expected - 1, expected + 3):
                candidate = normal_candidates.get(index)
                if candidate is None:
                    continue
                rival = max(
                    (
                        other.adjusted_score
                        for other_index, other in normal_candidates.items()
                        if other_index != index
                    ),
                    default=-1.0,
                )
                valid, quality, confidence = _candidate_is_valid(
                    candidate=candidate,
                    rival_score=rival,
                    local_min_score=min_score,
                    local_min_overlap=min_overlap,
                    threshold=min_confidence,
                    ambiguous_min_score=ambiguous_min_score,
                    ambiguous_min_confidence=ambiguous_min_confidence,
                )
                if not valid or quality is None:
                    continue
                jump = index - last_matched_index
                local_max_forward_jump = 1 if (awaiting_reacquire or reacquire_lock_ayahs_remaining > 0) else max_forward_jump_ayahs
                if jump < 1 or jump > local_max_forward_jump:
                    continue
                selected_index = index
                selected_candidate = candidate
                selected_quality = quality
                selected_confidence = confidence
                selected_from_recovery = False
                break

            if selected_index is None:
                recovery_start = max(expected, 0)
                recovery_end = min(len(corpus_entries), recovery_start + 60)
                recovery_best_index = -1
                recovery_best: CandidateEvidence | None = None
                recovery_best_conf = 0.0
                recovery_second = -1.0

                # Disable long-jump recovery while reacquiring after pause/non-recitation.
                if not awaiting_reacquire and reacquire_lock_ayahs_remaining <= 0:
                    for index in range(recovery_start, recovery_end):
                        candidate = evaluate_index(index)
                        if candidate is None:
                            continue
                        rival = recovery_best.adjusted_score if recovery_best is not None else -1.0
                        valid, quality, confidence = _candidate_is_valid(
                            candidate=candidate,
                            rival_score=rival,
                            local_min_score=max(min_score, 80),
                            local_min_overlap=max(min_overlap, 0.20),
                            threshold=max(min_confidence, 0.72),
                            ambiguous_min_score=max(ambiguous_min_score, 78),
                            ambiguous_min_confidence=max(ambiguous_min_confidence, 0.68),
                        )
                        if not valid or quality is None:
                            continue
                        jump = index - last_matched_index
                        if jump < 1:
                            continue
                        if jump > max_recovery_jump_ayahs:
                            continue
                        if last_marker_time >= 0:
                            approx_gap_seconds = float(candidate.start_time) - float(last_marker_time)
                            min_expected_gap = max(10.0, float(jump) * 2.0)
                            if approx_gap_seconds < min_expected_gap:
                                continue
                        if candidate.adjusted_score < 80 or candidate.overlap < 0.20 or confidence < 0.72:
                            continue
                        if recovery_best is None or candidate.adjusted_score > recovery_best.adjusted_score:
                            recovery_second = recovery_best.adjusted_score if recovery_best is not None else recovery_second
                            recovery_best = candidate
                            recovery_best_index = index
                            recovery_best_conf = confidence
                        elif candidate.adjusted_score > recovery_second:
                            recovery_second = candidate.adjusted_score

                if recovery_best is not None and recovery_best_index >= 0:
                    selected_index = recovery_best_index
                    selected_candidate = recovery_best
                    selected_quality = "high"
                    selected_confidence = max(0.72, recovery_best_conf)
                    selected_from_recovery = True

        if selected_index is None or selected_candidate is None or selected_quality is None:
            if repeat_detected:
                stale_segments = 0
                continue
            stale_segments += 1
            continue

        if repeat_detected and selected_candidate.adjusted_score < (repeat_detected_score + 1.0):
            stale_segments = 0
            continue

        if awaiting_reacquire:
            strict_reacquire_score = max(82.0, float(min_score) + 4.0)
            strict_reacquire_overlap = max(0.22, float(min_overlap) + 0.04)
            strict_reacquire_conf = max(0.78, float(min_confidence) + 0.12)
            if (
                selected_quality != "high"
                or selected_candidate.adjusted_score < strict_reacquire_score
                or selected_candidate.overlap < strict_reacquire_overlap
                or selected_confidence < strict_reacquire_conf
            ):
                stale_segments += 1
                continue

        entry = corpus_entries[selected_index]
        previous_marker = markers[-1] if markers else None
        marker_start, marker_end, matched_tokens = _resolve_marker_times(
            segment=segment,
            entry=entry,
            evidence=selected_candidate,
            transcript_segments=transcript_segments,
            segment_index=segment_index,
        )
        marker_end = max(marker_start, marker_end)

        previous_for_transition = previous_marker
        if previous_marker is not None and previous_marker.surah != entry.surah:
            tail_markers = _build_transition_tail_markers(
                previous=previous_marker,
                next_entry=entry,
                transition_time=marker_start,
                surah_totals=surah_totals,
                entry_lookup=entry_lookup,
                transcript_segments=transcript_segments,
                fatiha_reset_times=fatiha_reset_times,
                min_gap_seconds=min_gap_seconds,
                min_score=min_score,
                min_overlap=min_overlap,
                min_confidence=min_confidence,
                ambiguous_min_score=ambiguous_min_score,
                ambiguous_min_confidence=ambiguous_min_confidence,
                require_weak_support_for_inferred=require_weak_support_for_inferred,
            )
            if tail_markers:
                for tail_marker in tail_markers:
                    tail_key = (tail_marker.surah, tail_marker.ayah)
                    existing_tail_index = marker_positions.get(tail_key)
                    if existing_tail_index is not None and 0 <= existing_tail_index < len(markers):
                        existing_tail = markers[existing_tail_index]
                        if _should_replace_existing(existing_tail, tail_marker):
                            markers[existing_tail_index] = tail_marker
                    else:
                        markers.append(tail_marker)
                        marker_positions[tail_key] = len(markers) - 1
                previous_for_transition = max(tail_markers, key=lambda item: (item.ayah, item.time))

            previous_total = surah_totals.get(previous_marker.surah, previous_marker.ayah)
            if previous_for_transition is None or previous_for_transition.ayah < previous_total:
                stale_segments += 1
                continue

        if not _is_valid_forward_transition(
            previous=previous_for_transition,
            entry=entry,
            marker_time=marker_start,
            surah_totals=surah_totals,
            min_gap_seconds=min_gap_seconds,
            max_forward_jump_ayahs=(max_recovery_jump_ayahs if selected_from_recovery else max_forward_jump_ayahs),
        ):
            stale_segments += 1
            continue

        marker_candidate = Marker(
            time=marker_start,
            start_time=marker_start,
            end_time=marker_end,
            surah=entry.surah,
            surah_number=entry.surah_number,
            ayah=entry.ayah,
            juz=get_juz_for_ayah(entry.surah_number, entry.ayah),
            quality=selected_quality,
            confidence=round(selected_confidence, 3),
            matched_token_indices=matched_tokens,
        )

        key = (entry.surah, entry.ayah)
        existing_index = marker_positions.get(key)
        accepted_marker = marker_candidate
        if existing_index is not None:
            existing = markers[existing_index]
            if abs(marker_candidate.time - existing.time) <= duplicate_ayah_window_seconds:
                if _should_replace_existing(existing, marker_candidate):
                    markers[existing_index] = marker_candidate
                    accepted_marker = marker_candidate
                else:
                    accepted_marker = existing
            else:
                markers.append(marker_candidate)
                marker_positions[key] = len(markers) - 1
        else:
            markers.append(marker_candidate)
            marker_positions[key] = len(markers) - 1

        last_matched_index = max(last_matched_index, selected_index)
        last_marker_time = max(last_marker_time, accepted_marker.time)
        awaiting_reacquire = False
        pause_reacquire_until = None
        if reacquire_lock_ayahs_remaining > 0:
            reacquire_lock_ayahs_remaining -= 1
        stale_segments = 0

    inferred_markers: list[Marker] = []
    keyed_markers: dict[tuple[str, int], Marker] = {(marker.surah, marker.ayah): marker for marker in markers}
    entry_lookup: dict[tuple[str, int], AyahEntry] = {(entry.surah, entry.ayah): entry for entry in corpus_entries}
    anchors = [item for item in sorted(markers, key=lambda m: m.time) if _is_anchor_quality(item.quality)]

    for left, right in zip(anchors, anchors[1:]):
        if left.surah != right.surah:
            transition_tail = _build_transition_tail_markers(
                previous=left,
                next_entry=AyahEntry(
                    surah_number=right.surah_number or 0,
                    surah=right.surah,
                    ayah=right.ayah,
                    text="",
                    normalized="",
                    match_forms=[],
                ),
                transition_time=right.time,
                surah_totals=surah_totals,
                entry_lookup=entry_lookup,
                transcript_segments=transcript_segments,
                fatiha_reset_times=fatiha_reset_times,
                min_gap_seconds=min_gap_seconds,
                min_score=min_score,
                min_overlap=min_overlap,
                min_confidence=min_confidence,
                ambiguous_min_score=ambiguous_min_score,
                ambiguous_min_confidence=ambiguous_min_confidence,
                require_weak_support_for_inferred=require_weak_support_for_inferred,
            )
            for marker_to_add in transition_tail:
                key = (marker_to_add.surah, marker_to_add.ayah)
                if key in keyed_markers:
                    continue
                inferred_markers.append(marker_to_add)
                keyed_markers[key] = marker_to_add
            continue
        if left.ayah >= right.ayah:
            continue

        missing_count = right.ayah - left.ayah - 1
        if missing_count <= 0:
            continue
        strong_bridge = _is_strong_anchor_marker(left) and _is_strong_anchor_marker(right)

        gap_seconds = right.time - left.time
        if gap_seconds <= min_gap_seconds or gap_seconds > max_infer_gap_seconds:
            continue
        resets_between = _reset_points_between(fatiha_reset_times, left.time, right.time)
        if resets_between:
            search_floor = resets_between[0] + max(6, min_gap_seconds)
            searched = _recover_missing_gap_with_search(
                left=left,
                right=right,
                entry_lookup=entry_lookup,
                transcript_segments=transcript_segments,
                min_gap_seconds=min_gap_seconds,
                min_score=min_score,
                min_overlap=min_overlap,
                min_confidence=min_confidence,
                ambiguous_min_score=ambiguous_min_score,
                ambiguous_min_confidence=ambiguous_min_confidence,
                require_weak_support_for_inferred=require_weak_support_for_inferred,
                search_floor_time=search_floor,
                exhaustive_ahead_search=True,
            )
            for marker_to_add in searched:
                key = (marker_to_add.surah, marker_to_add.ayah)
                if key in keyed_markers:
                    continue
                inferred_markers.append(marker_to_add)
                keyed_markers[key] = marker_to_add
            if not searched:
                fallback_window_start = left.time + min_gap_seconds
                fallback_window_end = right.time - min_gap_seconds
                fallback_start = max(fallback_window_start, search_floor)
                fallback_span = right.time - fallback_start
                if fallback_span > min_gap_seconds and missing_count > 0:
                    fallback_step = fallback_span / float(missing_count + 1)
                    if fallback_step >= float(min_infer_step_seconds):
                        fallback_prev_time = int(left.time)
                        fallback_prev_ayah = int(left.ayah)
                        for offset_idx, ayah_number in enumerate(range(left.ayah + 1, right.ayah), start=1):
                            key = (left.surah, ayah_number)
                            if key in keyed_markers:
                                fallback_prev_ayah = ayah_number
                                existing = keyed_markers[key]
                                fallback_prev_time = int(existing.start_time or existing.time)
                                continue
                            entry = entry_lookup.get(key)
                            previous_entry = entry_lookup.get((left.surah, fallback_prev_ayah))
                            previous_duration = _estimated_ayah_duration_seconds(previous_entry)
                            long_ayah_hold = int(round(previous_duration * 0.72)) if previous_duration >= 28 else max(6, int(round(fallback_step * 0.7)))
                            start_floor_from_previous = fallback_prev_time + long_ayah_hold
                            if entry is not None and strong_bridge:
                                expected_time = int(round(fallback_start + (fallback_step * offset_idx)))
                                forward_start = max(fallback_window_start, expected_time, start_floor_from_previous)
                                forward_end = fallback_window_end
                                if forward_end > forward_start:
                                    best_forward = _find_best_ayah_timestamp(
                                        transcript_segments=transcript_segments,
                                        entry=entry,
                                        window_start=forward_start,
                                        window_end=forward_end,
                                        expected_time=expected_time,
                                        min_score=max(66, min_score - 10),
                                        min_overlap=max(0.05, min_overlap - 0.10),
                                        min_confidence=max(0.54, min_confidence - 0.10),
                                        ambiguous_min_score=max(62, ambiguous_min_score - 10),
                                        ambiguous_min_confidence=max(0.45, ambiguous_min_confidence - 0.08),
                                    )
                                    if best_forward is not None:
                                        matched_time, matched_end, quality, confidence = best_forward
                                        marker_to_add = Marker(
                                            time=max(fallback_window_start, min(fallback_window_end, int(matched_time))),
                                            start_time=max(fallback_window_start, min(fallback_window_end, int(matched_time))),
                                            end_time=max(
                                                max(fallback_window_start, min(fallback_window_end, int(matched_time))),
                                                min(fallback_window_end, int(matched_end)),
                                            ),
                                            surah=left.surah,
                                            surah_number=left.surah_number,
                                            ayah=ayah_number,
                                            juz=get_juz_for_ayah(left.surah_number or 1, ayah_number),
                                            quality=quality,
                                            confidence=round(confidence, 3),
                                        )
                                        inferred_markers.append(marker_to_add)
                                        keyed_markers[key] = marker_to_add
                                        fallback_prev_ayah = ayah_number
                                        fallback_prev_time = int(marker_to_add.start_time or marker_to_add.time)
                                        continue
                            inferred_time = int(round(fallback_start + (fallback_step * offset_idx)))
                            inferred_time = _defer_inferred_time_after_fatiha(
                                inferred_time=inferred_time,
                                fatiha_reset_times=fatiha_reset_times,
                            )
                            inferred_time = max(inferred_time, start_floor_from_previous)
                            inferred_time = max(fallback_window_start, min(fallback_window_end, inferred_time))
                            if inferred_time > fallback_window_end:
                                continue
                            if entry is not None:
                                wide_start = max(fallback_window_start, inferred_time)
                                wide_end = fallback_window_end
                                if wide_end > wide_start:
                                    wide_best = _try_wide_reground_before_infer(
                                        transcript_segments=transcript_segments,
                                        entry=entry,
                                        window_start=wide_start,
                                        window_end=wide_end,
                                        expected_time=max(wide_start, inferred_time),
                                        min_score=min_score,
                                        min_overlap=min_overlap,
                                        min_confidence=min_confidence,
                                        ambiguous_min_score=ambiguous_min_score,
                                        ambiguous_min_confidence=ambiguous_min_confidence,
                                    )
                                    if wide_best is not None:
                                        matched_time, matched_end, quality, confidence = wide_best
                                        marker_to_add = Marker(
                                            time=max(fallback_window_start, min(fallback_window_end, int(matched_time))),
                                            start_time=max(fallback_window_start, min(fallback_window_end, int(matched_time))),
                                            end_time=max(
                                                max(fallback_window_start, min(fallback_window_end, int(matched_time))),
                                                min(fallback_window_end, int(matched_end)),
                                            ),
                                            surah=left.surah,
                                            surah_number=left.surah_number,
                                            ayah=ayah_number,
                                            juz=get_juz_for_ayah(left.surah_number or 1, ayah_number),
                                            quality=quality,
                                            confidence=round(confidence, 3),
                                        )
                                        inferred_markers.append(marker_to_add)
                                        keyed_markers[key] = marker_to_add
                                        fallback_prev_ayah = ayah_number
                                        fallback_prev_time = int(marker_to_add.start_time or marker_to_add.time)
                                        continue
                            if entry is not None and require_weak_support_for_inferred:
                                support_start = max(fallback_window_start, inferred_time - max(12, int(round(fallback_step))))
                                support_end = min(fallback_window_end, inferred_time + max(12, int(round(fallback_step))))
                                if _has_low_data_gap(
                                    transcript_segments=transcript_segments,
                                    start_time=support_start,
                                    end_time=support_end,
                                    max_silence_seconds=16,
                                    min_density=0.14,
                                ):
                                    continue
                                if not _has_weak_local_support(
                                    transcript_segments=transcript_segments,
                                    entry=entry,
                                    window_start=support_start,
                                    window_end=support_end,
                                    min_score=max(60, ambiguous_min_score - 8),
                                    min_overlap=max(0.07, min_overlap - 0.05),
                                ):
                                    continue
                            marker_to_add = Marker(
                                time=inferred_time,
                                start_time=inferred_time,
                                end_time=inferred_time,
                                surah=left.surah,
                                surah_number=left.surah_number,
                                ayah=ayah_number,
                                juz=get_juz_for_ayah(left.surah_number or 1, ayah_number),
                                quality="inferred",
                                confidence=round(min(left.confidence or 0.56, right.confidence or 0.56, 0.56), 3),
                            )
                            inferred_markers.append(marker_to_add)
                            keyed_markers[key] = marker_to_add
                            fallback_prev_ayah = ayah_number
                            fallback_prev_time = int(marker_to_add.start_time or marker_to_add.time)
            continue
        if _has_low_data_gap(transcript_segments, left.time, right.time):
            continue

        step_seconds = gap_seconds / (missing_count + 1)
        if (
            missing_count > max_infer_gap_ayahs
            or step_seconds < float(min_infer_step_seconds)
            or step_seconds > float(max_infer_step_seconds)
        ):
            searched = _recover_missing_gap_with_search(
                left=left,
                right=right,
                entry_lookup=entry_lookup,
                transcript_segments=transcript_segments,
                min_gap_seconds=min_gap_seconds,
                min_score=min_score,
                min_overlap=min_overlap,
                min_confidence=min_confidence,
                ambiguous_min_score=ambiguous_min_score,
                ambiguous_min_confidence=ambiguous_min_confidence,
                require_weak_support_for_inferred=require_weak_support_for_inferred,
            )
            for marker_to_add in searched:
                key = (marker_to_add.surah, marker_to_add.ayah)
                if key in keyed_markers:
                    continue
                inferred_markers.append(marker_to_add)
                keyed_markers[key] = marker_to_add
            continue
        rolling_prev_time = int(left.time)
        rolling_prev_ayah = int(left.ayah)

        for offset in range(1, missing_count + 1):
            ayah_number = left.ayah + offset
            key = (left.surah, ayah_number)
            if key in keyed_markers:
                rolling_prev_ayah = ayah_number
                existing = keyed_markers[key]
                rolling_prev_time = int(existing.start_time or existing.time)
                continue

            expected_time = int(round(left.time + (step_seconds * offset)))
            window_half = max(10, int(round(step_seconds * 0.8)))
            window_start = max(left.time + min_gap_seconds, expected_time - window_half)
            window_end = min(right.time - min_gap_seconds, expected_time + window_half)

            entry = entry_lookup.get(key)
            inferred_time = int(round(left.time + (step_seconds * offset)))
            inferred_time = _defer_inferred_time_after_fatiha(
                inferred_time=inferred_time,
                fatiha_reset_times=fatiha_reset_times,
            )
            inferred_time = max(window_start, min(window_end, inferred_time))
            previous_entry = entry_lookup.get((left.surah, rolling_prev_ayah))
            previous_duration = _estimated_ayah_duration_seconds(previous_entry)
            long_ayah_hold = int(round(previous_duration * 0.72)) if previous_duration >= 28 else max(6, int(round(step_seconds * 0.7)))
            start_floor_from_previous = rolling_prev_time + long_ayah_hold
            inferred_time = max(inferred_time, start_floor_from_previous)
            if inferred_time > window_end:
                continue
            if entry is not None:
                wide_start = max(window_start, inferred_time)
                wide_end = right.time - min_gap_seconds
                if wide_end > wide_start:
                    wide_best_general = _try_wide_reground_before_infer(
                        transcript_segments=transcript_segments,
                        entry=entry,
                        window_start=wide_start,
                        window_end=wide_end,
                        expected_time=max(wide_start, inferred_time),
                        min_score=min_score,
                        min_overlap=min_overlap,
                        min_confidence=min_confidence,
                        ambiguous_min_score=ambiguous_min_score,
                        ambiguous_min_confidence=ambiguous_min_confidence,
                    )
                    if wide_best_general is not None:
                        matched_time, matched_end, quality, confidence = wide_best_general
                        marker_to_add = Marker(
                            time=max(window_start, min(window_end, int(matched_time))),
                            start_time=max(window_start, min(window_end, int(matched_time))),
                            end_time=max(
                                max(window_start, min(window_end, int(matched_time))),
                                min(window_end, int(matched_end)),
                            ),
                            surah=left.surah,
                            surah_number=left.surah_number,
                            ayah=ayah_number,
                            juz=get_juz_for_ayah(left.surah_number or 1, ayah_number),
                            quality=quality,
                            confidence=round(confidence, 3),
                        )
                        inferred_markers.append(marker_to_add)
                        keyed_markers[key] = marker_to_add
                        rolling_prev_ayah = ayah_number
                        rolling_prev_time = int(marker_to_add.start_time or marker_to_add.time)
                        continue
            if entry is not None:
                # Before inferring, always check a wider forward window to catch cases
                # where the current ayah is long and the next ayah starts noticeably later.
                forward_start = max(window_start, inferred_time + max(2, int(round(step_seconds * 0.35))), start_floor_from_previous)
                forward_end = right.time - min_gap_seconds
                if forward_end > forward_start:
                    best_forward_general = _find_best_ayah_timestamp(
                        transcript_segments=transcript_segments,
                        entry=entry,
                        window_start=forward_start,
                        window_end=forward_end,
                        expected_time=max(forward_start, inferred_time),
                        min_score=max(64, min_score - 10),
                        min_overlap=max(0.05, min_overlap - 0.10),
                        min_confidence=max(0.52, min_confidence - 0.10),
                        ambiguous_min_score=max(60, ambiguous_min_score - 10),
                        ambiguous_min_confidence=max(0.44, ambiguous_min_confidence - 0.08),
                    )
                    if best_forward_general is not None:
                        matched_time, matched_end, quality, confidence = best_forward_general
                        marker_to_add = Marker(
                            time=max(window_start, min(window_end, int(matched_time))),
                            start_time=max(window_start, min(window_end, int(matched_time))),
                            end_time=max(
                                max(window_start, min(window_end, int(matched_time))),
                                min(window_end, int(matched_end)),
                            ),
                            surah=left.surah,
                            surah_number=left.surah_number,
                            ayah=ayah_number,
                            juz=get_juz_for_ayah(left.surah_number or 1, ayah_number),
                            quality=quality,
                            confidence=round(confidence, 3),
                        )
                        inferred_markers.append(marker_to_add)
                        keyed_markers[key] = marker_to_add
                        rolling_prev_ayah = ayah_number
                        rolling_prev_time = int(marker_to_add.start_time or marker_to_add.time)
                        continue
            if entry is not None and strong_bridge:
                # For strong-anchor bridges, search all the way toward the right anchor before inferring.
                forward_start = max(window_start, inferred_time, start_floor_from_previous)
                forward_end = right.time - min_gap_seconds
                if forward_end > forward_start:
                    best_forward = _find_best_ayah_timestamp(
                        transcript_segments=transcript_segments,
                        entry=entry,
                        window_start=forward_start,
                        window_end=forward_end,
                        expected_time=inferred_time,
                        min_score=max(66, min_score - 10),
                        min_overlap=max(0.05, min_overlap - 0.10),
                        min_confidence=max(0.54, min_confidence - 0.10),
                        ambiguous_min_score=max(62, ambiguous_min_score - 10),
                        ambiguous_min_confidence=max(0.45, ambiguous_min_confidence - 0.08),
                    )
                    if best_forward is not None:
                        matched_time, matched_end, quality, confidence = best_forward
                        marker_to_add = Marker(
                            time=max(window_start, min(window_end, int(matched_time))),
                            start_time=max(window_start, min(window_end, int(matched_time))),
                            end_time=max(
                                max(window_start, min(window_end, int(matched_time))),
                                min(window_end, int(matched_end)),
                            ),
                            surah=left.surah,
                            surah_number=left.surah_number,
                            ayah=ayah_number,
                            juz=get_juz_for_ayah(left.surah_number or 1, ayah_number),
                            quality=quality,
                            confidence=round(confidence, 3),
                        )
                        inferred_markers.append(marker_to_add)
                        keyed_markers[key] = marker_to_add
                        rolling_prev_ayah = ayah_number
                        rolling_prev_time = int(marker_to_add.start_time or marker_to_add.time)
                        continue
            if require_weak_support_for_inferred and entry is not None:
                window_start = max(left.time + min_gap_seconds, inferred_time - max(12, int(round(step_seconds))))
                window_end = min(right.time - min_gap_seconds, inferred_time + max(12, int(round(step_seconds))))
                if _has_low_data_gap(
                    transcript_segments=transcript_segments,
                    start_time=window_start,
                    end_time=window_end,
                    max_silence_seconds=16,
                    min_density=0.14,
                ):
                    continue
                if not _has_weak_local_support(
                    transcript_segments=transcript_segments,
                    entry=entry,
                    window_start=window_start,
                    window_end=window_end,
                    min_score=max(60, ambiguous_min_score - 8),
                    min_overlap=max(0.07, min_overlap - 0.05),
                ):
                    continue
            marker_to_add = Marker(
                time=inferred_time,
                start_time=inferred_time,
                end_time=inferred_time,
                surah=left.surah,
                surah_number=left.surah_number,
                ayah=ayah_number,
                juz=get_juz_for_ayah(left.surah_number or 1, ayah_number),
                quality="inferred",
                confidence=round(min(left.confidence or 0.58, right.confidence or 0.58, 0.6), 3),
            )

            inferred_markers.append(marker_to_add)
            keyed_markers[key] = marker_to_add
            rolling_prev_ayah = ayah_number
            rolling_prev_time = int(marker_to_add.start_time or marker_to_add.time)

    # Backfill leading ayahs for each surah's first anchor when it starts after ayah 1.
    # This helps recover obvious transitions such as ending one surah and starting the next
    # when ayah 1 is weakly captured but ayah 2/3 are confidently matched.
    if anchors:
        first_anchor_per_surah: list[Marker] = []
        seen_surahs: set[str] = set()
        for marker in anchors:
            if marker.surah in seen_surahs:
                continue
            seen_surahs.add(marker.surah)
            first_anchor_per_surah.append(marker)

        for first in first_anchor_per_surah:
            if first.ayah <= 1 or first.ayah - 1 > max_leading_infer_ayahs:
                continue

            next_same_surah = next(
                (
                    marker
                    for marker in anchors
                    if marker.surah == first.surah and marker.ayah > first.ayah and marker.time > first.time
                ),
                None,
            )
            if next_same_surah is not None:
                delta_ayahs = max(1, next_same_surah.ayah - first.ayah)
                delta_time = max(1, next_same_surah.time - first.time)
                time_step = max(4, min(18, int(round(delta_time / delta_ayahs))))
            else:
                time_step = 8

            for ayah_number in range(first.ayah - 1, 0, -1):
                key = (first.surah, ayah_number)
                if key in keyed_markers:
                    continue

                offset = first.ayah - ayah_number
                leading_step = max(4, min(8, time_step))
                expected_time = max(0, first.time - (leading_step * offset))
                window_half = max(8, time_step)
                window_start = max(0, expected_time - window_half)
                window_end = min(max(0, first.time - min_gap_seconds), expected_time + window_half)
                if window_end <= window_start:
                    continue

                marker_to_add: Marker | None = None
                entry = entry_lookup.get(key)
                if entry is not None:
                    best = _find_best_ayah_timestamp(
                        transcript_segments=transcript_segments,
                        entry=entry,
                        window_start=window_start,
                        window_end=window_end,
                        expected_time=expected_time,
                        min_score=min_score,
                        min_overlap=max(min_overlap, 0.18),
                        min_confidence=min_confidence,
                        ambiguous_min_score=ambiguous_min_score,
                        ambiguous_min_confidence=ambiguous_min_confidence,
                    )
                    if best is not None:
                        matched_time, matched_end, quality, confidence = best
                        bounded_time = max(window_start, min(window_end, matched_time))
                        bounded_end = max(bounded_time, min(window_end, matched_end))
                        marker_to_add = Marker(
                            time=bounded_time,
                            start_time=bounded_time,
                            end_time=bounded_end,
                            surah=first.surah,
                            surah_number=first.surah_number,
                            ayah=ayah_number,
                            juz=get_juz_for_ayah(first.surah_number or 1, ayah_number),
                            quality=quality,
                            confidence=confidence,
                        )

                if marker_to_add is None and not allow_unverified_leading_infer:
                    continue

                if marker_to_add is None:
                    expected_time = _defer_inferred_time_after_fatiha(
                        inferred_time=expected_time,
                        fatiha_reset_times=fatiha_reset_times,
                    )
                    expected_time = max(window_start, min(window_end, expected_time))
                    if require_weak_support_for_inferred and not _has_weak_local_support(
                        transcript_segments=transcript_segments,
                        entry=entry,
                        window_start=window_start,
                        window_end=window_end,
                        min_score=max(60, ambiguous_min_score - 8),
                        min_overlap=max(0.07, min_overlap - 0.05),
                    ):
                        continue
                    marker_to_add = Marker(
                        time=expected_time,
                        start_time=expected_time,
                        end_time=expected_time,
                        surah=first.surah,
                        surah_number=first.surah_number,
                        ayah=ayah_number,
                        juz=get_juz_for_ayah(first.surah_number or 1, ayah_number),
                        quality="inferred",
                        confidence=round(min(first.confidence or 0.58, 0.58), 3),
                    )

                inferred_markers.append(marker_to_add)
                keyed_markers[key] = marker_to_add

    merged = markers + inferred_markers
    coverage_inferred = _fill_surah_coverage_markers(
        merged,
        entry_lookup,
        transcript_segments=transcript_segments,
        fatiha_reset_times=fatiha_reset_times,
        weak_support_score=max(60, ambiguous_min_score - 8),
        weak_support_overlap=max(0.07, min_overlap - 0.05),
        enforce_weak_support=require_weak_support_for_inferred,
        min_bridge_step_seconds=min_infer_step_seconds,
        max_bridge_step_seconds=max_infer_step_seconds,
    )
    merged.extend(coverage_inferred)
    merged = _dedupe_by_local_time_window(merged, window_seconds=90)
    merged = _apply_overlap_conflict_resolution(merged)
    merged = _refine_weak_boundary_markers(
        merged,
        transcript_segments=transcript_segments,
        entry_lookup=entry_lookup,
        fatiha_reset_times=fatiha_reset_times,
        min_score=min_score,
        min_overlap=min_overlap,
        min_confidence=min_confidence,
        ambiguous_min_score=ambiguous_min_score,
        ambiguous_min_confidence=ambiguous_min_confidence,
    )
    merged = _refine_inferred_markers_with_local_search(
        merged,
        transcript_segments=transcript_segments,
        entry_lookup=entry_lookup,
        fatiha_reset_times=fatiha_reset_times,
        min_score=min_score,
        min_overlap=min_overlap,
        min_confidence=min_confidence,
        ambiguous_min_score=ambiguous_min_score,
        ambiguous_min_confidence=ambiguous_min_confidence,
    )
    merged = _delay_weak_markers_after_resets(
        merged,
        fatiha_reset_times=fatiha_reset_times,
        transcript_segments=transcript_segments,
        hold_seconds=34,
    )
    merged = _quran_first_refine_weak_markers(
        merged,
        transcript_segments=transcript_segments,
        entry_lookup=entry_lookup,
        min_score=min_score,
        min_overlap=min_overlap,
        min_confidence=min_confidence,
        ambiguous_min_score=ambiguous_min_score,
        ambiguous_min_confidence=ambiguous_min_confidence,
    )
    merged = _redistribute_dense_weak_runs(merged)
    merged = _stabilize_weak_marker_durations(merged)
    merged = _extend_point_markers_to_next(merged, max_extension_seconds=90)
    merged = _prune_unrealistic_progression(merged)
    merged = _enforce_surah_transition_order(merged, surah_totals=surah_totals, min_gap_seconds=min_gap_seconds)
    merged = _enforce_long_ayah_inferred_floor(merged, entry_lookup=entry_lookup)
    # Final continuity pass on the stabilized timeline: recover remaining ayah holes
    # using anchor-aware interpolation only when pacing is plausible.
    post_fill = _fill_surah_coverage_markers(
        merged,
        entry_lookup,
        transcript_segments=transcript_segments,
        fatiha_reset_times=fatiha_reset_times,
        weak_support_score=max(60, ambiguous_min_score - 8),
        weak_support_overlap=max(0.07, min_overlap - 0.05),
        enforce_weak_support=False,
        min_bridge_step_seconds=min_infer_step_seconds,
        max_bridge_step_seconds=max_infer_step_seconds,
    )
    if post_fill:
        merged.extend(post_fill)
        merged = _dedupe_by_local_time_window(merged, window_seconds=90)
        merged = _apply_overlap_conflict_resolution(merged)
        merged = _prune_unrealistic_progression(merged)
        merged = _enforce_surah_transition_order(merged, surah_totals=surah_totals, min_gap_seconds=min_gap_seconds)
    merged = _enforce_long_ayah_inferred_floor(merged, entry_lookup=entry_lookup)
    merged = _enforce_sequential_ayah_order(merged)
    merged = sorted(merged, key=lambda marker: (marker.time, marker.surah_number or 0, marker.ayah))
    return merged
