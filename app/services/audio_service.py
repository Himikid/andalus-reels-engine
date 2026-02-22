import re
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.storage import ensure_dir


class AudioService:
    def __init__(self) -> None:
        ensure_dir(settings.cache_videos_dir)

    def resolve_video_source(self, youtube_url: str | None) -> Path:
        if not youtube_url:
            raise RuntimeError("No synced source URL available for this segment.")

        video_id = self._youtube_id(youtube_url)
        target = settings.cache_videos_dir / f"{video_id}.mp4"
        if target.exists():
            return target

        cmd = [
            "yt-dlp",
            "-f",
            "mp4/best",
            "-o",
            str(target),
            youtube_url,
        ]
        subprocess.run(cmd, check=True)
        return target

    def extract_audio_clip(self, video_path: Path, output_wav: Path, clip_start: float, duration: float) -> Path:
        ensure_dir(output_wav.parent)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(clip_start),
            "-t",
            str(duration),
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_wav),
        ]
        subprocess.run(cmd, check=True)
        return output_wav

    def extract_full_audio(self, video_path: Path, output_wav: Path) -> Path:
        ensure_dir(output_wav.parent)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_wav),
        ]
        subprocess.run(cmd, check=True)
        return output_wav

    @staticmethod
    def _youtube_id(url: str) -> str:
        match = re.search(r"(?:v=|youtu\.be/|/live/)([A-Za-z0-9_-]{8,})", url)
        if not match:
            return str(abs(hash(url)))
        return match.group(1)
