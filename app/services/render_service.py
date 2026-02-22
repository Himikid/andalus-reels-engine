import json
import subprocess
from pathlib import Path

from app.core.storage import ensure_dir


class RenderService:
    def __init__(self) -> None:
        self.make_reel_script = Path("scripts/make_reel.py")

    def try_generate_with_make_reel(
        self,
        *,
        day: int,
        surah_number: int,
        ayah_start: int,
        ayah_end: int,
        clip_start: float,
        duration: float,
        sheikh: str | None,
        youtube_url: str | None,
        source_video_path: Path | None,
        draft_video_out: Path,
        subtitle_map_out: Path,
        align_subtitles: bool,
    ) -> bool:
        if not self.make_reel_script.exists():
            return False

        ensure_dir(draft_video_out.parent)
        cmd = [
            "python3",
            str(self.make_reel_script),
            "--day",
            str(day),
            "--surah-number",
            str(surah_number),
            "--ayah",
            str(ayah_start),
            "--ayah-end",
            str(ayah_end),
            "--start",
            str(clip_start),
            "--duration",
            str(duration),
            "--sheikh",
            sheikh or "Sheikh",
            "--variants",
            "clean",
            "--style",
            "fit",
            "--output",
            str(draft_video_out),
            "--subtitle-map-output",
            str(subtitle_map_out),
        ]
        if align_subtitles:
            cmd.append("--align-subtitles")
        if source_video_path:
            cmd.extend(["--video-file", str(source_video_path)])
        elif youtube_url:
            cmd.extend(["--youtube-url", youtube_url])
        else:
            return False

        try:
            subprocess.run(cmd, check=True)
        except Exception:
            return False
        return draft_video_out.exists() and subtitle_map_out.exists()

    def generate_preview(self, video_path: Path, output_path: Path, clip_start: float, duration: float) -> Path:
        ensure_dir(output_path.parent)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(clip_start),
            "-t",
            str(duration),
            "-i",
            str(video_path),
            "-vf",
            "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def render_final_with_subtitles(self, video_path: Path, subtitle_map_path: Path, output_path: Path, clip_start: float, duration: float) -> Path:
        ensure_dir(output_path.parent)
        chunks = json.loads(subtitle_map_path.read_text(encoding="utf-8")).get("chunks", [])

        # Current scaffold keeps rendering deterministic and simple: high-quality clip render.
        # Subtitle burn-in step can be switched to make_reel service in next iteration.
        _ = chunks
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(clip_start),
            "-t",
            str(duration),
            "-i",
            str(video_path),
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-preset",
            "slow",
            "-crf",
            "20",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path
