from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REELS_", extra="ignore")

    data_dir: Path = Path("data")
    markers_dir: Path = Path("data/markers")
    summaries_dir: Path = Path("data/summaries")
    drafts_dir: Path = Path("data/drafts")
    cache_videos_dir: Path = Path("data/cache/videos")
    tmp_dir: Path = Path("data/tmp")
    max_workers: int = 1


settings = Settings()
