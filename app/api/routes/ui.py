from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["ui"])

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@router.get("/")
def ui_index() -> FileResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(path, media_type="text/html")
