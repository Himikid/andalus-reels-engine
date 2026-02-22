# andalus-reels-engine

FastAPI local backend for generating vertical Taraweeh reels from synced ayah markers.

## What this scaffold includes

- Filesystem-only storage (`data/markers`, `data/drafts`, no DB)
- Single worker background jobs (generate + final render)
- Deterministic draft dedupe by request hash
- Draft state machine and metadata persistence
- Marker sync endpoint for external website repo push
- Subtitle map update API (JSON payload, no file upload)
- Artifact streaming endpoint
- Built-in browser UI at `/` for full local workflow
- UI auto-estimates clip start/duration/source from synced markers (no manual marker JSON paste)
- Docker and docker-compose

## API

- `GET /health`
- `GET /` (local UI)
- `POST /markers/sync`
- `POST /draft/generate`
- `POST /draft/update-subtitles`
- `POST /render/final`
- `GET /draft/{draft_id}`
- `GET /draft/{draft_id}/artifact/{artifact_name}`

## Draft layout

`data/drafts/{draft_id}/`

- `metadata.json`
- `draft.mp4`
- `audio.wav`
- `subtitle_map.json`
- `subtitle_map_edited.json`
- `final.mp4`

## Status model

`created -> generating -> ready_for_edit -> rendering_final -> completed`

and terminal `failed`.

## Deduplication

`POST /draft/generate` computes request hash from:

- `day`
- `surah_number`
- `ayah_start`
- `ayah_end`
- `clip_start`
- `duration`

If hash exists, server returns the existing draft.

## Run locally

```bash
cd /Users/himi/Documents/firebase/andalus-reels-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
```

Open:

- UI: `http://localhost:8090/`
- API docs: `http://localhost:8090/docs`

## Run with Docker

```bash
docker compose up --build
```

Open:

- UI: `http://localhost:8090/`
- API docs: `http://localhost:8090/docs`

## Example marker sync

```bash
curl -X POST http://localhost:8090/markers/sync \
  -H "Content-Type: application/json" \
  -d '{
    "day": 4,
    "source_url": "https://youtube.com/watch?v=QxnylahNG_U",
    "full_refresh": true,
    "markers": [
      {"time": 1530, "surah": "Aal-i-Imran", "ayah": 92, "surah_number": 3}
    ]
  }'
```

## Notes

- UI is intentionally minimal and local-first; no separate frontend build step.
- Pipeline first tries `scripts/make_reel.py`; if unavailable or failing, it falls back to ffmpeg-based preview/final rendering.
- Final renders are immutable by draft, and can be cleaned up externally by client policy.
