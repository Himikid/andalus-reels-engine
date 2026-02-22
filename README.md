# andalus-reels-engine

FastAPI local backend for generating vertical Taraweeh reels from synced ayah markers.

## What this scaffold includes

- Filesystem-only storage (`data/markers`, `data/drafts`, no DB)
- Single worker background jobs (generate + final render)
- Deterministic draft dedupe by request hash
- Draft state machine and metadata persistence
- Marker sync endpoint for external website repo push
- Local bundle import for offline sync (`/sync/import-local` + `scripts/import_sync_bundle.py`)
- Subtitle map update API (JSON payload, no file upload)
- Ayah timing adjustment step before subtitle preparation
- Chained multi-segment reels across days/surahs with transitions
- Verified subtitle memory for reused ayah subtitle work
- Artifact streaming endpoint
- Built-in browser UI at `/` for full local workflow
- UI preview is audio-only during draft stage for cleaner review
- UI includes final render download action
- Single active draft session per server instance (one-at-a-time workflow)
- Final render uses `scripts/make_reel.py` (clean + fit) with edited subtitle map input for legacy-consistent subtitle styling
- UI includes CLI-style progress bar and favorites queue launcher
- UI auto-estimates clip start/duration/source from synced markers (no manual marker JSON paste)
- Source resolution is server-owned: sync provides URL references, and engine uses its own local cache directory.
- Docker and docker-compose

## API

- `GET /health`
- `GET /` (local UI)
- `POST /markers/sync`
- `POST /sync/import-local`
- `POST /favorites/sync`
- `GET /favorites`
- `POST /queue/from-favorites`
- `GET /queue/current`
- `GET /markers/available`
- `GET /markers/day/{day}/index`
- `POST /draft/generate`
- `GET /draft/current`
- `GET /draft/estimate`
- `POST /draft/update-ayah-timings`
- `POST /draft/prepare-subtitles`
- `POST /draft/update-subtitles`
- `POST /render/final`
- `DELETE /draft/{draft_id}`
- `DELETE /drafts/completed`
- `GET /draft/{draft_id}`
- `GET /drafts`
- `GET /subtitles/verified`
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

`created -> ready_for_timing -> generating -> ready_for_edit -> rendering_final -> completed`

and terminal `failed`.

## Draft workflow

1. Generate draft request (`POST /draft/generate`) to create draft + ayah timing map.
2. Adjust ayah start/end rows in UI and save (`POST /draft/update-ayah-timings`).
3. Prepare audio/subtitles (`POST /draft/prepare-subtitles`).
4. Edit subtitle rows (`POST /draft/update-subtitles`).
5. Render final (`POST /render/final`) and download final MP4.

`POST /draft/generate` returns the current active draft if one already exists.

## Deduplication

`POST /draft/generate` computes request hash from:

- `day`
- `surah_number`
- `ayah_start`
- `ayah_end`
- `clip_start`
- `duration`

If hash exists, server returns the existing draft.

For chained requests, hash is computed from all segment keys in order.

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

## Offline sync (no cross-repo HTTP)

From website repo, export bundle:

```bash
python3 /Users/himi/Documents/firebase/andalus-taraweeh/scripts/sync_reels_engine.py \
  --days 1 2 3 4 \
  --bundle-out /Users/himi/Documents/firebase/andalus-reels-engine/data/inbox/sync_bundle.json
```

Then import bundle inside reels engine:

```bash
cd /Users/himi/Documents/firebase/andalus-reels-engine
python3 scripts/import_sync_bundle.py --bundle data/inbox/sync_bundle.json
```

Or call the local API from the same machine:

```bash
curl -X POST http://localhost:8090/sync/import-local \
  -H "Content-Type: application/json" \
  -d '{"bundle_path":"data/inbox/sync_bundle.json"}'
```

## Direct marker sync (optional)

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
