# andalus-reels-engine

A local-first FastAPI engine for generating vertical Taraweeh reels with a clean human-in-the-loop workflow.

## At a glance

| Area | What it gives you |
| --- | --- |
| Draft workflow | Select ayat -> tune timing -> subtitle edit -> final render |
| Render quality | Uses `scripts/make_reel.py` pipeline for stylized output |
| Storage model | Filesystem only (`data/*`), no database |
| Runtime model | Single-worker, one active draft session per server |
| Sync model | Offline bundle import from website repo |
| Ops model | Docker-first, local network trusted |

## Core capabilities

- One-at-a-time draft session to keep operations simple and predictable.
- Ayah timing adjustment step before subtitle preparation.
- Subtitle row editing as structured JSON rows (no manual file uploads).
- Final render download flow with immutable artifact output per draft.
- Favorites sync + queue generation (day-based ayat/dua reel prep).
- Local cache ownership for source media resolution.

## Architecture

```mermaid
flowchart TD
    A[Website Repo Export Script] -->|sync bundle json| B[Local inbox bundle]
    B --> C[/sync/import-local]
    C --> D[data/markers]
    C --> E[data/summaries]
    C --> F[data/favorites]

    D --> G[Draft Generate]
    E --> G
    F --> H[Queue Build]

    G --> I[Ayah Timing Map]
    I --> J[Prepare Audio + Subtitle Map]
    J --> K[Subtitle Edit]
    K --> L[Render Final]
    L --> M[final.mp4]
```

## Repository layout

```text
andalus-reels-engine/
├─ app/
│  ├─ api/routes/                # FastAPI route groups
│  ├─ services/                  # Marker, draft, subtitle, render orchestration
│  ├─ models/                    # Pydantic schemas
│  └─ web/index.html             # Built-in operator UI
├─ scripts/
│  ├─ make_reel.py               # Stylized reel render pipeline
│  └─ import_sync_bundle.py      # CLI bundle import helper
├─ data/
│  ├─ markers/
│  ├─ summaries/
│  ├─ favorites/
│  ├─ queue/
│  ├─ drafts/
│  └─ cache/videos/
├─ Dockerfile
├─ docker-compose.yml
└─ requirements.txt
```

## API surface

### Health and UI

- `GET /health`
- `GET /`

### Sync and indexing

- `POST /sync/import-local`
- `POST /markers/sync`
- `GET /markers/available`
- `GET /markers/day/{day}/index`
- `POST /summaries/sync`
- `POST /favorites/sync`
- `GET /favorites`
- `POST /queue/from-favorites`
- `GET /queue/current`

### Draft lifecycle

- `POST /draft/generate`
- `GET /draft/current`
- `GET /draft/estimate`
- `POST /draft/update-ayah-timings`
- `POST /draft/prepare-subtitles`
- `POST /draft/update-subtitles`
- `POST /render/final`
- `GET /draft/{draft_id}`
- `DELETE /draft/{draft_id}`
- `DELETE /drafts/completed`
- `GET /draft/{draft_id}/artifact/{artifact_name}`

## Draft state model

```text
created -> ready_for_timing -> generating -> ready_for_edit -> rendering_final -> completed
                                 \-> failed
```

## Artifacts per draft

Each draft has its own isolated folder:

```text
data/drafts/{draft_id}/
├─ metadata.json
├─ ayah_timing_map.json
├─ ayah_timing_map_edited.json
├─ draft.mp4
├─ audio.wav
├─ subtitle_map.json
├─ subtitle_map_edited.json
└─ final.mp4
```

## Operator workflow

1. Create draft from selected day/surah/ayah range.
2. Adjust ayah timing rows (`start`, `end`).
3. Prepare draft audio + subtitle map.
4. Edit subtitle chunks.
5. Render final reel.
6. Download `final.mp4`.

## Quick start (Docker)

```bash
cd /Users/himi/Documents/firebase/andalus-reels-engine
docker compose up --build -d
```

Open:

- UI: `http://localhost:8090/`
- API docs: `http://localhost:8090/docs`

Stop:

```bash
docker compose down
```

## Quick start (local Python)

```bash
cd /Users/himi/Documents/firebase/andalus-reels-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
```

## Offline sync workflow

### 1) Export bundle from website repo

```bash
python3 /Users/himi/Documents/firebase/andalus-taraweeh/scripts/sync_reels_engine.py \
  --days 1 2 3 4 \
  --bundle-out /Users/himi/Documents/firebase/andalus-reels-engine/data/inbox/sync_bundle.json
```

### 2) Import bundle into reels engine

CLI method:

```bash
cd /Users/himi/Documents/firebase/andalus-reels-engine
python3 scripts/import_sync_bundle.py --bundle data/inbox/sync_bundle.json
```

API method:

```bash
curl -X POST http://localhost:8090/sync/import-local \
  -H "Content-Type: application/json" \
  -d '{"bundle_path":"data/inbox/sync_bundle.json"}'
```

## Queue flow for favorites

- Sync favorites from website day highlights.
- Pick day in UI (`Favorites Day`).
- Load selected day into chain.
- Generate and process as normal draft flow.

## Rendering notes

- Final render path uses `scripts/make_reel.py` for stylized output (legacy-consistent visual style).
- If `make_reel.py` fails at runtime, fallback rendering may produce plain output.
- Keep `scripts/make_reel.py` dependencies healthy (`ffmpeg`, `rapidfuzz`, `Pillow`, etc.) to preserve styled renders.

## Operational notes

- Server is local and designed for trusted local-network use.
- Laptop sleep pauses Docker containers; use an always-on machine for uninterrupted service.
- Final artifacts are immutable per draft; cleanup is explicit via delete endpoints.

## Troubleshooting

### UI shows plain video (no styled overlays)

- Confirm final render path executed `make_reel.py` successfully.
- Check container logs:

```bash
docker compose logs -f
```

### Queue appears empty

- Verify favorites exist:

```bash
curl -s http://localhost:8090/favorites
```

- Rebuild queue:

```bash
curl -X POST http://localhost:8090/queue/from-favorites \
  -H "Content-Type: application/json" \
  -d '{"days":[1,2,3,4],"include_theme_types":["Dua","Famous Ayah"],"full_refresh":true}'
```

### No active draft appears

- Drafts are single-session by design.
- If session was deleted/completed, create a new draft from UI.
