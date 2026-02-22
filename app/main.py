from fastapi import FastAPI

from app.api.routes import draft, favorites, health, local_sync, markers, summaries, ui

app = FastAPI(title="andalus-reels-engine", version="0.1.0")

app.include_router(ui.router)
app.include_router(health.router)
app.include_router(markers.router)
app.include_router(summaries.router)
app.include_router(local_sync.router)
app.include_router(favorites.router)
app.include_router(draft.router)
