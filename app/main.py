from fastapi import FastAPI

from app.api.routes import draft, health, markers

app = FastAPI(title="andalus-reels-engine", version="0.1.0")

app.include_router(health.router)
app.include_router(markers.router)
app.include_router(draft.router)
