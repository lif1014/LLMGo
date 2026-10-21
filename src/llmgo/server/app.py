"""Read-only autonomous chess research dashboard, including streamed game replay."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .rsi import router

STATIC = Path(__file__).resolve().parent/"static"
app = FastAPI(title="LLMGo autonomous chess research")
app.include_router(router)
app.mount("/static",StaticFiles(directory=STATIC),name="static")

@app.get("/")
@app.get("/rsi")
def index():
    return FileResponse(STATIC/"rsi.html")
