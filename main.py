"""Local Qwen API — LLM brain for chat/tools + file Q&A.

Base64 file flow:
  POST /api/v1/file-to-base64  → { file_base64 }
  POST /api/v1/file-qa-base64  → { filename, file_base64, question }

UI: GET /ui
Also: /generate (tools), /file-qa-form. See README.md.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.integration import router as integration_router
from api.pdf_qa import router as pdf_qa_router
from config import settings

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load GGUF once at startup (warmup), not per request."""
    import asyncio

    from llm.engine import warmup

    await asyncio.to_thread(warmup)
    yield


app = FastAPI(
    title="Qwen API",
    version="3.6.0",
    description="Local Qwen: /generate, /file-to-base64, /file-qa-base64, /file-qa-form, /ui",
    lifespan=lifespan,
)

app.include_router(integration_router)
app.include_router(pdf_qa_router)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root() -> dict:
    return {
        "service": "qwen-api",
        "docs": "See README.md",
        "ui": "GET /ui — simple file Q&A page",
        "health": "GET /api/v1/health",
        "generate": "POST /api/v1/generate",
        "file_to_base64": "POST /api/v1/file-to-base64",
        "file_qa_base64": "POST /api/v1/file-qa-base64",
        "file_qa_form": "POST /api/v1/file-qa-form",
    }


@app.get("/ui")
async def ui() -> FileResponse:
    """Single-page UI: upload file + question → base64 → Q&A."""
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "UI not found (static/index.html missing)")
    return FileResponse(page)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
