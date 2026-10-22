"""Integration API — POST /api/v1/generate for external chatbot apps."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from config import settings
from llm.engine import call_qwen_messages, get_llm, is_model_loaded

router = APIRouter(prefix="/api/v1", tags=["Qwen API"])


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    """
    Anthropic Messages API compatible body.

    - tools: false → text only
    - tools: [{ name, description, input_schema }, ...] → Qwen returns tool_use;
      your app runs the tool (same loop as Opus)
    """

    model: str | None = Field(default="qwen-local")
    max_tokens: int | None = Field(None, ge=1, le=8192)
    system: str | None = Field(default="You are Qwen, a helpful local assistant.")
    messages: list[ChatMessage] | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    tools: bool | list[Any] | None = Field(
        default=False,
        description="false = text only. List = your Anthropic tool schemas.",
    )
    user: str | None = Field(None, description="Legacy: single user string if messages omitted")

    @model_validator(mode="after")
    def require_messages_or_user(self) -> "GenerateRequest":
        if not self.messages and not (self.user and self.user.strip()):
            raise ValueError("Provide messages (Opus style) or user (legacy)")
        if self.tools is True:
            raise ValueError("Pass your tool schemas as a list, not tools=true")
        return self


class AnthropicContentBlock(BaseModel):
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    id: str = "msg_local"
    type: str = "message"
    role: str = "assistant"
    model: str = "qwen-local"
    content: list[AnthropicContentBlock]
    stop_reason: str
    reply: str | None = None


def _ensure_model() -> None:
    if not Path(settings.model_path).exists():
        raise HTTPException(
            503,
            f"Model not found at {settings.model_path}. Set MODEL_PATH in .env",
        )
    try:
        get_llm()
    except ModuleNotFoundError as exc:
        if "llama_cpp" in str(exc):
            raise HTTPException(
                503,
                "llama-cpp-python not installed in this Python. "
                "Start with: .venv\\Scripts\\python.exe main.py  (not system python)",
            ) from exc
        raise
    except (RuntimeError, OSError) as exc:
        raise HTTPException(503, f"LLM failed to load: {exc}") from exc


def _normalize_client_tools(tools: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        description = item.get("description") or ""
        input_schema = item.get("input_schema") or item.get("inputSchema")

        fn = item.get("function")
        if isinstance(fn, dict):
            name = name or fn.get("name")
            description = description or fn.get("description") or ""
            input_schema = input_schema or fn.get("parameters")

        if not name:
            continue
        normalized.append(
            {
                "name": str(name),
                "description": description,
                "input_schema": input_schema or {},
            }
        )
    return normalized


def _to_messages(req: GenerateRequest) -> list[dict[str, str]]:
    if req.messages:
        return [{"role": m.role, "content": m.content} for m in req.messages]
    return [{"role": "user", "content": (req.user or "").strip()}]


def _to_response(result: dict[str, Any], *, model: str) -> GenerateResponse:
    if result.get("type") == "tool_use":
        return GenerateResponse(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            model=model,
            content=[
                AnthropicContentBlock(
                    type="tool_use",
                    id=f"toolu_{uuid.uuid4().hex[:12]}",
                    name=result.get("tool") or "",
                    input=result.get("arguments") or {},
                )
            ],
            stop_reason="tool_use",
            reply=None,
        )

    text = result.get("content") or ""
    return GenerateResponse(
        id=f"msg_{uuid.uuid4().hex[:12]}",
        model=model,
        content=[AnthropicContentBlock(type="text", text=text)],
        stop_reason="end_turn",
        reply=text,
    )


@router.get("/health")
async def health() -> dict:
    model_path = Path(settings.model_path)
    model_ok = model_path.exists()
    return {
        "status": "ok" if model_ok and is_model_loaded() else ("model_missing" if not model_ok else "starting"),
        "service": "qwen-api",
        "model_path": str(model_path),
        "model_loaded": is_model_loaded(),
        "endpoints": {
            "generate": "POST /api/v1/generate — pick tool (tool_use) or text",
        },
    }


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Local Qwen brain — Opus replacement. Your app supplies tools and runs them."""
    await asyncio.to_thread(_ensure_model)

    client_tools: list[dict[str, Any]] | None = None
    if isinstance(req.tools, list):
        client_tools = _normalize_client_tools(req.tools)
        if not client_tools:
            raise HTTPException(400, "tools list must contain objects with name and input_schema")

    try:
        result = await asyncio.to_thread(
            lambda: call_qwen_messages(
                system=req.system or "You are Qwen, a helpful local assistant.",
                messages=_to_messages(req),
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                client_tools=client_tools,
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Generate failed: {exc}") from exc

    return _to_response(result, model=req.model or "qwen-local")
