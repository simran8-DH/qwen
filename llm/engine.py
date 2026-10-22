import json
import re
import threading
from typing import Any, TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    from llama_cpp import Llama

_llm: "Llama | None" = None
_llm_init_lock = threading.Lock()
_llm_inference_lock = threading.Lock()

CLIENT_TOOL_SELECT_PROMPT = """You are a tool router. Pick exactly one tool from the catalog for the user's request.

CRITICAL RULES:
- Respond with ONLY one JSON object. No markdown, no explanation, no other text.
- Use ONLY tool names listed below — copy the name exactly.
- Never invent tool names.

When a tool is needed:
{"tool": "<exact_name_from_catalog>", "arguments": {<keys from that tool's schema>}}

When no listed tool applies (greetings, general knowledge only):
{"tool": "none", "arguments": {}}

Available tools (use these names exactly):
{tools}"""


def build_client_tools_prompt(tool_defs: list[dict[str, Any]]) -> str:
    if not tool_defs:
        return "(no tools provided)"

    lines: list[str] = []
    for index, tool in enumerate(tool_defs, start=1):
        name = tool.get("name") or "unknown"
        schema = tool.get("input_schema") or {}
        lines.append(
            f"{index}. {name} — {tool.get('description') or 'No description'}\n"
            f"   input_schema: {json.dumps(schema, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def get_llm() -> "Llama":
    global _llm
    if _llm is not None:
        return _llm
    with _llm_init_lock:
        if _llm is not None:
            return _llm
        from llama_cpp import Llama

        model_path = settings.model_path
        if not __import__("os").path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Set MODEL_PATH in .env to your Qwen GGUF file path."
            )
        _llm = Llama(
            model_path=model_path,
            n_ctx=settings.n_ctx,
            n_gpu_layers=settings.n_gpu_layers,
            n_threads=settings.n_threads,
            n_batch=settings.n_batch,
            n_threads_batch=settings.n_threads,
            use_mmap=True,
            use_mlock=False,
            logits_all=False,
            embedding=False,
            chat_format="chatml",
            verbose=False,
        )
    return _llm


def warmup() -> None:
    """Load GGUF once at startup so the first client request does not crash the process."""
    get_llm()
    _llm_chat("You are Qwen.", "Reply with exactly: ok", max_tokens=8, temperature=0.0)


def is_model_loaded() -> bool:
    return _llm is not None


def reset_llm() -> None:
    global _llm
    _llm = None


def _strip_thinking(text: str) -> str:
    for tag in ("think", "redacted_thinking", "reasoning"):
        text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _llm_chat(system: str, user: str, max_tokens: int = 512, temperature: float | None = None) -> str:
    with _llm_inference_lock:
        llm = get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else settings.temperature,
        )
        return _strip_thinking(response["choices"][0]["message"]["content"] or "")


def select_client_tool(question: str, tool_defs: list[dict[str, Any]], *, strict: bool = False) -> str:
    catalog = build_client_tools_prompt(tool_defs)
    system = CLIENT_TOOL_SELECT_PROMPT.replace("{tools}", catalog)
    if strict:
        system += "\n\nREMINDER: Output ONLY the JSON object. Tool name must match the catalog exactly."
    user = question if question.rstrip().endswith("/no_think") else question + "\n\n/no_think"
    # Short JSON only — keep max_tokens low for latency
    return _llm_chat(system, user, max_tokens=128, temperature=0.0)


def _allowed_client_tool_names(tool_defs: list[dict[str, Any]]) -> set[str]:
    return {str(t["name"]) for t in tool_defs if t.get("name")}


def _resolve_client_tool_name(name: str, allowed: set[str]) -> str | None:
    if not name or name == "none":
        return None
    if name in allowed:
        return name
    if "__" in name:
        suffix = name.split("__", 1)[1]
        if suffix in allowed:
            return suffix
    for candidate in allowed:
        if candidate.lower() == name.lower():
            return candidate
    return None


def normalize_client_tool_call(data: dict[str, Any], allowed: set[str]) -> dict[str, Any] | None:
    if data.get("tool") == "none" or data.get("name") == "none":
        return None

    name = data.get("tool") or data.get("name")
    resolved = _resolve_client_tool_name(str(name or ""), allowed)
    if not resolved:
        return None

    arguments = data.get("arguments")
    if arguments is None:
        arguments = data.get("input")
    if arguments is None:
        arguments = data.get("parameters")
    if not isinstance(arguments, dict):
        arguments = {}

    return {"tool": resolved, "arguments": arguments}


def parse_tool_call(text: str) -> dict[str, Any] | None:
    text = _strip_thinking(text).strip()

    def _coerce(data: dict[str, Any]) -> dict[str, Any] | None:
        if data.get("tool") == "none" or data.get("name") == "none":
            return None
        if data.get("type") == "tool_use" and data.get("name"):
            return {
                "tool": data["name"],
                "arguments": data.get("input") or data.get("arguments") or {},
            }
        if "tool" in data:
            return {
                "tool": data["tool"],
                "arguments": data.get("arguments") or data.get("input") or {},
            }
        if "name" in data and ("input" in data or "arguments" in data):
            return {
                "tool": data["name"],
                "arguments": data.get("arguments") or data.get("input") or {},
            }
        return None

    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                coerced = _coerce(data)
                if coerced:
                    return coerced
        except json.JSONDecodeError:
            pass

    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                coerced = _coerce(data)
                if coerced:
                    return coerced
        except json.JSONDecodeError:
            continue

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            data = json.loads(code_block.group(1))
            if isinstance(data, dict):
                coerced = _coerce(data)
                if coerced:
                    return coerced
        except json.JSONDecodeError:
            pass
    return None


def _try_parse_client_tool_call(text: str, allowed: set[str]) -> dict[str, Any] | None:
    parsed = parse_tool_call(text)
    if parsed:
        return normalize_client_tool_call(parsed, allowed)

    stripped = _strip_thinking(text).strip()
    if not stripped:
        return None

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            normalized = normalize_client_tool_call(data, allowed)
            if normalized:
                return normalized
            if isinstance(data.get("content"), list):
                for block in data["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        normalized = normalize_client_tool_call(block, allowed)
                        if normalized:
                            return normalized
    except json.JSONDecodeError:
        pass

    return None


def _is_explicit_no_tool(text: str) -> bool:
    """True when the model clearly chose tool=none (do not burn a second inference)."""
    stripped = _strip_thinking(text).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
        if not match:
            return False
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return False
    if not isinstance(data, dict):
        return False
    return data.get("tool") == "none" or data.get("name") == "none"


def _client_tool_selection(question: str, tool_defs: list[dict[str, Any]]) -> dict[str, Any] | None:
    allowed = _allowed_client_tool_names(tool_defs)
    if not allowed:
        return None

    raw = select_client_tool(question, tool_defs, strict=False)
    resolved = _try_parse_client_tool_call(raw, allowed)
    if resolved:
        return {**resolved, "raw": raw}
    # Explicit none → skip strict retry (was doubling latency on every greeting)
    if _is_explicit_no_tool(raw):
        return None

    raw = select_client_tool(question, tool_defs, strict=True)
    resolved = _try_parse_client_tool_call(raw, allowed)
    if resolved:
        return {**resolved, "raw": raw}
    return None


def generate(
    user: str,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    system_prompt = (system or "You are Qwen, a helpful local assistant.") + " /no_think"
    user_prompt = user if user.rstrip().endswith("/no_think") else user + "\n\n/no_think"
    return _llm_chat(
        system_prompt,
        user_prompt,
        max_tokens=max_tokens or settings.max_tokens,
        temperature=temperature,
    )


def _last_user_message(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and (msg.get("content") or "").strip():
            return msg["content"].strip()
    return ""


def _format_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").upper()
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def call_qwen_messages(
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    client_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Anthropic-like call: text or tool_use from client-supplied tool schemas."""
    question = _last_user_message(messages) or _format_messages(messages)
    if not question:
        return {"type": "text", "content": "No user message provided.", "model": "qwen-local"}

    if client_tools:
        selection = _client_tool_selection(question, client_tools)
        if selection:
            return {
                "type": "tool_use",
                "tool": selection["tool"],
                "arguments": selection.get("arguments") or {},
                "raw": selection.get("raw"),
                "model": "qwen-local",
            }

    user_prompt = _format_messages(messages) if len(messages) > 1 else question
    content = generate(
        user=user_prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"type": "text", "content": content, "model": "qwen-local"}
