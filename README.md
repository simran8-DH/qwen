# Qwen API

Local Qwen GGUF service — LLM brain for your chatbot / tools / PDF Q&A.

Your app keeps WhatsApp, DB, Redis, business tools. This service only runs the model.

---

## Handoff (for the next person)

This folder is **only the local LLM**. FastAPI loads `Qwen3-8B-Q4_K_M.gguf` from `C:\Models\` and serves HTTP on **port 8000**. The chatbot app calls this API; tools (policy search, WhatsApp, DB) stay in that other app.

| Item | Where |
|------|--------|
| App entry | `main.py` |
| Chat / tools | `POST /api/v1/generate` in `api/integration.py` |
| File Q&A | `api/pdf_qa.py` (prompt stuffing, **no RAG**) |
| Model wrapper | `llm/engine.py` (llama.cpp, CPU) |
| Config | `.env` → `config.py` |
| UI | `http://127.0.0.1:8000/ui` |
| Postman | `postman/Qwen_Integration_API.postman_collection.json` |

Do **not** add WhatsApp/DB here. If a file is too long, `truncated: true` — RAG was skipped on purpose.

---

## Start Qwen

Need: Python 3.12, model file at `C:\Models\Qwen3-8B-Q4_K_M.gguf` (~4.8 GB).

### First time (new PC / empty folder)

PowerShell:

```powershell
cd C:\Users\darkhorse\Desktop\LLM
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
copy .env.example .env
```

Then open `.env` and set:

```text
MODEL_PATH=C:\Models\Qwen3-8B-Q4_K_M.gguf
N_CTX=4096
N_GPU_LAYERS=0
N_THREADS=8
N_BATCH=512
MAX_TOKENS=256
HOST=0.0.0.0
PORT=8000
```

Or double-click `run.bat` (creates venv, installs, copies `.env` if missing, then starts).

### Every day (venv already exists)

```powershell
cd C:\Users\darkhorse\Desktop\LLM
.\.venv\Scripts\python.exe main.py
```

Wait until the GGUF is loaded (first start can take a minute). Keep this window open.

Check it is up:

```powershell
curl http://127.0.0.1:8000/api/v1/health
```

`model_loaded` should be `true`.

| Who calls | URL |
|-----------|-----|
| Same PC | `http://127.0.0.1:8000` |
| Browser UI | `http://127.0.0.1:8000/ui` |
| Swagger | `http://127.0.0.1:8000/docs` |
| Other PC on LAN | `http://192.168.65.11:8000` (include **:8000**) |

Postman: `postman/Qwen_Integration_API.postman_collection.json`

Upload a file, type a question, click **Ask**.  
It calls `/file-to-base64` then `/file-qa-base64` and shows the answer below.

Stop: `Ctrl+C` in the terminal that is running `main.py`.

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | Health + model loaded? |
| POST | `/api/v1/generate` | Chat / tool routing (JSON) — **tools list still supported** |
| POST | `/api/v1/file-to-base64` | Upload file → `{ filename, file_base64 }` |
| POST | `/api/v1/file-qa-base64` | Q&A with `{ filename, file_base64, question }` |
| POST | `/api/v1/file-qa-form` | One-step form-data (file + question) |

Removed: `/files` and `/file-qa` (fileId flow).

---

## Base64 file flow

```text
1) POST /api/v1/file-to-base64
   form-data: file = your.pdf
   → { "filename": "your.pdf", "file_base64": "...", "size_bytes": N }

2) POST /api/v1/file-qa-base64
   {
     "filename": "your.pdf",
     "file_base64": "...",
     "question": "What is the Size of Order?",
     "max_tokens": 256
   }
```

In production, your app should base64-encode in code (no hand-paste). JSON body often passes firewalls better than multipart file upload.

---

## How File Q&A works (no RAG)

```text
file bytes → extract text → text + question in one prompt → Qwen answers once
```

- `truncated: false` → full text fit in `N_CTX`
- `truncated: true` → only the beginning was sent

---

## `/generate` (chat / tools)

```json
POST /api/v1/generate
{
  "messages": [{ "role": "user", "content": "look up policy 03511187" }],
  "tools": [
    {
      "name": "policy_search",
      "description": "Search by policy number",
      "input_schema": {
        "type": "object",
        "properties": { "policy_number": { "type": "string" } },
        "required": ["policy_number"]
      }
    }
  ]
}
```

- `tools: false` → text reply  
- `tools: [ {...} ]` → may return `tool_use` (list, not `tools: true`)

---

## Tips

- Always use port **`:8000`**
- CPU Qwen can take 1–2+ minutes — use a long client timeout
- `max_tokens` = answer tokens, not words
- Scanned/image PDFs need OCR — not supported
