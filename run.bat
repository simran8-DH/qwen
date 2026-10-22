@echo off
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
if not exist .env copy .env.example .env
echo.
echo Edit .env and set MODEL_PATH to your Qwen GGUF file.
echo API: POST http://localhost:8000/api/v1/generate
echo.
.\.venv\Scripts\python.exe main.py
