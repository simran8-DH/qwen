from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: str = r"C:\Models\Qwen3-8B-Q4_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    n_threads: int = 8
    n_batch: int = 512
    max_tokens: int = 256
    temperature: float = 0.7

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
