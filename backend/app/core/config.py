"""Configuration, read from environment variables (optionally from a `.env` file).

Nothing here is hard-coded to a specific Gemini model: `GEMINI_MODEL` must be set by the user and is
recorded in every experiment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    value = _env(name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- LLM ---
    llm_provider: str = "gemini"            # "gemini" | "mock" (mock = labelled simulated LLM for offline software tests)
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_use_response_schema: bool = False  # optional API-level schema enforcement (off by default)
    gemini_thinking_budget: Optional[int] = None
    temperature: float = 0.4
    max_output_tokens: int = 8192
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 4
    llm_use_seed: bool = True
    request_delay_s: float = 0.0
    price_input_per_m: Optional[float] = None   # USD per 1M input tokens (user supplied; never guessed)
    price_output_per_m: Optional[float] = None  # USD per 1M output tokens
    # --- method defaults ---
    max_repair_rounds: int = 3
    # --- solver ---
    solver_time_limit_s: float = 10.0
    solver_workers: int = 1
    # --- storage ---
    storage_backend: str = "file"           # "file" | "mongo"
    mongodb_uri: Optional[str] = None
    mongodb_database: str = "resq_verify"
    mongo_fallback_to_file: bool = True
    data_dir: Path = PROJECT_ROOT / "data"
    prompt_dir: Path = PROJECT_ROOT / "prompts"
    # --- servers ---
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    backend_url: str = "http://127.0.0.1:8000"

    @classmethod
    def from_env(cls) -> "Settings":
        host = _env("BACKEND_HOST", "127.0.0.1")
        port = _env_int("BACKEND_PORT", 8000)
        return cls(
            llm_provider=(_env("LLM_PROVIDER", "gemini") or "gemini").lower(),
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL"),
            gemini_api_base=_env("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
            gemini_use_response_schema=_env_bool("GEMINI_USE_RESPONSE_SCHEMA", False),
            gemini_thinking_budget=(_env_int("GEMINI_THINKING_BUDGET", -1) if _env("GEMINI_THINKING_BUDGET") else None),
            temperature=_env_float("DEFAULT_TEMPERATURE", 0.4),
            max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 8192),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", 120.0),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 4),
            llm_use_seed=_env_bool("LLM_USE_SEED", True),
            request_delay_s=_env_float("LLM_REQUEST_DELAY_S", 0.0),
            price_input_per_m=_env_float("PRICE_INPUT_PER_M_TOKENS", None),
            price_output_per_m=_env_float("PRICE_OUTPUT_PER_M_TOKENS", None),
            max_repair_rounds=_env_int("MAX_REPAIR_ROUNDS", 3),
            solver_time_limit_s=_env_float("SOLVER_TIME_LIMIT_S", 10.0),
            solver_workers=_env_int("SOLVER_WORKERS", 1),
            storage_backend=(_env("STORAGE_BACKEND", "file") or "file").lower(),
            mongodb_uri=_env("MONGODB_URI"),
            mongodb_database=_env("MONGODB_DATABASE", "resq_verify"),
            mongo_fallback_to_file=_env_bool("MONGO_FALLBACK_TO_FILE", True),
            data_dir=Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))),
            prompt_dir=Path(_env("PROMPT_DIR", str(PROJECT_ROOT / "prompts"))),
            backend_host=host,
            backend_port=port,
            backend_url=_env("BACKEND_URL", f"http://{host}:{port}"),
        )

    def public_dict(self) -> dict:
        """Settings safe to show in the UI / store in experiment metadata (no secrets)."""
        d = asdict(self)
        d["gemini_api_key"] = "set" if self.gemini_api_key else "not set"
        d["mongodb_uri"] = "set" if self.mongodb_uri else "not set"
        d["data_dir"] = str(self.data_dir)
        d["prompt_dir"] = str(self.prompt_dir)
        return d


def get_settings() -> Settings:
    """Read settings fresh from the environment (cheap, and keeps tests/monkeypatching simple)."""
    return Settings.from_env()
