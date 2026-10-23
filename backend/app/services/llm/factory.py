from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.core.errors import BadRequestError
from backend.app.services.llm.base import LLMClient
from backend.app.services.llm.gemini import GeminiClient
from backend.app.services.llm.mock import MockLLM


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "gemini":
        return GeminiClient(settings)
    if settings.llm_provider == "mock":
        return MockLLM(settings)
    raise BadRequestError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'mock'.")


def llm_status(settings: Settings) -> dict:
    """Describe the LLM configuration for /health without making any network call."""
    if settings.llm_provider == "mock":
        return {"provider": "mock", "configured": True, "is_simulated": True, "model": "simulated-llm",
                "note": "SIMULATED LLM for software testing only. Results are NOT experimental results."}
    configured = bool(settings.gemini_api_key and settings.gemini_model)
    missing = [n for n, v in (("GEMINI_API_KEY", settings.gemini_api_key), ("GEMINI_MODEL", settings.gemini_model)) if not v]
    return {"provider": settings.llm_provider, "configured": configured, "is_simulated": False,
            "model": settings.gemini_model, "missing": missing}
