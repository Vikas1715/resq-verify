"""Shared fixtures. No test calls a real LLM or needs MongoDB; the LLM is a scripted fake or the labelled mock."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import Settings  # noqa: E402


@pytest.fixture()
def settings(tmp_path) -> Settings:
    base = Settings.from_env()
    return dataclasses.replace(base, llm_provider="mock", gemini_api_key=None, data_dir=tmp_path / "data", storage_backend="file",
                               request_delay_s=0.0, solver_time_limit_s=5, price_input_per_m=None, price_output_per_m=None)
