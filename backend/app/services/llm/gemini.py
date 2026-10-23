"""Gemini client over the public REST API (`generateContent`) using `requests`.

Why REST and not an SDK: fewer dependencies for a student install, and every request/response field we need for
reproducibility (model version, token counts, finish reason) is explicit.

The API key is sent in the `x-goog-api-key` header (never in the URL) and is stripped from any error text.
"""
from __future__ import annotations

import random
import time
from typing import Any, Optional

import requests

from backend.app.core.config import Settings
from backend.app.core.errors import LLMCallError, LLMNotConfiguredError
from backend.app.services.llm.base import LLMClient, LLMResult

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiClient(LLMClient):
    provider = "gemini"
    is_simulated = False

    def __init__(self, settings: Settings, *, session: Optional[requests.Session] = None, sleep=time.sleep):
        if not settings.gemini_api_key:
            raise LLMNotConfiguredError(
                "GEMINI_API_KEY is not set. Put it in your .env file (see .env.example), or set LLM_PROVIDER=mock "
                "to try the software with the clearly-labelled simulated LLM.")
        if not settings.gemini_model:
            raise LLMNotConfiguredError("GEMINI_MODEL is not set. Put the exact Gemini model name in your .env file.")
        self.settings = settings
        self.model = settings.gemini_model
        self._key = settings.gemini_api_key
        self._session = session or requests.Session()
        self._sleep = sleep

    # ------------------------------------------------------------------ helpers
    def _url(self) -> str:
        base = self.settings.gemini_api_base.rstrip("/")
        model = self.model if self.model.startswith("models/") else f"models/{self.model}"
        return f"{base}/{model}:generateContent"

    def _clean(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _payload(self, prompt, system, temperature, max_output_tokens, seed, expect_json, response_schema) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "temperature": self.settings.temperature if temperature is None else temperature,
            "maxOutputTokens": max_output_tokens or self.settings.max_output_tokens,
        }
        if expect_json:
            cfg["responseMimeType"] = "application/json"
            if response_schema is not None and self.settings.gemini_use_response_schema:
                cfg["responseSchema"] = response_schema
        if seed is not None and self.settings.llm_use_seed:
            cfg["seed"] = int(seed)
        if self.settings.gemini_thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self.settings.gemini_thinking_budget}
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @staticmethod
    def _extract(data: dict[str, Any]) -> tuple[str, Optional[str]]:
        cands = data.get("candidates") or []
        if not cands:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise ValueError(f"no candidates returned (blockReason={block})")
        cand = cands[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return text, cand.get("finishReason")

    # ------------------------------------------------------------------ main entry
    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None,
                 max_output_tokens: Optional[int] = None, seed: Optional[int] = None,
                 expect_json: bool = True, response_schema: Optional[dict] = None) -> LLMResult:
        payload = self._payload(prompt, system, temperature, max_output_tokens, seed, expect_json, response_schema)
        headers = {"x-goog-api-key": self._key, "Content-Type": "application/json"}
        errors: list[str] = []
        t0 = time.perf_counter()
        max_retries = max(0, int(self.settings.llm_max_retries))

        for attempt in range(max_retries + 1):
            retry_after: Optional[float] = None
            try:
                resp = self._session.post(self._url(), headers=headers, json=payload, timeout=self.settings.llm_timeout_s)
            except (requests.Timeout, requests.ConnectionError) as exc:
                errors.append(self._clean(f"attempt {attempt + 1}: network error: {type(exc).__name__}: {exc}"))
                resp = None
            except requests.RequestException as exc:
                raise LLMCallError(self._clean(f"request failed: {exc}"), retry_count=attempt, errors=errors,
                                   latency_s=time.perf_counter() - t0)

            if resp is not None:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        text, finish = self._extract(data)
                    except ValueError as exc:
                        # a 200 with no usable text (blocked, empty) is not retried: it would be the same answer
                        errors.append(self._clean(f"attempt {attempt + 1}: unusable response: {exc}"))
                        raise LLMCallError(errors[-1], retry_count=attempt, errors=errors, latency_s=time.perf_counter() - t0)
                    usage = data.get("usageMetadata") or {}
                    out_tok = usage.get("candidatesTokenCount")
                    thoughts = usage.get("thoughtsTokenCount")
                    return LLMResult(
                        text=text, model=self.model, model_version=data.get("modelVersion"),
                        input_tokens=usage.get("promptTokenCount"),
                        output_tokens=out_tok, thoughts_tokens=thoughts,
                        total_tokens=usage.get("totalTokenCount"),
                        latency_s=time.perf_counter() - t0, retry_count=attempt, finish_reason=finish, errors=errors)
                msg = self._error_message(resp)
                errors.append(self._clean(f"attempt {attempt + 1}: HTTP {resp.status_code}: {msg}"))
                if resp.status_code not in RETRY_STATUS:
                    raise LLMCallError(errors[-1], retry_count=attempt, errors=errors, latency_s=time.perf_counter() - t0,
                                       details={"http_status": resp.status_code})
                ra = resp.headers.get("Retry-After")
                try:
                    retry_after = float(ra) if ra else None
                except ValueError:
                    retry_after = None

            if attempt < max_retries:
                delay = retry_after if retry_after is not None else min(60.0, (2 ** attempt) + random.uniform(0, 1))
                self._sleep(min(delay, 120.0))

        raise LLMCallError(f"Gemini request failed after {max_retries + 1} attempts: {errors[-1] if errors else 'unknown error'}",
                           retry_count=max_retries, errors=errors, latency_s=time.perf_counter() - t0)

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        try:
            j = resp.json()
            err = j.get("error", {}) if isinstance(j, dict) else {}
            return f"{err.get('status', '')} {err.get('message', '')}".strip() or resp.text[:200]
        except ValueError:
            return resp.text[:200]
