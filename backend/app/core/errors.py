"""Application-level errors with stable codes so the API and UI can react to them."""
from __future__ import annotations

from typing import Any, Optional


class ResQError(Exception):
    """Base class. `code` is machine-readable, `http_status` is used by the API layer."""

    code = "RESQ_ERROR"
    http_status = 500

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class NotFoundError(ResQError):
    code = "NOT_FOUND"
    http_status = 404


class BadRequestError(ResQError):
    code = "BAD_REQUEST"
    http_status = 400


class DuplicateScenarioError(ResQError):
    code = "DUPLICATE_SCENARIO"
    http_status = 409


class LLMNotConfiguredError(ResQError):
    code = "LLM_NOT_CONFIGURED"
    http_status = 503


class LLMCallError(ResQError):
    """The LLM could not be reached or refused to answer (after retries)."""

    code = "LLM_CALL_FAILED"
    http_status = 502

    def __init__(self, message: str, *, retry_count: int = 0, errors: Optional[list[str]] = None,
                 latency_s: float = 0.0, details: Optional[dict[str, Any]] = None):
        super().__init__(message, details=details)
        self.retry_count = retry_count
        self.errors = errors or [message]
        self.latency_s = latency_s


class SolverError(ResQError):
    code = "SOLVER_ERROR"
    http_status = 500


class StorageError(ResQError):
    code = "STORAGE_ERROR"
    http_status = 500


class ExperimentStateError(ResQError):
    code = "EXPERIMENT_STATE"
    http_status = 409
