"""FastAPI application factory. Start with:  uvicorn backend.app.api.app:app --port 8000"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.app.api.context import AppContext
from backend.app.api.routes import router
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import ResQError
from backend.app.experiments.manager import ExperimentManager
from backend.app.experiments.runner import ExperimentRunner
from backend.app.services.adhoc import AdhocService
from backend.app.services.store.base import Store
from backend.app.services.store.factory import make_store

log = logging.getLogger("resq.api")


def build_context(settings: Settings, store: Optional[Store] = None, **runner_kw) -> AppContext:
    info = {"requested": "injected", "active": getattr(store, "backend", "injected"), "note": ""}
    if store is None:
        store, info = make_store(settings)
    runner = ExperimentRunner(store, settings, **runner_kw)
    manager = ExperimentManager(store, runner)
    manager.mark_interrupted()
    return AppContext(settings, store, info, runner, manager, AdhocService(store, runner))


def create_app(settings: Optional[Settings] = None, store: Optional[Store] = None, **runner_kw) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="ResQ-Verify API", version="1.0.0",
                  description="Verifier-augmented LLM planning for emergency resource allocation — research platform API.")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.ctx = build_context(settings, store, **runner_kw)

    @app.exception_handler(ResQError)
    async def _resq_error(_: Request, exc: ResQError):
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        errs = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_REQUEST", "message": "Request validation failed", "details": {"errors": errs}}})

    @app.exception_handler(ValidationError)
    async def _model_error(_: Request, exc: ValidationError):
        errs = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=400, content={"error": {"code": "BAD_REQUEST", "message": "Invalid configuration or data", "details": {"errors": errs}}})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}", "details": {}}})

    app.include_router(router)
    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app() if __name__ != "__main__" else None  # `uvicorn backend.app.api.app:app`
