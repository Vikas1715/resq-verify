from __future__ import annotations

import logging

from backend.app.core.config import Settings
from backend.app.core.errors import StorageError
from backend.app.services.store.base import Store
from backend.app.services.store.file_store import FileStore
from backend.app.services.store.mongo_store import MongoStore

log = logging.getLogger("resq.store")


def make_store(settings: Settings) -> tuple[Store, dict]:
    """Return (store, info). With STORAGE_BACKEND=mongo and MongoDB down, fall back to files (if allowed)."""
    files = FileStore(settings.data_dir)
    if settings.storage_backend != "mongo":
        return files, {"requested": settings.storage_backend, "active": "file", "note": ""}
    if not settings.mongodb_uri:
        msg = "STORAGE_BACKEND=mongo but MONGODB_URI is not set"
    else:
        try:
            return MongoStore.connect(settings.mongodb_uri, settings.mongodb_database, mirror=files), \
                {"requested": "mongo", "active": "mongo", "note": ""}
        except StorageError as exc:
            msg = exc.message
    if settings.mongo_fallback_to_file:
        log.warning("MongoDB unavailable (%s). Falling back to file storage.", msg)
        return files, {"requested": "mongo", "active": "file", "note": f"MongoDB unavailable, using files: {msg}"}
    raise StorageError(msg)
