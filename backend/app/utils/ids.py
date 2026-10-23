from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_experiment_id() -> str:
    return "EXP-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def new_run_id() -> str:
    return "RUN-" + uuid.uuid4().hex[:12]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def derive_seed(*parts: Any) -> int:
    """Deterministic 31-bit integer from arbitrary parts (used for per-call LLM seeds)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF
