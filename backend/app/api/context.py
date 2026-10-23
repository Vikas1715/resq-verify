from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.core.config import Settings
from backend.app.experiments.manager import ExperimentManager
from backend.app.experiments.runner import ExperimentRunner
from backend.app.services.adhoc import AdhocService
from backend.app.services.store.base import Store


@dataclass
class AppContext:
    settings: Settings
    store: Store
    store_info: dict
    runner: ExperimentRunner
    manager: ExperimentManager
    adhoc: AdhocService
