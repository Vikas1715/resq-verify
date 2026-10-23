"""Load experiment configs from YAML files (experiments/configs/*.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from backend.app.core.config import PROJECT_ROOT
from backend.app.core.errors import BadRequestError, NotFoundError
from backend.app.schemas.experiment import ExperimentConfig

CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs"


def list_config_files(config_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(p.name for p in config_dir.glob("*.y*ml")) if config_dir.exists() else []


def load_config_file(path_or_name: str | Path, config_dir: Path = CONFIG_DIR) -> ExperimentConfig:
    p = Path(path_or_name)
    if not p.is_absolute() and not p.exists():
        p = config_dir / p.name  # name only: never allow path traversal through the API
    if not p.exists():
        raise NotFoundError(f"Config file not found: {path_or_name}")
    try:
        data: Any = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise BadRequestError(f"Invalid YAML in {p.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise BadRequestError(f"{p.name} must contain a YAML mapping")
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise BadRequestError(f"Invalid experiment config in {p.name}: {exc.errors()[:3]}") from exc
