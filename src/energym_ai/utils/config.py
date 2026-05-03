from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top = ["camera", "mediapipe", "exercises", "output"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Config kekurangan field: {missing}")

    return cfg
