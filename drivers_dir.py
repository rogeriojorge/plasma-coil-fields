"""Directory of the study's own drivers and helpers (``drivers/``), put first on ``sys.path``."""

from pathlib import Path


def drivers_dir() -> str:
    return str(Path(__file__).resolve().parent / "drivers")
