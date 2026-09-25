"""Locate the ESSOS ``examples/coil_optimization`` directory this study runs against.

Set ``ESSOS_EXAMPLE_DIR`` explicitly, or put an ESSOS source checkout (not a
wheel, which has no examples) on ``PYTHONPATH``.
"""

import importlib.util
import os
from pathlib import Path


def essos_examples() -> str:
    explicit = os.environ.get("ESSOS_EXAMPLE_DIR")
    if explicit:
        return explicit
    spec = importlib.util.find_spec("essos")
    if spec is None or spec.origin is None:
        raise ImportError("ESSOS is not importable; set ESSOS_EXAMPLE_DIR")
    path = Path(spec.origin).resolve().parents[1] / "examples" / "coil_optimization"
    if not (path / "shafranov_shift.py").exists():
        raise ImportError(f"{path} has no ESSOS examples; set ESSOS_EXAMPLE_DIR to a checkout")
    return str(path)
