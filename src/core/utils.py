import functools
import json
import logging
import math
import pickle
import joblib
import time
from pathlib import Path
from typing import Iterable


def _nan_to_none(obj: object) -> object:
    """Recursively replace float NaN with None for JSON serialization."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def save_to_json(data: object, file_path: str | Path) -> None:
    """Save data to a JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(_nan_to_none(data), f, indent=4)


def load_from_json(file_path: str | Path) -> object:
    """Load data from a JSON file."""
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_to_pickle(data: object, file_path: str | Path) -> None:
    """Save data to a pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(file_path: str | Path) -> object:
    """Load data from a pickle file."""
    file_path = Path(file_path)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_to_joblib(data: object, file_path: str | Path) -> None:
    """Save data (typically a sklearn estimator) via joblib."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, file_path)


def load_from_joblib(file_path: str | Path) -> object:
    """Load data previously written with save_to_joblib."""
    return joblib.load(Path(file_path))


_TIMING_RECORDS: list[dict] = []


def timed(fn):
    """Measure wall-clock execution time of fn, log elapsed time, and record it."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        output = fn(*args, **kwargs)
        elapsed_s = time.perf_counter() - t0
        logging.getLogger(fn.__module__).info(
            "%s completed in %.2f s", fn.__qualname__, elapsed_s
        )
        _TIMING_RECORDS.append({"function": fn.__qualname__, "duration_s": elapsed_s})
        return output

    return wrapper


def flush_timing(path: str | Path) -> None:
    """Append accumulated timing records to JSON file and clear the in-memory list."""
    path = Path(path)
    existing = load_from_json(path) if path.exists() else []
    save_to_json(existing + _TIMING_RECORDS, path)
    _TIMING_RECORDS.clear()


def skip_if_exists(
    markers: Path | Iterable[Path], force: bool, stage_name: str
) -> bool:
    """Return True (and log) when all marker paths exist and force is False."""
    if force:
        return False
    paths = [markers] if isinstance(markers, Path) else list(markers)
    if paths and all(Path(p).exists() for p in paths):
        logging.getLogger(__name__).info(
            "Skipping %s — outputs present (force=true to recompute).", stage_name
        )
        return True
    return False
