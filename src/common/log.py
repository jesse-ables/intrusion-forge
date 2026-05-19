import logging
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from src.plot.base import Plot
from .utils import save_to_json, save_to_pickle


def setup_logger(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s: %(message)s",
    date_fmt: str = "%H:%M:%S",
    console: bool = True,
    log_file: str | None = None,
    file_level: int | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure the root logger with optional console and rolling file handlers.

    Handlers are only added once, even if you call this multiple times.
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    if console and not any(isinstance(h, RichHandler) for h in root.handlers):
        rich_handler = RichHandler(
            rich_tracebacks=True, show_time=False, show_path=False, markup=True
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(level)
        root.addHandler(rich_handler)

    if log_file:
        file_level = file_level or level
        resolved_path = str(Path(log_file).resolve())
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        existing = [
            h
            for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == resolved_path
        ]
        if not existing:
            fh = RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setFormatter(formatter)
            fh.setLevel(file_level)
            root.addHandler(fh)

    return root


@dataclass
class LogBundle:
    """Structured payload routed to subscribers.

    Each field is a dict keyed by relative path (without extension), resolved
    against the corresponding subscriber's base directory.
    Build from a prefixed-key dict via `LogBundle.from_dict`.
    """

    figures: dict[str, Plot] = field(default_factory=dict)
    json: dict[str, Any] = field(default_factory=dict)
    pickle: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LogBundle":
        """Build a LogBundle from a flat dict with type-prefixed keys.

        Recognised prefixes: 'figure/', 'json/', 'pickle/'. Keys without a
        recognised prefix are silently ignored.
        """
        figures: dict[str, Plot] = {}
        json_: dict[str, Any] = {}
        pickle_: dict[str, Any] = {}
        for key, value in d.items():
            if key.startswith("figure/"):
                figures[key[len("figure/") :]] = value
            elif key.startswith("json/"):
                json_[key[len("json/") :]] = value
            elif key.startswith("pickle/"):
                pickle_[key[len("pickle/") :]] = value
        return cls(figures=figures, json=json_, pickle=pickle_)


class LogDispatcher:
    """Routes LogBundle events to all registered subscribers."""

    def __init__(self) -> None:
        self._subscribers: list = []

    def subscribe(self, sub) -> None:
        """Register a subscriber. sub must implement on_log(bundle: LogBundle)."""
        self._subscribers.append(sub)

    def clear(self) -> None:
        """Remove all registered subscribers."""
        self._subscribers.clear()

    def publish(self, bundle: LogBundle) -> None:
        """Send bundle to all registered subscribers."""
        for sub in self._subscribers:
            sub.on_log(bundle)


class FilesystemFigureSubscriber:
    """Writes figures from LogBundle as image files under base_path / {name}.{format}."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def on_log(self, bundle: LogBundle) -> None:
        for name, plot in bundle.figures.items():
            out = self._base_path / f"{name}.{plot.format}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(plot.data)


class JSONSubscriber:
    """Saves json artifacts from LogBundle under base_path / f"{name}.json"."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def on_log(self, bundle: LogBundle) -> None:
        for name, value in bundle.json.items():
            save_to_json(value, self._base_path / f"{name}.json")


class PickleSubscriber:
    """Saves pickle artifacts from LogBundle under base_path / f"{name}.pkl"."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def on_log(self, bundle: LogBundle) -> None:
        for name, value in bundle.pickle.items():
            save_to_pickle(value, self._base_path / f"{name}.pkl")
