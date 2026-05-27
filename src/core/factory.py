import importlib
import logging
import pkgutil
import re
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar, Type

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Factory(Generic[T]):
    """Generic factory for creating instances from configuration."""

    def __init__(self, component_type_name: str = "component"):
        self._registry: dict[str, Type[T]] = {}
        self._component_type_name = component_type_name

    def register(self, name: str | None = None) -> Callable:
        """Decorator to register a class under an optional name (defaults to snake_case)."""

        def decorator(cls: Type[T]) -> Type[T]:
            self._registry[name or _to_snake_case(cls.__name__)] = cls
            return cls

        return decorator

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def get(self, name: str) -> Type[T]:
        """Return the registered class/callable for name, without instantiation."""
        cls = self._registry.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown {self._component_type_name}: {name!r}. "
                f"Available: {sorted(self._registry)}"
            )
        return cls

    def create(self, class_name: str, params: dict | None = None) -> T:
        """Create an instance of a registered class."""
        cls = self._registry.get(class_name)
        if cls is None:
            raise ValueError(
                f"Unknown {self._component_type_name}: {class_name!r}. "
                f"Available: {sorted(self._registry)}"
            )
        return cls(**params) if params else cls()

    def create_from_list(self, names: list[str], params_list: list[dict]) -> list[T]:
        """Create multiple instances from parallel name/params lists."""
        if len(names) != len(params_list):
            raise ValueError(
                f"Length mismatch: {len(names)} names vs {len(params_list)} params"
            )
        return [self.create(name, params) for name, params in zip(names, params_list)]

    def get_available(self) -> list[str]:
        """Return sorted list of all registered type names."""
        return sorted(self._registry)

    def get_registry(self) -> dict[str, Type[T]]:
        """Return a copy of the registry."""
        return self._registry.copy()


def _to_snake_case(name: str) -> str:
    return re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    ).lower()


def discover_and_import_modules(package_path: Path, package_name: str) -> list[str]:
    """Recursively import all modules in a package, triggering decorator registration."""
    imported: list[str] = []
    for _, modname, _ in pkgutil.walk_packages(
        path=[str(package_path)], prefix=f"{package_name}."
    ):
        if "__pycache__" in modname:
            continue
        try:
            importlib.import_module(modname)
            imported.append(modname)
        except Exception as e:
            logger.warning(f"Failed to import {modname}: {e}")
    return imported
