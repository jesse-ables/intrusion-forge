from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.core.utils import save_to_json


def load_config(
    *,
    config_path: str | Path = "configs",
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load and return a DictConfig using Hydra's Compose API.

    Args:
        config_path: Path to the configuration folder. Can be:
            - Relative path from CWD (e.g., "configs", "../configs")
            - Absolute path (e.g., "/Users/user/project/configs")
        config_name: Name of the main configuration file (without .yaml extension)
        overrides: List of overrides like ["db.user=admin", "db.port=1234"]

    Returns:
        DictConfig: The composed configuration
    """
    overrides = overrides or []

    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = Path.cwd() / config_path

    config_dir = config_dir.resolve()

    if not config_dir.exists():
        raise ValueError(
            f"Configuration directory does not exist: {config_dir}\n"
            f"Current working directory: {Path.cwd()}"
        )

    # Use initialize_config_dir for absolute paths
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    cfg = DictConfig(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist the fully-resolved config to a JSON file."""
    save_to_json(
        OmegaConf.to_container(cfg, resolve=True),
        path,
    )


def to_container(cfg) -> dict:
    """Convert an OmegaConf config (or sub-node) to plain Python types."""
    return OmegaConf.to_container(cfg, resolve=True)
