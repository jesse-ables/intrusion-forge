from pathlib import Path

from src.core.factory import Factory, discover_and_import_modules
from src.domain.clustering.base import ClusterFn, FitFn, grid_search

ClusteringFactory = Factory[FitFn](component_type_name="clustering_algorithm")

_package_path = Path(__file__).parent
discover_and_import_modules(package_path=_package_path, package_name=__name__)

from src.domain.clustering.compose import build_cluster_fn

__all__ = ["ClusteringFactory", "ClusterFn", "FitFn", "build_cluster_fn", "grid_search"]
