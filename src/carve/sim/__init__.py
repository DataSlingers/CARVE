# Lightweight, minimal API surface and version | should modify later
__version__ = "0.1.0"

# Re-export key functions/classes for a clean top-level API
from .api import simulate_clusters

__all__ = ["__version__", "simulate_clusters"]