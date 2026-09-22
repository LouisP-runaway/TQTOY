"""tqtoy: a 0D multi-fluid toy model of the tokamak thermal quench."""

__version__ = "1.0.0"

from .config import Config, ConfigError  # noqa: F401
from .solver import RunResult, simulate  # noqa: F401

__all__ = ["Config", "ConfigError", "RunResult", "simulate", "__version__"]
