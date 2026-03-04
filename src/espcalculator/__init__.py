"""espcalculator: Cryo-EM Electrostatic Potential computation with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from .atom_stack import AtomStack
from .lattice import Lattice
from .cryoesp_calculator import (
    compute_esp,
    compute_esp_stencil_compiled,
    setup_esp_batch_calculator,
)

try:
    __version__ = version("torch-calculate-electrostatic-potential")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "AtomStack",
    "Lattice",
    "compute_esp",
    "compute_esp_stencil_compiled",
    "setup_esp_batch_calculator",
    "__version__",
]
