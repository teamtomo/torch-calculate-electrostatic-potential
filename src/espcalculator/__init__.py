"""espcalculator: public API for the ESP calculator."""

from importlib.metadata import PackageNotFoundError, version

from cryoesp.atom_stack import AtomStack
from cryoesp.lattice import Lattice
from cryoesp.cryoesp_calculator import (
    compute_volume_over_insertable_matrices,
    compute_volume_stencil,
    setup_fast_esp_solver,
)

try:
    __version__ = version("torch-calculate-electrostatic-potential")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "AtomStack",
    "Lattice",
    "compute_volume_over_insertable_matrices",
    "compute_volume_stencil",
    "setup_fast_esp_solver",
    "__version__",
]

