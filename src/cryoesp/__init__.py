"""torch-calculate-electrostatic-potential: Cryo-EM Electrostatic Potential computation with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-calculate-electrostatic-potential")
except PackageNotFoundError:
    __version__ = "uninstalled"
