"""torch-cryoesp-calculator: Cryo-EM Electrostatic Potential computation with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-cryoesp-calculator")
except PackageNotFoundError:
    __version__ = "uninstalled"
