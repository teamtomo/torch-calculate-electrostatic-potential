# torch-cryoesp-calculator

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20--%203.14-blue)](https://python.org)
[![CI](https://github.com/volodymyr-masalitin/torch-cryoesp-calculator/actions/workflows/ci.yml/badge.svg)](https://github.com/volodymyr-masalitin/torch-cryoesp-calculator/actions/workflows/ci.yml)

Cryo-EM Electrostatic Potential (ESP) calculator built with PyTorch. Computes 3D ESP volumes from atom coordinates using Peng 1996 scattering factors.

## Features

- **compute_volume_over_insertable_matrices**: Standard ESP volume computation with optional subvolume masks
- **compute_volume_stencil**: Optimized stencil-based computation with `torch.compile`
- **setup_fast_esp_solver**: Batch solver for multiple volumes (e.g. hypothesis scoring)

## Installation

```sh
pip install torch-cryoesp-calculator
```

Or from GitHub:

```sh
pip install git+https://github.com/volodymyr-masalitin/torch-cryoesp-calculator.git
```

### Python 3.14+

Python 3.14 requires PyTorch nightly for `torch.compile` support:

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install torch-cryoesp-calculator
```

With [uv](https://github.com/astral-sh/uv): `uv pip install torch-cryoesp-calculator` (correct PyTorch nightly selected automatically on 3.14+).

## Usage

Create an `AtomStack` from coordinates and atom names, then compute the ESP volume:

```python
import torch
from cryoesp import AtomStack, Lattice
from cryoesp.cryoesp_calculator import compute_volume_over_insertable_matrices

# Create AtomStack (coordinates in Angstroms, shape [B, N, 3])
coords = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]])  # 3 atoms
atom_names = ["C", "N", "O"]
atom_stack = AtomStack.from_coords_and_names(coords, atom_names, device="cpu")
atom_stack.fill_constant_bfactor(8 * 3.14159**2 * 0.5**2)  # B-factor in Å²

# Create lattice
lattice = Lattice.from_voxel_sizes_and_corner_points(
    voxel_sizes_in_A=(1.0, 1.0, 1.0),
    left_bottom_point_in_A=(-5.0, -5.0, -5.0),
    right_upper_point_in_A=(5.0, 5.0, 5.0),
    sublattice_radius_in_A=5.0,
    device="cpu",
)

# Compute ESP volume
volume = compute_volume_over_insertable_matrices(
    atom_stack=atom_stack,
    lattice=lattice,
    B=64,
    per_voxel_averaging=True,
    verbose=False,
)
# volume shape: (Dx, Dy, Dz)
```

## Testing

From the repo root (with the package and test deps installed):

```sh
pytest
```

or `python -m pytest`. With coverage: `pytest --cov=cryoesp --cov-report=html`. (With [uv](https://github.com/astral-sh/uv): `uv run pytest`.)

**Warnings**: You may see `DeprecationWarning: torch.jit.script_method is deprecated`. These come from PyTorch internals when `torch.compile` runs; they are not from this package and can be ignored.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (nightly for Python 3.14+)
- gemmi, numpy, tqdm

## License

BSD 3-Clause License
