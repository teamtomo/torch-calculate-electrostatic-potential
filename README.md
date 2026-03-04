# torch-calculate-electrostatic-potential

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20--%203.14-blue)](https://python.org)
[![CI](https://github.com/teamtomo/torch-calculate-electrostatic-potential/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-calculate-electrostatic-potential/actions/workflows/ci.yml)

Cryo-EM Electrostatic Potential (ESP) calculator built with PyTorch. Computes 3D ESP volumes from atom coordinates using Peng 1996 scattering factors.

## Features

- **compute_volume_over_insertable_matrices**: Standard ESP volume computation with optional subvolume masks
- **compute_volume_stencil**: Optimized stencil-based computation with `torch.compile`
- **setup_fast_esp_solver**: Batch solver for multiple volumes (e.g. hypothesis scoring)

## Installation

```sh
# From GitHub (main branch)
pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
```

### Python 3.14+

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
```

With [uv](https://github.com/astral-sh/uv): `uv pip install torch-calculate-electrostatic-potential`.

## Usage

Create an `AtomStack` from coordinates and atom names, then compute the ESP volume:

```python
import torch
from espcalculator import AtomStack, Lattice, compute_volume_over_insertable_matrices

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

Install the package together with test dependencies:

```sh
pip install "torch-calculate-electrostatic-potential[test]" @ git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
pytest
```

With coverage: `pytest --cov=espcalculator --cov-report=html`.

**Warnings**: You may see `DeprecationWarning: torch.jit.script_method is deprecated`. These come from PyTorch internals when `torch.compile` runs; they are not from this package and can be ignored.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (nightly for Python 3.14+)
- gemmi, numpy, tqdm

## License

BSD 3-Clause License
