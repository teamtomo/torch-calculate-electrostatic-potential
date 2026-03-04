# torch-calculate-electrostatic-potential

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20--%203.14-blue)](https://python.org)
[![CI](https://github.com/teamtomo/torch-calculate-electrostatic-potential/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-calculate-electrostatic-potential/actions/workflows/ci.yml)

Cryo-EM Electrostatic Potential (ESP) calculator built with PyTorch. Computes 3D ESP volumes from atom coordinates using Peng 1996 scattering factors.

## Features

- **compute_esp**: Dense ESP computation. Backprop through this path is memory-heavy; use the stencil path when you need gradients. Complexity is $O(N_{\text{atoms}} \cdot D_x D_y D_z)$, where $D_x$, $D_y$, $D_z$ are the side lengths of the insertable matrix (sublattice).
- **compute_esp_stencil_compiled**: Compiled stencil-based ESP (`torch.compile`). Designed for backprop VRAM optimization.
- **setup_esp_batch_calculator**: Builds a callable for multiple ESP volumes. Returns `(compute_batch, compute_batch_from_coords)`, where `compute_batch` takes a list of `AtomStack` objects and `compute_batch_from_coords` works directly from coordinate tensors. Preferred when the forward pass is run many times over the same type and size of structures (e.g. density alignment).

Use the stencil / batch path when you need gradients; the dense path is memory-heavy for backprop.

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

Create an `AtomStack` from coordinates and atom names, then compute a single ESP volume:

```python
import torch
from espcalculator import AtomStack, Lattice, compute_esp

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

# Compute ESP volume (dense path)
esp_volume = compute_esp(
    atom_stack=atom_stack,
    lattice=lattice,
    B=64,
    per_voxel_averaging=True,
    verbose=False,
)
# esp_volume shape: (Dx, Dy, Dz)
```

Single volume with the stencil-compiled path (backprop-friendly):

```python
from espcalculator import compute_esp_stencil_compiled

# Re-use atom_stack and lattice from above
esp_volume = compute_esp_stencil_compiled(
    atom_stack=atom_stack,
    lattice=lattice,
    B=4096,
    per_voxel_averaging=True,
    verbose=False,
)
# esp_volume shape: (Dx, Dy, Dz)
```

For multiple ESP volumes in one call (batch calculator):

```python
from espcalculator import setup_esp_batch_calculator

# Re-use atom_stack and lattice from above
compute_batch, compute_batch_from_coords = setup_esp_batch_calculator(
    atom_stack=atom_stack,
    lattice=lattice,
    per_voxel_averaging=True,
)

# Example: two hypotheses with the same structure
volumes = compute_batch([atom_stack, atom_stack])
# volumes shape: (2, Dx, Dy, Dz)
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
