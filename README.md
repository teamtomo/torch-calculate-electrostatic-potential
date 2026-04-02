# torch-calculate-electrostatic-potential

[![License](https://img.shields.io/github/license/teamtomo/torch-calculate-electrostatic-potential?style=flat-square)](https://github.com/teamtomo/torch-calculate-electrostatic-potential/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20--%203.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/teamtomo/torch-calculate-electrostatic-potential/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/teamtomo/torch-calculate-electrostatic-potential/actions/workflows/ci.yml)

Cryo-EM Electrostatic Potential (ESP) calculator built with PyTorch. Computes 3D ESP volumes from atom coordinates using Peng 1996 scattering factors. Uses efficient per-atom computations and CUDA-optimized stencil-based compiled kernels.

## Features

- **calculate_esp**: Dense ESP computation. Backprop through this path is memory-heavy; use the stencil path when you need gradients of large structures and/or volumes. Complexity is $O(N_{\text{atoms}} \cdot D_x D_y D_z)$, where the dimensions $D_x$, $D_y$, $D_z$ are the dimensions of the per-atom insertable matrix—the local region where the forward model for each atom is evaluated and inserted (`scatter_add`) into the big output volume.
- **calculate_esp_stencil_compiled**: Compiled stencil-based ESP (`torch.compile`). Designed for backprop VRAM optimization.
- **setup_batch_esp_calculator**: Builds a callable for multiple ESP volumes. Returns `(calculate_esp_batch, calculate_esp_batch_from_coords)`, where `calculate_esp_batch` takes a list of `AtomStack` objects and `calculate_esp_batch_from_coords` works directly from coordinate tensors. Preferred when the forward pass is run many times over the same type and size of structures (e.g. density alignment).
- **Lattice**: Defines the metaparameters of the resulting volume—resolution (`voxel_sizes_in_A`) and spatial extent (corner points). The parameter `sublattice_radius_in_A` (Å) affects computational cost the most: it sets the dimensions $D_x$, $D_y$, $D_z$  of the per-atom insertable matrix. Increase it when B-factors are high or voxel sizes are low.

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
from espcalculator import AtomStack, Lattice, calculate_esp

# Create AtomStack (coordinates in Angstroms, shape [B, N, 3])
coords = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]])  # 3 atoms
atom_names = ["C", "N", "O"]
atom_stack = AtomStack.from_coords_and_names(coords, atom_names, device="cpu")
atom_stack.fill_constant_bfactor(8 * 3.14159**2 * 0.5**2)  # B-factor in Å²

# Create lattice. sublattice_radius_in_A is the main complexity lever: larger = more voxels per atom.
# It should scale with B-factors and voxel size (e.g. larger radius for larger B or finer voxels).
# One value in Å for all dimensions; per-axis voxel counts are derived from voxel_sizes_in_A.
lattice = Lattice.from_voxel_sizes_and_corner_points(
    voxel_sizes_in_A=(1.0, 1.0, 1.0),
    left_bottom_point_in_A=(-5.0, -5.0, -5.0),
    right_upper_point_in_A=(5.0, 5.0, 5.0),
    sublattice_radius_in_A=5.0,
    device="cpu",
)

# Compute ESP volume (dense path)
esp_volume = calculate_esp(
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
from espcalculator import calculate_esp_stencil_compiled

# Re-use atom_stack and lattice from above
esp_volume = calculate_esp_stencil_compiled(
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
from espcalculator import setup_batch_esp_calculator

# Re-use atom_stack and lattice from above
calculate_esp_batch, calculate_esp_batch_from_coords = setup_batch_esp_calculator(
    atom_stack=atom_stack,
    lattice=lattice,
    per_voxel_averaging=True,
)

# Example: two hypotheses with the same structure
volumes = calculate_esp_batch([atom_stack, atom_stack])
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
