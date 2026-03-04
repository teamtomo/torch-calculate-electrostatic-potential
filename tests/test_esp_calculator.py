"""
Tests for ESP (Electrostatic Potential) calculator functions.

Tests cover:
1. compute_volume_over_insertable_matrices (standard)
2. compute_volume_stencil (fused single volume)
3. setup_fast_esp_solver -> compute_batch (fused multiple volumes)

Uses 8OSK.pdb for tests that require real atom coordinates within lattice bounds.
Same lattice parameters as original: pixel_size 0.845, padding 10, sublattice_radius 5.0.

Note on warnings: pytest may show DeprecationWarning about `torch.jit.script_method`.
These come from PyTorch internals when torch.compile runs; they are not from this package.
"""

import torch
from pathlib import Path

import gemmi

from espcalculator import (
    AtomStack,
    Lattice,
    compute_volume_over_insertable_matrices,
    compute_volume_stencil,
    setup_fast_esp_solver,
)


# ---------------------------------------------------------------------------
# Test data setup - PDB loading helper (same logic as from_pdb_file)
# ---------------------------------------------------------------------------

def get_test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "examples" / "data"


def _load_atom_stack_from_pdb(pdb_file: Path, device: str = "cpu") -> AtomStack:
    """
    Load AtomStack from PDB file using gemmi (same logic as original from_pdb_file).
    PDB files are already in Angstroms.
    """
    pdb_struc = gemmi.read_pdb(str(pdb_file))
    model = pdb_struc[0]
    chains = [chain for chain in model]

    atom_coordinates = torch.tensor(
        [atom.pos.tolist() for chain in chains for residue in chain for atom in residue],
        device=device,
    ).unsqueeze(0)  # Shape (1, N, 3)
    atom_names = [atom.name[0] for chain in chains for residue in chain for atom in residue]
    bfactors = torch.tensor(
        [atom.b_iso for chain in chains for residue in chain for atom in residue],
        device=device,
    ).unsqueeze(0).unsqueeze(-1)  # Shape (1, N, 1)

    return AtomStack(
        atom_coordinates=atom_coordinates,
        atom_names=atom_names,
        bfactors=bfactors,
        device=device,
    )


def get_small_atom_stack():
    """Load first 30 atoms from 8OSK.pdb for fast CPU tests. Same as original."""
    test_data_dir = get_test_data_dir()
    pdb_file = test_data_dir / "8OSK.pdb"
    if not pdb_file.exists():
        raise FileNotFoundError(
            "Test data required: examples/data/8OSK.pdb not found. "
            "Provide this file to run tests (e.g. copy from parent repo)."
        )

    atom_stack = _load_atom_stack_from_pdb(pdb_file, device="cpu")
    N = min(30, atom_stack.atom_coordinates.shape[1])
    atom_stack = AtomStack(
        atom_coordinates=atom_stack.atom_coordinates[:, :N, :],
        atom_names=atom_stack.atom_names[:N],
        bfactors=atom_stack.bfactors[:, :N, :] if atom_stack.bfactors is not None else None,
        device="cpu",
        occupancies=atom_stack.occupancies,
    )

    if atom_stack.bfactors is None:
        atomic_radius = 0.5  # Angstroms
        bfactor = 8 * torch.pi**2 * atomic_radius**2
        atom_stack.fill_constant_bfactor(bfactor)

    return atom_stack


def get_small_lattice(atom_stack):
    """Create lattice that fits the atom coordinates. Same params as original: pixel_size 0.845, padding 10, sublattice_radius 5.0."""
    coords = atom_stack.atom_coordinates[0]
    min_coords = coords.min(dim=0)[0]
    max_coords = coords.max(dim=0)[0]
    padding = 10.0
    left_bottom = (min_coords - padding).tolist()
    right_upper = (max_coords + padding).tolist()
    pixel_size = 0.8450000277777778  # Angstroms per voxel (standard from 8OSK structure)
    sublattice_radius = 5.0

    lattice = Lattice.from_voxel_sizes_and_corner_points(
        voxel_sizes_in_A=(pixel_size, pixel_size, pixel_size),
        left_bottom_point_in_A=left_bottom,
        right_upper_point_in_A=right_upper,
        sublattice_radius_in_A=sublattice_radius,
        dtype=torch.float32,
        device="cpu",
    )
    lattice._initialize_lattice_coordinates()
    return lattice


# ---------------------------------------------------------------------------
# TestESPCalculatorStandard
# ---------------------------------------------------------------------------

class TestESPCalculatorStandard:
    """Tests for compute_volume_over_insertable_matrices (standard function)."""

    def test_single_batch_no_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        volume = compute_volume_over_insertable_matrices(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.dtype == torch.float32
        assert torch.all(volume >= 0.0)
        assert volume.sum() > 0

    def test_single_batch_with_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        small_atom_stack.occupancies = torch.tensor([1.0], device="cpu")
        volume = compute_volume_over_insertable_matrices(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.sum() > 0

    def test_multiple_batches_no_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        batched_stack = small_atom_stack.replicate_ensemble(B=3)
        volume = compute_volume_over_insertable_matrices(
            atom_stack=batched_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.sum() > 0

    def test_multiple_batches_with_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        batched_stack = small_atom_stack.replicate_ensemble(B=3)
        batched_stack.occupancies = torch.tensor([0.5, 0.3, 0.2], device="cpu")
        volume = compute_volume_over_insertable_matrices(
            atom_stack=batched_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.sum() > 0

    def test_per_voxel_averaging_vs_point_sampling(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        volume_avg = compute_volume_over_insertable_matrices(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        volume_point = compute_volume_over_insertable_matrices(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=False,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume_avg.shape == volume_point.shape
        assert not torch.allclose(volume_avg, volume_point, atol=1e-6)
        assert volume_avg.sum() > 0
        assert volume_point.sum() > 0

    def test_empty_atom_stack(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        empty_stack = AtomStack(
            atom_coordinates=torch.zeros((1, 0, 3), dtype=torch.float32),
            atom_names=[],
            bfactors=None,
            device="cpu",
        )
        volume = compute_volume_over_insertable_matrices(
            atom_stack=empty_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert torch.allclose(volume, torch.zeros_like(volume))


# ---------------------------------------------------------------------------
# TestESPCalculatorFusedSingle
# ---------------------------------------------------------------------------

class TestESPCalculatorFusedSingle:
    """Tests for compute_volume_stencil (fused single volume)."""

    def test_single_batch_no_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        volume = compute_volume_stencil(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.dtype == torch.float32
        assert torch.all(volume >= 0.0)
        assert volume.sum() > 0

    def test_single_batch_with_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        small_atom_stack.occupancies = torch.tensor([1.0], device="cpu")
        volume = compute_volume_stencil(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.sum() > 0

    def test_multiple_batches_no_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        batched_stack = small_atom_stack.replicate_ensemble(B=3)
        volume = compute_volume_stencil(
            atom_stack=batched_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.sum() > 0

    def test_multiple_batches_with_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        batched_stack = small_atom_stack.replicate_ensemble(B=3)
        batched_stack.occupancies = torch.tensor([0.5, 0.3, 0.2], device="cpu")
        volume = compute_volume_stencil(
            atom_stack=batched_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume.shape == tuple(small_lattice.grid_dimensions.tolist())
        assert volume.sum() > 0

    def test_per_voxel_averaging_vs_point_sampling(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        volume_avg = compute_volume_stencil(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        volume_point = compute_volume_stencil(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=False,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume_avg.shape == volume_point.shape
        assert not torch.allclose(volume_avg, volume_point, atol=1e-6)
        assert volume_avg.sum() > 0
        assert volume_point.sum() > 0

    def test_subvolume_mask_not_supported(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        mask = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        try:
            compute_volume_stencil(
                atom_stack=small_atom_stack,
                lattice=small_lattice,
                B=32,
                per_voxel_averaging=True,
                subvolume_mask_in_indices=mask,
                verbose=False,
            )
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass


# ---------------------------------------------------------------------------
# TestESPCalculatorFusedMultiple
# ---------------------------------------------------------------------------

class TestESPCalculatorFusedMultiple:
    """Tests for setup_fast_esp_solver -> compute_batch (fused multiple volumes)."""

    def test_single_volume_no_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        compute_batch, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes = compute_batch([small_atom_stack])
        assert volumes.shape == (1,) + tuple(small_lattice.grid_dimensions.tolist())
        assert volumes.dtype == torch.float32
        assert torch.all(volumes >= 0.0)
        assert volumes[0].sum() > 0

    def test_single_volume_with_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        small_atom_stack.occupancies = torch.tensor([1.0], device="cpu")
        compute_batch, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes = compute_batch([small_atom_stack])
        assert volumes.shape == (1,) + tuple(small_lattice.grid_dimensions.tolist())
        assert volumes[0].sum() > 0

    def test_multiple_volumes_same_structure(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        stack1 = AtomStack(
            atom_coordinates=small_atom_stack.atom_coordinates.clone(),
            atom_names=small_atom_stack.atom_names.copy(),
            bfactors=small_atom_stack.bfactors.clone() if small_atom_stack.bfactors is not None else None,
            device="cpu",
            occupancies=torch.tensor([1.0], device="cpu"),
        )
        stack2 = AtomStack(
            atom_coordinates=small_atom_stack.atom_coordinates.clone(),
            atom_names=small_atom_stack.atom_names.copy(),
            bfactors=small_atom_stack.bfactors.clone() if small_atom_stack.bfactors is not None else None,
            device="cpu",
            occupancies=torch.tensor([1.0], device="cpu"),
        )
        compute_batch, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes = compute_batch([stack1, stack2])
        assert volumes.shape == (2,) + tuple(small_lattice.grid_dimensions.tolist())
        assert torch.allclose(volumes[0], volumes[1], atol=1e-5)

    def test_multiple_volumes_with_batches(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        batched_stack1 = small_atom_stack.replicate_ensemble(B=2)
        batched_stack1.occupancies = torch.tensor([0.6, 0.4], device="cpu")
        batched_stack2 = small_atom_stack.replicate_ensemble(B=2)
        batched_stack2.occupancies = torch.tensor([0.7, 0.3], device="cpu")
        compute_batch, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes = compute_batch([batched_stack1, batched_stack2])
        assert volumes.shape == (2,) + tuple(small_lattice.grid_dimensions.tolist())
        assert volumes[0].sum() > 0
        assert volumes[1].sum() > 0

    def test_multiple_volumes_different_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        stack1 = small_atom_stack.replicate_ensemble(B=2)
        stack1.occupancies = torch.tensor([0.6, 0.4], device="cpu")
        stack2 = small_atom_stack.replicate_ensemble(B=2)
        stack2.occupancies = torch.tensor([0.8, 0.2], device="cpu")
        compute_batch, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes = compute_batch([stack1, stack2])
        assert volumes.shape == (2,) + tuple(small_lattice.grid_dimensions.tolist())
        assert torch.allclose(volumes[0], volumes[1], atol=1e-4)

    def test_per_voxel_averaging_vs_point_sampling(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        compute_batch_avg, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        compute_batch_point, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=False,
        )
        volumes_avg = compute_batch_avg([small_atom_stack])
        volumes_point = compute_batch_point([small_atom_stack])
        assert volumes_avg.shape == volumes_point.shape
        assert not torch.allclose(volumes_avg[0], volumes_point[0], atol=1e-6)
        assert volumes_avg[0].sum() > 0
        assert volumes_point[0].sum() > 0


# ---------------------------------------------------------------------------
# TestESPCalculatorConsistency
# ---------------------------------------------------------------------------

class TestESPCalculatorConsistency:
    """Tests to verify consistency between different ESP calculator functions."""

    def test_standard_vs_fused_single(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        volume_standard = compute_volume_over_insertable_matrices(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        volume_fused = compute_volume_stencil(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume_standard.shape == volume_fused.shape
        max_diff = (volume_standard - volume_fused).abs().max()
        mean_diff = (volume_standard - volume_fused).abs().mean()
        assert max_diff < 1e-4, f"Max difference: {max_diff:.6f}"
        assert mean_diff < 1e-5, f"Mean difference: {mean_diff:.6f}"

    def test_standard_vs_fused_multiple_single_volume(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        volume_standard = compute_volume_over_insertable_matrices(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        compute_batch, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes_fused = compute_batch([small_atom_stack])
        volume_fused = volumes_fused[0]
        assert volume_standard.shape == volume_fused.shape
        max_diff = (volume_standard - volume_fused).abs().max()
        mean_diff = (volume_standard - volume_fused).abs().mean()
        assert max_diff < 1e-4, f"Max difference: {max_diff:.6f}"
        assert mean_diff < 1e-5, f"Mean difference: {mean_diff:.6f}"

    def test_fused_single_vs_fused_multiple_single_volume(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        volume_fused_single = compute_volume_stencil(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        compute_batch, _ = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes_fused = compute_batch([small_atom_stack])
        volume_fused_multiple = volumes_fused[0]
        assert volume_fused_single.shape == volume_fused_multiple.shape
        max_diff = (volume_fused_single - volume_fused_multiple).abs().max()
        mean_diff = (volume_fused_single - volume_fused_multiple).abs().mean()
        assert max_diff < 1e-4, f"Max difference: {max_diff:.6f}"
        assert mean_diff < 1e-5, f"Mean difference: {mean_diff:.6f}"

    def test_same_structure_different_occupancies(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        stack1 = AtomStack(
            atom_coordinates=small_atom_stack.atom_coordinates.clone(),
            atom_names=small_atom_stack.atom_names.copy(),
            bfactors=small_atom_stack.bfactors.clone() if small_atom_stack.bfactors is not None else None,
            device="cpu",
            occupancies=torch.tensor([1.0], device="cpu"),
        )
        batched_stack = small_atom_stack.replicate_ensemble(B=2)
        batched_stack.occupancies = torch.tensor([0.6, 0.4], device="cpu")
        volume1 = compute_volume_over_insertable_matrices(
            atom_stack=stack1,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        volume_batched = compute_volume_over_insertable_matrices(
            atom_stack=batched_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert torch.allclose(volume1, volume_batched, atol=1e-4)

    def test_batched_ensemble_consistency(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        single_stack = AtomStack(
            atom_coordinates=small_atom_stack.atom_coordinates.clone(),
            atom_names=small_atom_stack.atom_names.copy(),
            bfactors=small_atom_stack.bfactors.clone() if small_atom_stack.bfactors is not None else None,
            device="cpu",
            occupancies=torch.tensor([1.0], device="cpu"),
        )
        batched_stack = small_atom_stack.replicate_ensemble(B=3)
        batched_stack.occupancies = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], device="cpu")
        volume_single = compute_volume_over_insertable_matrices(
            atom_stack=single_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        volume_batched = compute_volume_over_insertable_matrices(
            atom_stack=batched_stack,
            lattice=small_lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert torch.allclose(volume_single, volume_batched, atol=1e-5)


# ---------------------------------------------------------------------------
# TestAtomStackReplicateEnsemble
# ---------------------------------------------------------------------------

class TestAtomStackReplicateEnsemble:
    """Tests for AtomStack.replicate_ensemble method."""

    def test_replicate_single_to_multiple(self):
        small_atom_stack = get_small_atom_stack()
        assert small_atom_stack.atom_coordinates.shape[0] == 1
        replicated = small_atom_stack.replicate_ensemble(B=3)
        assert replicated.atom_coordinates.shape[0] == 3
        assert replicated.atom_coordinates.shape[1] == small_atom_stack.atom_coordinates.shape[1]
        assert replicated.atom_coordinates.shape[2] == 3
        assert torch.allclose(replicated.atom_coordinates[0], replicated.atom_coordinates[1])
        assert torch.allclose(replicated.atom_coordinates[0], replicated.atom_coordinates[2])
        assert torch.allclose(replicated.atom_coordinates[0], small_atom_stack.atom_coordinates[0])

    def test_replicate_preserves_bfactors(self):
        small_atom_stack = get_small_atom_stack()
        replicated = small_atom_stack.replicate_ensemble(B=2)
        if small_atom_stack.bfactors is not None:
            assert replicated.bfactors is not None
            assert replicated.bfactors.shape[0] == 2
            assert torch.allclose(replicated.bfactors[0], replicated.bfactors[1])
            assert torch.allclose(replicated.bfactors[0], small_atom_stack.bfactors[0])

    def test_replicate_occupancies_normalized(self):
        small_atom_stack = get_small_atom_stack()
        replicated = small_atom_stack.replicate_ensemble(B=3)
        assert replicated.occupancies.shape[0] == 3
        assert torch.allclose(replicated.occupancies.sum(), torch.tensor(1.0))
        assert torch.allclose(replicated.occupancies, torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]))

    def test_replicate_preserves_atom_names(self):
        small_atom_stack = get_small_atom_stack()
        replicated = small_atom_stack.replicate_ensemble(B=2)
        assert replicated.atom_names == small_atom_stack.atom_names

    def test_replicate_requires_single_batch(self):
        small_atom_stack = get_small_atom_stack()
        multi_batch = small_atom_stack.replicate_ensemble(B=2)
        try:
            multi_batch.replicate_ensemble(B=3)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    def test_replicate_with_different_B_values(self):
        small_atom_stack = get_small_atom_stack()
        for B in [1, 2, 3, 5, 10]:
            replicated = small_atom_stack.replicate_ensemble(B=B)
            assert replicated.atom_coordinates.shape[0] == B
            assert torch.allclose(replicated.occupancies.sum(), torch.tensor(1.0))
            assert len(replicated.occupancies) == B


# ---------------------------------------------------------------------------
# TestExampleLatticeFusedVsNonfused
# ---------------------------------------------------------------------------

class TestExampleLatticeFusedVsNonfused:
    """Test that the lattice_fused_vs_nonfused.py example runs correctly with small atom count."""

    def test_example_runs_with_small_atom_count(self):
        test_data_dir = get_test_data_dir()
        pdb_file = test_data_dir / "8OSK.pdb"
        if not pdb_file.exists():
            raise FileNotFoundError(
                "Test data required: examples/data/8OSK.pdb not found. "
                "Provide this file to run tests (e.g. copy from parent repo)."
            )

        atom_stack = _load_atom_stack_from_pdb(pdb_file, device="cpu")
        N = min(20, atom_stack.atom_coordinates.shape[1])
        atom_stack = AtomStack(
            atom_coordinates=atom_stack.atom_coordinates[:, :N, :],
            atom_names=atom_stack.atom_names[:N],
            bfactors=atom_stack.bfactors[:, :N, :] if atom_stack.bfactors is not None else None,
            device="cpu",
            occupancies=atom_stack.occupancies,
        )
        atomic_radius = 0.5
        bfactor = 8 * torch.pi**2 * atomic_radius**2
        atom_stack.fill_constant_bfactor(bfactor)

        voxel_size = 0.845
        left_bottom = [0.0, 0.0, 0.0]
        right_upper = [303.0, 303.0, 303.0]
        sublattice_radius = 5.0
        lattice = Lattice.from_voxel_sizes_and_corner_points(
            voxel_sizes_in_A=(voxel_size, voxel_size, voxel_size),
            left_bottom_point_in_A=left_bottom,
            right_upper_point_in_A=right_upper,
            sublattice_radius_in_A=sublattice_radius,
            dtype=torch.float32,
            device="cpu",
        )

        volume_nonfused = compute_volume_over_insertable_matrices(
            atom_stack=atom_stack,
            lattice=lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        volume_fused = compute_volume_stencil(
            atom_stack=atom_stack,
            lattice=lattice,
            B=32,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            verbose=False,
        )
        assert volume_nonfused.shape == volume_fused.shape
        assert volume_nonfused.shape == tuple(lattice.grid_dimensions.tolist())


# ---------------------------------------------------------------------------
# TestComputeBatchFromCoords
# ---------------------------------------------------------------------------

class TestComputeBatchFromCoords:
    """Tests for compute_batch_from_coords (vectorized coord-based volume computation)."""

    def test_compute_batch_from_coords_matches_compute_batch(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        compute_batch, compute_batch_from_coords = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        volumes_method1 = compute_batch([small_atom_stack])
        coords = small_atom_stack.atom_coordinates
        bfactors = small_atom_stack.bfactors
        atomic_numbers = small_atom_stack.atomic_numbers
        occupancies = small_atom_stack.occupancies
        coords_batch = coords.unsqueeze(0)  # [1, B_ens, N, 3]
        volumes_method2 = compute_batch_from_coords(
            coords_batch, bfactors, atomic_numbers, occupancies
        )
        assert volumes_method1.shape == volumes_method2.shape
        assert torch.allclose(volumes_method1, volumes_method2, atol=1e-5)

    def test_compute_batch_from_coords_multiple_hypotheses(self):
        small_atom_stack = get_small_atom_stack()
        small_lattice = get_small_lattice(small_atom_stack)
        compute_batch, compute_batch_from_coords = setup_fast_esp_solver(
            atom_stack=small_atom_stack,
            lattice=small_lattice,
            per_voxel_averaging=True,
        )
        coords = small_atom_stack.atom_coordinates
        B_ens, N, _ = coords.shape
        translations = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]).unsqueeze(1).expand(3, B_ens, 3)
        coords_batch = coords.unsqueeze(0) + translations.unsqueeze(2)  # [k=3, B_ens, N, 3]
        bfactors = small_atom_stack.bfactors
        atomic_numbers = small_atom_stack.atomic_numbers
        occupancies = small_atom_stack.occupancies
        volumes = compute_batch_from_coords(
            coords_batch, bfactors, atomic_numbers, occupancies
        )
        assert volumes.shape[0] == 3
        for hyp_idx in range(3):
            transformed_stack = AtomStack(
                atom_coordinates=coords_batch[hyp_idx],
                atom_names=small_atom_stack.atom_names,
                bfactors=bfactors,
                device=coords.device,
                occupancies=occupancies,
            )
            vol_single = compute_batch([transformed_stack])
            assert torch.allclose(volumes[hyp_idx], vol_single[0], atol=1e-5), f"Hypothesis {hyp_idx} mismatch"
