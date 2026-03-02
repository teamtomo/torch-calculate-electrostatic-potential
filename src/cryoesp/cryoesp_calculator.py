from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint
from .lattice import Lattice
from .atom_stack import AtomStack
from cryoesp.utils.peng_model import ScatteringAttributes
from cryoesp.utils.torch_utils import batched_with_indices
from tqdm import tqdm



def compute_volume_over_insertable_matrices(
        atom_stack: AtomStack,
        lattice: Lattice,
        B: int = 64,
        per_voxel_averaging: bool = True, 
        subvolume_mask_in_indices: torch.Tensor | None = None,
        use_checkpointing: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
    """
    Compute electrostatic potential volume over lattice from atom stack.
    
    Parameters:
    - atom_stack: Atoms with coordinates, identities, B-factors, and occupancies
    - lattice: Target lattice for volume computation
    - B: Batch size for atoms (performance tuning)
    - per_voxel_averaging: If True, average signal over voxel; if False, use center point
    - subvolume_mask_in_indices: Optional indices (in 'ij' order) for compact subvolume. Shape (M,)
    - use_checkpointing: If True, use gradient checkpointing to reduce memory usage (trades compute for memory).
                        Useful when gradients are needed. If False, faster but uses more memory.
    - verbose: If True, show tqdm progress bars. Default is False.
    
    Returns:
    - volume: Full grid (D,D,D) or compact subvolume (M,)
    """

    # Initialize volume and index mapping
    if subvolume_mask_in_indices is not None:
        volume = torch.zeros((len(subvolume_mask_in_indices), 1), dtype=lattice.dtype, device=atom_stack.device)
        # Ensure indices are sorted for binary search (typically already sorted by lattice 'ij' convention)
        if len(subvolume_mask_in_indices) > 1 and not torch.all(subvolume_mask_in_indices[1:] >= subvolume_mask_in_indices[:-1]):
            sorted_indices, inverse_perm = torch.sort(subvolume_mask_in_indices)
            index_mapping = (sorted_indices, inverse_perm)
        else:
            index_mapping = (subvolume_mask_in_indices, None)
    else:
        volume = torch.zeros((torch.prod(lattice.grid_dimensions), 1), dtype=lattice.dtype, device=atom_stack.device)
        index_mapping = None
    
    scattering_attributes = ScatteringAttributes(atom_stack.device)
    ensemble_size = atom_stack.atom_coordinates.shape[0]
    num_atoms = atom_stack.atom_coordinates.shape[1]
    batches_per_ensemble = torch.ceil(torch.as_tensor(num_atoms / B)).item()

    def compute_insertion_matrices(atom_batch, insertion_matrix_voxels, a_jk, b_jk, bfactor_batch, occupancy, voxel_sizes, per_voxel_averaging):
        voxel_to_atom_differences = insertion_matrix_voxels - atom_batch[:, None, :]
        if per_voxel_averaging:
            multiplier = (4*torch.pi**2 / (b_jk + bfactor_batch)).sqrt()[:, None, None, :]  
            insertion_matrices = (torch.prod(
                ( (voxel_to_atom_differences + voxel_sizes[None, None, :] / 2)[..., None] * multiplier ).erf() - \
                ( (voxel_to_atom_differences - voxel_sizes[None, None, :] / 2)[..., None] * multiplier ).erf(),
                dim=2
            ) * a_jk[:, None, :]).sum(dim=-1) / ( 8 * voxel_sizes.prod() )
        else:
            squared_distances = torch.sum(voxel_to_atom_differences**2, dim=2, keepdim=True)
            gaussian_widths = 1 / (b_jk + bfactor_batch)
            insertion_matrices = (
                a_jk[:, None, :] * (4 * torch.pi)**(3/2) * 
                ( gaussian_widths**(3/2) )[:, None, :] * 
                torch.exp(-4 * torch.pi**2 * squared_distances * gaussian_widths[:, None, :]) 
            ).sum(dim=-1)
        return insertion_matrices * occupancy

    for ensemble_idx in range(ensemble_size):
        ensemble_coords = atom_stack.atom_coordinates[ensemble_idx]
        ensemble_bfactors = atom_stack.bfactors[ensemble_idx] if atom_stack.bfactors is not None else None
        occupancy = atom_stack.occupancies[ensemble_idx]

        for indices, atom_batch in tqdm(batched_with_indices(ensemble_coords, B), total=batches_per_ensemble, desc=f"Ensemble {ensemble_idx+1}/{ensemble_size}", disable=not verbose):
            bfactor_batch = ensemble_bfactors[indices] if ensemble_bfactors is not None else torch.zeros_like(indices, device=atom_stack.device, dtype=torch.float32)
            atom_identities_batch = atom_stack.atomic_numbers[indices].flatten()

            with torch.no_grad():
                insertion_matrix_indices, insertion_matrix_voxels = lattice.extract_closest_submatrices(atom_batch)
            
            insertion_matrix_voxels = insertion_matrix_voxels.detach()
            
            a_jk, b_jk = scattering_attributes(atom_identities_batch)
            if use_checkpointing:
                insertion_matrices = checkpoint(compute_insertion_matrices, atom_batch, insertion_matrix_voxels, a_jk, b_jk, bfactor_batch, occupancy, lattice.voxel_sizes_in_A, per_voxel_averaging, use_reentrant=False)
            else:
                insertion_matrices = compute_insertion_matrices(atom_batch, insertion_matrix_voxels, a_jk, b_jk, bfactor_batch, occupancy, lattice.voxel_sizes_in_A, per_voxel_averaging)
            
            if subvolume_mask_in_indices is not None:
                sorted_indices, inverse_perm = index_mapping
                positions = torch.searchsorted(sorted_indices, insertion_matrix_indices)
                in_bounds = positions < len(sorted_indices)
                positions_clamped = torch.clamp(positions, 0, len(sorted_indices) - 1)
                valid_mask = in_bounds & (sorted_indices[positions_clamped] == insertion_matrix_indices)
                compact_indices = inverse_perm[positions[valid_mask]] if inverse_perm is not None else positions[valid_mask]
                volume.scatter_add_(0, compact_indices.unsqueeze(1).to(torch.int64), insertion_matrices[valid_mask].unsqueeze(1))
            else:
                volume.scatter_add_(0, insertion_matrix_indices.reshape(-1, 1).to(torch.int64), insertion_matrices.reshape(-1, 1))

    if subvolume_mask_in_indices is not None:
        return volume.squeeze(-1)
    else:
        return volume.reshape(tuple(lattice.grid_dimensions.tolist()))


# ============================================================================
# Optimized stencil-based volume computation with fused kernels
# ============================================================================

def _compute_density_kernel(atom_coords, voxel_coords, a, b, B_val, occ, voxel_sizes, is_averaged):
    """
    Compute density values for atoms at voxel coordinates.
    Supports both per-voxel averaging and point-sampling modes.
    """
    # Geometry Diff: (N, K, 3) - (N, 1, 3) -> (N, K, 3, 1)
    diff = (voxel_coords - atom_coords.unsqueeze(1)).unsqueeze(-1)

    # Ensure broadcast-friendly shapes
    if B_val.ndim == 1:
        B_val = B_val.unsqueeze(-1)  # (N, 1)
    if occ.ndim == 1:
        occ = occ.unsqueeze(-1)      # (N, 1)
    
    if is_averaged:
        gamma = (4 * torch.pi**2 / (b + B_val)).sqrt().unsqueeze(1)
        v_x, v_y, v_z = voxel_sizes[0] / 2.0, voxel_sizes[1] / 2.0, voxel_sizes[2] / 2.0
        
        term_x = torch.erf((diff[..., 0, :] + v_x) * gamma) - torch.erf((diff[..., 0, :] - v_x) * gamma)
        term_y = torch.erf((diff[..., 1, :] + v_y) * gamma) - torch.erf((diff[..., 1, :] - v_y) * gamma)
        term_z = torch.erf((diff[..., 2, :] + v_z) * gamma) - torch.erf((diff[..., 2, :] - v_z) * gamma)

        spatial_integral = term_x * term_y * term_z
        weights = (occ * a).unsqueeze(1)
        return (spatial_integral * weights).sum(dim=-1).squeeze(-1) / (8 * voxel_sizes.prod())
    else:
        squared_distances = (diff ** 2).sum(dim=2) 
        sigma_inv = 1.0 / (b + B_val).unsqueeze(1)
        prefactor = (a * occ).unsqueeze(1) * (4 * torch.pi)**(3/2)
        width_factor = sigma_inv ** (3/2)
        term = torch.exp(-4 * torch.pi**2 * squared_distances * sigma_inv)
        return (prefactor * width_factor * term).sum(dim=-1).squeeze(-1)


def _fused_stencil_kernel(
    atom_batch,                  # (N, 3)
    anchor_coords,               # (N, 3) [Translation vector for coords]
    anchor_indices_flat,         # (N, 1) [Translation vector for indices, flattened]
    stencil_coords,              # (K, 3) [Sublattice shape constant]
    stencil_indices_flat,        # (K, 1) [Sublattice indices constant]
    volume, 
    a, b, B_val, occ, voxel_sizes, is_averaged
):
    """
    Fused kernel that generates voxel coordinates and indices on-the-fly using anchor+stencil pattern.
    Computes density values and scatters them directly to the volume tensor.
    """
    # Generate voxel coordinates: Anchor (N, 3) + Stencil (K, 3) -> (N, K, 3)
    generated_voxels = anchor_coords.unsqueeze(1) + stencil_coords.unsqueeze(0)
    
    # Compute density values
    values = _compute_density_kernel(atom_batch, generated_voxels, a, b, B_val, occ, voxel_sizes, is_averaged)
    
    # Generate target indices: Anchor (N,) + Stencil (K,) -> (N, K)
    target_indices = anchor_indices_flat.view(-1, 1) + stencil_indices_flat.view(1, -1)
    volume.scatter_add_(0, target_indices.view(-1, 1), values.view(-1, 1))


# Create separate compiled functions for each boolean value to avoid torch.compile specialization issues
# torch.compile treats Python booleans as compile-time constants, so we need separate compiled versions
def _fused_stencil_kernel_averaged(
    atom_batch, anchor_coords, anchor_indices_flat, stencil_coords, stencil_indices_flat,
    volume, a, b, B_val, occ, voxel_sizes
):
    """Wrapper with is_averaged=True baked in."""
    return _fused_stencil_kernel(
        atom_batch, anchor_coords, anchor_indices_flat, stencil_coords, stencil_indices_flat,
        volume, a, b, B_val, occ, voxel_sizes, True
    )


def _fused_stencil_kernel_point_sampled(
    atom_batch, anchor_coords, anchor_indices_flat, stencil_coords, stencil_indices_flat,
    volume, a, b, B_val, occ, voxel_sizes
):
    """Wrapper with is_averaged=False baked in."""
    return _fused_stencil_kernel(
        atom_batch, anchor_coords, anchor_indices_flat, stencil_coords, stencil_indices_flat,
        volume, a, b, B_val, occ, voxel_sizes, False
    )


_compiled_stencil_kernel_averaged = torch.compile(_fused_stencil_kernel_averaged, mode="max-autotune", dynamic=True)
_compiled_stencil_kernel_point_sampled = torch.compile(_fused_stencil_kernel_point_sampled, mode="max-autotune", dynamic=True)


def _fused_multi_volume_kernel(
    atom_batch,              # (Total_Atoms, 3)   flattened across B volumes
    anchor_coords,           # (Total_Atoms, 3)
    anchor_indices_flat,     # (Total_Atoms,)
    batch_offsets,           # (Total_Atoms,) integer offsets = volume_id * GridSize
    stencil_coords,          # (K, 3)
    stencil_indices_flat,    # (K,)
    volume,                  # (B * GridSize, 1)
    a, b, B_val, occ, voxel_sizes, is_averaged
):
    """
    Compute multiple independent volumes in a single fused kernel launch.
    target_indices = anchor_indices_flat + stencil_indices_flat + batch_offsets
    """
    generated_voxels = anchor_coords.unsqueeze(1) + stencil_coords.unsqueeze(0)
    values = _compute_density_kernel(atom_batch, generated_voxels, a, b, B_val, occ, voxel_sizes, is_averaged)

    target_indices = (
        anchor_indices_flat.view(-1, 1)
        + stencil_indices_flat.view(1, -1)
        + batch_offsets.view(-1, 1)
    )

    volume.scatter_add_(0, target_indices.view(-1, 1), values.view(-1, 1))


# Create separate compiled functions for each boolean value to avoid torch.compile specialization issues
def _fused_multi_volume_kernel_averaged(
    atom_batch, anchor_coords, anchor_indices_flat, batch_offsets,
    stencil_coords, stencil_indices_flat, volume, a, b, B_val, occ, voxel_sizes
):
    """Wrapper with is_averaged=True baked in."""
    return _fused_multi_volume_kernel(
        atom_batch, anchor_coords, anchor_indices_flat, batch_offsets,
        stencil_coords, stencil_indices_flat, volume, a, b, B_val, occ, voxel_sizes, True
    )


def _fused_multi_volume_kernel_point_sampled(
    atom_batch, anchor_coords, anchor_indices_flat, batch_offsets,
    stencil_coords, stencil_indices_flat, volume, a, b, B_val, occ, voxel_sizes
):
    """Wrapper with is_averaged=False baked in."""
    return _fused_multi_volume_kernel(
        atom_batch, anchor_coords, anchor_indices_flat, batch_offsets,
        stencil_coords, stencil_indices_flat, volume, a, b, B_val, occ, voxel_sizes, False
    )


_compiled_multi_volume_kernel_averaged = torch.compile(_fused_multi_volume_kernel_averaged, mode="max-autotune", dynamic=True)
_compiled_multi_volume_kernel_point_sampled = torch.compile(_fused_multi_volume_kernel_point_sampled, mode="max-autotune", dynamic=True)


def compute_volume_stencil(
    atom_stack,
    lattice,
    B: int = 4096,
    per_voxel_averaging: bool = True,
    subvolume_mask_in_indices=None,
    verbose=False,
):
    """
    Compute electrostatic potential volume using stencil-based fused kernels.

    Optimized path: uses anchor+stencil pattern and torch.compile. No subvolume
    masks; for those use compute_volume_over_insertable_matrices.

    Parameters
    ----------
    atom_stack : AtomStack
        Atoms with coordinates, identities, B-factors, occupancies.
    lattice : Lattice
        Target lattice (full grid).
    B : int
        Batch size for atoms.
    per_voxel_averaging : bool
        If True, average over voxel; if False, sample at center.
    subvolume_mask_in_indices : tensor or None
        Not supported; must be None.
    verbose : bool
        If True, show progress bars.

    Returns
    -------
    volume : tensor
        Shape (Dx, Dy, Dz), dtype lattice.dtype.
    """
    if subvolume_mask_in_indices is not None:
        raise NotImplementedError(
            "Subvolume masks are not supported with stencil-based computation. "
            "Use compute_volume_over_insertable_matrices instead."
        )

    volume = torch.zeros((torch.prod(lattice.grid_dimensions), 1), dtype=lattice.dtype, device=atom_stack.device)
    scattering_attributes = ScatteringAttributes(atom_stack.device)

    flat_stencil = lattice.convert_cubic_index_to_flat_index(lattice.sublattice_cubic_indices).contiguous()
    stencil_coords = lattice.sublattice_coordinates.contiguous()

    ensemble_size = atom_stack.atom_coordinates.shape[0]
    num_atoms = atom_stack.atom_coordinates.shape[1]
    batches_per_ensemble = torch.ceil(torch.as_tensor(num_atoms / B)).item()

    for ensemble_idx in range(ensemble_size):
        ensemble_coords = atom_stack.atom_coordinates[ensemble_idx]
        ensemble_bfactors = atom_stack.bfactors[ensemble_idx] if atom_stack.bfactors is not None else None
        occupancy = atom_stack.occupancies[ensemble_idx]

        for indices, atom_batch in tqdm(batched_with_indices(ensemble_coords, B), total=batches_per_ensemble, desc=f"Ensemble {ensemble_idx+1}/{ensemble_size}", disable=not verbose):
            bfactor_batch = ensemble_bfactors[indices] if ensemble_bfactors is not None else torch.zeros_like(indices, device=atom_stack.device, dtype=torch.float32)
            atom_identities_batch = atom_stack.atomic_numbers[indices].flatten()

            with torch.no_grad():
                anchor_flat, anchor_coord = lattice.get_stencil_anchor_translations(atom_batch)
                atom_batch = atom_batch.contiguous()

            a_jk, b_jk = scattering_attributes(atom_identities_batch)

            compiled_kernel = _compiled_stencil_kernel_averaged if per_voxel_averaging else _compiled_stencil_kernel_point_sampled
            compiled_kernel(
                atom_batch,
                anchor_coord,
                anchor_flat,
                stencil_coords,
                flat_stencil,
                volume,
                a_jk,
                b_jk,
                bfactor_batch,
                occupancy,
                lattice.voxel_sizes_in_A,
            )

    return volume.reshape(tuple(lattice.grid_dimensions.tolist()))


def setup_fast_esp_solver(
    atom_stack,
    lattice,
    per_voxel_averaging: bool = True,
):
    """
    Prepare a high-performance multi-volume stencil solver.

    Returns a function that computes multiple volumes in a single batch.
    Input to the returned `compute_batch` MUST be a list of AtomStack, one per output volume.
    Each AtomStack may have an ensemble axis (B_ensemble, N, 3); all ensemble members are
    accumulated into that volume using its occupancies. List length == number of volumes.
    """
    Dx, Dy, Dz = lattice.grid_dimensions
    grid_size = int(torch.prod(lattice.grid_dimensions).item())

    flat_stencil = lattice.convert_cubic_index_to_flat_index(lattice.sublattice_cubic_indices).contiguous()
    stencil_coords = lattice.sublattice_coordinates.contiguous()

    scattering_attributes = ScatteringAttributes(atom_stack.device)

    def compute_batch(atom_stacks: list[AtomStack]):
        """
        atom_stacks: list of AtomStack, one per output volume.
        Each AtomStack shape: (B_ensemble, N, 3); all share the same (B_ensemble, N, 3).
        Returns: [len(atom_stacks), Dx, Dy, Dz]
        """
        assert isinstance(atom_stacks, list) and len(atom_stacks) > 0, "Input must be a non-empty list of AtomStack"
        assert all(isinstance(s, AtomStack) for s in atom_stacks), "All elements must be AtomStack"

        B_vol = len(atom_stacks)
        N = atom_stacks[0].atom_coordinates.shape[1]
        B_ens = atom_stacks[0].atom_coordinates.shape[0]
        for s in atom_stacks:
            assert s.atom_coordinates.shape == (B_ens, N, 3), "All AtomStacks must share shape (B_ensemble, N, 3)"

        coords_flat = []
        bfac_flat = []
        atom_ids_flat = []
        occ_flat = []
        batch_offsets_list = []

        for vol_idx, s in enumerate(atom_stacks):
            coords = s.atom_coordinates.to(lattice.device)
            coords_flat.append(coords.reshape(-1, 3))

            if s.bfactors is None:
                raise ValueError("bfactors must be provided for all AtomStacks; zeros are disallowed")
            bf = s.bfactors.to(lattice.device)
            if torch.any(bf == 0):
                raise ValueError("bfactors contain zeros; expected non-zero values for all atoms")
            bfac_flat.append(bf.reshape(-1))

            ids = s.atomic_numbers.to(lattice.device).repeat(B_ens, 1)
            atom_ids_flat.append(ids.reshape(-1))

            occ_per_atom = s.occupancies.to(lattice.device).repeat_interleave(N)
            occ_flat.append(occ_per_atom.to(lattice.dtype))

            batch_offsets_list.append(
                torch.full(
                    (B_ens * N,),
                    vol_idx * grid_size,
                    device=lattice.device,
                    dtype=torch.int64,
                )
            )

        atoms_flat = torch.cat(coords_flat, dim=0).contiguous()
        bfactor_batch = torch.cat(bfac_flat, dim=0).contiguous()
        atom_identities_batch = torch.cat(atom_ids_flat, dim=0).contiguous()
        occupancy_vec = torch.cat(occ_flat, dim=0).contiguous()
        batch_offsets = torch.cat(batch_offsets_list, dim=0).contiguous()
        B_volumes = B_vol

        # Select the appropriate compiled kernel based on per_voxel_averaging
        compiled_multi_kernel = _compiled_multi_volume_kernel_averaged if per_voxel_averaging else _compiled_multi_volume_kernel_point_sampled

        with torch.no_grad():
            anchor_flat, anchor_coord = lattice.get_stencil_anchor_translations(atoms_flat)
            atoms_flat = atoms_flat.contiguous()

        a_jk, b_jk = scattering_attributes(atom_identities_batch)

        volume = torch.zeros(
            (B_volumes * grid_size, 1), dtype=lattice.dtype, device=atoms_flat.device
        )
        compiled_multi_kernel(
            atoms_flat,
            anchor_coord,
            anchor_flat,
            batch_offsets,
            stencil_coords,
            flat_stencil,
            volume,
            a_jk,
            b_jk,
            bfactor_batch,
            occupancy_vec,
            lattice.voxel_sizes_in_A,
        )
        return volume.view(B_volumes, Dx, Dy, Dz)

    def compute_batch_from_coords(
        coords_batch: torch.Tensor,      # [k, B_ens, N, 3] pre-transformed coordinates
        bfactors: torch.Tensor,          # [B_ens, N, 1] shared across hypotheses
        atomic_numbers: torch.Tensor,    # [N, 1] shared across hypotheses
        occupancies: torch.Tensor,       # [B_ens] shared across hypotheses
    ) -> torch.Tensor:
        """
        Vectorized volume computation from pre-transformed coordinates.
        
        Avoids AtomStack creation overhead by accepting raw tensors directly.
        All hypotheses share the same bfactors, atomic_numbers, and occupancies.
        
        Args:
            coords_batch: Pre-transformed coordinates [k, B_ens, N, 3]
            bfactors: B-factors [B_ens, N, 1], shared across all k hypotheses
            atomic_numbers: Atomic numbers [N, 1], shared across all hypotheses
            occupancies: Occupancies [B_ens], shared across all hypotheses
            
        Returns:
            volumes: [k, Dx, Dy, Dz] computed volumes
        """
        k, B_ens, N, _ = coords_batch.shape
        B_volumes = k
        
        # Flatten coords: [k, B_ens, N, 3] -> [k * B_ens * N, 3]
        atoms_flat = coords_batch.reshape(-1, 3).to(lattice.device).contiguous()
        
        # Expand bfactors: [B_ens, N, 1] -> [k, B_ens, N] -> [k * B_ens * N]
        bfactor_batch = bfactors.squeeze(-1).unsqueeze(0).expand(k, -1, -1).reshape(-1).to(lattice.device).contiguous()
        
        # Expand atomic_numbers: [N, 1] -> [k, B_ens, N] -> [k * B_ens * N]
        atom_ids_expanded = atomic_numbers.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(k, B_ens, -1)
        atom_identities_batch = atom_ids_expanded.reshape(-1).to(lattice.device).contiguous()
        
        # Expand occupancies: [B_ens] -> [k, B_ens, N] -> [k * B_ens * N]
        # Each atom in ensemble member i gets occupancy[i]
        occ_expanded = occupancies.unsqueeze(0).unsqueeze(-1).expand(k, -1, N)
        occupancy_vec = occ_expanded.reshape(-1).to(lattice.device, dtype=lattice.dtype).contiguous()
        
        # Build batch_offsets: each hypothesis gets a separate volume
        # For hypothesis h, all its atoms (B_ens * N) map to volume h
        batch_offsets = torch.arange(k, device=lattice.device, dtype=torch.int64).repeat_interleave(B_ens * N) * grid_size
        batch_offsets = batch_offsets.contiguous()
        
        # Select the appropriate compiled kernel based on per_voxel_averaging
        compiled_multi_kernel = _compiled_multi_volume_kernel_averaged if per_voxel_averaging else _compiled_multi_volume_kernel_point_sampled
        
        with torch.no_grad():
            anchor_flat, anchor_coord = lattice.get_stencil_anchor_translations(atoms_flat)
            atoms_flat = atoms_flat.contiguous()

        a_jk, b_jk = scattering_attributes(atom_identities_batch)

        volume = torch.zeros(
            (B_volumes * grid_size, 1), dtype=lattice.dtype, device=atoms_flat.device
        )
        compiled_multi_kernel(
            atoms_flat, anchor_coord, anchor_flat, batch_offsets,
            stencil_coords, flat_stencil, volume,
            a_jk, b_jk, bfactor_batch, occupancy_vec,
            lattice.voxel_sizes_in_A,
        )
        return volume.view(B_volumes, Dx, Dy, Dz)

    return compute_batch, compute_batch_from_coords