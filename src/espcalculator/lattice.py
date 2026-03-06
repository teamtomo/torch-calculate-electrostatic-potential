"""3D lattice for ESP volume computation."""

from __future__ import annotations

import numpy as np
import torch


class Lattice:
    """3D lattice for ESP volume computation.

    Defines the metaparameters of the resulting volume: resolution (voxel_sizes_in_A)
    and spatial extent (corner points). It affects computational cost the most:
    complexity scales with higher resolution and thus bigger dimensions D_x, D_y, D_z.
    Increase sublattice_radius_in_A (Å) when B-factors are high or voxel sizes are low.
    """

    def __init__(
        self,
        grid_dimensions: tuple[int, int, int] | torch.Tensor,
        voxel_sizes_in_A: tuple[float, float, float] | torch.Tensor,
        left_bottom_point_in_A: tuple[float, float, float] | torch.Tensor | None = None,
        right_upper_point_in_A: tuple[float, float, float] | torch.Tensor | None = None,
        sublattice_radius_in_A: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device: torch.DeviceObjType = torch.device("cpu"),
        **kwargs,
    ):
        self.grid_dimensions = torch.as_tensor(grid_dimensions, dtype=torch.int32, device=device)
        self.grid_flat_size = self.grid_dimensions.prod().item()
        self.voxel_sizes_in_A = torch.as_tensor(voxel_sizes_in_A, dtype=dtype, device=device)
        self.device = device
        self.dtype = dtype
        self.grid_side_lengths_in_A = (self.grid_dimensions - 1) * self.voxel_sizes_in_A
        self.physical_extent_in_A = self.grid_dimensions * self.voxel_sizes_in_A

        if left_bottom_point_in_A is None or right_upper_point_in_A is None:
            left_bottom_point_in_A = torch.zeros(3, dtype=dtype, device=device)
            right_upper_point_in_A = self.grid_side_lengths_in_A.to(dtype)

        self.left_bottom_point = torch.as_tensor(left_bottom_point_in_A, dtype=dtype, device=device)
        self.right_upper_point = torch.as_tensor(right_upper_point_in_A, dtype=dtype, device=device)
        self.physical_left_bottom = self.left_bottom_point - self.voxel_sizes_in_A / 2.0
        self.physical_right_upper = self.right_upper_point + self.voxel_sizes_in_A / 2.0

        self.frequency_grid_in_m_2d = self.compute_frequency_grid_in_m_2d(
            voxel_sizes_in_A[0], grid_dimensions[0], dtype=dtype, device=device
        )
        self.sublattice_radius_in_A = sublattice_radius_in_A

        sublattice_radius_voxels_xyz = (
            int(np.ceil((sublattice_radius_in_A / self.voxel_sizes_in_A[0].item()) - 0.5)),
            int(np.ceil((sublattice_radius_in_A / self.voxel_sizes_in_A[1].item()) - 0.5)),
            int(np.ceil((sublattice_radius_in_A / self.voxel_sizes_in_A[2].item()) - 0.5)),
        )
        self._initialize_sublattice_coordinates_and_indices(sublattice_radius_voxels_xyz)
        self.frequency_grid_in_A_2d = self.frequency_grid_in_m_2d / 1e10

    def _initialize_lattice_coordinates(self) -> torch.Tensor:
        D1, D2, D3 = self.grid_dimensions
        x_left, y_left, z_left = self.left_bottom_point
        x_right, y_right, z_right = self.right_upper_point
        x = torch.linspace(x_left, x_right, D1, device=self.device, dtype=self.dtype)
        y = torch.linspace(y_left, y_right, D2, device=self.device, dtype=self.dtype)
        z = torch.linspace(z_left, z_right, D3, device=self.device, dtype=self.dtype)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        self.lattice_coordinates = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)
        return self.lattice_coordinates

    def _initialize_sublattice_coordinates_and_indices(
        self, sublattice_radius_voxels_xyz: tuple[int, int, int]
    ) -> None:
        radius_x, radius_y, radius_z = sublattice_radius_voxels_xyz
        extent_x = min(2 * radius_x + 1, self.grid_dimensions[0])
        extent_y = min(2 * radius_y + 1, self.grid_dimensions[1])
        extent_z = min(2 * radius_z + 1, self.grid_dimensions[2])
        extent_x = extent_x if extent_x % 2 == 1 else extent_x - 1
        extent_y = extent_y if extent_y % 2 == 1 else extent_y - 1
        extent_z = extent_z if extent_z % 2 == 1 else extent_z - 1
        self.sublattice_dimensions = torch.tensor(
            (extent_x, extent_y, extent_z), dtype=torch.int32, device=self.device
        )

        x = torch.arange(extent_x, device=self.device, dtype=self.dtype)
        y = torch.arange(extent_y, device=self.device, dtype=self.dtype)
        z = torch.arange(extent_z, device=self.device, dtype=self.dtype)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        cubic_indices = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)

        self.sublattice_center_cubic_index = torch.tensor(
            (extent_x // 2, extent_y // 2, extent_z // 2),
            dtype=torch.int32,
            device=self.device,
        )
        self.sublattice_center_index_flat = self.convert_cubic_index_to_flat_index(
            self.sublattice_center_cubic_index
        )
        self.sublattice_center_coordinate = (
            self.left_bottom_point + self.voxel_sizes_in_A * self.sublattice_center_cubic_index
        )

        coords = cubic_indices * self.voxel_sizes_in_A + self.left_bottom_point
        self.sublattice_coordinates = coords
        self.sublattice_cubic_indices = cubic_indices.to(torch.int32)

    def convert_cubic_index_to_flat_index(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert (..., 3) cubic indices to (...) flat indices (row-major: x slowest, z fastest)."""
        Dx, Dy, Dz = self.grid_dimensions
        return (
            indices[..., 0].to(torch.int64) * (Dy * Dz)
            + indices[..., 1].to(torch.int64) * Dz
            + indices[..., 2].to(torch.int64)
        )

    def get_stencil_anchor_translations(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For stencil-based volume code: return clamped anchor (flat indices, coordinates)
        so that anchor + stencil stays inside the grid. Same clamp as in extract_closest_submatrices.
        Returns (anchor_flat, anchor_coord) with shapes (N,) and (N, 3).
        """
        (
            closest_voxel_cubic_indices,
            _,
            closest_voxel_centers,
        ) = self.find_closest_voxel_center_coordinates_and_indices(points)
        trans_idx = closest_voxel_cubic_indices - self.sublattice_center_cubic_index
        trans_idx = trans_idx.clamp(
            min=torch.zeros(3, dtype=torch.int32, device=self.device),
            max=self.grid_dimensions - self.sublattice_dimensions,
        )
        trans_coord = closest_voxel_centers - self.sublattice_center_coordinate
        max_trans = (self.voxel_sizes_in_A * (self.grid_dimensions - self.sublattice_dimensions)).to(
            self.dtype
        )
        trans_coord = trans_coord.clamp(
            min=torch.zeros(3, device=self.device, dtype=self.dtype),
            max=max_trans,
        )
        anchor_flat = self.convert_cubic_index_to_flat_index(trans_idx)
        return anchor_flat.contiguous(), trans_coord.contiguous()

    def flat_indices_to_boolean_mask(self, flat_indices: torch.Tensor) -> torch.Tensor:
        Dx, Dy, Dz = self.grid_dimensions
        mask = torch.zeros(Dx * Dy * Dz, dtype=torch.bool, device=self.device)
        mask[flat_indices] = True
        return mask.reshape(Dx, Dy, Dz)

    def boolean_mask_to_flat_indices(self, boolean_mask: torch.Tensor) -> torch.Tensor:
        Dx, Dy, Dz = self.grid_dimensions
        assert boolean_mask.shape == tuple(self.grid_dimensions.tolist())
        return torch.where(boolean_mask.reshape(-1))[0]

    def insert_compact_subvolume_into_full_volume(
        self,
        compact_values: torch.Tensor,
        flat_indices: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        Dx, Dy, Dz = self.grid_dimensions
        volume = torch.zeros(Dx * Dy * Dz, dtype=dtype, device=self.device)
        volume[flat_indices] = compact_values
        return volume.reshape(Dx, Dy, Dz)

    def find_closest_voxel_center_coordinates_and_indices(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert points.shape[1] == 3
        x = points - self.left_bottom_point
        closest_voxel_cubic_indices = torch.floor(
            (x + self.voxel_sizes_in_A / 2) / self.voxel_sizes_in_A
        ).to(torch.int32)
        closest_voxel_cubic_indices = torch.clamp(
            closest_voxel_cubic_indices,
            min=torch.zeros(3, dtype=torch.int32, device=self.device),
            max=self.grid_dimensions - 1,
        )
        return (
            closest_voxel_cubic_indices,
            self.convert_cubic_index_to_flat_index(closest_voxel_cubic_indices),
            (closest_voxel_cubic_indices * self.voxel_sizes_in_A + self.left_bottom_point).to(
                self.dtype
            ),
        )

    def extract_closest_submatrices(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            closest_voxel_cubic_indices,
            _,
            closest_voxel_centers,
        ) = self.find_closest_voxel_center_coordinates_and_indices(points)

        trans_idx = closest_voxel_cubic_indices - self.sublattice_center_cubic_index
        trans_idx = trans_idx.clamp(
            min=torch.zeros(3, dtype=torch.int32, device=self.device),
            max=self.grid_dimensions - self.sublattice_dimensions,
        )
        translated_flat = self.convert_cubic_index_to_flat_index(
            self.sublattice_cubic_indices + trans_idx.unsqueeze(1)
        )

        trans_coord = closest_voxel_centers - self.sublattice_center_coordinate
        max_trans = (self.voxel_sizes_in_A * (self.grid_dimensions - self.sublattice_dimensions)).to(
            self.dtype
        )
        trans_coord = trans_coord.clamp(
            min=torch.zeros(3, device=self.device, dtype=self.dtype),
            max=max_trans,
        )
        translated_coords = self.sublattice_coordinates + trans_coord.unsqueeze(1)

        return translated_flat, translated_coords

    @staticmethod
    def compute_frequency_grid_in_m_2d(
        pixel_size_in_A: float,
        D: int,
        dtype: torch.dtype = torch.float64,
        device: torch.DeviceObjType = torch.device("cpu"),
    ) -> torch.Tensor:
        nyquist = 1 / (2 * pixel_size_in_A)
        freqs_1d = torch.linspace(-nyquist, nyquist, D, device=device, dtype=dtype)
        fx, fy = torch.meshgrid(freqs_1d, freqs_1d, indexing="ij")
        freqs_2d = torch.stack([fx.ravel(), fy.ravel()], dim=1)
        return freqs_2d * 1e10

    @classmethod
    def from_grid_dimensions_and_voxel_sizes(
        cls,
        grid_dimensions: tuple[int, int, int],
        voxel_sizes_in_A: tuple[float, float, float] | torch.Tensor,
        left_bottom_point_in_A=None,
        right_upper_point_in_A=None,
        sublattice_radius_in_A: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device: torch.DeviceObjType = torch.device("cpu"),
        **kwargs,
    ):
        return cls(
            grid_dimensions=grid_dimensions,
            voxel_sizes_in_A=voxel_sizes_in_A,
            left_bottom_point_in_A=left_bottom_point_in_A,
            right_upper_point_in_A=right_upper_point_in_A,
            sublattice_radius_in_A=sublattice_radius_in_A,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_grid_dimensions_and_corner_points(
        cls,
        grid_dimensions: tuple[int, int, int] | torch.Tensor,
        left_bottom_point_in_A: tuple[float, float, float] | torch.Tensor,
        right_upper_point_in_A: tuple[float, float, float] | torch.Tensor,
        sublattice_radius_in_A: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device: torch.DeviceObjType = torch.device("cpu"),
        **kwargs,
    ):
        grid_dimensions = torch.as_tensor(grid_dimensions, dtype=torch.int32, device=device)
        left_bottom_point_in_A = torch.as_tensor(left_bottom_point_in_A, dtype=dtype, device=device)
        right_upper_point_in_A = torch.as_tensor(right_upper_point_in_A, dtype=dtype, device=device)
        voxel_sizes_in_A = (right_upper_point_in_A - left_bottom_point_in_A) / (grid_dimensions - 1)
        return cls(
            grid_dimensions=grid_dimensions,
            voxel_sizes_in_A=voxel_sizes_in_A,
            left_bottom_point_in_A=left_bottom_point_in_A,
            right_upper_point_in_A=right_upper_point_in_A,
            sublattice_radius_in_A=sublattice_radius_in_A,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_voxel_sizes_and_corner_points(
        cls,
        voxel_sizes_in_A: tuple[float, float, float],
        left_bottom_point_in_A: tuple[float, float, float],
        right_upper_point_in_A: tuple[float, float, float],
        sublattice_radius_in_A: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device: torch.DeviceObjType = torch.device("cpu"),
    ):
        voxel_sizes = torch.as_tensor(voxel_sizes_in_A, dtype=dtype, device=device)
        right_upper_point_in_A = torch.as_tensor(right_upper_point_in_A, dtype=dtype, device=device)
        left_bottom_point_in_A = torch.as_tensor(left_bottom_point_in_A, dtype=dtype, device=device)
        span_of_centers = right_upper_point_in_A - left_bottom_point_in_A
        grid_dimensions = torch.ceil(span_of_centers / voxel_sizes).to(torch.int32) + 1
        recomputed_voxel_sizes_in_A = span_of_centers / (grid_dimensions - 1)
        return cls(
            grid_dimensions=grid_dimensions,
            voxel_sizes_in_A=recomputed_voxel_sizes_in_A,
            left_bottom_point_in_A=left_bottom_point_in_A,
            right_upper_point_in_A=right_upper_point_in_A,
            sublattice_radius_in_A=sublattice_radius_in_A,
            device=device,
            dtype=dtype,
        )
