"""Minimal AtomStack: coordinates, atomic numbers, B-factors, occupancies. No IO, no transforms."""

from __future__ import annotations

from copy import deepcopy

import gemmi
import torch


class AtomStack:
    """
    Atoms with coordinates, identities, B-factors, and occupancies.
    Create directly via constructor, from_coords_and_names, or from_coords_and_atomic_numbers.
    """

    def __init__(
        self,
        atom_coordinates: torch.Tensor,  # Shape (B, N, 3)
        atom_names: list[str],  # len = N
        bfactors: torch.Tensor | float | None,  # Shape (B, N, 1) or float
        device: torch.DeviceObjType = torch.device("cpu"),
        occupancies: torch.Tensor | None = None,  # Shape (B)
    ):
        assert atom_coordinates.ndim == 3 and atom_coordinates.shape[-1] == 3
        self.atom_coordinates = atom_coordinates.to(device)
        self.atom_names = atom_names
        self.atomic_numbers = torch.tensor(
            [gemmi.Element(atom_name).atomic_number for atom_name in atom_names],
            dtype=torch.int32,
            device=device,
        ).unsqueeze(-1)
        self.device = device

        B, N = atom_coordinates.shape[0], atom_coordinates.shape[1]
        if bfactors is None:
            self.bfactors = None
        elif isinstance(bfactors, float):
            self.bfactors = torch.ones((B, N, 1), device=device, dtype=torch.float32) * bfactors
        elif isinstance(bfactors, torch.Tensor):
            assert bfactors.ndim == 3 and bfactors.shape == (B, N, 1)
            self.bfactors = bfactors.to(device)
        else:
            raise TypeError(f"bfactors must be None, float, or Tensor, got {type(bfactors)}")

        self.occupancies = (
            occupancies
            if occupancies is not None
            else torch.ones((B,), device=device, dtype=torch.float32) / B
        )
        assert torch.allclose(self.occupancies.sum(), torch.ones(1, device=device))
        assert torch.all(self.occupancies >= 0.0) and torch.all(self.occupancies <= 1.0)

    @classmethod
    def from_coords_and_names(
        cls,
        atom_coordinates: torch.Tensor,
        atom_names: list[str],
        device: torch.DeviceObjType = torch.device("cpu"),
    ):
        return cls(atom_coordinates=atom_coordinates, atom_names=atom_names, bfactors=None, device=device)

    @classmethod
    def from_coords_and_atomic_numbers(
        cls,
        atom_coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        device: torch.DeviceObjType = torch.device("cpu"),
    ):
        atom_names = [gemmi.Element(int(z)).name for z in atomic_numbers.flatten()]
        return cls(atom_coordinates=atom_coordinates, atom_names=atom_names, bfactors=None, device=device)

    def fill_constant_bfactor(self, bfactor: float):
        B, N = self.atom_coordinates.shape[0], self.atom_coordinates.shape[1]
        self.bfactors = torch.ones((B, N, 1), device=self.device, dtype=torch.float32) * bfactor

    def copy(self) -> AtomStack:
        return deepcopy(self)

    def replicate_ensemble(self, B: int) -> AtomStack:
        """Replicate single-member stack into B-member ensemble."""
        assert self.atom_coordinates.shape[0] == 1
        coords = self.atom_coordinates.repeat(B, 1, 1)
        bf = self.bfactors.repeat(B, 1, 1) if self.bfactors is not None else None
        occ = torch.ones((B,), device=self.device, dtype=torch.float32) / B
        return AtomStack(
            atom_coordinates=coords,
            atom_names=self.atom_names,
            bfactors=bf,
            device=self.device,
            occupancies=occ,
        )
