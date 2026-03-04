"""Peng 1996 element scattering factor parameters."""

import os

import numpy as np
import torch


def _load_peng_element_scattering_factor_parameter_table():
    path = os.path.join(os.path.dirname(__file__), "peng1996_element_params.npy")
    return np.load(path)


class ScatteringAttributes:
    """Scattering attributes from Peng 1996 parameter table."""

    def __init__(
        self,
        device: torch.DeviceObjType = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.peng_element_scattering_factor_parameter_table = torch.from_numpy(
            _load_peng_element_scattering_factor_parameter_table()
        ).to(device).to(dtype=dtype)

    def __call__(self, atom_identities):
        return self.peng_element_scattering_factor_parameter_table[:, atom_identities]
