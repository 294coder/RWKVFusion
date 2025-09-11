# -*- coding: utf-8 -*-
import numpy as np
from utils.sharpening_index.tools import fir_filter_wind, gaussian2d, kaiser2d


def genMTF_pan(ratio, sensor, nbands):
    N = 41

    if sensor == "QB":
        GNyq = 0.15
    elif sensor == "IKONOS":
        GNyq = 0.17
    elif sensor in ["GeoEye1", "WV4"]:
        GNyq = 0.16
    elif sensor == "WV2":
        GNyq = 0.11
    elif sensor == "WV3":
        GNyq = 0.14
    else:
        GNyq = 0.15

    """MTF"""
    fcut = 1 / ratio
    # h = np.zeros((N, N, nbands))
    alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = np.real(fir_filter_wind(Hd, w))

    return h


# * ==========================================================
# * Torch version

import torch
from utils.sharpening_index.tools import (
    fir_filter_wind_torch,
    gaussian2d_torch,
    kaiser2d_torch,
)


def genMTF_pan_torch(ratio: int, sensor: str, nbands: int):
    N = 41

    # Define the Nyquist frequency based on the sensor type
    if sensor == "QB":
        GNyq = 0.15
    elif sensor == "IKONOS":
        GNyq = 0.17
    elif sensor in ["GeoEye1", "WV4"]:
        GNyq = 0.16
    elif sensor == "WV2":
        GNyq = 0.11
    elif sensor == "WV3":
        GNyq = 0.14
    else:
        GNyq = 0.15

    """MTF"""
    fcut = 1 / ratio

    # Calculate alpha and generate Gaussian filter using PyTorch tensors
    alpha = torch.sqrt(
        torch.tensor(((N - 1) * (fcut / 2)) ** 2 / (-2 * torch.log(torch.tensor(GNyq))))
    )
    H = gaussian2d_torch(
        N, alpha.item()
    )  # Assuming gaussian2d still returns a NumPy array
    Hd = H / H.max()

    # Generate Kaiser window and apply it to the filter
    w = kaiser2d_torch(N, 0.5)
    h = torch.real(fir_filter_wind_torch(Hd, w))

    # Convert the final filter to a PyTorch tensor before returning
    return torch.tensor(h, dtype=torch.float32)


if __name__ == "__main__":
    h_numpy = genMTF_pan(4, "GF2", 4)
    h_torch = genMTF_pan_torch(4, "GF2", 4)
    print("NumPy MTF:", h_numpy)
    print("PyTorch MTF:", h_torch.numpy())
