# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description:
           Generate a bank of filters shaped on the MTF of the sensor. Each filter
           corresponds to a band acquired by the sensor.

 Interface:
           h = genMTF(ratio, sensor, nbands)

 Inputs:
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
           nbands:             Number of spectral bands.

 Outputs:
           h:                  Gaussian filter mimicking the MTF of the MS sensor.

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import numpy as np
from utils.sharpening_index.tools import (
    fir_filter_wind,
    fir_filter_wind_torch,
    gaussian2d,
    gaussian2d_torch,
    kaiser2d,
    kaiser2d_torch,
)


def genMTF(ratio, sensor, nbands):
    N = 41

    if sensor == "QB":
        GNyq = np.asarray(
            [0.34, 0.32, 0.30, 0.22], dtype="float32"
        )  # Band Order: B,G,R,NIR
    elif sensor == "IKONOS":
        GNyq = np.asarray(
            [0.26, 0.28, 0.29, 0.28], dtype="float32"
        )  # Band Order: B,G,R,NIR
    elif sensor == "GeoEye1" or sensor == "WV4":
        GNyq = np.asarray(
            [0.23, 0.23, 0.23, 0.23], dtype="float32"
        )  # Band Order: B,G,R,NIR
    elif sensor == "WV2":
        GNyq = [0.35 * np.ones(nbands), 0.27]
    elif sensor == "WV3":
        GNyq = np.asarray(
            [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315], dtype="float32"
        )
    else:
        GNyq = 0.3 * np.ones(nbands)

    """MTF"""
    h = np.zeros((N, N, nbands))

    fcut = 1 / ratio

    h = np.zeros((N, N, nbands))
    for ii in range(nbands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[ii])))
        H = gaussian2d(N, alpha)
        Hd = H / np.max(H)
        w = kaiser2d(N, 0.5)
        h[:, :, ii] = np.real(fir_filter_wind(Hd, w))

    return h


# * ==========================================================
# * torch version

import torch


def genMTF_torch(ratio, sensor, nbands):
    N = 41

    if sensor == "QB":
        GNyq = torch.tensor(
            [0.34, 0.32, 0.30, 0.22], dtype=torch.float32
        )  # Band Order: B,G,R,NIR
    elif sensor == "IKONOS":
        GNyq = torch.tensor(
            [0.26, 0.28, 0.29, 0.28], dtype=torch.float32
        )  # Band Order: B,G,R,NIR
    elif sensor in ["GeoEye1", "WV4"]:
        GNyq = torch.tensor(
            [0.23, 0.23, 0.23, 0.23], dtype=torch.float32
        )  # Band Order: B,G,R,NIR
    elif sensor == "WV2":
        GNyq = torch.tensor([0.35] * nbands + [0.27], dtype=torch.float32)
    elif sensor == "WV3":
        GNyq = torch.tensor(
            [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315],
            dtype=torch.float32,
        )
    else:
        GNyq = 0.3 * torch.ones(nbands, dtype=torch.float32)

    h = torch.zeros((N, N, nbands), dtype=torch.float32)

    fcut = 1 / ratio

    for ii in range(nbands):
        alpha = torch.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * torch.log(GNyq[ii])))
        H = gaussian2d_torch(N, alpha)
        Hd = H / H.max()
        w = kaiser2d_torch(N, 0.5)
        h[:, :, ii] = fir_filter_wind_torch(Hd, w).real

    return h


if __name__ == "__main__":
    mtf_np = genMTF(4, "QB", 4)
    mtf_torch = genMTF_torch(4, "QB", 4)

    print((torch.as_tensor(mtf_np, dtype=torch.float32) - mtf_torch).abs().max())
