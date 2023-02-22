# -*- coding: utf-8 -*-
"""
Cumulative irradiance figure.
"""

# Module importation
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt


from source.radiance import ImageRadiancei360

if __name__ == "__main__":

    # Create ImageRadiance instance
    path_to_file = "data/IMG_20180831_181226_094.dng"
    im_rad = ImageRadiancei360(path_to_file, "water")

    # Retrieve radiance
    im_rad.get_radiance_angular_distribution()

    # Calculate irradiance value from 1D curves
    zen_rad = np.arange(0, 181, 1) * np.pi / 180

    inte_r = np.polynomial.legendre.legval(np.cos(zen_rad), im_rad.legendre_coefficients) * np.sin(zen_rad) * np.cos(zen_rad)
    ed_rad_smoothed = 2 * np.pi * simps(inte_r, x=zen_rad)

    amax_ed = np.arange(0, 91, 1)
    amax_eu = np.arange(90, 181, 1)

    ed = np.zeros((amax_ed.shape[0], 3))
    ed_raw = ed.copy()

    eu = np.zeros((amax_eu.shape[0], 3))

    for i, a in enumerate(zip(amax_ed, amax_eu)):

        # Downwelling
        ed[i, :] = im_rad.irradiance(0, a[0], planar=True, extrapolation=True)  # with extrapolation
        ed_raw[i, :] = im_rad.irradiance(0, a[0], planar=True, extrapolation=False)  # no extrapolation

        # Upwelling
        eu[i, :] = im_rad.irradiance(90, a[1], planar=True, extrapolation=True)

    # Figure
    plt.style.use("../../figurestyle.mplstyle")
    fig1, ax1 = plt.subplots(1, 3)

    ax1[0].plot(amax_ed, ed[:, 0])
    ax1[1].plot(amax_ed, ed[:, 1])
    ax1[2].plot(amax_ed, ed[:, 2])

    ax1[0].plot(amax_ed, ed_raw[:, 0])
    ax1[1].plot(amax_ed, ed_raw[:, 1])
    ax1[2].plot(amax_ed, ed_raw[:, 2])

    ax1[0].scatter(90.0, ed_rad_smoothed, marker="o")

    #ax1[0].plot(amax_eu - 90, eu[:, 0]/np.max(eu[:, 0]))
    #ax1[1].plot(amax_eu - 90, eu[:, 1]/np.max(eu[:, 1]))
    #ax1[2].plot(amax_eu - 90, eu[:, 2]/np.max(eu[:, 2]))

    ax1[0].set_xlabel("Angle of incidence (°)")
    ax1[0].set_ylabel("Cumulative downwelling irradiance (W/m²)")

    ax1[1].set_xlabel("Angle of incidence (°)")
    ax1[1].set_ylabel("Cumulative downwelling irradiance (W/m²)")

    ax1[2].set_xlabel("Angle of incidence (°)")
    ax1[2].set_ylabel("Cumulative downwelling irradiance (W/m²)")

    fig1.tight_layout()

    plt.show()
