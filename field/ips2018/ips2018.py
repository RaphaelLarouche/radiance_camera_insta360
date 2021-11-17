# -*- coding: utf-8 -*-
"""
IPS2018 on Amundsen icebreaker, cam optic on ice floe.
"""

# Module importation
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.radiance import ImageRadiancei360
import source.radiance as r
from field.oden2018 import oden_2018_irradiance_profile as irr_oden_2018

if __name__ == "__main__":

    # Object ProcessImage
    process_im = ProcessImage()

    # Path to dng files
    ips_path = process_im.folder_choice()  # Backup_dossier_OneDrive_UL/Insta360/Insta360_Data_Amundsen/Before_bug/"
    ips_imname = glob.glob(ips_path + "/*.dng")
    ips_imname.sort()

    # # Mask
    wanted_images = ["006", "007", "008", "009", "010", "011", "013", "014", "015", "017", "018", "019", "020",
                     "021", "022"]
    #wanted_images = ["006", "007", "008", "009", "010", "011", "013", "014"]
    m = [os.path.splitext(os.path.basename(i).split("_")[3])[0] in wanted_images for i in ips_imname]
    depths = np.array([0.1 + i*0.1 for i in range(len(wanted_images))])

    # Figure pre-allocation
    plt.style.use("../../figurestyle.mplstyle")
    fig1, ax1 = plt.subplots(2, 3, sharex=True, figsize=(8, 4.59))

    colo = matplotlib.cm.get_cmap("viridis", len(wanted_images))
    cmit = iter(colo.colors)

    ips_imname = np.array(ips_imname)

    # Loop
    Ed = np.zeros(len(wanted_images), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))
    Eu, Eo, Edo, Euo = Ed.copy(), Ed.copy(), Ed.copy(), Ed.copy()

    for pn, p in enumerate(ips_imname[np.array(m)]):

        im_rad = ImageRadiancei360(p, "water")

        # Build radiance map
        im_rad.get_radiance(dark_metadata=False)
        im_rad.map_radiance(angular_resolution=1.0)  # 1 deg in angular resolution (zenith and azimuth)

        # Azimuthal average
        az_average = im_rad.azimuthal_average()

        # Interpolation
        im_rad.interpolation_gaussian_function()

        cl = next(cmit)
        for n in range(az_average.shape[1]):

            integration = az_average[:, n].copy()
            zenith = im_rad.zenith_mesh[:, 0].copy() * 180 / np.pi

            condzero = np.where(integration == 0)
            integration[condzero] = np.nan
            zenith[condzero] = np.nan

            integration_norm = (integration / np.nanmax(integration)) * 100

            # Ax1 - absolute
            ax1[0, n].plot(zenith, integration, linewidth=2, color=cl)
            ax1[0, n].set_yscale("log")
            ax1[0, n].set_xlim((20, 160))

            # Ax2 - normalization
            ax1[1, n].plot(zenith, integration_norm, linewidth=2, color=cl)
            ax1[1, n].set_yticks(np.arange(0, 120, 20))
            ax1[1, n].set_xlabel("Zenith angle [˚]")

        # Irradiances
        Ed[pn] = tuple(im_rad.irradiance(0, 90, planar=True, interpolation=True))
        Edo[pn] = tuple(im_rad.irradiance(0, 90, planar=False, interpolation=True))

        # Up-welling irradiance
        Eu[pn] = tuple(im_rad.irradiance(90, 180, planar=True, interpolation=True))
        Euo[pn] = tuple(im_rad.irradiance(90, 180, planar=False, interpolation=True))

        # Scalar irradiance
        Eo[pn] = tuple(im_rad.irradiance(0, 180, planar=False, interpolation=True))

    # Figure 2
    fig2, ax2 = plt.subplots(1, 3, sharey=True)
    ax2[0].plot(Ed["r"], depths, color="r")
    ax2[0].plot(Eu["r"], depths, color="r")
    ax2[0].plot(Eo["r"], depths, color="r")

    ax2[1].plot(Ed["g"], depths, color="g")
    ax2[1].plot(Eu["g"], depths, color="g")
    ax2[1].plot(Eo["g"], depths, color="g")

    ax2[2].plot(Ed["b"], depths, color="b")
    ax2[2].plot(Eu["b"], depths, color="b")
    ax2[2].plot(Eo["b"], depths, color="b")

    ax2[0].invert_yaxis()

    plt.show()
