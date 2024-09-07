# -*- coding: utf-8 -*-
"""
Baie des Chaleurs by Bastien.
"""

# Module importation
import os
import glob
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.radiance import ImageRadiancei360


def create_label(text_file_path):
    """
    Get depth from image name.
    :param text_file_path: path to txt file
    :return: depth_label (dictionary)
    """

    op_txt = open(text_file_path, "r")
    lines = op_txt.readlines()
    idx_start = lines.index('depth\tfilename\n')
    newlines = lines[idx_start + 1:]
    dc = {}

    for i in newlines:
        sep = i.strip().split("    ")
        dc[float(sep[0])] = sep[1]

    return dc


def get_ice_freeboard(text_file_path):
    """
    Return ice freeboard from readme file.
    :param text_file_path: path to txt file
    :return: ice freaboard
    """
    open_txtfile = open(text_file_path, "r")
    all_lines = open_txtfile.readlines()
    thickness_str = [s for s in all_lines if "ice thickness:" in s]
    if len(thickness_str) == 1:
        thickness_str = thickness_str[0]

    ice_draft = thickness_str.split(',')[1].strip()

    return float(ice_draft.split(":")[1].split("\\")[0])


def save_radiance_data(station_name, rad_dct, path_name="data/baiedeschaleurs-03232022.h5"):
    """

    :param path_name:
    :param station_name:
    :param rad_dct:
    :return:
    """
    with h5py.File(path_name, "a") as hf:
        for ke in rad_dct.keys():
            data_name = station_name + "/" + ke
            if data_name in hf:
                dt = hf[data_name]
                dt[...] = rad_dct[ke]
            else:
                hf.create_dataset(data_name, data=rad_dct[ke])


if __name__ == "__main__":

    pim = ProcessImage()

    # Chose profile
    #path_to_data = pim.folder_choice(r"D:\data-i360\field\baideschaleurs-03232022")
    path_to_data = "/Volumes/MYBOOK/data-i360/field/baideschaleurs-03232022/station_4"
    impath = glob.glob(path_to_data + "/*.dng")
    impath.sort()

    # Create label
    txt_file = glob.glob(path_to_data + "/*.txt")[0]
    depth_label = create_label(txt_file)

    # Get ice freeboard
    ice_fb = get_ice_freeboard(txt_file)

    # Station numero
    station = os.path.basename(path_to_data)

    # Figure pre-allocation
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")
    fig1, ax1 = plt.subplots(2, 3, sharex=True, figsize=(8, 4.59))

    # Colormap
    colornormdict = dict(zip(depth_label.values(), depth_label.keys()))
    colo = matplotlib.cm.get_cmap("viridis", len(colornormdict.values()))
    cmit = iter(colo.colors)

    # Initialize irradiance
    ed = np.zeros((len(depth_label.keys()), 3))
    eu, eo = ed.copy(), ed.copy()

    # Initialize radiance
    rad = {}

    # Loop
    for d, k in enumerate(depth_label.keys()):

        # Print current depth processed
        current_depth = "{0} cm".format(k)
        print(current_depth)

        # Water medium or air ?
        if k >= ice_fb:
            im_rad = ImageRadiancei360(path_to_data + "/" + depth_label[k], "water")
        else:
            im_rad = ImageRadiancei360(path_to_data + "/" + depth_label[k], "air")

        # Build radiance map
        #im_rad.get_radiance(dark_metadata=False)
        #im_rad.map_radiance(angular_resolution=1.0)  # 1 deg in angular resolution (zenith and azimuth)
        im_rad.get_radiance_angular_distribution()

        # Azimuthal average
        az_average = im_rad.azimuthal_average()

        # Radiance
        if d == 0:
            rad["zenith"] = im_rad.zenith_mesh.copy()
            rad["azimuth"] = im_rad.azimuth_mesh.copy()
        rad[current_depth] = im_rad.mappedradiance.copy()

        # Irradiance
        extra = False
        #interpo = False
        if im_rad.medium == "water":
            #im_rad.extrapolation_gaussian_function()
            extra = True

        ed[d, :] = im_rad.irradiance(0, 90, extrapolation=extra)
        eu[d, :] = im_rad.irradiance(90, 180, extrapolation=extra)
        eo[d, :] = im_rad.irradiance(0, 180, planar=False, extrapolation=extra)

        # Plot
        cl = next(cmit)
        for i in range(az_average.shape[1]):
            integration = az_average[:, i].copy()
            zenith = im_rad.zenith_mesh[:, 0].copy() * 180 / np.pi

            condzero = np.where(integration == 0)
            integration[condzero] = np.nan
            zenith[condzero] = np.nan

            integration_norm = (integration / np.nanmax(integration)) * 100

            # Ax1 - absolute
            ax1[0, i].plot(zenith, integration, linewidth=2, color=cl, label=depth_label[k])
            ax1[0, i].set_yscale("log")
            ax1[0, i].set_xlim((20, 160))
            #ax1[0, i].set_ylim((1e-5, 0.1))

            # Ax2 - normalization
            ax1[1, i].plot(zenith, integration_norm, linewidth=2, color=cl, label=depth_label[k])
            ax1[1, i].set_yticks(np.arange(0, 120, 20))
            ax1[1, i].set_xlabel("Zenith angle [˚]")

    # Saving data
    #save_radiance_data(station, rad)
    save_radiance_data(station, rad, path_name="data/baiedeschaleurs-03232022-imf-fluo.h5")

    # Figure 1
    ax1[0, 0].set_ylabel(r"$\overline{{L}}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
    ax1[1, 0].set_ylabel(r"$\frac{{\overline{{L}}}}{{\overline{{L}}_{{max}}}}$ [%]")

    fig1.tight_layout()

    # Figure 2 - irradiance
    fig2, ax2 = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(6.136, 3.784))

    band_name = ["r", "g", "b"]
    lstyle = ["-", "--", ":", "-."]

    depths = np.array(list(depth_label))

    ax2[0].plot(ed[:, 0], depths, linewidth=0.8, color="#a6cee3", linestyle=lstyle[0], label="$E_{d}$")
    ax2[0].plot(eu[:, 0], depths, linewidth=0.8, color="#1f78b4", linestyle=lstyle[0], label="$E_{u}$")
    ax2[0].plot(eo[:, 0], depths, linewidth=0.8, color="#b2df8a", linestyle=lstyle[0], label="$E_{0}$")

    ax2[1].plot(ed[:, 1], depths, linewidth=0.8, color="#a6cee3", linestyle=lstyle[0], label="$E_{d}$")
    ax2[1].plot(eu[:, 1], depths, linewidth=0.8, color="#1f78b4", linestyle=lstyle[0], label="$E_{u}$")
    ax2[1].plot(eo[:, 1], depths, linewidth=0.8, color="#b2df8a", linestyle=lstyle[0], label="$E_{0}$")

    ax2[2].plot(ed[:, 2], depths, linewidth=0.8, color="#a6cee3", linestyle=lstyle[0], label="$E_{d}$")
    ax2[2].plot(eu[:, 2], depths, linewidth=0.8, color="#1f78b4", linestyle=lstyle[0], label="$E_{u}$")
    ax2[2].plot(eo[:, 2], depths, linewidth=0.8, color="#b2df8a", linestyle=lstyle[0], label="$E_{0}$")

    ax2[0].set_xscale("log")
    ax2[0].invert_yaxis()

    ax2[0].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[1].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[2].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[0].legend(loc="best", frameon=False, fontsize=6)
    ax2[1].legend(loc="best", frameon=False, fontsize=6)
    ax2[2].legend(loc="best", frameon=False, fontsize=6)

    ax2[0].set_ylabel("Depth [cm]")
    fig2.suptitle(station)
    fig2.tight_layout()

    fig2.savefig("figures/irradiance_{0}.png".format(station), dpi=600, format="png")

    plt.show()
