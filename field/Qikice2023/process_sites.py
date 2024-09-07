"""
Process data from Insta360 X3 cam.
"""

# Module importation
import h5py
import os
import glob
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

from source.radiance import ImageRadiancei360x3
from source.processing import ProcessImage


# Classes and functions
def save_radiance_data(station_name, rad_dct, path_name="data/baiedeschaleurs-03232022.h5"):
    """
    Save radiance data as h5 format.
    :param station_name:
    :type station_name:
    :param rad_dct:
    :type rad_dct:
    :param path_name:
    :type path_name:
    :return:
    :rtype:
    """
    with h5py.File(path_name, "a") as hf:
        for ke in rad_dct.keys():
            data_name = station_name + "/" + ke
            if data_name in hf:
                dt = hf[data_name]
                dt[...] = rad_dct[ke]
            else:
                hf.create_dataset(data_name, data=rad_dct[ke])


def create_label(txt_file, start_str_line="Filename, Depth \n"):
    """

    :param txt_file:
    :type txt_file:
    :return:
    :rtype:
    """
    op_txt = open(txt_file, "r")
    lines = op_txt.readlines()
    idx_start = lines.index(start_str_line)
    newlines = lines[idx_start+1:]

    dc = {}
    for i in newlines:
        sep = i.strip().split(",")
        try:
            dc[float(sep[1][:-2])] = sep[0]
        except Exception as err:
            print(err)
    # Freeboard
    freeboard_str = [s for s in lines if "Freeboard:" in s]
    ice_draft = freeboard_str[0].split(":")[1].strip()

    # Snow
    snow_str = [s for s in lines if "Snow cover:" in s]
    snow_cover = snow_str[0].split(":")[1].strip()

    # Ice thick
    ice_th_str = [s for s in lines if "Ice thickness:" in s]
    ice_th_str = ice_th_str[0].split(":")[1].strip()

    # Cam wet
    cwet = [s for s in lines if "Cam wet:" in s]
    if cwet:
        cwet = float(cwet[0].split(":")[1].strip()[:-2])
    else:
        cwet = None
    return dc, float(ice_draft[:-2]), float(snow_cover[:-2]), float(ice_th_str[:-2]), cwet


if __name__ == "__main__":
    # ******** HEADER **********
    # Information about camera
    #sn = "2BW7X7"
    #cover = "cover"
    sn = "2C9JCA"  # Camera serial number ("2C9JCA" or "2BW7X7")
    cover = "nocover"  # Camera with or without cover ("cover" or "nocover")

    # Normalized naming for the profile
    station = 2  # Station number (TO CHANGE)
    site = 1  # Site number  (TO CHANGE)
    num = 0  # Profile number  (TO CHANGE)

    norm_name = f"QI{station:02}{site:1}{num:1}"  # data id (TO CHANGE according to mission)

    # Choose profile
    # path_to_data: path to the raw .dng files (TO CHANGE according to mission)
    path_to_data = f"/Volumes/MYBOOK/QikIce2023/Qik2023/QI{station:02}/{norm_name}_radiance_raw"
    # ******** HEADER **********

    # Initialize objects
    process_im = ProcessImage()

    impath = glob.glob(path_to_data + "/*.dng")
    impath.sort()

    # Create label
    # Always create a README.txt with the raw images. Has specific format to respect !! (see example)
    txt_file = glob.glob(path_to_data + "/README.txt")[0]
    depth_label, ice_fb, snow, ice_thick, cam_wet = create_label(txt_file)

    # Figure pre-allocation
    #plt.style.use("../../figurestyle.mplstyle")
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

    # Water
    if cam_wet:
        water_lev = cam_wet
    else:
        water_lev = ice_fb + snow
    print(f'Water level: {water_lev} cm')

    # Loop
    for d, k in enumerate(depth_label.keys()):

        # Print current depth processed
        current_depth = "{0} cm".format(k)
        print(current_depth)

        if k >= water_lev:

            im_rad = ImageRadiancei360x3(path_to_data + f"/{depth_label[k]}.dng",
                                         cam_sn=sn, medium="water", cover=cover)
        else:
            im_rad = ImageRadiancei360x3(path_to_data + f"/{depth_label[k]}.dng",
                                         cam_sn=sn, medium="air", cover=cover)

        # Azimuthal average
        im_rad.get_radiance_angular_distribution()

        # Azimuthal average
        az_average = im_rad.azimuthal_average()

        # Radiance
        if d == 0:
            rad["zenith"] = im_rad.zenith_mesh.copy()
            rad["azimuth"] = im_rad.azimuth_mesh.copy()
        rad[current_depth] = im_rad.mapped_radiance.copy()
        #rad[current_depth] = im_rad.mapped_radiance_4pi.copy()

        # Irradiance calculation !
        extra = False
        if im_rad.medium == "water":
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

    # Save radiance data
    # Within the folder of the data, it will create a subfolder "processed_data"
    filename = norm_name + f"_radiance.h5"  # Name of the h5 file ({id}_radiance.h5) to save in the new folder
    path_to_save = path_to_data + "/processed_data"

    if not os.path.isdir(path_to_save):
        os.mkdir(path_to_save)

    save_radiance_data(norm_name, rad, path_to_save + f"/{filename}")

    # Figure 1
    ax1[0, 0].set_ylabel(r"$\overline{{L}}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
    ax1[1, 0].set_ylabel(r"$\frac{{\overline{{L}}}}{{\overline{{L}}_{{max}}}}$ [%]")

    fig1.suptitle(norm_name)
    fig1.tight_layout()

    # Figure 2 - irradiance
    fig2, ax2 = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(6.136, 3.784))

    band_name = ["r", "g", "b"]
    lstyle = ["-", "--", ":", "-."]

    depths = np.array(list(depth_label))

    ax2[0].plot(ed[:, 0], depths, linewidth=0.8, color="#a6cee3", linestyle=lstyle[0], label="$E_{d}$")
    ax2[0].plot(eu[:, 0], depths, linewidth=0.8, color="#1f78b4", linestyle=lstyle[0], label="$E_{u}$")
    ax2[0].plot(eo[:, 0], depths, linewidth=0.8, color="#b2df8a", linestyle=lstyle[0], label="$E_{0}$")

    ax2[0].axhline(snow, linestyle="-", linewidth=0.6, color="k")
    ax2[0].axhline(snow + ice_fb, linestyle="--", linewidth=0.6, color="k")
    ax2[0].axhline(snow + ice_thick, linestyle="-.", linewidth=0.6, color="k")

    ax2[1].plot(ed[:, 1], depths, linewidth=0.8, color="#a6cee3", linestyle=lstyle[0], label="$E_{d}$")
    ax2[1].plot(eu[:, 1], depths, linewidth=0.8, color="#1f78b4", linestyle=lstyle[0], label="$E_{u}$")
    ax2[1].plot(eo[:, 1], depths, linewidth=0.8, color="#b2df8a", linestyle=lstyle[0], label="$E_{0}$")

    ax2[1].axhline(snow, linestyle="-", linewidth=0.6, color="k")
    ax2[1].axhline(snow + ice_fb, linestyle="--", linewidth=0.6, color="k")
    ax2[1].axhline(snow + ice_thick, linestyle="-.", linewidth=0.6, color="k")

    ax2[2].plot(ed[:, 2], depths, linewidth=0.8, color="#a6cee3", linestyle=lstyle[0], label="$E_{d}$")
    ax2[2].plot(eu[:, 2], depths, linewidth=0.8, color="#1f78b4", linestyle=lstyle[0], label="$E_{u}$")
    ax2[2].plot(eo[:, 2], depths, linewidth=0.8, color="#b2df8a", linestyle=lstyle[0], label="$E_{0}$")

    ax2[2].axhline(snow, linestyle="-", linewidth=0.6, color="k")
    ax2[2].axhline(snow + ice_fb, linestyle="--", linewidth=0.6, color="k")
    ax2[2].axhline(snow + ice_thick, linestyle="-.", linewidth=0.6, color="k")

    ax2[0].set_xscale("log")
    ax2[0].invert_yaxis()

    ax2[0].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[1].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[2].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[0].legend(loc="best", frameon=False, fontsize=6)
    ax2[1].legend(loc="best", frameon=False, fontsize=6)
    ax2[2].legend(loc="best", frameon=False, fontsize=6)

    ax2[0].set_ylabel("Depth [cm]")
    fig2.suptitle(norm_name)
    fig2.tight_layout()

    # Save images
    # Saving radiance and irradiance figures in the "processed_data" folder
    fig1.savefig(path_to_save + f"/{norm_name}_radiance.png", dpi=600, format="png")
    fig2.savefig(path_to_save + f"/{norm_name}_irradiance.png", dpi=600, format="png")

    plt.show()
