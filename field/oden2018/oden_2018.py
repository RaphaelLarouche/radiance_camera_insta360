# -*- coding: utf-8 -*-
"""
Oden icebreaker A02018 mission, cam optic.
"""

# Module importation
import os
import h5py
import glob
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.radiance import ImageRadiancei360


# Function and classes
def imagelabel(path):
    """
    Get label (detph) related to image name from the readme file.

    :param path: absolute path to the readme file (str)
    :return: dictionary of label with the image name for key (dct)
    """
    labels = {}
    with open(path, "r") as f:
        for lines in f:
            name, depth = lines.split("*")
            labels[name] = depth.strip()
    return labels


def save_radiance_image_hdf5(path_name, data_name, dat):
    """
    Save radiance image in hdf5 files.

    :param path_name: absolute path to save data (str)
    :param data_name: tag for data (depth, zenith or azimuth), (str)
    :param dat: radiance image (array)
    :return:
    """

    datapath = data_name
    with h5py.File(path_name) as hf:
        if datapath in hf:
            d = hf[datapath]  # load the data
            d[...] = dat
        else:
            dset = hf.create_dataset(data_name, data=dat)


if __name__ == "__main__":

    # Object ProcessImage
    process_im = ProcessImage()

    # Object FigureFunction
    ff = FigureFunctions()

    # Oden images
    oden_path = "/Volumes/MYBOOK/data-i360/field/oden-08312018/"
    oden_impath = glob.glob(oden_path + "*.dng")
    oden_impath.sort()

    im_labels = imagelabel(oden_path + "ReadMe_python.txt")
    wanted_depth = ["zero minus",
                    "20 cm (in water)",
                    "40 cm", "60 cm", "80 cm", "100 cm", "120 cm", "140 cm", "160 cm", "180 cm", "200 cm"]

    oden_impath_filtered = oden_impath[2:-9:]
    oden_impath_filtered[-1] = oden_impath[-9]

    # Data saving parameters
    path_save_rad = "data/oden-08312018.h5"
    answ = process_im.save_results(text="Do you want to save the radiance angular distributions?")
    cond_save = answ == "y"

    # Figure pre-allocation
    plt.style.use("../../figurestyle.mplstyle")
    fig1, ax1 = plt.subplots(2, 3, sharex=True, figsize=(8, 4.59))

    # Colormap
    colornormdict = dict(zip(wanted_depth, np.arange(0, 220, 20)))
    colo = matplotlib.cm.get_cmap("viridis", len(colornormdict.values()))
    cmit = iter(colo.colors)

    # Loop
    for f in oden_impath_filtered:

        # Get corresponding depth of the current path
        _, tail = os.path.split(os.path.splitext(f)[0])
        depth = im_labels[tail]

        # Filter image according to wanted depth
        if depth in wanted_depth:

            print("Evaluation current depth: {}".format(depth))

            if depth == "zero minus":
                im_rad = ImageRadiancei360(f, "air")
            else:
                im_rad = ImageRadiancei360(f, "water")

            # Build radiance map
            im_rad.get_radiance(dark_metadata=False)
            im_rad.map_radiance(angular_resolution=1.0)  # 1 deg in angular resolution (zenith and azimuth)

            # Azimuthal average
            az_average = im_rad.azimuthal_average()

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
                ax1[0, i].plot(zenith, integration, linewidth=2, color=cl, label=depth)
                ax1[0, i].set_yscale("log")
                ax1[0, i].set_xlim((20, 160))
                ax1[0, i].set_ylim((1e-5, 0.1))

                # Ax2 - normalization
                ax1[1, i].plot(zenith, integration_norm, linewidth=2, color=cl, label=depth)
                ax1[1, i].set_yticks(np.arange(0, 120, 20))
                ax1[1, i].set_xlabel("Zenith angle [˚]")

            # Saving data
            if cond_save:
                save_radiance_image_hdf5(path_save_rad, depth, im_rad.mappedradiance.copy())
                save_radiance_image_hdf5(path_save_rad, "zenith", im_rad.zenith_mesh.copy() * 180 / np.pi)
                save_radiance_image_hdf5(path_save_rad, "azimuth", im_rad.azimuth_mesh.copy() * 180 / np.pi)
        else:
            continue

    # Figure parameters
    # Figure 1
    ax1[0, 0].set_ylabel(r"$\overline{{L}}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
    ax1[1, 0].set_ylabel(r"$\frac{{\overline{{L}}}}{{\overline{{L}}_{{max}}}}$ [%]")

    fig1.tight_layout()

    plt.show()
