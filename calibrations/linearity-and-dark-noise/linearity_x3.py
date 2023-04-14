# -*- coding: utf-8 -*-
"""
Assessment of the linearity of the X3 camera
"""
import glob
# Module importation
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.geometric_rolloff import MatlabGeometricMengine
from linearity_fit import plot_linear_regression


if __name__ == "__main__":

    # Processing instance
    pim = ProcessImage()

    # Parameters
    bayer_pattern = "GBRG"
    mask_deg = 5  # degrees
    # Choosing lens
    cover = "nocover"
    sn = "2BW7X7"
    wlens = "back"

    # General path
    gen_path = f"/Volumes/MYBOOK/data-i360X3/calibrations/linearity/{sn}/{cover}/{wlens}/20230321"

    # Geometric calibration
    geo_calib = h5py.File("../geometric-calibration/calibrationfiles/geometric-calibration-2W7X7-air.h5")
    if wlens.lower() == "front":
        geo_calib = geo_calib["lens-front"]["20230212_115034"]
    elif wlens.lower() == "back":
        geo_calib = geo_calib["lens-front"]["20230212_115034"]

    geo = {}
    zen = {}
    for i in geo_calib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geo_calib["fp"][i], geo_calib["ierror"][i])
        r, z, a, = geo[i].angular_coordinates()
        zen[i] = z
    channel = {0: "red", 1: "green", 2: "blue"}  # Channel 3rd dimension correspondances


    # Ambiance list
    list_images_amb = glob.glob(gen_path + "/amb/*.dng")
    amb_imstack_exp, amb_exp, amb_iso, _ = pim.imagestack(list_images_amb, wlens, cam="x3")

    # Creating stack of averaged image
    folder_list = glob.glob(gen_path + "/data/*_*")
    bright_im_stack = np.empty((amb_imstack_exp.shape[0], amb_imstack_exp.shape[1], len(folder_list)))
    all_exposures = np.array([])  # all the exposure time appended

    for i, f in enumerate(folder_list):  # Loop for all the folders (each having 4 images) in /data directory

        print(os.path.basename(f))
        imstack_c, exp_c, _, _ = pim.imagestack(glob.glob(f + "/*.dng"), wlens, cam="x3")
        bright_im_stack[:, :, i] = imstack_c.mean(axis=2)

        all_exposures = np.append(all_exposures, exp_c)

    # Sorting
    unique_exposure = all_exposures[::4]
    argsort_exp = np.argsort(unique_exposure)
    unique_exposure = np.sort(unique_exposure)

    bright_im_stack = bright_im_stack[:, :, argsort_exp]

    # Loop to extract data
    exp_dn = np.zeros((bright_im_stack.shape[2], 3))
    exp_dn_noise = np.zeros((bright_im_stack.shape[2], 3))

    for i in range(bright_im_stack.shape[2]):

        # Dark removal
        imdrm = bright_im_stack[:, :, i].astype(float) - amb_imstack_exp[:, :, i].astype(float)
        #imdrm = bright_im_stack[:, :, i] - 1024.0

        # Downsampling
        im_dws = pim.dwnsampling(imdrm, bayer_pattern)

        for b in range(im_dws.shape[2]):
            curr_im = im_dws[:, :, b]
            mask_angular = zen[channel[b]] <= mask_deg

            pixel_values = curr_im[mask_angular]
            exp_dn[i, b] = pixel_values.mean()
            exp_dn_noise[i, b] = pixel_values.std()

    # Figures
    ff = FigureFunctions()
    plt.style.use("../../figurestyle.mplstyle")
    col = ['#d95f02', '#1b9e77', '#7570b3']

    # Linestyle
    ls = ["-", "-.", ":"]
    ms = ["o", "s", "^"]

    # Fig 1
    fs = ff.set_size(subplots=(2, 3))

    fig1, ax1 = plt.subplots(1, 3, sharey="row", figsize=(fs[0], fs[1] * 1.7))

    tx_exp = "$DN_{{i}} = m \cdot t_{{int}} + b$\n$m = ({0:.2f}\pm{1:.2f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$"

    for b in range(3):
        ax1[b].errorbar(unique_exposure * 1000, exp_dn[:, b], yerr=exp_dn_noise[:, b], color=col[b], linestyle="none", marker=ms[b], markersize=3, label="averaged $DN$")

        plot_linear_regression(ax1[b], unique_exposure * 1000, exp_dn[:, b], tx_exp, ls[b], postext=(0.45, 0.05), color=col[b])

        ax1[b].set_xlabel("exposure time $t_{int}$ [ms]")
        ax1[b].set_xscale("log")
        ax1[b].set_yscale("log")

    ax1[0].set_ylabel("$DN_{i}$ [ADU]")
    fig1.tight_layout()

    plt.show()
