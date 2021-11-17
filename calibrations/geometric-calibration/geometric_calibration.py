# -*- coding: utf-8 -*-
"""
Geometric calibration using Scaramuzza et al. (2006) algorithm.
"""

# Module importation
import os
import matlab.engine  # Matlab engine needed for geometric calibration see: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
import time
import string
import timeit
import deepdish
import numpy as np
import matplotlib.pyplot as plt

# Other module importation
import source.processing as processing
from source.geometric_rolloff import MatlabGeometricMengine


# Functions
def show_reprojectionerrors(band_data, matlab_data, geodict, imagenumber, band="red"):
    """

    :param band_data:
    :param matlab_data:
    :param geodict:
    :param band:
    :return:
    """

    if np.divmod(np.sqrt(imagenumber), 1)[1] == 0:

        images = band_data[band]["image_tot"]
        repropoints = geodict[band].fisheye_params["ReprojectedPoints"]
        corners = np.array(matlab_data[band])

        images = images[:imagenumber, :, :]
        repropoints = repropoints[:, :, :imagenumber]
        corners = corners[:, :, :imagenumber]

        fig, ax = plt.subplots(np.sqrt(imagenumber).astype(int), np.sqrt(imagenumber).astype(int))

        for i, a in enumerate(ax.ravel()):

            a.plot(corners[:, 0, i], corners[:, 1, i], "o", markersize=4, markeredgecolor="g", markerfacecolor="none", label="algorithm detection")
            a.plot(repropoints[:, 0, i], repropoints[:, 1, i], "r+", markersize=4, label="reprojected points")

            xlm = a.get_xlim()
            ylm = a.get_ylim()

            a.imshow(images[i, :, :], cmap="viridis")

            a.set_xlim(xlm)
            a.set_ylim(ylm)

            a.set_xticklabels([])
            a.set_yticklabels([])

        ax[0, 0].legend(loc="best")

    else:
        raise ValueError("Invalid number of image")

    return fig, ax


if __name__ == "__main__":

    # Figure size instance
    ff = processing.FigureFunctions()

    # Processimage instance
    process = processing.ProcessImage()

    # List of image path
    imlist = process.imageslist(process.folder_choice("/Volumes/MYBOOK/data-i360/calibrations/geometric"))  # Lensclose1 & Lensfar2 --- bests

    if "lensclose" in imlist[0]:
        which = "close"
    elif "lensfar" in imlist[0]:
        which = "far"
    else:
        raise ValueError("Invalid naming.")

    # Med
    if "water" in imlist[0]:
        med = "water"
    elif "air" in imlist[0]:
        med = "air"
    else:
        raise ValueError("Invalid path image.")

    # Date
    creation_time = os.path.getmtime(imlist[0])
    date = time.strftime("%Y%m%d_%H%M", time.gmtime(creation_time))

    data = {"i": 0, "point_x": np.array([]), "point_y": np.array([]), "image_tot": np.array([])}
    band_data = {"red": data, "green": data.copy(), "blue": data.copy()}

    # Corner detection algorithm
    for a, path in enumerate(imlist):
        print("Processing image number {}".format(a+1))

        im, met = process.readDNG_insta360_np(path, which)

        gray = process.raw2gray(im, met, 3.0)
        gray_dws = process.dwnsampling(gray, "RGGB").astype(np.uint8)

        for n, k in enumerate(band_data.keys()):
            corners = process.detect_corners(gray_dws[:, :, n], vis=False)

            if np.any(corners):

                band_data[k]["i"] += 1
                band_data[k]["point_x"] = np.append(band_data[k]["point_x"], corners[:, 0])
                band_data[k]["point_y"] = np.append(band_data[k]["point_y"], corners[:, 1])
                band_data[k]["image_tot"] = np.append(band_data[k]["image_tot"], gray_dws[:, :, n].flatten())

    band_matlab_data = {"red": np.array([]), "green": np.array([]), "blue": np.array([])}
    imshape = process.imageshape(imlist[0], "close")

    for ke in band_data.keys():

        band_matlab_data[ke] = np.empty((48, 2, band_data[ke]["i"]))
        band_matlab_data[ke][:, 0, :] = band_data[ke]["point_x"].reshape((band_data[ke]["i"], 48)).T
        band_matlab_data[ke][:, 1, :] = band_data[ke]["point_y"].reshape((band_data[ke]["i"], 48)).T

        band_matlab_data[ke] = np.delete(band_matlab_data[ke], np.s_[::8], 0)
        band_matlab_data[ke] = np.delete(band_matlab_data[ke], np.s_[6::7], 0)
        band_matlab_data[ke] = matlab.double(band_matlab_data[ke].tolist())

        band_data[ke]["image_tot"] = band_data[ke]["image_tot"].reshape((-1, imshape[0], imshape[1]))

    # Matlab geometric calibration
    # Pre-allocation
    plt.style.use("../../figurestyle.mplstyle")

    fig1, ax1 = plt.figure(figsize=ff.set_size(443.86319, height_ratio=0.45)), []
    ax1.append(fig1.add_subplot(121))
    ax1.append(fig1.add_subplot(122, projection="3d"))

    ls = ["-", "-.", ":"]
    lab = ["602 nm", "544 nm", "484 nm"]
    color = ['#d62728', '#2ca02c', '#1f77b4']
    geo = {}
    fp = {}
    ierror = {}

    starttime = timeit.default_timer()
    eng = matlab.engine.start_matlab()
    print("Opening Matlab engine took :", timeit.default_timer() - starttime)

    for n, k in enumerate(band_matlab_data.keys()):

        print("Band number {}".format(n))

        eng.cd(os.path.dirname(__file__))

        fp[k], ierror[k] = eng.scaramuzza_calibration(matlab.double([imshape[0]]), matlab.double([imshape[1]]), band_matlab_data[k], med, nargout=2)

        geo[k] = MatlabGeometricMengine(fp[k], ierror[k])
        geo[k].print_results()
        res = geo[k].get_results()

        # Interpolation
        r_dws = np.linspace(0, 810 * 1.05, 1000)
        z_dws = np.interp(r_dws, res[0], res[1])  # interpolation
        ax1[0].plot(r_dws, z_dws, color=color[n], linestyle=ls[n], label=lab[n])

    ax1[0].plot(np.interp(76, res[1], res[0]), 76, markersize=5, marker="d", markerfacecolor="none", color="k")
    ax1[0].annotate("76˚ FoV limit", xy=(np.interp(76, res[1], res[0]), 76), xytext=(-50, -70), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", shrinkA=3, shrinkB=3, color="red"), fontsize=9)

    # Figures
    ax1[0].set_xticks(np.arange(0, 1000, 100))
    ax1[0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax1[0].transAxes, size=11, weight='bold')

    ax1[1] = geo["red"].draw_targets(ax1[1])
    ax1[1].view_init(elev=35, azim=-122)

    ax1[1].set_xlim((-150, 150))
    ax1[1].set_zlim((-50, 150))
    ax1[1].set_ylim((0, 200))

    ax1[1].set_yticks(np.arange(0, 250, 50))
    ax1[1].set_zticks(np.arange(-50, 200, 50))

    ax1[1].text2D(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax1[1].transAxes, size=11, weight='bold')

    ax1[0].legend(loc="best")
    ax1[0].set_xlabel(r"$\rho$ [px]")
    ax1[0].set_ylabel(r"$\theta$ [˚]")

    # Figure 2 - Plot differential between imaging function in pixels
    fr = geo["red"].imagingfunction(r_dws, geo["red"].mapping_coefficients)
    fg = geo["green"].imagingfunction(r_dws, geo["green"].mapping_coefficients)
    fb = geo["blue"].imagingfunction(r_dws, geo["blue"].mapping_coefficients,)

    fig2, ax2 = plt.subplots(2, 1, figsize=ff.set_size(443.86319, subplots=(2, 1), fraction=0.8), sharex=True)

    ax2[0].plot(r_dws, fr - fr, linestyle="-", label="red pixels")
    ax2[0].plot(r_dws, fr - fg, linestyle="-.", label="green pixels")
    ax2[0].plot(r_dws, fr - fb, linestyle=":", label="blue pixels")

    angle_diff_r = np.absolute((180 / np.pi) * np.arctan2(r_dws, fr) - (180 / np.pi) * np.arctan2(r_dws, fr))
    angle_diff_g = np.absolute((180 / np.pi) * np.arctan2(r_dws, fr) - (180 / np.pi) * np.arctan2(r_dws, fg))
    angle_diff_b = np.absolute((180 / np.pi) * np.arctan2(r_dws, fr) - (180 / np.pi) * np.arctan2(r_dws, fb))

    ax2[1].plot(r_dws, angle_diff_r)
    ax2[1].plot(r_dws, angle_diff_g)
    ax2[1].plot(r_dws, angle_diff_b)

    ax2[0].set_yticks(np.arange(-2, 8, 2))

    ax2[0].set_xlabel(r"$\rho$ [px]")
    ax2[0].set_ylabel(r"$f_{red}(\rho) - f(\rho)$ [px]")

    ax2[0].legend(loc="best")

    fig3, ax3 = show_reprojectionerrors(band_data, band_matlab_data, geo, 4,  band="red")

    fig1.tight_layout(w_pad=1.5)
    fig2.tight_layout()
    fig3.tight_layout()

    # Saving calibration results
    answer = process.save_results()
    if answer == "y":

        # Type of medium
        acq_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.stat(imlist[0])[-2]))

        savd = {"lens-{}".format(which): {acq_time: {"fp": fp, "ierror": ierror}}}
        deepdish.io.save("calibrationfiles/geometric-calibration-{}.h5".format(med), savd)
        fig1.savefig("figures/projection_{0}_{1}.pdf".format(med, which), format="pdf", dpi=600)

    plt.show()
