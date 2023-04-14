# -*- coding: utf-8 -*-
"""
Roll-off calibration for X3 cameras.
"""


# Module importation
import os
import glob
import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Others module
import calibrations.calibrations_info
from source.processing import ProcessImage
from source.geometric_rolloff import MatlabGeometricMengine


# Function and classes
def sort_folders(dir_list):
    """
    Function that sorts the folder in order of angles.

    :param dir_list: list of all the directories
    :type dir_list: list
    :return:
    :rtype:
    """
    angles = []
    for i in dir_list:
        val = float(os.path.basename(i))
        if val >= 180:
            val -= 360
        angles.append(val)
    asort = np.argsort(angles)
    return sorted(angles), [dir_list[f] for f in asort]


def values_around_centroid(image, centroid, radius):
    """
    Taking the image, a centroid from a connected region in the threshold image (region properties), the function
    return the data around the centroid according to a given radius.

    Can be done for image with stack RGB data? To be tested

    :param image: Image
    :param centroid: Centroid from region properties with scikit-image
    :param radius: Radius around centroid
    :return:
    """
    # Rounding centroid
    centroid_y, centroid_x = round(centroid[0]), round(centroid[1])
    imshape = image.shape
    # Pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(0, imshape[1], 1), np.arange(0, imshape[0], 1))
    # Subtraction of centroid to pixel coordinates
    ngrid_x, ngrid_y = grid_x - centroid_x, grid_y - centroid_y

    # Norm calculation
    norm = np.sqrt(ngrid_x ** 2 + ngrid_y ** 2)
    # Binary image of norm below or equal to radius
    bin = norm <= radius

    return bin, image[bin]


def rolloff_polynomial(x, a0, a2, a4, a6, a8):
    """
    Polynomial fit with even coefficients (degree 0 to 8) for roll-off fitting.

    :param x:
    :param a0:
    :param a2:
    :param a4:
    :param a6:
    :param a8:
    :return:
    """
    return a0 + a2*x**2 + a4*x**4 + a6*x**6 + a8*x**8


if __name__ == "__main__":

    # Objets
    processim = ProcessImage()

    # Parameters
    cover = "cover"
    sn = "2BW7X7"
    #sn = "2C9JCA"
    wlens = "front"
    #wlens = "front"
    #date = "20230212"
    date = "20230413"
    med = "water"

    npixel = 15

    # General path
    gen_path = f"/Volumes/MYBOOK/data-i360X3/calibrations/rolloff/{sn}/{cover}/{med}/{wlens}/{date}"

    # Listing folders
    folders = glob.glob(gen_path + "/*[0-9]")
    ang, fs = sort_folders(folders)

    # Dark
    im_ambiance, _, _, _ = processim.imagestack(glob.glob(gen_path + "/amb/*.dng"), wlens, cam="x3")
    im_ambiance = im_ambiance.mean(axis=2)

    # Open geometric calib
    # Geometric calibration
    #geo_calib_air = h5py.File("../geometric-calibration/calibrationfiles/geometric-calibration-2W7X7-air.h5", "r")
    #geo_calib = h5py.File(f"../geometric-calibration/calibrationfiles/geometric-calibration-{sn}.h5", "r")
    geocalib = h5py.File(f"../geometric-calibration/calibrationfiles/geometric-calibration-{sn}.h5", "r")
    geo_id = calibrations.calibrations_info.geometric[f"{sn}"][f"{cover}"]["water"][f"{wlens}"]
    geocalib = geocalib[f"{med}/{cover}/{wlens}/{geo_id}"]

    geo = {}
    for i in geocalib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geocalib["fp"][i], geocalib["ierror"][i])

    # Roll-off centroid Loop
    # Pre-allocation
    image_total = np.zeros((im_ambiance.shape[0]//2, im_ambiance.shape[0]//2, 3))

    centroids = np.empty((len(fs), 3), dtype=[("y", "float32"), ("x", "float32")])
    rolloff_data = np.empty((len(fs), 3), dtype=[("a", "float32"), ("DN_avg", "float32"), ("DN_std", "float32")])

    centroids.fill(np.nan)
    rolloff_data.fill(np.nan)

    ke = {0: "red", 1: "green", 2: "blue"}

    for n, p in enumerate(fs):

        print(f"Angle: {ang[n]} deg.")

        # Open three images
        all_im, _, _, _ = processim.imagestack(glob.glob(p + "/*.dng"), wlens, cam="x3")

        # Dark subtraction
        all_im -= im_ambiance[:, :, None]

        # Down-sampling
        im_dws = processim.dwnsampling(np.clip(all_im.mean(axis=2), 0, None), "GBRG", ave=True)

        # Image total
        image_total += im_dws

        # Region property
        _, region_properties = processim.region_properties(im_dws[:, :, 0], 1000, 17000)  # Red image
        rp = region_properties

        if rp:
            for j, k in enumerate(["red", "green", "blue"]):
                _, zen_dwsa, _ = geo[k].angular_coordinates()
                yc, xc = rp[0].centroid

                # To be changed for other type of roll-off processing
                _, pixval = values_around_centroid(im_dws[:, :, j], (yc, xc), npixel)  # Using 15 pixels

                centroids[n, j] = yc, xc  # storing centroid
                rolloff_data[n, j] = zen_dwsa[int(round(yc)), int(round(xc))], np.mean(pixval), np.std(pixval)

    # Normalization
    rolloff_norm = rolloff_data.copy()
    rolloff_norm["DN_avg"] /= np.nanmax(rolloff_data["DN_avg"], axis=0)
    rolloff_norm["DN_std"] /= np.nanmax(rolloff_data["DN_avg"], axis=0)

    # Figures
    fig1, ax1 = plt.subplots(1, 1)
    ax1.axhline(y=geo["red"].intrinsics["DistortionCenter"][1])
    ax1.axvline(x=geo["red"].intrinsics["DistortionCenter"][0])
    ax1.imshow(image_total[:, :, 0])

    ax1.plot(centroids["x"][:, 0], centroids["y"][:, 0], "r+")
    ax1.plot(centroids["x"][:, 0], centroids["y"][:, 0], "r+")

    fig2, ax2 = plt.subplots(3, 1, sharex=True, sharey=True)
    fig3, ax3 = plt.subplots(1, 1, figsize=(6.4, 6.4 * 1/1.61803))

    color = iter(['#d62728', '#2ca02c', '#1f77b4'])
    lab = ["red", "green", "blue"]
    lab_fit = ["fit red: ", "fit green: ", "fit blue: "]
    ls = ["-", "-.", ":"]
    marker = ["o", "s", "d"]

    theta_fit = np.linspace(0, rolloff_norm["a"].max() + 5, 101)
    negative_multiplicator = np.ones(len(ang))
    negative_multiplicator[np.array(ang) < 0] *= -1

    fit_coeffs_results = np.empty((3, 5))

    for band in range(rolloff_norm.shape[1]):

        # Change of color
        col = next(color)

        # Fit
        popt_az0, pcov_az0 = curve_fit(rolloff_polynomial, rolloff_norm["a"][:, band], rolloff_norm["DN_avg"][:, band])

        fit_coeffs_results[band, :] = popt_az0.copy()

        ax2[band].plot(rolloff_norm["a"][:, band], rolloff_norm["DN_avg"][:, band], marker=marker[band], markersize=3, linestyle="none", markeredgecolor=col, markerfacecolor="none", alpha=0.9, label=lab[band])
        ax2[band].plot(theta_fit, rolloff_polynomial(theta_fit, *popt_az0), color=col, linestyle=ls[band], label=lab_fit[band])  # Fit

        ax2[band].set_ylabel(r"$R(\theta)$")
        ax2[band].legend(loc="best")

        # Figure 3
        ax3.plot(rolloff_norm["a"][:, band] * negative_multiplicator, rolloff_norm["DN_avg"][:, band], linestyle=ls[band], color=col, label=lab[band])

    ax2[2].set_xlabel(r"$\theta$ [˚]")
    fig2.tight_layout()
    ax3.set_ylabel(r"$R(\theta)$")
    ax3.set_xlabel(r"$\theta$ [˚]")
    fig3.tight_layout()

    # Save data
    filename = f"roll-off-{sn}.h5"
    pathname = "calibrationfiles/" + filename
    group_name = f"{cover}/{med}/{wlens}/{date}"
    saved_answer = processim.save_results()
    if saved_answer == "y":

        processim.save_x3_hdf5(pathname, group_name, "fit-coefficients", fit_coeffs_results)


    plt.show()
