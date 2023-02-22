# -*- coding: utf-8 -*-
"""

"""

# Importation of modules
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks

from source.processing import ProcessImage
from source.geometric_rolloff import MatlabGeometricMengine


# Functions
def fit_fov(imagestack, radiustest, geometric, fov_mask):
    """
    Using circular Hough transform with a set of radius to fit the best radius from distortion center obtained from
    OcamCalib algorithm.

    :param imagestack: Stack of grayscale 8 bit images
    :param radiustest: Radius to test with Hough transform circle
    :param geometric: Geometric calibration
    :return:
    """
    fov = [np.array([]), np.array([]), np.array([])]  # Pre-allocation
    rad_array = [np.array([]), np.array([]), np.array([])]
    channel = {0: "red", 1: "green", 2: "blue"}

    for n in range(imagestack.shape[3]):
        print("Processing image number {}".format(n))
        curr_im = imagestack[:, :, :, n]

        for b in range(curr_im.shape[2]):
            _, zen, _ = geometric[channel[b]].angular_coordinates()
            maskz = zen >= fov_mask
            edges = canny(curr_im[:, :, b], sigma=6, low_threshold=5, high_threshold=10, mask=maskz)

            # Hough transform
            h = hough_circle(edges, radiustest)

            # Max of circular Hough transform at the distortion center
            m = np.argmax(h[:, geometric[channel[b]].center[1].astype(int), geometric[channel[b]].center[0].astype(int)])
            rmax = radiustest[m]
            print(rmax)

            rad_array[b] = np.append(rad_array[b], rmax)

            circy, circx = circle_coordinates(rmax, geometric[channel[b]].center, 1000)
            fov[b] = np.append(fov[b], zen[circy, circx])

    return edges, fov, rad_array, circx, circy


def circle_coordinates(radius, center, npoints):
    """

    :param radius:
    :param center:
    :param npoints: number of points on the perimeter
    :return:
    """
    theta = np.linspace(0, 2*np.pi, npoints)
    y = (radius * np.sin(theta)) + center[1].astype(int)
    x = (radius * np.cos(theta)) + center[0].astype(int)

    return y.astype(int), x.astype(int)


if __name__ == "__main__":

    # ProcessImage instance
    process = ProcessImage()

    # Choose general folder (between air-water)
    gen_path = process.folder_choice("D:\data-i360\calibrations\geometric")

    # Open geometric calibration and images
    if "water" in gen_path.lower():
        m = "water"
        mask_angle = 70
    elif "air" in gen_path.lower():
        m = "air"
        mask_angle = 88
    else:
        raise Exception("Unable to determine the medium being analyzed.")

    # medium calibration dictionary
    mcd_cl = {"water": "20200730_112353", "air": "20190104_192404"}  # lens 1 (close)
    mcd_fr = {"water": "20200730_143716", "air": "20190104_214037"}  # lens 2 (far)

    # Choose cam
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    geo_file = h5py.File(os.path.dirname(__file__) + "/calibrationfiles/geometric-calibration-{0}.h5".format(m))
    if answer.lower() == "c":
        which = "close"
        imagelist = process.imageslist("D:\data-i360\calibrations\geometric\{0}\lensclose".format(m))
        geocalib = geo_file["lens-close"][mcd_cl[m]]

    elif answer.lower() == "f":
        which = "far"
        imagelist = process.imageslist("D:\data-i360\calibrations\geometric\{0}\lensfar".format(m))
        geocalib = geo_file["lens-far"][mcd_fr[m]]
    else:
        raise Exception("Unable to determine which lens is being analyzed.")

    imagelist = imagelist[:10]

    # Geometric calibration
    geo = {}
    for i in geocalib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geocalib["fp"][i], geocalib["ierror"][i])
    channel_correspondance = {0: "red", 1: "green", 2: "blue"}

    # Pre-allocation
    stackimagegray = np.empty((1728, 1728, 3, len(imagelist)))
    fov = [np.array([]), np.array([]), np.array([])]

    for n, imagepath in enumerate(imagelist):
        print("Processing image number {}".format(n))
        image, met = process.readDNG_insta360_np(imagepath, which)
        image_dws = process.dwnsampling(image, "RGGB")
        stackimagegray[:, :, :, n] = process.raw2gray(image_dws, met, 6)

    # Field of view fit algorithm
    e, fov, radius, cx, cy = fit_fov(stackimagegray, np.arange(790, 814, 1), geo, mask_angle)

    for i, f in enumerate(fov):
        print("Band {0}: mean radius {3}, mean FoV {1}, standard deviation FoV{2}".format(i, f.mean(), f.std(), radius[i].mean()))

    # Figures
    # Figure 1 - last edge and x,y coordinates
    fig1, ax1 = plt.subplots(1, 1)

    ax1.imshow(e)
    ax1.plot(cx, cy, "r.")

    # Figure 2
    fig2, ax2 = plt.subplots(1, 1)

    ax2.imshow(stackimagegray[:, :, 1, -1])
    ax2.plot(cx, cy, "r.", markersize=2)

    # Results are for the camera with the case

    plt.show()
