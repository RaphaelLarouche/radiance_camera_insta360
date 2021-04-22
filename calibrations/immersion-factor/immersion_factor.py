# -*- coding: utf-8 -*-
"""
Immersion factor calibration.
"""

# Importation of modules
import os
import time
import h5py
import string
import deepdish
import datetime
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from refractivesqlite import dboperations as DB   # https://github.com/HugoGuillen/refractiveindex.info-sqlite

# Importation of other modules
from source.processing import ProcessImage, FigureFunctions
from source.geometric_rolloff import MatlabGeometricMengine


# Functions
def radiance_increase(nw, na):
    """
    Radiance increase below water (n-squared law)

    :param nw:
    :param na:
    :return:
    """
    return transmittance(nw, na) * nw ** 2


def transmittance(n1, n2):
    """
    Fresnel equation fro transmittance from medium 1 toward medium 2.

    :param n1:
    :param n2:
    :return:
    """

    return 1 - (((n1 - n2) ** 2) / ((n1 + n2) ** 2))


def imagestack_averaging(imagelist, framenumber, dframe, which):
    """

    :param imagelist:
    :param framenumber:
    :param dframe:
    :param which:
    :return:
    """

    p = ProcessImage()
    iternumber = int(len(imagelist) / framenumber)
    imstack = np.zeros((darkframe.shape[0], darkframe.shape[0], iternumber))

    for n, i in enumerate(list(range(0, len(imagelist), framenumber))):

        print("----Stack number: {}----".format(n + 1))
        s, _, _, _ = p.imagestack(imagelist[i:i+framenumber], which)

        s -= dframe[:, :, None]
        imstack[:, :, n] = s.mean(axis=2)

    return imstack


def replica_trapz(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    diff = x[1:] - x[:-1]
    if len(y.shape) == 1:
        return np.sum((y[1:] + y[:-1]) * diff/2, axis=0)
    else:
        return np.sum((y[1:] + y[:-1]) * diff[:, None] / 2, axis=0)


def n_from_immersionfactor(cim, nw):
    """
    Fitting refractive index of external surface from the immersion factor and the refractive index of water.

    :param cim:
    :param nw:
    :return:
    """
    c1 = (cim / nw) - 1
    c2 = 2 * ((cim / nw) - nw)
    c3 = (cim / nw) - nw ** 2

    ro = np.empty(c1.shape)
    for i in range(c1.shape[0]):

        coeff = np.array([c1[i], c2[i], c3[i]])
        r = np.roots(coeff)
        ro[i] = r[r>=0]
    return ro


def standard_immersionfactor(nwater, ng):
    """
    Immersion factor computed for small angles and by considering only the transmittance at the water(air)-glass
    interface. Equation 9 of Zibordi (2006).
    :param nw:
    :param nm:
    :return:
    """
    return (nwater * (nwater + ng) ** 2) / (1 + ng) ** 2


def estimators_std(slope, intercept, x, y):
    """
    Standard deviation of the linear regression estimates (slope and intercept).

    :param slope:
    :param intercept:
    :param x:
    :param y:
    :return:
    """
    y_i = slope * x + intercept
    ssr = np.sum((y - y_i) ** 2)  # sum of squared residual
    mse = np.sqrt(ssr / (x.shape[0] - 2))

    std_slope = mse * np.sqrt(1 / np.sum((x - x.mean()) ** 2))
    std_intercept = std_slope * np.sqrt(np.sum(x ** 2) / x.shape[0])

    return std_slope, std_intercept


def extract_datetime(imlist):
    """

    :param imlist:
    :return:
    """
    dtstr = []
    for i in imlist:
        split = os.path.splitext(os.path.basename(i))[0].split("_")
        dtstr.append(split[1] + "_" + split[2])

    convertime = np.vectorize(datestr_to_datetime)
    return convertime(dtstr)


def datestr_to_datetime(x):
    """

    :param x:
    :return:
    """
    return datetime.datetime.strptime(x, "%Y%m%d_%H%M%S")


def total_seconds(x):
    """

    :param x:
    :return:
    """
    return x.total_seconds()


if __name__ == "__main__":

    # Instance of process image
    process = ProcessImage()

    # Instance of figurefunction
    ff = FigureFunctions()

    path_i360 = os.path.dirname(os.path.dirname(__file__))

    # Choice of lens
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    if answer.lower() == "c":

        # Path to images
        imagespath = "/Volumes/MYBOOK/data-i360-tests/calibrations/immersion-factor/09102020/lensclose"

        # Geometric calibration
        geocalib = deepdish.io.load(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5", "/lens-close/20200730_112353/")

        # Camera spectral response
        srdata = h5py.File(path_i360 + "/relative-spectral-response/calibrationfiles/rsr_20200610.h5", "r")
        srdata = srdata["lens-close"]

        # Water level
        camera_z = 4.6  # cm
        water_level = np.arange(5, 10.5, 0.5)  # cm

        wlens = "close"

    elif answer.lower() == "f":

        # Path to images
        imagespath = "/Volumes/MYBOOK/data-i360-tests/calibrations/immersion-factor/09102020/lensfar"

        # Geometric calibration
        geocalib = deepdish.io.load(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5", "/lens-far/20200730_143716/")

        # Camera spectral response
        srdata = h5py.File(path_i360 + "/relative-spectral-response/calibrationfiles/rsr_20200710.h5", "r")
        srdata = srdata["lens-far"]

        # Water level
        camera_z = 4.5  # cm
        water_level = np.arange(4.5, 10, 0.5)  # cm

        wlens = "far"

    else:
        raise ValueError("Not valid choice.")

    # Refractive index of pure water
    db = DB.Database("refractive.db")
    pageID_Hale = 2707
    nwHale = np.array(db.get_material(pageID_Hale).get_complete_refractive())
    nwHale[:, 0] *= 1000
    nwHale = nwHale[(400 <= nwHale[:, 0]) & (nwHale[:, 0] <= 800)]

    # Geometric classes
    geo = {}
    for i in geocalib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geocalib["fp"][i], geocalib["ierror"][i])
    channel = {0: "red", 1: "green", 2: "blue"}

    # Effective wavelength calculation
    wl_rsr = srdata["wavelength"][:]
    rsr = srdata["rsr_peak_norm"][:]
    effective_wavelength = replica_trapz(wl_rsr, rsr * wl_rsr[:, None]) / replica_trapz(wl_rsr, rsr)

    # Image list
    im_amb = process.imageslist_dark(imagespath, prefix="AMB")  # without open light source
    im_air = process.imageslist_dark(imagespath, prefix="AIR")
    im_water = process.imageslist(imagespath)

    # Stacking
    im_amb_stack, _, _, _ = process.imagestack(im_amb, wlens)
    im_air_stack, _, _, _ = process.imagestack(im_air, wlens)

    # Dark subtraction
    darkframe = im_amb_stack.mean(axis=2)
    im_air_stack -= darkframe[:, :, None]
    im_water_stack = imagestack_averaging(im_water, 5, darkframe, wlens)

    # Downsampling
    im_air_dws = process.dwnsampling(im_air_stack.mean(axis=2), "RGGB")

    # ______ Loops
    plt.style.use("/Users/raphaellarouche/PycharmProjects/radiance_camera_i360/figurestyle.mplstyle")
    z = water_level - camera_z

    zenith_mask = 1.5  # degrees

    # Air measurements loop
    # Pre-allocation
    dn_air_mean = np.empty((3, 1))
    dn_air_std = np.empty((3, 1))

    for b in range(im_air_dws.shape[2]):

        _, zenith, _ = geo[channel[b]].angular_coordinates()  # Have to use in-air calibration?
        angular_mask = zenith <= zenith_mask

        data_a = im_air_dws[:, :, b][angular_mask]
        dn_air_mean[b, :] = data_a.mean()
        dn_air_std[b, :] = data_a.std()

    err_air = dn_air_std / dn_air_mean

    # Water measurements loop
    # Pre-allocation
    dn_water_mean = np.empty((im_water_stack.shape[2], 3))
    dn_water_std = dn_water_mean.copy()

    for n in range(im_water_stack.shape[2]):

        im_water_dws = process.dwnsampling(im_water_stack[:, :, n], "RGGB")

        for b in range(im_water_dws.shape[2]):

            _, zenith, _ = geo[channel[b]].angular_coordinates()
            angular_mask = zenith <= zenith_mask

            data_w = im_water_dws[:, :, b][angular_mask]
            dn_water_mean[n, b] = data_w.mean()
            dn_water_std[n, b] = data_w.std()

    # Correction for increase in radiance below water
    refractive_index = np.interp(effective_wavelength, nwHale[:, 0], nwHale[:, 1])  # n at effective wavelengths
    dn_water_mean /= radiance_increase(refractive_index, 1.00)

    # Intercepts
    # Pre-allocation
    dn_water_zero = np.empty((1, 3))
    inter = np.empty((1, 3))
    inter_std = np.empty((1, 3))
    tx = "$ln~DN_{{i}}(z) = m \cdot z + b$\n$m = ({0:.3f}\pm{1:.3f})$\n$b = ({2:.3f}\pm{3:.3f})$\n$R^{{2}} = {4:.3f}$"

    fig1, ax1 = plt.subplots(3, 1, figsize=(ff.set_size(subplots=(2, 1), fraction=0.7)[0], ff.set_size(subplots=(2, 1))[1] * 0.8))

    linestl = ["-", "-.", ":"]
    for b in range(dn_water_mean.shape[1]):
        slope, intercept, rval, _, stderror = stats.linregress(z, np.log(dn_water_mean[:, b]))

        _, std_int = estimators_std(slope, intercept, z, np.log(dn_water_mean[:, b]))

        dn_water_zero[:, b] = np.exp(intercept)
        inter[:, b] = intercept
        inter_std[:, b] = std_int * 1.96

        z_graph = np.linspace(0, z.max() * 1.1, 50)
        ax1[b].plot((slope * z_graph + intercept), z_graph, linestyle=linestl[1], color="k", label="Linear fit")

        ax1[b].text(np.log(dn_water_zero[:, b]) * 1.01, 3, tx.format(slope, stderror * 1.96, intercept, std_int * 1.96, rval**2), fontsize=8)

    immersion = dn_air_mean.ravel() / dn_water_zero

    # Uncertainties
    unc_abs_dn_water_zero = np.exp(inter) * inter_std
    unc_rel_dn_water_zero = unc_abs_dn_water_zero / dn_water_zero

    unc_abs_immersion = immersion * np.sqrt((unc_rel_dn_water_zero) ** 2 + (err_air.ravel()) ** 2)

    # ______ Figures
    # Figure 1
    for b in range(dn_water_mean.shape[1]):
        ax1[b].plot(np.log(dn_water_mean[:, b]), z, "o", markersize=4, markeredgecolor="k", markerfacecolor="none", label="Water measurements")
        # ax1[b].errorbar(np.log(dn_water_mean[:, b]), z, xerr=dn_water_std[:, b]/dn_water_mean[:, b], marker="o",
        #                 linestyle="none", markersize=4, markerfacecolor="none", label="Water measurements")
        ax1[b].errorbar(np.log(dn_air_mean[b]), 0, xerr=err_air.T[0][b], color="grey", marker="s", markersize=3,  markerfacecolor="none",
                        linestyle="none", label="$DN(0^{+})$")
        ax1[b].errorbar(np.log(dn_water_zero[0][b]), 0, color="grey", xerr=inter_std[0][b], marker="^", markersize=3,  markerfacecolor="none",
                     linestyle="none", label="$DN(0^{-})$")

        ax1[b].set_ylabel("z [cm]")
        ax1[b].set_xlabel("$\ln~DN_{i}$")

        ax1[b].legend(loc="lower right")

        ax1[b].invert_yaxis()

        ax1[b].text(-0.05, 1.05, "(" + string.ascii_lowercase[b] + ")", transform=ax1[b].transAxes, size=11, weight='bold')

    # Figure 2
    pageID_NBK7 = 805  # pageid 805 for Schott NBK7 glass
    data_NBK7 = np.array(db.get_material(pageID_NBK7).get_complete_refractive())
    nNBK7 = np.interp(nwHale[:, 0], data_NBK7[:, 0] * 1000, data_NBK7[:, 1])
    If_NBK7 = standard_immersionfactor(nwHale[:, 1], nNBK7)

    fig2, ax2 = plt.subplots(1, 1)

    ax2.plot(nwHale[:, 0], If_NBK7, label="NBk7")
    ax2.plot(effective_wavelength, immersion[0])

    ax2.set_ylim((1.65, 1.75))

    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel("$I_{f}$")

    # Figure 3
    n_glass_inverse = n_from_immersionfactor(immersion[0], np.interp(effective_wavelength, nwHale[:, 0], nwHale[:, 1]))

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(effective_wavelength, n_glass_inverse)
    ax3.plot(nwHale[:, 0], nNBK7)

    ax3.set_xlabel("Wavelength [nm]")
    ax3.set_ylabel("refractive index")

    # Saving figures
    fig1.tight_layout()

    # Saving results
    save_file = process.save_results()

    if save_file == "y":

        pathname = "calibrationfiles/" + "immersion_factor" + ".h5"
        timestr = time.strftime("%Y%m%d", time.localtime(os.stat(im_water[0])[-1]))

        if answer.lower() == "c":
            process.create_hdf5_dataset(pathname, "lens-close/" + timestr, "immersion", immersion)
        elif answer.lower() == "f":
            process.create_hdf5_dataset(pathname, "lens-far/" + timestr, "immersion", immersion)

        optics_correspondance = {"c": "close", "f": "far"}
        fig1.savefig("figures/immersion-factor-{0}.pdf".format(optics_correspondance[answer.lower()]), format="pdf", dpi=600)

    plt.show()
