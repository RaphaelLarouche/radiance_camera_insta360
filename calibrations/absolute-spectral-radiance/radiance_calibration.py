# -*- coding: utf-8 -*-
"""
Absolute spectral radiance calibration
"""

# Module importation
import os
import time
import string
import deepdish
import h5py
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt

# Other modules
import source.processing as proccessing
from source.geometric_rolloff import MatlabGeometricMengine


# Functions
def radiance_planck(wavelength, T):
    """
    Planck black body radiance distribution.

    :param wavelength:
    :param T:
    :return:
    """
    h = 6.62607015e-34
    c = 299792458
    k = 1.380649e-23

    lamb = wavelength * 1e-9
    expo = np.exp((h * c) / (lamb * k * T))

    rad = 1e-9 * ((2 * h * c ** 2) / (lamb ** 5)) * (1 / (expo - 1))

    return rad, rad / replica_trapz(wavelength, rad)


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


if __name__ == "__main__":

    # Instance of ProcessImage
    process = proccessing.ProcessImage()

    # Instance of FigureFunctions
    ff = proccessing.FigureFunctions()

    # General path to all data
    path = process.folder_choice("/Volumes/MYBOOK/data-i360-tests/")
    path_i360 = os.path.dirname(os.path.dirname(__file__))

    # Choice of camera
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    if answer.lower() == "c":

        impath = path + "/nofilter/lensclose"

        imlist = process.imageslist(impath)
        imlistdark = process.imageslist_dark(impath, prefix="AMB")

        geocalib = deepdish.io.load(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-air.h5", "/lens-close/20190104_192404/")

        srdata = h5py.File(path_i360 + "/relative-spectral-response/calibrationfiles/rsr_20200610.h5", "r")
        srdata = srdata["lens-close"]

    elif answer.lower() == "f":

        impath = path + "/nofilter/lensfar"

        imlist = process.imageslist(impath)
        imlistdark = process.imageslist_dark(impath, prefix="AMB")

        geocalib = deepdish.io.load(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-air.h5", "/lens-far/20190104_214037/")

        srdata = h5py.File(path_i360 + "/relative-spectral-response/calibrationfiles/rsr_20200710.h5", "r")
        srdata = srdata["lens-far"]

    else:
        raise ValueError("Not valid choice.")

    # Open spectrometer data
    spectro = proccessing.FlameSpectrometer(path)
    spectro.calibration_coefficient("light")  # Calibration for absolute spectrum
    _, spectral_rad_15, spectral_rad_15_unc, cops_wl, cops_val = spectro.source_spectral_radiance("labsphere", [589, 589, 589], 0)
    w_s, spectral_rad_35, _, _, _ = spectro.source_spectral_radiance("labsphere", [589, 589, 589], 1)
    _, spectral_rad_45, _ , _, _ = spectro.source_spectral_radiance("labsphere", [589, 589, 589], 2)

    condwl = (w_s <= 700) & (w_s >= 400)
    spectral_rad_norm_15 = spectral_rad_15 / replica_trapz(w_s, spectral_rad_15)
    _, rad_planck_norm = radiance_planck(w_s, 2796)

    # Effective radiance in bands
    wl_rsr = srdata["wavelength"][:]
    rsr = srdata["rsr_peak_norm"][:]

    spectral_rad_source = np.interp(wl_rsr, w_s, spectral_rad_15)
    effective_rad = replica_trapz(wl_rsr, rsr * spectral_rad_source[:, None]) / replica_trapz(wl_rsr, rsr)
    effective_lambda = replica_trapz(wl_rsr, rsr * wl_rsr[:, None]) / replica_trapz(wl_rsr, rsr)

    # i360 camera data
    whichim = {"c": "close", "f": "far"}
    imstack, exp, gain, blevel = process.imagestack(imlist, whichim[answer.lower()])
    ambstack, _, _, _ = process.imagestack(imlistdark, whichim[answer.lower()])

    imstack -= ambstack.mean(axis=2)[:, :, None]
    imstack = np.clip(imstack, 0, None)
    im_dws = process.dwnsampling(imstack.mean(axis=2), "RGGB")

    geo = {}
    for i in geocalib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geocalib["fp"][i], geocalib["ierror"][i])
    channel_correspondance = {0: "red", 1: "green", 2: "blue"}

    zenith_max = 5.0
    imdownsampling = 250

    # Pre-allocation
    plt.style.use("../../figurestyle.mplstyle")

    dn_avg = np.empty(3)
    dn_std = np.empty(3)

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=ff.set_size())

    for i in range(im_dws.shape[2]):

        im = im_dws[:, :, i]
        curr_geo = geo[channel_correspondance[i]]
        _, z, _ = curr_geo.angular_coordinates()

        maskdegree = z <= zenith_max

        dn_avg[i] = im[maskdegree].mean()
        dn_std[i] = im[maskdegree].std()

        print(im[maskdegree].shape)

        # Figure
        region = skimage.measure.regionprops(maskdegree.astype(int))
        draw_circle = plt.Circle((region[0].centroid[1], region[0].centroid[0]), region[0].equivalent_diameter / 2,
                                 fill=False, linestyle=":")

        imsh = ax[i].imshow(im)  # vmin=dn_avg[i]*0.9, vmax=dn_avg[i]*1.1

        ax[i].plot(curr_geo.center[0], curr_geo.center[1], "r+")
        ax[i].add_artist(draw_circle)

        cb = fig.colorbar(imsh, ax=ax[i], orientation="vertical", fraction=0.046, pad=0.04)
        cb.ax.set_title("$DN_{i}$", fontsize=10)

        mask_sphere = (im >= dn_avg[i]*0.9) & (im <= dn_avg[i]*1.1)
        region_sphere = skimage.measure.regionprops(mask_sphere.astype(int))

        ax[i].set_xlim((int(region_sphere[0].centroid[1] - imdownsampling), int(region_sphere[0].centroid[1] + imdownsampling)))
        ax[i].set_ylim((int(region_sphere[0].centroid[0] - imdownsampling), int(region_sphere[0].centroid[0] + imdownsampling)))

        ax[i].text(-0.1, 1.1, "(" + string.ascii_lowercase[i] + ")", transform=ax[i].transAxes, size=11, weight='bold')

        ax[i].set_xlabel("$x$ [px]")

    ax[0].set_ylabel("$y$ [px]")

    coeff = effective_rad / (dn_avg / (exp[0] * gain[0] / 100))
    print(coeff)

    cvf = np.array([2.397e-8, 8.460e-9, 1.362e-8])

    # Uncertainty on calibration coefficient
    unc_effective_rad = np.interp(effective_lambda, w_s, spectral_rad_15_unc)
    unc_coeff = np.sqrt((unc_effective_rad) ** 2 + (dn_std / dn_avg) ** 2)

    # Figures
    fig1, ax1 = plt.subplots(1, 2)

    ax1[0].plot(w_s[condwl], spectral_rad_15[condwl], label="1.5 cm")
    ax1[0].plot(w_s[condwl], spectral_rad_35[condwl], label="3.5 cm")
    ax1[0].plot(w_s[condwl], spectral_rad_45[condwl], label="4.5 cm")
    ax1[0].plot(effective_lambda, effective_rad, "o")

    ax1[0].set_xlabel("wavelength [nm]")
    ax1[0].set_ylabel("$L_{source}$ [$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")

    ax1[0].legend(loc="best")

    ax1[1].plot(w_s[condwl], spectral_rad_norm_15[condwl], label="1.5 cm")
    ax1[1].plot(w_s[condwl], rad_planck_norm[condwl], label="Planck $T = 2796$ K")

    ax1[1].set_xlabel("wavelength [nm]")
    ax1[1].set_ylabel("normalized radiance [$\mathrm{nm^{-1}}$]")

    ax1[1].legend(loc="best")

    # Figure 2
    fig2, ax2 = plt.subplots(1, 1, figsize=ff.set_size(fraction=0.7))

    ax2.plot(w_s[condwl], spectral_rad_15[condwl], color="k", label="$L_{source}(\lambda)$")
    ax2.fill_between(w_s[condwl], spectral_rad_15[condwl] * (1 - spectral_rad_15_unc[condwl]),
                     spectral_rad_15[condwl] * (1 + spectral_rad_15_unc[condwl]), color="gray", alpha=0.6)
    ax2.plot(cops_wl, cops_val,  color="k", marker="^", markersize=6, linestyle="None", markeredgecolor="k",
             markerfacecolor="none", label="C-OPS at 589 nm")
    ax2.errorbar(effective_lambda, effective_rad, xerr=replica_trapz(wl_rsr, rsr) / 2, color="k", marker="o",
                 markersize=5, linestyle="None", markeredgecolor="k", markerfacecolor="none",
                 label="$\overline{L}_{i, source}$")

    ax2.set_yscale("log")
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel("$L~[\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}]$")

    ax2.legend(loc='lower right')

    # Figure 3 - uncertainties of the Ocean Optic calibration source
    fig3, ax4 = plt.subplots(1, 1, figsize=ff.set_size())

    ax4.plot(spectro.oo_lamp_uncertainty[:, 0], spectro.oo_lamp_uncertainty[:, 1] * 100,)

    ax4.set_xlabel("Wavelength [nm]")
    ax4.set_ylabel("Uncertainty (k=1)[%]")

    # Saving figure
    fig.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()

    # Saving results
    save_answer = process.save_results()

    if save_answer == "y":

        filename = "absolute_radiance" + ".h5"
        pathname = "calibrationfiles/" + filename

        timestr = time.strftime("%Y%m%d", time.localtime(os.stat(imlist[0])[-1]))

        correspond_optic = {"c": "close", "f": "far"}

        if answer.lower() == "c":
            process.create_hdf5_dataset(pathname, "lens-close/" + timestr, "cal-coefficients", coeff)
        else:
            process.create_hdf5_dataset(pathname, "lens-far/" + timestr, "cal-coefficients", coeff)

        fig.savefig("figures/output_sphere_{}.pdf".format(correspond_optic[answer.lower()]), format="pdf", dpi=600, bbox_inches='tight')
        fig2.savefig("figures/spectral_radiance_{}.pdf".format(correspond_optic[answer.lower()]), format="pdf", dpi=600, bbox_inches='tight')

    plt.show()
