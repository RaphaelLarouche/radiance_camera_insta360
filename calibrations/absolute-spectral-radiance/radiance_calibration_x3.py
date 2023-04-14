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
import calibrations.calibrations_info
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

    # Which cam conditions
    sn = "2BW7X7"
    cover = "cover"
    which_lens = "back"
    date = "20230407"

    # General path to all data
    #path = process.folder_choice() + r"\data-i360\calibrations\absolute-radiance\09082020"
    path = f"/Volumes/MYBOOK/data-i360X3/calibrations/absolute_radiance/{sn}/{cover}/{which_lens}/{date}"

    impath = path + "/images"
    imlist = process.imageslist(impath)
    imlistdark = process.imageslist_dark(impath, prefix="AMB")

    # RSR data
    #srdata = h5py.File("../relative-spectral-response/calibrationfiles/rsr_fluorolog_x3_20230313.h5", "r")
    srdata = h5py.File(f"../relative-spectral-response/calibrationfiles/rsr-x3-{sn}.h5")
    date_rsr = calibrations.calibrations_info.rsr[f"{sn}"][f"{cover}"][f"{which_lens}"]
    srdata = srdata[f"{cover}/{which_lens}/{date_rsr}"]

    # Geometric calibration
    geocalib = h5py.File(f"../geometric-calibration/calibrationfiles/geometric-calibration-{sn}.h5")
    calib_id = calibrations.calibrations_info.geometric[f"{sn}"][f"{cover}"]["air"][f"{which_lens}"]
    geocalib = geocalib[f"air/{cover}/{which_lens}/{calib_id}"]

    # Open spectrometer data
    spectro = proccessing.FlameSpectrometer(path)
    spectro.calibration_coefficient("light")  # Calibration for absolute spectrum
    w_s, spectral_rad_00, spectral_rad_00_unc, cops_wl, cops_val = spectro.source_spectral_radiance("labsphere", [589, 589], 1)
    _, spectral_rad_10, _, _, _ = spectro.source_spectral_radiance("labsphere", [589, 589], 0)
    #_, spectral_rad_45, _ , _, _ = spectro.source_spectral_radiance("labsphere", [589, 589, 589], 2)

    condwl = (w_s <= 700) & (w_s >= 400)
    spectral_rad_norm_00 = spectral_rad_00 / replica_trapz(w_s, spectral_rad_00)
    _, rad_planck_norm = radiance_planck(w_s, 2796)

    # Effective radiance in bands
    wl_rsr = srdata["wavelength [nm]"][:]
    rsr = srdata["rsr [-]"][:]

    spectral_rad_source = np.interp(wl_rsr, w_s, spectral_rad_00)
    effective_rad = replica_trapz(wl_rsr, rsr * spectral_rad_source[:, None]) / replica_trapz(wl_rsr, rsr)
    effective_lambda = replica_trapz(wl_rsr, rsr * wl_rsr[:, None]) / replica_trapz(wl_rsr, rsr)

    # i360 camera data
    imstack, exp, gain, blevel = process.imagestack(imlist, which_lens, cam="x3")
    ambstack, _, _, _ = process.imagestack(imlistdark, which_lens, cam="x3")

    imstack -= ambstack.mean(axis=2)[:, :, None]
    imstack = np.clip(imstack, 0, None)
    im_dws = process.dwnsampling(imstack.mean(axis=2), "GBRG")

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

    fig, ax = plt.subplots(2, 3, figsize=ff.set_size(height_ratio=0.8))
    ax_row1 = list(ax[0, :])
    ax_row2 = list(ax[1, :])
    ax_row1[0].get_shared_y_axes().join(*ax_row1)
    ax_row2[0].get_shared_y_axes().join(*ax_row2)

    fig5, ax5 = plt.subplots(1, 3, figsize=ff.set_size(height_ratio=0.5), sharey=True)

    for i in range(im_dws.shape[2]):

        im = im_dws[:, :, i]
        curr_geo = geo[channel_correspondance[i]]
        _, z, _ = curr_geo.angular_coordinates()

        maskdegree = z <= zenith_max

        dn_avg[i] = im[maskdegree].mean()
        dn_std[i] = im[maskdegree].std()

        print(im[maskdegree].shape)

        # Figure
        # First row
        # Images
        imsh = ax[0, i].imshow(im)  # vmin=dn_avg[i]*0.9, vmax=dn_avg[i]*1.1
        #cb = fig.colorbar(imsh, ax=ax[0, i], orientation="vertical", fraction=0.046, pad=0.04)
        cb = fig.colorbar(imsh, ax=ax[0, i], orientation="horizontal", fraction=0.046, pad=0.04)
        cb.set_label("$DN_{i}$ [ADU]", fontsize=9)
        #cb.ax.set_title("$DN_{i}$", fontsize=10)

        # Sphere
        region = skimage.measure.regionprops(maskdegree.astype(int))
        draw_circle = plt.Circle((region[0].centroid[1], region[0].centroid[0]), region[0].equivalent_diameter / 2, fill=False, linestyle=":")

        ax[0, i].plot(curr_geo.center[0], curr_geo.center[1], "r+")
        ax[0, i].add_artist(draw_circle)

        mask_sphere = (im >= dn_avg[i]*0.9) & (im <= dn_avg[i]*1.1)
        region_sphere = skimage.measure.regionprops(mask_sphere.astype(int))

        ax[0, i].set_xlim((int(region_sphere[0].centroid[1] - imdownsampling), int(region_sphere[0].centroid[1] + imdownsampling)))
        ax[0, i].set_ylim((int(region_sphere[0].centroid[0] - imdownsampling), int(region_sphere[0].centroid[0] + imdownsampling)))

        ax[0, i].xaxis.set_label_position('top')
        ax[0, i].xaxis.set_ticks_position('top')
        if i != 0:
            ax[0, i].set_yticklabels([])

        ax[0, i].text(-0.1, 1.1, "(" + string.ascii_lowercase[i] + ")", transform=ax[0, i].transAxes, size=11, weight='bold')
        ax[0, i].set_xlabel("$x$ position [px]")

        # Figure 5
        # Images
        imsh_f5 = ax5[i].imshow(im)  # vmin=dn_avg[i]*0.9, vmax=dn_avg[i]*1.1
        cb_f5 = fig.colorbar(imsh_f5, ax=ax5[i], orientation="horizontal", fraction=0.046, pad=0.04)
        cb_f5.set_label("$DN_{i}$ [ADU]", fontsize=9)

        draw_circle_f5 = plt.Circle((region[0].centroid[1], region[0].centroid[0]), region[0].equivalent_diameter / 2,
                                 fill=False, linestyle=":")

        ax5[i].plot(curr_geo.center[0], curr_geo.center[1], "r+")
        ax5[i].add_artist(draw_circle_f5)

        ax5[i].set_xlim((int(region_sphere[0].centroid[1] - imdownsampling), int(region_sphere[0].centroid[1] + imdownsampling)))
        ax5[i].set_ylim((int(region_sphere[0].centroid[0] - imdownsampling), int(region_sphere[0].centroid[0] + imdownsampling)))

        ax5[i].xaxis.set_label_position('top')
        ax5[i].xaxis.set_ticks_position('top')
        if i != 0:
            ax5[i].set_yticklabels([])

        ax5[i].text(-0.1, 1.1, "(" + string.ascii_lowercase[i] + ")", transform=ax5[i].transAxes, size=11, weight='bold')
        ax5[i].set_xlabel("$x$ position [px]")

        # Second row
        percent_error = 100*(im - dn_avg[i]) / dn_avg[i]
        wssize = int((region[0].equivalent_diameter / 2) + 10)

        sub_im_perr = percent_error[int(region[0].centroid[0]) - wssize:int(region[0].centroid[0]) + wssize + 1,
                                    int(region[0].centroid[1]) - wssize:int(region[0].centroid[1]) + wssize + 1]

        n_centr_y, n_centr_x = sub_im_perr.shape[0]//2, sub_im_perr.shape[1]//2

        imerr = ax[1, i].imshow(sub_im_perr)
        cb = fig.colorbar(imerr, ax=ax[1, i], orientation="horizontal", fraction=0.046, pad=0.04)
        cb.set_label("% from average", fontsize=9)
        #cb.ax.set_title("%", fontsize=10)

        # Sphere
        ax[1, i].plot(n_centr_x, n_centr_y, "r+")
        draw_circle_2 = plt.Circle((n_centr_x, n_centr_y), region[0].equivalent_diameter / 2, fill=False, linestyle=":")
        ax[1, i].add_artist(draw_circle_2)

        ax[1, i].tick_params(axis='x', label1On=False)
        ax[1, i].tick_params(axis='y', label1On=False)
        ax[1, i].text(-0.1, 1.1, "(" + string.ascii_lowercase[3 + i] + ")", transform=ax[1, i].transAxes, size=11, weight='bold')

    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel("$y$ position [px]")

    ax5[0].invert_yaxis()
    ax5[0].set_ylabel("$y$ position [px]")

    # Calibration coefficient calculation
    coeff = effective_rad / (dn_avg / (exp[0] * gain[0] / 100))
    print(coeff)

    cvf = np.array([2.397e-8, 8.460e-9, 1.362e-8])

    # Uncertainty on calibration coefficient
    unc_effective_rad = np.interp(effective_lambda, w_s, spectral_rad_00_unc)
    unc_coeff = np.sqrt((unc_effective_rad) ** 2 + (dn_std / dn_avg) ** 2)

    # Figures
    fig1, ax1 = plt.subplots(1, 2)

    ax1[0].plot(w_s[condwl], spectral_rad_00[condwl], label="0.0 cm")
    ax1[0].plot(w_s[condwl], spectral_rad_10[condwl], label="1.0 cm")
    #ax1[0].plot(w_s[condwl], spectral_rad_45[condwl], label="4.5 cm")

    ax1[0].plot(w_s[condwl], rad_planck_norm[condwl], label="Planck $T = 2796$ K")
    ax1[0].plot(effective_lambda, effective_rad, "o")

    ax1[0].set_xlabel("wavelength [nm]")
    ax1[0].set_ylabel("$L_{source}$ [$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")

    ax1[0].legend(loc="best")

    ax1[1].plot(w_s[condwl], 100 * (spectral_rad_00[condwl] - spectral_rad_00[condwl])/spectral_rad_00[condwl], label="0.0 cm")
    ax1[1].plot(w_s[condwl], 100 * (spectral_rad_00[condwl] - spectral_rad_10[condwl]) / spectral_rad_00[condwl], label="1.0 cm")
    #ax1[1].plot(w_s[condwl], 100 * (spectral_rad_00[condwl] - spectral_rad_45[condwl]) / spectral_rad_00[condwl], label="4.5 cm")

    ax1[1].set_ylabel("Error [%]")
    ax1[1].legend(loc="best")

    # Figure 2
    #fig2, ax2 = plt.subplots(1, 1, figsize=ff.set_size(fraction=0.7))
    fig2, ax2 = plt.subplots(1, 1, figsize=ff.set_size(fraction=0.6, height_ratio=0.75))

    ax2.plot(w_s[condwl], spectral_rad_00[condwl], color="k", label="$L_{source}(\lambda)$")
    ax2.fill_between(w_s[condwl], spectral_rad_00[condwl] * (1 - spectral_rad_00_unc[condwl]), spectral_rad_00[condwl] * (1 + spectral_rad_00_unc[condwl]), color="gray", alpha=0.6)
    ax2.plot(cops_wl, cops_val,  color="k", marker="d", markersize=6, linestyle="None", markeredgecolor="k", markerfacecolor="none", label="C-OPS at 589 nm")
    ax2.errorbar(effective_lambda, effective_rad, xerr=replica_trapz(wl_rsr, rsr) / 2, color="k", marker="o", markersize=5, linestyle="None", markeredgecolor="k", markerfacecolor="none", label="$\overline{L}_{i, source}$")

    #marker = ["o", "s", "^"]
    #colo = ['#d95f02', '#1b9e77', '#7570b3']
    #lstyle = ["-", "-.", ":"]
    #lab = ["red", "green", "blue"]
    #for n in range(3):
    #    ax2.errorbar(effective_lambda[n], effective_rad[n], xerr=replica_trapz(wl_rsr, rsr[:, n]) / 2,
    #                 color=colo[n], marker=marker[n], markersize=7, linestyle="None", markeredgecolor=colo[n],
    #                 markerfacecolor="none", label="$\overline{{L}}_{{{0}, source}}$".format(lab[n]))

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
    fig3.tight_layout()
    fig5.tight_layout()

    # Saving results
    #correspond_optic = {"c": "close", "f": "far"}
    #save_answer = process.save_results()

    # Save figure
    #fig.savefig("figures/output_sphere_{}.pdf".format(correspond_optic[answer.lower()]), format="pdf", dpi=600,
    #            bbox_inches='tight')
    #fig.savefig("figures/output_sphere_{}.png".format(correspond_optic[answer.lower()]), format="png", dpi=600,
    #            bbox_inches='tight')  # png
    #fig2.savefig("figures/spectral_radiance_{}.pdf".format(correspond_optic[answer.lower()]), format="pdf", dpi=600,
    #             bbox_inches='tight')
    #fig2.savefig("figures/spectral_radiance_{}.png".format(correspond_optic[answer.lower()]), format="png", dpi=600,
    #             bbox_inches='tight') # png

    #fig5.savefig("figures/output_sphere_1row_{}.png".format(correspond_optic[answer.lower()]), format="png", dpi=600,
    #             bbox_inches='tight')  # png

    # Saving calibration
    #if save_answer == "y":
    #
    #    filename = "absolute_radiance_fluorolog" + ".h5"
    #    pathname = "calibrationfiles/" + filename

    #    timestr = time.strftime("%Y%m%d", time.localtime(os.stat(imlist[0])[-1]))

    #    correspond_optic = {"c": "close", "f": "far"}

    #    if answer.lower() == "c":
    #        process.create_hdf5_dataset(pathname, "lens-close/" + timestr, "cal-coefficients", coeff)
    #    else:
    #        process.create_hdf5_dataset(pathname, "lens-far/" + timestr, "cal-coefficients", coeff)

    plt.show()
