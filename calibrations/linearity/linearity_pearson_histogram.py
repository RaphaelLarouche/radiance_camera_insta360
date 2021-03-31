# -*- coding: utf-8 -*-
"""
Linearity histogram. Pearson coefficient histogram.
"""

# Importation of standard modules
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

# Importation of other modules
from source.processing import ProcessImage, FigureFunctions
from source.geometric_rolloff import MatlabGeometric


# Functions
def pearson_coefficient(exp, pixeldata, verbose=True):
    """

    :param exp:
    :param pixeldata:
    :param verbose:
    :return:
    """
    pearson = np.array([])
    for n in range(pixeldata.shape[0]):
        p, _ = stats.pearsonr(exp, pixeldata[n, :])
        pearson = np.append(pearson, p)

    if verbose:
        print("Pearson coefficient iso stats\n"
              "Median: {0}\n"
              "Minimum: {1}\n"
              "Maximum: {2}\n"
              "Mean: {3}".format(np.median(pearson), pearson.min(), pearson.max(), pearson.mean()))

    return pearson


def darkremoval(imstack, blstack):
    """

    :param imstack:
    :param blstack:
    :return:
    """

    incr = int(imstack.shape[2]/blstack.shape[2])
    first_indx = np.arange(0, imstack.shape[2], incr)
    rimstack = imstack.copy()
    for i, ind in enumerate(first_indx):
        rimstack[:, :, ind:ind+incr] -= blstack[:, :, i][:, :, None]
    return rimstack


def maskdatacolor(imstack, nummean, zenith, zenlim):
    """

    :param imstack:
    :param nummean:
    :return:
    """
    processim = ProcessImage()

    imstack_avg = block_reduce(imstack, block_size=(1, 1, nummean), func=np.mean)  # Average

    zen_dws = processim.dwnsampling(zenith, "RGGB")  # Downsampling zenith=
    data = {"r": np.array([]), "g": np.array([]), "b": np.array([])}  # Pre-allocation

    for i in range(imstack_avg.shape[2]):

        currim_dws = processim.dwnsampling(imstack_avg[:, :, i], "RGGB")

        for n, label in enumerate(data.keys()):
            if i==0:
                data[label] = currim_dws[zen_dws[:, :, n]<=zenlim, n].reshape(-1, 1).copy()
            else:
                data[label] = np.concatenate((data[label], currim_dws[zen_dws[:, :, n]<=zenlim, n].reshape(-1, 1)), axis=1)
    return data


def analysis_oneimage(zen, zenlim, imlist, step, num_mean, wlens):
    """

    :param zen:
    :param zenlim:
    :param imlist:
    :param step:
    :param num_mean:
    :param wlens:
    :return:
    """

    processimage = ProcessImage()

    # Loop for image processing
    exp = np.array([])
    iso = np.array([])
    for n, pathname in enumerate(imlist[::step]):

        # Pre-allocation
        im = np.empty((3456, 3456, num_mean))

        wh = imlist.index(pathname)
        print(wh)
        for nim in range(num_mean):
            # Opening image
            im_oth, met_oth = processimage.readDNG_insta360_np(imlist[wh+nim], which_image=wlens)

            im_oth = im_oth.astype(float)
            im_oth -= float(str(met_oth["Image Tag 0xC61A"]))

            im[:, :, nim] = im_oth

        image = np.mean(im, axis=2)
        image_std = np.std(im, axis=2)

        data = image[zen <= zenlim].reshape(-1, 1)
        datastd = image_std[zen <= zenlim].reshape(-1, 1)

        # RGB pixels
        image_dws = processimage.dwnsampling(image, "RGGB")  # mean pixel values
        zen_dws = processimage.dwnsampling(zen, "RGGB")

        # Storing data
        exp = np.append(exp, float(processimage.extract_integrationtime(met_oth)))
        iso = np.append(iso, float(str(met_oth["Image ISOSpeedRatings"])))

        if n==0:
            alldata = data.copy()
            alldatastd = datastd.copy()

            alldata_r = image_dws[zen_dws[:, :, 0] <= zenlim, 0].reshape(-1, 1).copy()
            alldata_g = image_dws[zen_dws[:, :, 1] <= zenlim, 1].reshape(-1, 1).copy()
            alldata_b = image_dws[zen_dws[:, :, 2] <= zenlim, 2].reshape(-1, 1).copy()
        else:
            alldata = np.concatenate((alldata, data), axis=1)
            alldatastd = np.concatenate((alldatastd, datastd), axis=1)

            alldata_r = np.concatenate((alldata_r, image_dws[zen_dws[:, :, 0] <= zenlim, 0].reshape(-1, 1)), axis=1)
            alldata_g = np.concatenate((alldata_g, image_dws[zen_dws[:, :, 1] <= zenlim, 1].reshape(-1, 1)), axis=1)
            alldata_b = np.concatenate((alldata_b, image_dws[zen_dws[:, :, 2] <= zenlim, 2].reshape(-1, 1)), axis=1)

    return exp, iso, alldata, alldatastd, {"r": alldata_r, "g": alldata_g, "b": alldata_b}


def rel_uncertainty_mult_div(relative_unc_x, relative_unc_y):
    return np.sqrt(relative_unc_x**2 + relative_unc_y**2)


if __name__ == "__main__":

    # Instance figurefunctions
    ff = FigureFunctions()

    # Instance of of class ProcessImage
    processimage = ProcessImage()

    # Files path (MYBOOK)
    filepath_exp = "/Volumes/MYBOOK/data-i360/calibrations/linearity/integration-time/"
    filepath_iso = "/Volumes/MYBOOK/data-i360/calibrations/linearity/iso-gain/"

    # Input lens to analyzed
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    imlist_exp, imlist_iso, imlist_exp_bl, wlens = [], [], [], ""

    if answer.lower() == "c":

        # Imlist (MYBOOK)
        imlist_exp = processimage.imageslist(filepath_exp + "lensclose")[4:-6]
        imlist_iso = processimage.imageslist(filepath_iso + "lensclose")
        wlens = "close"

        imlist_exp_bl = processimage.imageslist_dark(filepath_exp + "lensclose")

        # Geometric calibration air
        geo = MatlabGeometric("../geometric-calibration/calibrationfiles/FishParamsClose_01_05_2019.mat")

    elif answer.lower() == "f":

        # Imlist (MYBOOK)
        imlist_exp = processimage.imageslist(filepath_exp + "lensfar")
        imlist_iso = processimage.imageslist(filepath_iso + "lensfar")
        wlens = "far"

        imlist_exp_bl = processimage.imageslist_dark(filepath_exp + "lensfar")

        # Geometric calibration air
        geo = MatlabGeometric("../geometric-calibration/calibrationfiles/FishParamsFar_01_05_2019.mat")

    # Angular coordinates
    _, zen, azi = geo.angular_coordinates()

    # Illuminated light stack
    imstack_exp, exp_expln, iso_expln, bl_expln = processimage.imagestack(imlist_exp, wlens)
    imstack_iso, exp_isoln, iso_isoln, bl_isoln = processimage.imagestack(imlist_iso, wlens)

    # Dark image stack (only for exposure)
    imstack_exp_bl, exp_bl, iso_bl, _ = processimage.imagestack(imlist_exp_bl, wlens)

    # Dark removal
    imstack_exp -= bl_expln[None, None, :]
    imstack_iso -= bl_isoln[None, None, :]

    # Average
    mask_zenith = 5

    # Normalization exposure time
    interval = 4  # interval between image
    firstim = 0  # first image to take

    # Normalization exposure time
    alldata_exp = imstack_exp[zen<=mask_zenith, firstim::interval]
    exptil = np.tile(exp_expln[::interval], (alldata_exp.shape[0], 1))
    norm_exp = exptil * np.tile(iso_expln[::interval], (alldata_exp.shape[0], 1))
    dn_norm_exp = alldata_exp.astype(float) / norm_exp

    # Normalization iso
    alldata_iso = imstack_iso[zen<=mask_zenith, firstim::interval]
    isotil = np.tile(iso_isoln[::interval], (alldata_iso.shape[0], 1))
    norm_iso = isotil * np.tile(exp_isoln[::interval], (alldata_exp.shape[0], 1))
    dn_norm_iso = alldata_iso.astype(float) / norm_iso

    # Pearson coefficient
    p = pearson_coefficient(exptil[0, :], alldata_exp)  # exposure time
    p_iso = pearson_coefficient(isotil[0, :], alldata_iso)  # iso gain
    p = p ** 2
    p_iso = p_iso ** 2

    # Color separation
    nummean = 4
    alldata_exp_cl = maskdatacolor(imstack_exp, nummean, zen, mask_zenith)
    alldata_iso_cl = maskdatacolor(imstack_iso, nummean, zen, mask_zenith)

    # Figures
    plt.style.use("../../figurestyle.mplstyle")

    # Figure 2
    fig2 = plt.figure(figsize=ff.set_size(fraction=0.7, height_ratio=0.75))
    ax2 = fig2.add_subplot(111)
    alphisto = 0.7

    ax2.hist(p, range=(0.92, 1.00), bins=100, alpha=alphisto, color="k", label="integration time")
    ax2.hist(p_iso, range=(0.92, 1.00), bins=100, alpha=alphisto, color="gray", label="ISO gain")
    ax2.axvline(p.mean(), color="k", linestyle="--", alpha=alphisto)
    ax2.axvline(p_iso.mean(), color="gray", linestyle="--", alpha=alphisto)
    ax2.annotate('$\mu_{{r_{{t}}^2}}={0:.5f}$'.format(p.mean()), xy=(p.mean(), 400), xytext=(-150, 0),
                 textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle='->'))
    ax2.annotate('$\mu_{{r_{{ISO}}^2}}={0:.5f}$'.format(p_iso.mean()), xy=(p_iso.mean(), 200), xytext=(-100, 0),
                 textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle='->'))

    ax2.set_yscale("log")
    ax2.set_ylim(None, 30000)

    ax2.text(0.92, 1000, "{0} individual pixels".format(p.shape[0]), fontsize=8)

    ax2.set_xlabel("Squared Pearson coefficient $r^2$")
    ax2.set_ylabel("Counts")

    ax2.legend(loc="upper left")

    fig2.tight_layout()

    # Figure 3
    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(exptil[::100, :].T, dn_norm_exp[::100, :].T, color="grey", alpha=0.2)

    ax3.set_xscale("log")

    ax3.set_ylabel("DN/($t_{int}\cdot$ISO)")
    ax3.set_xlabel("$t_{int}$ [s]")

    fig3.tight_layout()

    # Figure 4
    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(isotil[::100, :].T, dn_norm_iso[::100, :].T, color="grey", alpha=0.2)

    ax4.set_xscale("log")

    ax4.set_ylabel("DN/($t_{int}\cdot$ISO)")
    ax4.set_xlabel("ISO")

    fig4.tight_layout()

    # Figure 5 -
    fig5, ax5 = plt.subplots(1, 1)

    ls = ["-", "-.", "--"]
    for num, k in enumerate(alldata_exp_cl.keys()):
        exp_repmat = np.tile(exp_expln[::interval], (alldata_exp_cl[k].shape[0], 1))
        iso_repmat = np.tile(iso_expln[::interval], (alldata_exp_cl[k].shape[0], 1))
        alldatacolor = alldata_exp_cl[k]/(iso_repmat * exp_repmat)

        ax5.plot(exp_repmat[::100, :].T, alldatacolor[::100, :].T, linestyle=ls[num], color="grey", alpha=0.2)

    ax5.set_xscale("log")
    ax5.set_ylabel("DN/($t_{int}\cdot$ISO)")
    ax5.set_xlabel("$t_{int}$ [s]")

    fig5.tight_layout()

    # Figure 6 - figure iso and integration time

    fig6, ax6 = plt.subplots(1, 2, figsize=ff.set_size(height_ratio=0.45))
    fig7, ax7 = plt.subplots(3, 2, figsize=(12.8, 8), sharex=True)
    fig8, ax8 = plt.subplots(2, 3, figsize=ff.set_size(fraction=1))

    m = iter(["o", "s", ">"])
    ls = iter(["-", "--", "-."])
    col = iter(["#d62728", "#2ca02c", "#084594"])

    for k, lab in enumerate(["r", "g", "b"]):

        # Data ax6
        x_exp, y_exp = exp_expln[::interval] / exp_expln[0], alldata_exp_cl[lab].mean(axis=0) / alldata_exp_cl[lab].mean(axis=0)[0]
        x_iso, y_iso = iso_isoln[::interval] / iso_isoln[0], alldata_iso_cl[lab].mean(axis=0) / alldata_iso_cl[lab].mean(axis=0)[0]

        rel_inc_y_exp = alldata_exp_cl[lab].std(axis=0) / alldata_exp_cl[lab].mean(axis=0)
        yerr_exp = rel_uncertainty_mult_div(rel_inc_y_exp, rel_inc_y_exp[0]) * y_exp

        rel_inc_y_iso = alldata_iso_cl[lab].std(axis=0) / alldata_iso_cl[lab].mean(axis=0)
        yerr_iso = rel_uncertainty_mult_div(rel_inc_y_iso, rel_inc_y_iso[0]) * y_iso

        reg_exp = stats.linregress(x_exp, y_exp)
        reg_iso = stats.linregress(x_iso, y_iso)

        # Data axe 7
        exp_repmat = np.tile(exp_expln[::interval], (alldata_exp_cl[lab].shape[0], 1))
        exp_rempat_norm = exp_repmat / np.tile(exp_repmat[:, 0], (alldata_exp_cl[lab].shape[1], 1)).T
        iso_repmat = np.tile(iso_isoln[::interval], (alldata_iso_cl[lab].shape[0], 1))
        iso_rempat_norm = iso_repmat / np.tile(iso_repmat[:, 0], (alldata_iso_cl[lab].shape[1], 1)).T

        # Data axe 8
        expm, expv = alldata_exp_cl[lab].mean(axis=0), alldata_exp_cl[lab].var(axis=0)
        photon_curves = stats.linregress(expm, expv)
        print(photon_curves)
        xphoton_curves = np.linspace(expm.min() * 0.95, expm.max() * 1.05, 50)
        lin_curves = stats.linregress(exp_expln[::interval], expm)
        x_expln = np.linspace(exp_expln[::interval].min() * 0.95, exp_expln[::interval].max() * 1.05, 50)

        # Marker and linestyle
        colo = next(col)
        mar = next(m)
        lsty = next(ls)

        ax6[0].errorbar(x_exp, y_exp, yerr=alldata_exp_cl[lab].std(axis=0) / alldata_exp_cl[lab].mean(axis=0)[0],
                        marker=mar, ms=6, markeredgecolor=colo, markerfacecolor="none",
                        markeredgewidth=1, linestyle="none")
        ax6[1].errorbar(x_iso, y_iso, yerr=alldata_iso_cl[lab].std(axis=0) / alldata_iso_cl[lab].mean(axis=0)[0],
                        marker=mar, ms=6, markeredgecolor=colo, markerfacecolor="none",
                        markeredgewidth=1, linestyle="none")

        lege1 = "$m = {0:.3f}$, $b = {1:.3f}$, $r^2= {2:.6f}$".format(reg_exp[0], reg_exp[1], reg_exp[2]**2)
        lege2 = "$m = {0:.3f}$, $b = {1:.3f}$, $r^2= {2:.5f}$".format(reg_iso[0], reg_iso[1], reg_iso[2] ** 2)

        ax6[0].plot(np.linspace(0, 33, 50), np.linspace(0, 33, 50) * reg_exp[0] + reg_exp[1], color=colo, linestyle=lsty, label=lege1)
        ax6[1].plot(np.linspace(0, 33, 50), np.linspace(0, 33, 50) * reg_iso[0] + reg_iso[1], color=colo, linestyle=lsty, label=lege2)

        dn_iso_rel = alldata_iso_cl[lab].astype(float) / np.tile(alldata_iso_cl[lab].astype(float)[:, 0], (alldata_iso_cl[lab].shape[1], 1)).T
        dn_exp_rel = alldata_exp_cl[lab].astype(float) / np.tile(alldata_exp_cl[lab].astype(float)[:, 0], (alldata_exp_cl[lab].shape[1], 1)).T

        npixelexp = exp_rempat_norm[::100, :].T.shape[1]
        #ax7[k, 0].plot(np.nan, np.nan, linestyle="-.", color="grey", alpha=0.3, label="individual pixels")  # dummy
        ax7[k, 0].plot(exp_rempat_norm[::100, :].T, dn_exp_rel[::100, :].T, linestyle="-.", color="grey", alpha=0.3)
        ax7[k, 0].errorbar(x_exp, y_exp, yerr=alldata_exp_cl[lab].std(axis=0) / alldata_exp_cl[lab].mean(axis=0)[0], marker=">", ms=6, markeredgecolor="k", markerfacecolor="none", markeredgewidth=1.3, linestyle="none")
        ax7[k, 0].plot(np.linspace(0, 35, 100), np.linspace(0, 35, 100)*reg_exp[0] + reg_exp[1], color="k", linestyle="-", label=lege1)
        ax7[k, 0].set_ylabel("$\mathrm{DN}_{norm}$")

        npixeliso = iso_rempat_norm[::100, :].T.shape[1]
        #ax7[k, 1].plot(np.nan, np.nan, linestyle="-.", color="grey", alpha=0.3, label="individual pixels")  # dummy
        ax7[k, 1].plot(iso_rempat_norm[::100, :].T, dn_iso_rel[::100, :].T, linestyle="-.", color="grey", alpha=0.3)
        ax7[k, 1].errorbar(x_iso, y_iso, yerr=alldata_iso_cl[lab].std(axis=0) / alldata_iso_cl[lab].mean(axis=0)[0], marker=">", ms=6, markeredgecolor="k", markerfacecolor="none", markeredgewidth=1.3, linestyle="none")
        ax7[k, 1].plot(np.linspace(0, 35, 50), np.linspace(0, 35, 50) * reg_iso[0] + reg_iso[1], color="k", linestyle="-", label=lege2)
        ax7[k, 1].set_ylabel("$\mathrm{DN}_{norm}$")
        ax7[k, 0].legend(loc="lower right")
        ax7[k, 1].legend(loc="lower right")

        # Axe 8
        ax8[0, k].plot(expm, expv, linestyle="none", marker=".")
        ax8[0, k].plot(xphoton_curves, xphoton_curves * photon_curves[0] + photon_curves[1])

        ax8[0, k].set_xlabel("signal average [DN]")
        ax8[0, k].set_ylabel("signal variance [$\mathrm{DN^2}$]")

        ax8[1, k].plot(exp_expln[::interval], expm, linestyle="none", marker=".")
        ax8[1, k].plot(x_expln, x_expln * lin_curves[0] + lin_curves[1])

        ax8[1, k].set_xlabel("$t_{int}$ [s]")
        ax8[1, k].set_ylabel("signal average [DN]")

    # Fig 6
    ax6[0].xaxis.grid(linestyle="--")
    ax6[0].yaxis.grid(linestyle="--")

    ax6[0].set_xticks(np.arange(0, 40, 5))
    ax6[0].set_xlim((0, 35))
    ax6[0].set_ylim((0, 35))

    ax6[0].set_xlabel("$t_{int, norm}$")
    ax6[0].set_ylabel(r"$\mathrm{DN}_{t_{int, norm}}$")

    ax6[0].legend(loc="best", fontsize=7)

    ax6[1].xaxis.grid(linestyle="--")
    ax6[1].yaxis.grid(linestyle="--")

    ax6[1].set_xticks(np.arange(0, 40, 5))
    ax6[1].set_xlim((0, 35))
    ax6[1].set_ylim((0, 35))

    ax6[1].set_xlabel("$\mathrm{ISO}_{norm}$")
    ax6[1].set_ylabel(r"$\mathrm{DN}_{\mathrm{ISO}_{norm}}$")

    ax6[1].legend(loc="best", fontsize=7)

    fig6.tight_layout()

    # Fig 7
    ax7[2, 0].set_xlabel("$t_{int, norm}$")
    ax7[2, 1].set_xlabel("$\mathrm{ISO}_{norm}$")

    # Fig 8
    fig7.tight_layout()
    fig8.tight_layout()

    # Saving figures
    optics_correspondance = {"c": "close", "f": "far"}
    fig2.savefig("figures/linearity-histogram-{}.pdf".format(optics_correspondance[answer.lower()]), format="pdf", dpi=600)

    plt.show()
