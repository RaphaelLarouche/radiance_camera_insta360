# -*- coding: utf-8 -*-
"""
Oden data 2018, cam optics folder.
"""

# Module importation
import os
import h5py
import glob
import pandas
import string
import numpy as np
import matplotlib.pyplot as plt

# Other modules


# Function
def imagelabel(path):
    """

    :param path:
    :return:
    """
    labels = {}
    with open(path, "r") as f:
        for lines in f:
            name, depth = lines.split("*")
            labels[name] = depth.strip()
    return labels


def rsr_statistics(wavelength, rsr, verbose=True):
    """
    Printing spectral statistics.
    :param wavelength:
    :param rsr:
    :return:
    """
    eff_bw = np.zeros(3)
    eff_wl = np.zeros(3)
    max_wl = np.zeros(3)
    for band in range(rsr.shape[1]):
        eff_bw[band] = np.trapz(rsr[:, band], x=wavelength)
        eff_wl[band] = np.trapz(rsr[:, band] * wavelength, x=wavelength) / eff_bw[band]
        max_wl[band] = wavelength[np.argmax(rsr[:, band])]

        if verbose:
            print("Band no. {0} statistics".format(band))
            print("Effective bw: {0:.4f}, effective wl: {1:.4f}, maximum wl: {2:.4f}". format(eff_bw[band], eff_wl[band], max_wl[band]))
    return eff_wl, max_wl, eff_bw


def irradiance(rad, zeni, azi, zenimin, zenimax, planar=True):
    """

    :param rad:
    :param zeni: in degrees
    :param azi: in degrees
    :return:
    """
    mask = (zenimin <= zeni) & (zeni <= zenimax)

    zenith = zeni * np.pi / 180
    azimuth = azi * np.pi / 180

    if planar:
        integrand = rad[mask] * np.absolute(np.cos(zenith[mask])) * np.sin(zenith[mask])
    else:
        integrand = rad[mask] * np.sin(zenith[mask])

    azimuth_inte = integrate.simps(integrand.reshape((-1, azimuth.shape[1])), azimuth[mask].reshape((-1, azimuth.shape[1])), axis=1)

    return integrate.simps(azimuth_inte, zenith[mask].reshape((-1, azimuth.shape[1]))[:, 0], axis=0)


def colorbardepth(f, a, cmapp, diction):
    """
    Function to add colorbar to the profile figure.

    :param f:
    :param a:
    :param cmapp:
    :param diction:
    :return:
    """
    valdum = np.array(list(diction.values()))
    dcax = a.scatter(valdum, valdum, c=np.arange(1, len(valdum) + 1), cmap=cmapp)

    a.cla()
    cb = f.colorbar(dcax, ax=a, orientation="vertical", fraction=0.05, pad=0.04)
    cb.ax.locator_params(nbins=cmapp.colors.shape[0])
    cb.ax.set_yticklabels(["{0:.0f}".format(j) for j in np.array(list(diction.values()))])
    cb.ax.set_title("depth [cm]", fontsize=6)

    return f, a


def anglefromY(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return:
    """
    return np.arctan2((x ** 2 + z ** 2) ** (1 / 2), y)


def save_radiance_image_hdf5(path_name, dataname, dat):
    """

    :param path_name:
    :param dataname:
    :param mapped_radiance:
    :param zenith_mesh: meshgrid of zenith in radians
    :param azimuth_mesh: meshgrid of azimuth in radians
    :return:
    """

    datapath = dataname

    with h5py.File(path_name) as hf:
        if datapath in hf:
            d = hf[datapath]  # load the data
            d[...] = dat
        else:
            dset = hf.create_dataset(dataname, data=dat)


def saveresults():
    """

    :return:
    """
    ans = ""
    while ans not in ["y", "n"]:
        ans = input("Do you want to save this simulation?")

    return ans.lower()


if __name__ == "__main__":

    # Instance processimage
    processim = classes_i360.ProcessImage()

    # Instance of FigureFunctions
    ff = classes_i360.FigureFunctions()

    # Geometric calibration
    mgeometric = classes_i360.MatlabGeometric("../calibrations/geometric/calibrationfiles_water/CloseFisheyeParams.mat")
    _, zen, azi = mgeometric.angular_coordinates()

    # Oden data
    odenpath = "/Volumes/MYBOOK/data-i360/field/oden-08312018/"
    oden_impath = glob.glob(odenpath + "*.dng")
    oden_impath.sort()

    # Oden radiometer data
    openmatlab = classes_i360.OpenMatlabFiles()
    radiometer = openmatlab.loadmat(odenpath + "radiometerdata/2018R4_300025060010720.mat")
    radiometer = radiometer["raw"]

    locale.setlocale(locale.LC_ALL, 'en_US')
    timestamps = pandas.to_datetime(radiometer["irrad_t_1"] - 719529, unit='D')  # 719529 1 january 2000 Matlab
    #timestamps = timestamps.tz_localize("UTC").tz_convert("Etc/GMT-5")
    df_time = pandas.DataFrame({"Date": timestamps.to_period('H')})
    #mask = (df_time["Date"] <= "2018-08-31 12:00") & (df_time["Date"] >= "2018-08-31 1:00")
    mask = (df_time["Date"] <= "2018-08-31 16:00") & (df_time["Date"] >= "2018-08-31 8:00")
    #mask = (df_time["Date"] <= "2018-09-01 00:00") & (df_time["Date"] >= "2018-08-31 0:00")
    inx = df_time["Date"][mask].index

    timestamps = timestamps.strftime("%B %d %Y, %H:%M:%S")

    irradiance_incom = radiometer["incom_int_1"][inx].T / 1000  # mW to W
    irradiance_below = radiometer["irrad_int_1"][inx].T / 1000  # mW to W

    # Spectral curves
    srdata = h5py.File("../calibrations/relative_spectral_response/calibrationfiles/rsr_20200610.h5", "r")
    srdata = srdata["lens-close"]

    eff_wl, max_wl, eff_bw = rsr_statistics(srdata["wavelength"][:], srdata["rsr_peak_norm"][:])

    # Pre-allocation
    plt.style.use("../figurestyle.mplstyle")
    fig1, ax1 = plt.subplots(2, 3, sharex=True, figsize=(8, 4.59))
    fig4 = plt.figure(figsize=ff.set_size(subplots=(2, 2)))

    Ed = np.zeros(11, dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))
    Eu = np.zeros(11, dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))
    Eo = np.zeros(11, dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))

    ax4 = np.ndarray((2, 2), dtype=object)
    ax4[0, 0] = fig4.add_subplot(221)
    ax4[0, 1] = fig4.add_subplot(222, sharey=ax4[0, 0])
    ax4[1, 0] = fig4.add_subplot(223, sharex=ax4[0, 0])
    ax4[1, 1] = fig4.add_subplot(224, sharex=ax4[0, 1], sharey=ax4[1, 0])

    # Read profile metadata
    labels = imagelabel(odenpath + "ReadMe_python.txt")
    wanted_depth = ["zero minus", "20 cm (in water)", "40 cm", "60 cm", "80 cm", "100 cm", "120 cm", "140 cm", "160 cm", "180 cm", "200 cm"]
    # 180 cm likely 165 cm as selfie stick came up compressed
    imagepath_filtered = oden_impath[2:-9:]
    imagepath_filtered[-1] = oden_impath[-9]

    bounds = np.arange(0, 220, 20)

    # Colormap
    colornormdict = dict(zip(wanted_depth, bounds))
    colo = matplotlib.cm.get_cmap("viridis", len(colornormdict.values()))
    cmit = iter(colo.colors)

    # Dummy
    fig4, ax4[0, 1] = colorbardepth(fig4, ax4[0, 1], colo, colornormdict)
    fig4, ax4[1, 1] = colorbardepth(fig4, ax4[1, 1], colo, colornormdict)

    num = 0
    dicband = {0: "r", 1: "g", 2: "b"}

    # Data saving parameters
    path_save_rad = "data/oden-08312018_good.h5"

    answ = saveresults()
    cond_s = answ == "y"
    # Loop
    for f in imagepath_filtered:

        _, tail = os.path.split(f)

        if labels[tail[:-4]] in wanted_depth:

            head, tail = os.path.split(f)

            print(labels[tail[:-4]])

            im_op, met_op = processim.readDNG_insta360_np(f, "close")

            # Printing exposure paramters
            # print(processim.extract_iso(met_op))
            # print(processim.extract_integrationtime(met_op))

            datanoise = im_op[zen >= 90]

            if labels[tail[:-4]] == "zeros minus":
                imageradiance = classes_i360.ImageRadiancei360(f, "air")
            else:
                imageradiance = classes_i360.ImageRadiancei360(f, "water")

            # Post-processing to get radiance angular distribution from images
            _ = imageradiance.getradiance(dark_metadata=False)
            _, _, _ = imageradiance.radiancemap()
            integration = imageradiance.azimuthal_average()  # Integration

            # Saving data
            if cond_s:
                save_radiance_image_hdf5(path_save_rad, labels[tail[:-4]], imageradiance.mappedradiance.copy())
                save_radiance_image_hdf5(path_save_rad, "zenith", imageradiance.zenith_mesh.copy() * 180 / np.pi)
                save_radiance_image_hdf5(path_save_rad, "azimuth", imageradiance.azimuth_mesh.copy() * 180 / np.pi)

            cl = next(cmit)
            for i in range(integration.shape[1]):
                cond0 = np.where(integration[:, i] == 0)
                integra = integration[:, i].copy()
                zenith = imageradiance.zenith_mesh[:, 0].copy()

                integra[cond0] = np.nan
                zenith[cond0] = np.nan

                # Irradiances, interpolations
                X, Y, Z = imageradiance.points_3d(imageradiance.zenith_mesh.copy(), imageradiance.azimuth_mesh.copy())
                rad = imageradiance.mappedradiance[:, :, i].copy()
                aY = anglefromY(X, Y, Z)
                fov = 70
                cond = (aY <= fov * np.pi / 180) | (np.pi - aY <= fov * np.pi / 180)

                condzero = ~cond
                rad[condzero] = griddata((X[~condzero], Y[~condzero], Z[~condzero]),
                                         rad[~condzero], (X[condzero], Y[condzero], Z[condzero]),
                                         method="nearest")

                Ed[dicband[i]][num] = irradiance(rad, imageradiance.zenith_mesh.copy() * 180 / np.pi, imageradiance.azimuth_mesh.copy() * 180 / np.pi, 0, 90)
                Eu[dicband[i]][num] = irradiance(rad, imageradiance.zenith_mesh.copy() * 180 / np.pi, imageradiance.azimuth_mesh.copy() * 180 / np.pi, 90, 180)
                Eo[dicband[i]][num] = irradiance(rad, imageradiance.zenith_mesh.copy() * 180 / np.pi, imageradiance.azimuth_mesh.copy() * 180 / np.pi, 0, 180, planar=False)

                # Ax1
                ax1[0, i].plot(zenith * 180 / np.pi, integra, linewidth=2, color=cl, label=labels[tail[:-4]])
                ax1[0, i].set_yscale("log")
                ax1[0, i].set_xlim((20, 160))
                ax1[0, i].set_ylim((1e-5, 0.1))

                # Ax2
                ax1[1, i].plot(zenith * 180 / np.pi, 100*(integra/np.nanmax(integra)), linewidth=2, color=cl, label=labels[tail[:-4]])
                ax1[1, i].set_yticks(np.arange(0, 120, 20))
                ax1[1, i].set_xlabel("Zenith angle [˚]")

                if i == 1:  # Green band
                    ax4[0, 1].plot(zenith * 180 / np.pi, integra, linewidth=2, color=cl, label=labels[tail[:-4]])
                    ax4[0, 1].set_yscale("log")
                    ax4[0, 1].set_xlim((20, 160))
                    #ax4[0, 1].set_ylim((2.5842052131699986e-05, 0.033625178581195549))

                    ax4[1, 1].plot(zenith * 180 / np.pi, 100 * (integra / np.nanmax(integra)), linewidth=2, color=cl, label=labels[tail[:-4]])
                    ax4[1, 1].set_yticks(np.arange(0, 120, 20))
                    ax4[1, 1].set_xlabel("Zenith [˚]")

            if labels[tail[:-4]] == "60 cm":
                fig_cont, ax_c = plt.subplots(1, 3, figsize=ff.set_size(443.86319), subplot_kw=dict(projection='polar'))
                fig_cont, ax_c = imageradiance.polar_plot_contourf(fig_cont, ax_c, 20)
            num += 1
        else:
            continue

    # DORT2002 simulations - wavelength 540 nm (near effective wl of green band)
    number_layer = 4.0
    layer_thickness = matlab.double([100, 0.1, 2.0, 10.0])
    #depth = matlab.double(list(np.arange(100.2, 102.1, 0.1)))  # Between 20 cm and 200 cm each 10 cm
    depth = matlab.double(list(np.arange(100, 102.1, 0.1)))

    # d = [100.0] + list(np.arange(100.2, 102.1, 0.1))
    # depth = matlab.double(d)

    # Christian IOPs
    b = matlab.double([0, 250, 25, 0.1])  # scattering coefficient
    a = matlab.double([0, 0.15, 0.15, 0.15]) # absorption coefficient

    # IOPs most probable at 544 nm (effective wavelength of green band)
    # b = matlab.double([0, 250, 55, 0.1])  # scattering coefficient
    # a = matlab.double([0, 0.0683, 0.0683, 0.0683])  # absorption 0.0683 m-1 Perovich and Grenfell 1981 at 540 nm

    g = matlab.double([0, 0.9, 0.9, 0.9])
    n = matlab.double([1, 1.33, 1.33, 1.33])

    # Incoming lambertian radiance (at 544 nm) from RAMSES TriOS irradiance measurements above sea ice
    wl_dort = np.round(eff_wl[1])  # At effective wl of green band
    arg_wl_dort = np.where(wl_dort == radiometer["wl"])
    rad_inc = irradiance_incom.mean(axis=1)[arg_wl_dort]/np.pi
    rad_inc_std = irradiance_incom.std(axis=1)[arg_wl_dort]/np.pi

    eng = matlab.engine.start_matlab()
    dort_p, dort_r = eng.dort_simulation(number_layer, layer_thickness, b, a, g, n, depth, matlab.double(list(rad_inc)), nargout=2)
    rad_dist_dort, zen_mesh_dort, azi_mesh_dort = compute_results_dort(dort_p, dort_r, bounds)

    e_dort_surf = irradiance(rad_dist_dort["0"], zen_mesh_dort, azi_mesh_dort, 0, 90)
    e_dort_below = irradiance(rad_dist_dort["200"], zen_mesh_dort, azi_mesh_dort, 0, 90)

    print("DORT2002 transmittance --> {}".format(e_dort_below / e_dort_surf))
    print("Camera transmittances --> red: {}, green: {}, blue: {}".format(Ed["r"][-1] / Ed["r"][0], Ed["g"][-1] / Ed["g"][0], Ed["b"][-1] / Ed["b"][0]))

    # Figures ____
    # Fig 1 -
    ax1[0, 0].set_ylabel(r"$\overline{{L}}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
    ax1[1, 0].set_ylabel(r"$\frac{{\overline{{L}}}}{{\overline{{L}}_{{max}}}}$ [%]")

    # Fig 2
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(ff.set_size(subplots=(1, 1))[0], ff.set_size(subplots=(1, 1))[1] * 1.5))

    maskwl = (radiometer["wl"] <= 700) & (radiometer["wl"] >= 400)
    trans = irradiance_below[maskwl] / irradiance_incom[maskwl]
    argtr = np.argwhere(radiometer["wl"][maskwl] == np.round(eff_wl).reshape(-1, 1))

    lines0 = ax2[0].plot(radiometer["wl"][maskwl], irradiance_incom[maskwl])
    lines1 = ax2[1].plot(radiometer["wl"][maskwl], irradiance_below[maskwl])
    #lines2 = ax2[2].plot(radiometer["wl"][maskwl], trans)
    #ax2[2].errorbar(np.round(eff_wl), trans[argtr[:, 1], :].mean(axis=1), xerr=eff_bw/2, marker="o")

    ax2[0].set_ylabel("$E_{d, surface}(\lambda)$ [$\mathrm{W \cdot m^{-2} \cdot nm^{-1}}$]")
    ax2[1].set_ylabel("$E_{d, below}(\lambda)$ [$\mathrm{W \cdot m^{-2} \cdot nm^{-1}}$]")  # W m-2 nm-1
    #ax2[2].set_ylabel("Transmittance")

    ax2[1].set_xlabel("Wavelength [nm]")

    ax2[0].legend(lines0, timestamps[inx].values, fontsize=5, loc="best")
    ax2[1].legend(lines1, timestamps[inx].values, fontsize=5, loc="best")
    #ax2[2].legend(lines2, timestamps[inx].values, fontsize=5, loc="best")

    ax2[0].text(-0.1, 1.1, string.ascii_lowercase[0], transform=ax2[0].transAxes, size=11, weight='bold')
    ax2[1].text(-0.1, 1.1, string.ascii_lowercase[1], transform=ax2[1].transAxes, size=11, weight='bold')
    #ax2[2].text(-0.1, 1.1, string.ascii_lowercase[2], transform=ax2[2].transAxes, size=11, weight='bold')

    fig2.tight_layout()

    # Figure 3 - triOS transmittance
    fig3, ax3 = plt.subplots(1, 1, figsize=ff.set_size())

    lines2 = ax3.plot(radiometer["wl"][maskwl], trans)
    ax3.errorbar(np.round(eff_wl), trans[argtr[:, 1], :].mean(axis=1), xerr=eff_bw/2,
                 linestyle="none", marker="o", ecolor="k", markeredgecolor="k", markerfacecolor="None")

    ax3.legend(lines2, timestamps[inx].values, fontsize=5, loc="best")

    ax3.set_ylabel("Transmittance")
    ax3.set_xlabel("Wavelength [nm]")


    # Fig 4 - results dort
    plot_dort_absolute_radiance(ax4[0, 0], rad_dist_dort, zen_mesh_dort, colo)
    plot_dort_relative_radiance(ax4[1, 0], rad_dist_dort, zen_mesh_dort, colo)

    ax4[0, 0].set_yscale("log")
    #ax4[0, 0].set_ylim((2.5842052131699986e-05, 0.033625178581195549))

    ax4[0, 0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
    ax4[1, 0].set_ylabel("Relative radiance [%]")
    ax4[1, 0].set_xlabel("Zenith [˚]")

    ax4[0, 0].text(-0.1, 1.1, string.ascii_lowercase[0], transform=ax4[0, 0].transAxes, size=11, weight='bold')
    ax4[0, 1].text(-0.1, 1.1, string.ascii_lowercase[1], transform=ax4[0, 1].transAxes, size=11, weight='bold')

    # Fig 5 - polar plot DORT2002
    fig5, ax5 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=ff.set_size())
    contour_plot_dort(fig5, ax5, rad_dist_dort["60"], zen_mesh_dort, azi_mesh_dort, 20)

    # Saving figures ___
    fig_cont.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    fig_cont.savefig("figures/oden_contour.pdf", format="pdf", dpi=600)
    fig2.savefig("figures/triOS.pdf", format="pdf", dpi=600)
    fig3.savefig("figures/triOS_transmittance", format="pdf", dpi=600)
    fig4.savefig("figures/oden_profiles.pdf", format="pdf", dpi=600)

    plt.show()
