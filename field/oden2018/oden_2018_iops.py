"""
Oden 2018 iops tunning.
"""

# Module importation
import os
import pandas
import matplotlib
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt

# Other module importations
import source.radiance as r
from source.geometric_rolloff import OpenMatlabFiles
from source.processing import ProcessImage, FigureFunctions


# Function and classes
def open_TriOS_data(min_date="2018-08-31 8:00", max_date="2018-08-31 16:00", path_trios="data/2018R4_300025060010720.mat"):
    """

    :param min_date:
    :param max_date:
    :param path_trios:
    :return:
    """
    openmatlab = OpenMatlabFiles()
    radiometer = openmatlab.loadmat(path_trios)
    radiometer = radiometer["raw"]

    timestamps = pandas.to_datetime(radiometer["irrad_t_1"] - 719529, unit='D')   # 719529 = 1 january 2000 Matlab
    df_time = pandas.DataFrame({"Date": timestamps.to_period('H')})
    mask = (df_time["Date"] <= max_date) & (df_time["Date"] >= min_date)
    inx = df_time["Date"][mask].index

    timestamps = timestamps.strftime("%B %d %Y, %H:%M:%S")
    irradiance_incom = radiometer["incom_int_1"][inx].T / 1000  # mW to W
    irradiance_below = radiometer["irrad_int_1"][inx].T / 1000  # mW to W

    return timestamps, radiometer["wl"], irradiance_incom, irradiance_below


def compute_results_dort(p, r, wdepth):
    """

    :param p:
    :param r:
    :param wdepth:
    :return:
    """
    polar = np.arccos(p["mu_interpol"]) * 180 / np.pi
    zenith = np.concatenate((polar[::-1], 180 - polar), axis=0)
    azimuth = np.array(dort_p["phi_interpol"]) * 180 / np.pi

    az_mesh, zen_mesh = np.meshgrid(azimuth, zenith)

    # Pre-allocation
    rad_dist = {}

    for i, de in enumerate(p["depth"]):

        # Depth
        decm = round((float(np.array(de)) - 100) * 100)

        if decm in wdepth:

            radiance_d = np.array(r["I_ac_" + str(i+1)])
            radiance_d = radiance_d[::-1, :]
            radiance_d[45:, :] = radiance_d[45:, :][::-1, :]

            rad_dist[str(decm)] = radiance_d

    return rad_dist, zen_mesh, az_mesh


def colorbardepth(f, a, cmapp, valdum):
    """
    Function to add colorbar to the profile figure.

    :param f:
    :param a:
    :param cmapp:
    :param diction:
    :return:
    """

    dcax = a.scatter(valdum, valdum, c=np.arange(1, cmapp.N+1), cmap=cmapp)

    a.cla()
    cb = f.colorbar(dcax, ax=a, orientation="vertical")

    cb.ax.locator_params(nbins=cmapp.N)
    cb.ax.set_yticklabels(["{0:.0f}".format(j) for j in valdum])
    cb.ax.set_title("depth [cm]", fontsize=6)
    cb.ax.invert_yaxis()

    return f, a


def build_cmap_2cond_color(cmap_name, d):
    """

    :param cmap_name:
    :return:
    """

    CMA = matplotlib.cm.get_cmap(cmap_name, len(d) + 1)
    colooor = CMA(np.arange(1, CMA.N))
    custom_cmap = matplotlib.colors.ListedColormap(colooor[::-1])
    return custom_cmap


def scattering_coeff_figure(scattering_coefficient, layer_t, fsize):
    """

    :param scattering_coefficient:
    :param layer_depth:
    :return:
    """
    figure, axe = plt.subplots(1, 1, figsize=fsize)
    lims = np.cumsum(np.squeeze(np.array(layer_t))) - 100
    scat = np.squeeze(np.array(scattering_coefficient))
    labs = ["Surface scattering layer (SSL)", "Drained layer (DL)", "Interior ice (II)"]
    ls = [":", "-.", "-"]

    for z in range(lims.shape[0] - 2):

        depth_vect = np.linspace(lims[z], lims[z + 1], 30)
        s = np.ones(depth_vect.shape[0]) * scat[z + 1]

        axe.plot(s, depth_vect, linestyle=ls[z], color="gray", label=labs[z])

    axe.invert_yaxis()
    axe.set_xscale("log")
    axe.set_xlabel("Scattering coefficient $b~[\mathrm{m^{-1}}]$ ")
    axe.set_ylabel("Depth [m]")

    axe.legend(loc=4)

    return figure, axe


def create_figure_axe_3x3(fisiz):
    """

    :param fisiz:
    :return:
    """

    f = plt.figure(figsize=fisiz)
    a00 = f.add_subplot(3, 3, 1)
    a01 = f.add_subplot(3, 3, 2, sharex=a00, sharey=a00)
    a02 = f.add_subplot(3, 3, 3, sharex=a00)

    a10 = f.add_subplot(3, 3, 4, sharex=a00, sharey=a00)
    a11 = f.add_subplot(3, 3, 5, sharex=a00, sharey=a00)
    a12 = f.add_subplot(3, 3, 6, sharex=a00, sharey=a02)

    a20 = f.add_subplot(3, 3, 7, sharex=a00, sharey=a00)
    a21 = f.add_subplot(3, 3, 8, sharex=a00, sharey=a00)
    a22 = f.add_subplot(3, 3, 9, sharex=a00, sharey=a02)

    a = np.array([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])

    return f, a


if __name__ == "__main__":

    # Instance of FigureFunctions
    ff = FigureFunctions()
    plt.style.use("../../figurestyle.mplstyle")

    # Object ProcessImage
    process = ProcessImage()

    # Setting incoming radiance
    time_stamp, radiometer_wl, irr_incom, irr_below = open_TriOS_data()
    eff_wl = np.array([602, 544, 484])

    rad_dort_band = {}

    # Open Field measurements
    ze_mesh, az_mesh, rad_profile = process.open_radiance_data(path="data/oden-08312018.h5")

    # DORT SIMULATIONS
    eng = matlab.engine.start_matlab()
    nstreams = 30
    number_layer = 5.0
    layer_thickness = matlab.double([100, 0.1, 0.7, 1.2, 10.0])  # in meters
    depth = matlab.double(list(np.arange(100, 102.1, 0.1)))

    # Scattering properties ______
    #bnp = np.array([0, 2300, 300, 80, 0.1])  # For g = 0.98
    #bnp = np.array([0,  2313.21860285,   308.42726122,    80.75973847, 0.1])
    # bnp = np.array([0, 2500, 2000, 100, 0.1])  # For g = 0.98
    bnp = np.array([0, 2277, 303, 79, 0.1])

    bnp[1:-1] = bnp[1:-1]  # Scale to respect the effective scattering coefficient
    b = matlab.double(bnp.tolist())  # Scattering coefficient matlab format
    phase_type = matlab.double([1, 1, 1, 1, 1])

    # Henyey-Greenstein First Moment, Assymmetry parameter
    g_all = 0.98
    g = eng.cell(1, 5)
    g[0], g[4] = matlab.double([0.]), matlab.double([0.9])
    g[1], g[2], g[3] = matlab.double([g_all]), matlab.double([g_all]), matlab.double([g_all])

    # Refractive index
    n = matlab.double([1.00, 1.00, 1.30, 1.30, 1.33])

    for ii in range(3):

        arg_wl_dort = np.where(eff_wl[ii] == radiometer_wl)
        rad_inc = irr_incom.mean(axis=1)[arg_wl_dort]/np.pi
        rad_inc_std = irr_incom.std(axis=1)[arg_wl_dort]/np.pi

        # DORT simulation paramters (absorption coefficient change according to wavelength)
        if ii == 0:
           a = matlab.double([0, 0.12, 0.12, 0.12, 0.12])  # absorption coefficient red
        elif ii == 1:
           a = matlab.double([0, 0.0683, 0.0683, 0.0683, 0.0683])  # absorption coefficient green
        else:
           a = matlab.double([0, 0.0430, 0.0430, 0.0430, 0.0430])  # absorption coefficient blue

        opt_thickness = np.squeeze((np.array(a) + np.array(b)) * np.squeeze(np.array(layer_thickness)))  # (a+b) * H
        reduced_b = np.squeeze(np.array(b)) * (1 - np.squeeze(np.array(g)))
        w0 = np.squeeze(np.array(b)) / (np.squeeze(np.array(a) + np.squeeze(np.array(b))))

        print("Optical thickness: {0:.2f}, {1:.2f}, {2:.2f}".format(opt_thickness[1], opt_thickness[2], opt_thickness[3]))
        print("Reduced scattering: {0:.2f}, {1:.2f}, {2:.2f}".format(reduced_b[1], reduced_b[2], reduced_b[3]))
        print("Single scattering albedo: {0:.9f}, {1:.9f}, {2:.9f}".format(w0[1], w0[2], w0[3]))

        # DORT simulation
        eng.cd(os.path.dirname(__file__))
        dort_p, dort_r = eng.dort_simulation_oden(matlab.double([nstreams]), number_layer,
                                                  layer_thickness, b, a, phase_type, g, n, depth,
                                                  matlab.double(list(rad_inc)), nargout=2)

        bounds = np.arange(0, 220, 20)
        rad_dist_dort, zen_mesh_dort, azi_mesh_dort = compute_results_dort(dort_p, dort_r, bounds)

        # Data in dict
        rad_dort_band[str(ii)] = rad_dist_dort

    # Loop
    # Pre-allocation
    fig1, ax1 = plt.subplots(2, 3, sharey=True)
    fig2, ax2 = plt.subplots(3, 1, sharey=True, figsize=ff.set_size(subplots=(2, 1)))
    fig3, ax3 = create_figure_axe_3x3((ff.set_size(subplots=(2, 1))[0], 5.74))

    depth_keys_order = ["20 cm (in water)",
                        "40 cm",
                        "60 cm",
                        "80 cm",
                        "100 cm",
                        "120 cm",
                        "140 cm",
                        "160 cm"]

    conversion_dict = dict(zip(["zero minus",
                        "20 cm (in water)",
                        "40 cm",
                        "60 cm",
                        "80 cm",
                        "100 cm",
                        "120 cm",
                        "140 cm",
                        "160 cm",
                        "180 cm",
                        "200 cm"], list(rad_dist_dort.keys())))
    # Irradiances
    Ed = np.zeros(len(depth_keys_order), dtype=([('0', 'f4'), ('1', 'f4'), ('2', 'f4')]))
    Eu = np.zeros(len(depth_keys_order), dtype=([('0', 'f4'), ('1', 'f4'), ('2', 'f4')]))
    Eo = np.zeros(len(depth_keys_order), dtype=([('0', 'f4'), ('1', 'f4'), ('2', 'f4')]))

    # Colormap
    colo = matplotlib.cm.get_cmap("cividis", len(depth_keys_order)+1)

    depth_color = np.array([int(i.split(" ")[0]) for i in depth_keys_order])

    colo_reds = build_cmap_2cond_color("Reds", depth_color)
    colo_greens = build_cmap_2cond_color("Greens", depth_color)
    colo_blues = build_cmap_2cond_color("Blues", depth_color)

    cm_it = iter(colo.colors)
    cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
    cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
    cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

    colorbardepth(fig3, ax3[0, 2], colo_reds, depth_color)
    colorbardepth(fig3, ax3[1, 2], colo_greens, depth_color)
    colorbardepth(fig3, ax3[2, 2], colo_blues, depth_color)

    # Other param
    mre_profile = np.empty((len(depth_keys_order), 3))

    for i, d_keys in enumerate(depth_keys_order):

        current_rad_map = rad_profile[d_keys]

        col = next(cm_it)
        col_r = next(cm_it_r)
        col_g = next(cm_it_g)
        col_b = next(cm_it_b)

        current_dort_rad_map = np.stack((rad_dort_band["0"][conversion_dict[d_keys]],
                                        rad_dort_band["1"][conversion_dict[d_keys]],
                                        rad_dort_band["2"][conversion_dict[d_keys]]), axis=2)

        Ed[i] = tuple(r.irradiance(zen_mesh_dort, azi_mesh_dort, current_dort_rad_map, 0, 90))
        Eu[i] = tuple(r.irradiance(zen_mesh_dort, azi_mesh_dort, current_dort_rad_map, 90, 180))
        Eo[i] = tuple(r.irradiance(zen_mesh_dort, azi_mesh_dort, current_dort_rad_map, 0, 180, planar=False))

        # Azimuthal average Measurements
        az_average = r.azimuthal_average(current_rad_map)

        # Azimuthal average DORT
        az_average_dort = r.azimuthal_average(current_dort_rad_map)

        for band in range(current_rad_map.shape[2]):

            az_average_dort_interp = np.interp(ze_mesh[:, 0], zen_mesh_dort[:, 0], az_average_dort[:, band])

            # Mean relative error (MUPD)
            ref_values = 0.5 * (az_average_dort_interp + az_average[:, band])
            rel_err = (az_average[:, band] - az_average_dort_interp) / ref_values
            mre_profile[i, band] = np.nanmean(rel_err)

            # Plot
            # Axe 1
            ax1[0, band].plot(ze_mesh[:, 0], az_average[:, band], color=col)

            ax1[0, band].set_yscale("log")
            ax1[0, band].set_xlim((20, 160))

            ax1[1, band].plot(ze_mesh[:, 0], az_average_dort_interp, color=col)
            ax1[1, band].set_yscale("log")
            ax1[1, band].set_xlim((20, 160))

            ax1[1, band].set_xlabel("Zenith angle [˚]")

            # Axe 2
            if i == 0:
                ax2[band].plot(ze_mesh[:, 0], az_average[:, band], color=col, label="Field data")
                ax2[band].plot(ze_mesh[:, 0], az_average_dort_interp, linestyle="-.", color=col, label="Simulations")
            else:
                ax2[band].plot(ze_mesh[:, 0], az_average[:, band], color=col)
                ax2[band].plot(ze_mesh[:, 0], az_average_dort_interp, linestyle="-.", color=col)

            ax2[band].set_yscale("log")
            ax2[band].set_xlim((20, 160))

            ax2[band].set_ylabel("Absolute radiance")

            # Axe 3
            if band == 0:
                curr_col = col_r
            elif band == 1:
                curr_col = col_g
            else:
                curr_col = col_b

            ax3[band, 0].plot(ze_mesh[:, 0], az_average[:, band], linestyle="-", color=curr_col)
            ax3[band, 1].plot(ze_mesh[:, 0], az_average_dort_interp, linestyle="-.", color=curr_col)
            ax3[band, 2].plot(ze_mesh[:, 0], rel_err * 100, color=curr_col)

            ax3[band, 0].set_yscale("log")
            ax3[band, 1].set_xlim((20, 160))
            ax3[band, 1].set_yscale("log")

            ax3[band, 0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax3[band, 1].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax3[band, 2].set_ylabel("relative error [%]")

    mupd = mre_profile.mean(axis=0) * 100
    print(mupd)
    print(mupd.mean())

    # Mean cosine
    # mu_r = utils.meancosine(Ed["0"], Eu["0"], Eo["0"])
    # mu_g = utils.meancosine(Ed["1"], Eu["1"], Eo["1"])
    # mu_b = utils.meancosine(Ed["2"], Eu["2"], Eo["2"])

    # Figures
    # Figure 1
    ax1[0, 0].set_ylabel("Absolute radiance")
    ax1[1, 0].set_ylabel("Absolute radiance")

    # Figure 2
    ax2[2].set_xlabel("Zenith angle [˚]")
    ax2[0].legend(loc="best")

    # Figure 3
    ax3[2, 0].set_xlabel("Zenith [˚]")
    ax3[2, 1].set_xlabel("Zenith [˚]")
    ax3[2, 2].set_xlabel("Zenith [˚]")

    ax3[0, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[0]), fontsize=6)
    ax3[1, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[1]), fontsize=6)
    ax3[2, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[2]), fontsize=6)

    # Figure 4
    fig4, ax4 = scattering_coeff_figure(b, layer_thickness, ff.set_size(subplots=(1, 1)))

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    # Save figure
    fig3.savefig("figures/dort_simulations.pdf", format="pdf", dpi=600)

    plt.show()
