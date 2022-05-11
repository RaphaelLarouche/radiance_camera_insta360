"""

"""

# Module importation
import h5py
import string
import pandas
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy import integrate
import matplotlib.cm
from scipy.optimize import curve_fit

from radiance import attenuation_coefficient
from processing import FigureFunctions

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer


import warnings
warnings.filterwarnings("ignore")



# Functions and classes
class OpenMatlabFiles:
    """

    """
    def loadmat(self, filename):
        """
        Loading fisheyeparams matlab.

        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        """
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self, matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            else:
                dict[strg] = elem
        return dict


def azimuthal_average(rad):
    """
    Average of radiance in azimuth direction.

    :return:
    """
    condzero = rad == 0
    rad2 = rad.copy()
    rad2[condzero] = np.nan
    return np.nanmean(rad2, axis=1)


def irradiance(zeni, azi, radm, zenimin, zenimax, planar=True):
    """
    Estimate irradiance from the radiance angular distribution. By default, it calculates the planar irradiance.
    By setting the parameter planar to false, the scalar irradiance is computed. Zenimin = 0˚ and Zenimax = 90˚ gives
    the downwelling irradiance, while Zenimin = 90° and Zenimax = 180˚ gives the upwelling irradiance.

    :param zeni: zenith meshgrid in degrees
    :param azi: azimuth meshgrid in degrees
    :param radm: radiance angular distribution
    :param zenimin: min zenith in degrees
    :param zenimax: max zenith in degrees
    :param planar: if True - planar radiance, if false - scalar (bool)
    :return:
    """

    mask = (zenimin <= zeni) & (zeni <= zenimax)
    irr = np.array([])
    zeni_rad = zeni * np.pi / 180
    azi_rad = azi * np.pi / 180
    for b in range(radm.shape[2]):

        # Integrand
        if planar:
            integrand = radm[:, :, b][mask] * np.absolute(np.cos(zeni_rad[mask])) * np.sin(zeni_rad[mask])
        else:
            integrand = radm[:, :, b][mask] * np.sin(zeni_rad[mask])

        azimuth_inte = integrate.simps(integrand.reshape((-1, azi_rad.shape[1])), azi_rad[mask].reshape((-1, azi_rad.shape[1])), axis=1)
        e = integrate.simps(azimuth_inte, zeni_rad[mask].reshape((-1, azi_rad.shape[1]))[:, 0], axis=0)

        irr = np.append(irr, e)

    return irr


def open_radiance_data(path="data/oden-08312018.h5"):
    """
    Function to open data stored in hdf5 file.

    :param path: relative or absolute path to file
    :return: (zenith meshgrid, azimuth meshgrid, radiance) (dct)
    """

    radiance_profile = {}
    with h5py.File(path) as hf:
        data = hf
        for k in data.keys():
            if k not in ["azimuth", "zenith"]:
                radiance_profile[k] = data[k][:]

        zenith_mesh = data["zenith"][:]
        azimuth_mesh = data["azimuth"][:]

    return zenith_mesh, azimuth_mesh, radiance_profile


def general_gaussian(x, a, b, c):
    """

    :param x: 
    :param a: 
    :param b: 
    :param c: 
    :param d: 
    :return: 
    """""
    return np.exp(-(x * a - b) ** 2) + c


def extrapolation(zenith_meshgrid, angular_radiance_distribution):
    """
    Extrapolation of missing angles (due reduced FOV due to water refractive index) using a gaussian function.

    :param zenith_meshgrid: zenith meshgrid in degrees (array)
    :param angular_radiance_distribution: current radiance angular distribution (array)
    :return: interpolated radiance angular distribution (array)
    """

    ard = angular_radiance_distribution.copy()  # Angular radiance distribution
    rad_zen = azimuthal_average(ard)  # Perform azimuthal average

    for b in range(rad_zen.shape[1]):

        # Condition for non-nan data
        co = ~np.isnan(rad_zen[:, b])

        # Normalization
        norm_val = np.mean(rad_zen[:, b][co][:5])  # 5 first values
        rad_zen_norm = rad_zen[:, b][co] / norm_val

        # Fit ()
        popt, pcov = curve_fit(general_gaussian, zenith_meshgrid[:, 0][co] * np.pi / 180, rad_zen_norm, p0=[-0.7, 0, 0.1])

        ard_c = ard[:, :, b].copy()

        ard_c[ard_c == 0] = general_gaussian(zenith_meshgrid[ard_c == 0] * np.pi / 180, *popt) * norm_val
        ard[:, :, b] = ard_c

    return ard


def create_irradiance_data(zenith_mesh, azimuth_mesh, radiance_mesh, keys_ordered, oden=True):
    """
    Function that output irradiance data from radiance simulations using DORT2002.
    :param zenith_mesh:
    :param azimuth_mesh:
    :param radiance_mesh:
    :return:
    """
    ed = np.zeros(len(keys_ordered), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))
    eu, eo = ed.copy(), ed.copy()

    # LOOP
    for i, ke in enumerate(keys_ordered):

        dort_rad = radiance_mesh[ke]

        if oden:
            dort_rad = extrapolation(zenith_mesh, radiance_mesh[ke])

        ed[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 0, 90))
        eu[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 90, 180))
        eo[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 0, 180, planar=False))

    return ed, eu, eo


def graph_dort_vs_measurements(depths, irradiances, irradiance_labels):
    """

    :param irradiance_dort:
    :param irradiance_meas:
    :return:
    """

    #i_d, i_u, i_o = irradiance_meas
    #i_d_dort, i_u_dort, i_o_dort = irradiance_dort

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(6.136, 3.784))

    band_name = ["r", "g", "b"]
    lstyle = ["-", "--", ":", "-."]

    n_irr = len(irradiances)
    txt_annotate = ""
    for n in range(n_irr):
        txt_annotate+="{0} : {1}\n".format(lstyle[n], irradiance_labels[n])

    for i, a in enumerate(irradiances):

        i_d, i_u, i_o = a

        for b in range(3):

            ax[b].plot(i_d[band_name[b]], depths, linewidth=0.8, color="#a6cee3", linestyle=lstyle[i], label="$E_{d}$")
            ax[b].plot(i_u[band_name[b]], depths, linewidth=0.8, color="#1f78b4", linestyle=lstyle[i], label="$E_{u}$")
            ax[b].plot(i_o[band_name[b]], depths, linewidth=0.8, color="#b2df8a", linestyle=lstyle[i], label="$E_{0}$")

            #ax[b].plot(i_d_dort[band_name[b]], depths, linewidth=0.8, color="#a6cee3", linestyle="--")
            #ax[b].plot(i_u_dort[band_name[b]], depths, linewidth=0.8, color="#1f78b4", linestyle="--")
            #ax[b].plot(i_o_dort[band_name[b]], depths, linewidth=0.8, color="#b2df8a", linestyle="--")

            if i == 0:
                ax[b].set_xscale("log")
                ax[b].invert_yaxis()

                ax[b].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
                ax[b].text(-0.05, 1.05, "(" + string.ascii_lowercase[b] + ")", transform=ax[b].transAxes, size=11, weight='bold')
                ax[b].legend(loc="best", frameon=False, fontsize=6)
                ax[b].annotate(txt_annotate, (0.04, 0.7), xycoords="axes fraction", fontsize=6)

    ax[0].set_ylabel("Depth [cm]")
    fig.tight_layout()

    return fig, ax


def azimuthal_average(mapped_radiance):
    """
    Average of radiance in azimuth direction.

    :return:
    """
    if len(mapped_radiance.shape) > 1:
        map_rad = mapped_radiance.copy()
        mask_zero = mapped_radiance == 0

        map_rad[mask_zero] = np.nan

        return np.nanmean(map_rad, axis=1)
    else:
        raise ValueError("Build radiance map before any integration.")


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
    cb.ax.set_title("depth [cm]", fontsize=8)
    cb.ax.invert_yaxis()

    return f, a


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

    print(radiometer)

    timestamps = pandas.to_datetime(radiometer["irrad_t_1"] - 719529, unit='D')   # 719529 = 1 january 2000 Matlab
    df_time = pandas.DataFrame({"Date": timestamps.to_period('H')})
    mask = (df_time["Date"] <= max_date) & (df_time["Date"] >= min_date)
    inx = df_time["Date"][mask].index

    timestamps = timestamps.strftime("%B %d %Y, %H:%M:%S")
    irradiance_incom = radiometer["incom_int_1"][inx].T / 1000  # mW to W
    irradiance_below = radiometer["irrad_int_1"][inx].T / 1000  # mW to W

    return timestamps, radiometer["wl"], irradiance_incom, irradiance_below


def build_cmap_2cond_color(cmap_name, d):
    """

    :param cmap_name:
    :return:
    """

    CMA = matplotlib.cm.get_cmap(cmap_name, len(d) + 1)
    colooor = CMA(np.arange(1, CMA.N))
    custom_cmap = matplotlib.colors.ListedColormap(colooor[::-1])
    return custom_cmap


def graph_cam_vs_simulations(radiance_camera, zenith_camera, radiance_simul, zenith_simul, cam_keys_order, simul_keys_order):

    # Figure creation
    fig_func = FigureFunctions()
    fig = plt.figure(figsize=(fig_func.set_size(subplots=(2, 1))[0], 5.74))

    a00 = fig.add_subplot(3, 3, 1)
    a01 = fig.add_subplot(3, 3, 2, sharex=a00, sharey=a00)
    a02 = fig.add_subplot(3, 3, 3, sharex=a00)

    a10 = fig.add_subplot(3, 3, 4, sharex=a00, sharey=a00)
    a11 = fig.add_subplot(3, 3, 5, sharex=a00, sharey=a00)
    a12 = fig.add_subplot(3, 3, 6, sharex=a00, sharey=a02)

    a20 = fig.add_subplot(3, 3, 7, sharex=a00, sharey=a00)
    a21 = fig.add_subplot(3, 3, 8, sharex=a00, sharey=a00)
    a22 = fig.add_subplot(3, 3, 9, sharex=a00, sharey=a02)

    ax = np.array([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])

    # Build colorbar
    depth_color = np.array([int(i.split(" ")[0]) for i in simul_keys_order])

    colo_reds = build_cmap_2cond_color("Reds", depth_color)
    colo_greens = build_cmap_2cond_color("Greens", depth_color)
    colo_blues = build_cmap_2cond_color("Blues", depth_color)

    cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
    cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
    cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

    colorbardepth(fig3, ax[0, 2], colo_reds, depth_color)
    colorbardepth(fig3, ax[1, 2], colo_greens, depth_color)
    colorbardepth(fig3, ax[2, 2], colo_blues, depth_color)

    # MUPD
    mre_profile = np.empty((len(simul_keys_order), 3))

    for i, (cam_k, simul_k) in enumerate(zip(cam_keys_order, simul_keys_order)):

        # Radiances
        rad_cam = radiance_camera[cam_k]
        rad_sim = radiance_simul[simul_k]

        # Color increment
        col_r = next(cm_it_r)
        col_g = next(cm_it_g)
        col_b = next(cm_it_b)

        # Azimuthal average
        rad_cam_avg = azimuthal_average(rad_cam)
        rad_sim_avg = azimuthal_average(rad_sim)

        for b in range(rad_cam.shape[2]):

            # Interpolation
            rad_sim_avg_interpo = np.interp(zenith_camera[:, 0], zenith_simul[:, 0], rad_sim_avg[:, b])

            # MUPD
            ref_values = 0.5 * (rad_sim_avg_interpo + rad_cam_avg[:, b])
            rel_err = (rad_sim_avg_interpo - rad_cam_avg[:, b]) / ref_values
            mre_profile[i, b] = np.nanmean(rel_err)

            #
            if b == 0:
                curr_col = col_r
            elif b == 1:
                curr_col = col_g
            else:
                curr_col = col_b

            ax[b, 0].plot(zenith_camera[:, 0], rad_cam_avg[:, b], linestyle="-", color=curr_col)
            ax[b, 1].plot(zenith_camera[:, 0], rad_sim_avg_interpo, linestyle="-.", color=curr_col)
            ax[b, 2].plot(zenith_camera[:, 0], rel_err * 100, color=curr_col)

            ax[b, 0].set_yscale("log")
            ax[b, 1].set_xlim((20, 160))
            ax[b, 1].set_yscale("log")

            ax[b, 0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax[b, 1].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax[b, 2].set_ylabel("relative error [%]")
            ax[b, 2].set_ylim((-40, 40))


    # Average mre
    mupd = np.abs(mre_profile).mean(axis=0) * 100

    ax[2, 0].set_xlabel("Zenith [˚]")
    ax[2, 1].set_xlabel("Zenith [˚]")
    ax[2, 2].set_xlabel("Zenith [˚]")

    ax[0, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[0]), fontsize=6)
    ax[1, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[1]), fontsize=6)
    ax[2, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[2]), fontsize=6)

    fig.tight_layout()

    return fig, ax


def graph_cam_vs_HE60_simulations(radiance_camera, zenith_camera, root_name, cam_keys_order, he60_depths):
    HE60_results = DataViewer(root_name=root_name)

    # Figure creation
    fig_func = FigureFunctions()
    fig = plt.figure(figsize=(fig_func.set_size(subplots=(2, 1))[0], 5.74))

    a00 = fig.add_subplot(3, 3, 1)
    a01 = fig.add_subplot(3, 3, 2, sharex=a00, sharey=a00)
    a02 = fig.add_subplot(3, 3, 3, sharex=a00)

    a10 = fig.add_subplot(3, 3, 4, sharex=a00, sharey=a00)
    a11 = fig.add_subplot(3, 3, 5, sharex=a00, sharey=a00)
    a12 = fig.add_subplot(3, 3, 6, sharex=a00, sharey=a02)

    a20 = fig.add_subplot(3, 3, 7, sharex=a00, sharey=a00)
    a21 = fig.add_subplot(3, 3, 8, sharex=a00, sharey=a00)
    a22 = fig.add_subplot(3, 3, 9, sharex=a00, sharey=a02)

    ax = np.array([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])

    # Build colorbar
    depth_color = np.array([int(i.split(" ")[0]) for i in cam_keys_order])

    colo_reds = build_cmap_2cond_color("Reds", depth_color)
    colo_greens = build_cmap_2cond_color("Greens", depth_color)
    colo_blues = build_cmap_2cond_color("Blues", depth_color)

    cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
    cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
    cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

    colorbardepth(fig, ax[0, 2], colo_reds, depth_color)
    colorbardepth(fig, ax[1, 2], colo_greens, depth_color)
    colorbardepth(fig, ax[2, 2], colo_blues, depth_color)

    # MUPD
    mre_profile = np.empty((len(cam_keys_order), 3))

    for i, (cam_k, he60_depth) in enumerate(zip(cam_keys_order, he60_depths)):

        # Radiances
        rad_cam = radiance_camera[cam_k]

        # Color increment
        col_r = next(cm_it_r)
        col_g = next(cm_it_g)
        col_b = next(cm_it_b)

        # Azimuthal average
        rad_cam_avg = azimuthal_average(rad_cam)
        wavelengths = [600, 540, 480]
        x_red, zenith_red = HE60_results.get_zenith_radiance_profile_at_depth(he60_depth, wavelengths[0])
        x_green, zenith_green = HE60_results.get_zenith_radiance_profile_at_depth(he60_depth, wavelengths[1])
        x_blue, zenith_blue = HE60_results.get_zenith_radiance_profile_at_depth(he60_depth, wavelengths[2])

        rad_sim_avg_interpo = np.stack((zenith_red, zenith_green, zenith_blue), axis=1)

        for b in range(rad_cam.shape[2]):

            # MUPD
            ref_values = 0.5 * (rad_sim_avg_interpo[:, b] + rad_cam_avg[:, b])
            rel_err = ( rad_sim_avg_interpo[:, b] - rad_cam_avg[:, b]) / ref_values
            mre_profile[i, b] = np.nanmean(rel_err)
            print("Comparison", rad_sim_avg_interpo[:, b], "\n" ,rad_cam_avg[:, b])

            #
            if b == 0:
                curr_col = col_r
            elif b == 1:
                curr_col = col_g
            else:
                curr_col = col_b

            ax[b, 0].plot(zenith_camera[:, 0], rad_cam_avg[:, b], linestyle="-", color=curr_col)
            ax[b, 1].plot(zenith_camera[:, 0], rad_sim_avg_interpo[:, b], linestyle="-.", color=curr_col)
            ax[b, 2].plot(zenith_camera[:, 0], rel_err * 100, color=curr_col)

            ax[b, 0].set_yscale("log")
            ax[b, 1].set_xlim((20, 160))
            ax[b, 1].set_yscale("log")

            ax[b, 0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax[b, 1].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax[b, 2].set_ylabel("relative error [%]")

    # Average mre
    mupd = np.abs(mre_profile).mean(axis=0) * 100

    ax[2, 0].set_xlabel("Zenith [˚]")
    ax[2, 1].set_xlabel("Zenith [˚]")
    ax[2, 2].set_xlabel("Zenith [˚]")

    ax[0, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[0]), fontsize=6)
    ax[1, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[1]), fontsize=6)
    ax[2, 2].text(25, np.round(np.max(mre_profile * 100)) + 8, "MUAPD = {0:.2f} %".format(mupd[2]), fontsize=6)

    fig.tight_layout()

    return fig, ax

def draw_radiance_figure(rootname):
    plt.style.use("../figurestyle.mplstyle")
    zen_oden, azi_oden, rad_oden = open_radiance_data(path="oden-08312018.h5")
    fig6, ax6 = graph_cam_vs_HE60_simulations(rad_oden, zen_oden, rootname, ["20 cm (in water)",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm"], [0.20,
                                                                                                  0.40,
                                                                                                  0.60,
                                                                                                  0.80,
                                                                                                  1.00,
                                                                                                  1.20,
                                                                                                  1.41,
                                                                                                  1.60])
    fig6.savefig(f"data/{rootname}/radiance_comparison.png", dpi=600)
    plt.show()
    return 0

if __name__ == "__main__":

    # Open TrioS data
    time_stamp, trios_wl, ed_trios_incom, ed_trios_below = open_TriOS_data(min_date="2018-08-31 8:00",
                                                                           max_date="2018-08-31 16:00",
                                                                           path_trios="2018R4_300025060010720.mat")

    ed_trios_incom[np.where(ed_trios_incom == 0)] = np.nan

    # Open measurements
    zen_oden, azi_oden, rad_oden = open_radiance_data(path="oden-08312018.h5")  # Path à changer
    ed_oden, eu_oden, eo_oden = create_irradiance_data(zen_oden, azi_oden, rad_oden, ["zero minus",
                                                                                      "20 cm (in water)",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm",
                                                                                      "180 cm",
                                                                                      "200 cm"], oden=True)

    # Open dort simulation
    zen_dort, azi_dort, rad_dort = open_radiance_data(path="dort-simulation.h5")  # Path à changer
    ed_dort, eu_dort, eo_dort = create_irradiance_data(zen_dort, azi_dort, rad_dort, ["0 cm",
                                                                                      "20 cm",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm",
                                                                                      "180 cm",
                                                                                      "200 cm"])

    z = np.arange(0, 220, 20)
    fig1, ax1 = graph_dort_vs_measurements(z, ((ed_dort, eu_dort, eo_dort), (ed_oden, eu_oden, eo_oden)),
                                           ("measurements", "simulations"))

    # Gershun law estimation of absorption coefficient
    absorption = np.zeros(len(z), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))

    absorption["r"] = attenuation_coefficient((ed_dort["r"] - eu_dort["r"]), z) * ((ed_dort["r"] - eu_dort["r"]) / ed_dort["r"])
    absorption["g"] = attenuation_coefficient((ed_dort["g"] - eu_dort["g"]), z) * ((ed_dort["g"] - eu_dort["g"]) / ed_dort["g"])
    absorption["b"] = attenuation_coefficient((ed_dort["b"] - eu_dort["b"]), z) * ((ed_dort["b"] - eu_dort["b"]) / ed_dort["b"])

    # Transmittance
    tr_trios = ed_trios_below / ed_trios_incom
    tr_cam = np.array([ed_oden[-1]["r"]/ed_oden[0]["r"], ed_oden[-1]["g"]/ed_oden[0]["g"], ed_oden[-1]["b"]/ed_oden[0]["b"]])
    tr_dort = np.array([ed_dort[-1]["r"]/ed_dort[0]["r"], ed_dort[-1]["g"]/ed_dort[0]["g"], ed_dort[-1]["b"]/ed_dort[0]["b"]])

    # Figure 2 - absorption coefficient
    plt.style.use("../figurestyle.mplstyle")
    ff = FigureFunctions()
    fig2, ax2 = plt.subplots(1, 1)

    ax2.plot(absorption["r"], z)
    ax2.plot(absorption["g"], z)
    ax2.plot(absorption["b"], z)

    ax2.invert_yaxis()

    # Figure 3
    # fig3, ax3 = graph_dort_vs_measurements(z, ((ed_dort, eu_dort, eo_dort), (ed_dort_2, eu_dort_2, eo_dort_2)),
    #                                        ("DORT n=1.00", "DORT n=1.355"))
    fig3, ax3 = graph_dort_vs_measurements(z, ((ed_dort, eu_dort, eo_dort), (ed_dort, eu_dort, eo_dort)),
                                           ("DORT n=1.00", "DORT n=1.00"))

    # Figure 4 - transmittance figure
    # fig4, ax4 = plt.subplots(1, 3, figsize=(6.4 , 2.976))
    fig4, ax4 = plt.subplots(3, 1, sharex=True, figsize=ff.set_size(height_ratio=0.9))

    # Incoming irradiance
    ax4[0].plot(trios_wl[:-100], ed_trios_incom[:-100, 0], label="TriOS")
    ax4[0].plot(trios_wl[:-100], ed_trios_incom[:-100, 1:])

    ax4[0].plot(np.array([603, 544, 484]), np.array([ed_oden[0]["r"], ed_oden[0]["g"], ed_oden[0]["b"]]), label="Cam", marker="o", markersize=4,
             markerfacecolor="none", markeredgecolor="k", linestyle="none")
    ax4[0].plot(np.array([603, 544, 484]), np.array([ed_dort[0]["r"], ed_dort[0]["g"], ed_dort[0]["b"]]),  label="DORT", marker="s", markersize=4,
             markerfacecolor="none", markeredgecolor="red", linestyle="none")

    ax4[0].set_title("Incoming irradiance", fontsize=9)
    #ax4[0].set_xlabel("Wavelength [nm]")
    ax4[0].set_ylabel("$E_{d}~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax4[0].legend(loc="best", fontsize=7)

    # Irradiance below

    ax4[1].plot(trios_wl[:-100], ed_trios_below[:-100, 0], label="TriOS")
    ax4[1].plot(trios_wl[:-100], ed_trios_below[:-100, 1:])

    ax4[1].plot(np.array([603, 544, 484]), np.array([ed_oden[-1]["r"], ed_oden[-1]["g"], ed_oden[-1]["b"]]), label="Cam", marker="o", markersize=4,
             markerfacecolor="none", markeredgecolor="k", linestyle="none")
    ax4[1].plot(np.array([603, 544, 484]), np.array([ed_dort[-1]["r"], ed_dort[-1]["g"], ed_dort[-1]["b"]]),  label="DORT", marker="s", markersize=4,
             markerfacecolor="none", markeredgecolor="red", linestyle="none")

    ax4[1].set_title("Irradiance below ice",  fontsize=9)
    #ax4[1].set_xlabel("Wavelength [nm]")
    ax4[1].set_ylabel("$E_{d}~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax4[1].legend(loc="best", fontsize=7)

    # Transmittance
    ax4[2].plot(trios_wl[:-100], tr_trios[:-100, 0], label="TriOS")
    ax4[2].plot(trios_wl[:-100], tr_trios[:-100, 1:])

    ax4[2].plot(np.array([603, 544, 484]), tr_cam, label="Cam", marker="o", markersize=4,
             markerfacecolor="none", markeredgecolor="k", linestyle="none")
    ax4[2].plot(np.array([603, 544, 484]), tr_dort, label="DORT", marker="s", markersize=4,
             markerfacecolor="none", markeredgecolor="red", linestyle="none")

    ax4[2].set_xlabel("Wavelength [nm]")
    ax4[2].set_ylabel("Transmittance [a.u.]")
    ax4[2].legend(loc="best", fontsize=7)

    fig4.tight_layout()

    # Figure 5
    fig5, ax5 = graph_cam_vs_simulations(rad_oden, zen_oden, rad_dort, zen_dort, ["20 cm (in water)",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm"], ["20 cm",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm"])


    fig6, ax6 = graph_cam_vs_HE60_simulations(rad_oden, zen_oden, 'he60_comp_dort_brine', ["20 cm (in water)",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm"], [0.20,
                                                                                                  0.40,
                                                                                                  0.60,
                                                                                                  0.80,
                                                                                                  1.00,
                                                                                                  1.20,
                                                                                                  1.41,
                                                                                                  1.60])

    plt.show()
