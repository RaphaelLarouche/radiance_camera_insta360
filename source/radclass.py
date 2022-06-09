import h5py
import numpy as np
from scipy import integrate

def attenuation_coefficient(E, d):
    """
    Computation of attenuation coefficient of down-welling irradiance.

    :param E: irradiance in W m-2 nm-1 (array)
    :param d: depths in cm (array)
    :return: attenuation coefficient
    """
    return -np.gradient(E, d/100, edge_order=2) * (1/E)

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

        # Azimuthal integration
        azimuth_inte = integrate.simps(integrand.reshape((-1, azi_rad.shape[1])), azi_rad[mask].reshape((-1, azi_rad.shape[1])), axis=1)
        # Zenithal integration
        e = integrate.simps(azimuth_inte, zeni_rad[mask].reshape((-1, azi_rad.shape[1]))[:, 0], axis=0)

        irr = np.append(irr, e)

    return irr

class RadClass:

    def __init__(self, data_path="data/oden-08312018_v02.h5", station="station_1", data_type="camera", freeboard=20.0):
        """

        :param data_path:
        :param data_type:
        :param freeboard:
        """

        # Save attributes
        self.data_path = data_path
        self.data_type = data_type
        self.freeboard = freeboard  # cm
        self.station = station

        # Open data
        if "oden" in self.data_path:
            self.zenith_meshgrid, self.azimuth_meshgrid, self.radiance_profile = self.open_radiance_data()
        elif "baiedeschaleurs" in self.data_path:
            self.zenith_meshgrid, self.azimuth_meshgrid, self.radiance_profile = self.open_radiance_data(site="bdc")
            self.zenith_meshgrid *= 180 / np.pi
            self.azimuth_meshgrid *= 180 / np.pi
        else:
            self.zenith_meshgrid, self.azimuth_meshgrid, self.radiance_profile = self.open_radiance_data()

        # Order the depth keys
        self.ordered_keys, self.dct_depth = self.order_keys()
        self.keys_from_depth = dict((v, k) for k, v in self.dct_depth.items())

        # Smoothed data
        if self.data_type == "camera":
            self.legendre_coeff = self.fit_radiance_curves()
        elif self.data_type == "simulation":
            self.legendre_coeff = None
        else:
            raise ValueError("Wrong data_type variable value. Should be 'camera' or 'simulation'.")

        # Irradiance data
        self.ed, self.eu, self.eo, self.edo, self.euo = self.create_irradiance_data()

        # Absorption coefficient
        self.mu_a = self.calculate_mua()

        # Average cosines
        self.u_d, self.u_u, self.u = self.calculate_average_cosines()

    def open_radiance_data(self, site="oden"):
        """
        Function to open data stored in hdf5 file.

        :param path: relative or absolute path to file
        :return: (zenith meshgrid, azimuth meshgrid, radiance) (dct)
        """

        radiance_profile = {}
        with h5py.File(self.data_path) as hf:
            if site == "bdc":
                data = hf[self.station]
            else:
                data = hf
            for k in data.keys():
                if k not in ["azimuth", "zenith"]:
                    radiance_profile[k] = data[k][:]

            zenith_mesh = data["zenith"][:]
            azimuth_mesh = data["azimuth"][:]

        # print(radiance_profile.keys())

        return zenith_mesh, azimuth_mesh, radiance_profile

    def order_keys(self):
        """
        Method to order radiance profile keys.
        :return:
        """

        original_keys = self.radiance_profile.keys()
        de = np.array([])
        dct_depth = {}

        # Loop
        for i in original_keys:

            if i == "zero plus":
                depth = -0.00001
            elif i == "zero minus":
                depth = 0.0
            else:
                depth = float(i.split(" ")[0])  # get depth in cm

            de = np.append(de, depth)
            dct_depth[i] = depth

        aso = np.argsort(de)
        sorted_keys = np.array(list(original_keys))[aso]

        return sorted_keys, dct_depth

    def fit_radiance_curves(self):
        """

        :return:
        """

        dct_legendre_coeff = {}
        leg_deg = 5  # Degree of Legendre Polynomials
        # LOOP
        for i, ke in enumerate(self.ordered_keys):
            rad = self.radiance_profile[ke]
            radiance_az_average = self.azimuthal_average(rad)  # Azimuthal average
            zenith = self.zenith_meshgrid[:, 0].copy()

            coeff_array = np.zeros((leg_deg + 1, radiance_az_average.shape[1]))

            dep = self.dct_depth[ke]

            if dep >= self.freeboard:

                # Loop for each band
                for b in range(radiance_az_average.shape[1]):

                    # Get not NaN values
                    curr_radiance_az_avg = radiance_az_average[:, b]
                    mask_co = ~np.isnan(curr_radiance_az_avg)  # not NaN values
                    radiance_val = curr_radiance_az_avg[mask_co]
                    zenith_val = zenith[mask_co]

                    # Further mask for value over 25 degrees (because of camera drastic drop at the edges)
                    mask_deg = zenith_val >= (25.0 * np.pi / 180)
                    radiance_val = radiance_val[mask_deg]
                    zenith_val = zenith_val[mask_deg]

                    # Legendre fit
                    coeff_array[:, b] = self.legendre_fit(zenith_val * np.pi/180, radiance_val, leg_deg)

                # Save array
                dct_legendre_coeff[ke] = coeff_array
            else:
                dct_legendre_coeff[ke] = None

        return dct_legendre_coeff

    def create_irradiance_data(self):
        """
        Function that output irradiance data from radiance simulations using DORT2002.

        :param zenith_mesh:
        :param azimuth_mesh:
        :param radiance_mesh:
        :return:
        """
        ed = np.zeros(self.ordered_keys.shape[0], dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))
        eu, eo = ed.copy(), ed.copy()
        edo, euo = ed.copy(), ed.copy()

        # LOOP
        for i, ke in enumerate(self.ordered_keys):

            # print(ke)
            rad = self.radiance_profile[ke].copy()  # radiance angular disttribution

            dep = self.dct_depth[ke]  # current depth

            lc = self.legendre_coeff[ke]  # legendre polynomials

            if np.any(lc):
                for b in range(rad.shape[2]):
                    curr_radiance = rad[:, :, b]
                    cond_zero = curr_radiance == 0

                    curr_radiance[cond_zero] = self.compute_legendre_polynomials(self.zenith_meshgrid[cond_zero]
                                                                                 * np.pi/180,
                                                                                 lc[:, b])
                    rad[:, :, b] = curr_radiance

            ed[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 0, 90)) + (dep, )
            edo[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 0, 90, planar=False)) + (dep, )
            eu[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 90, 180)) + (dep, )
            euo[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 90, 180, planar=False)) + (dep, )
            eo[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 0, 180, planar=False)) + (dep, )

        return ed, eu, eo, edo, euo

    def calculate_mua(self):
        """
        Method to calculate the absorption coefficient [m-1] using the Gershun's law.
        :return:
        """

        mask_zero_z = np.where(self.ed["depth"] >= 0)

        net_irr_r = self.ed[mask_zero_z]["r"] - self.eu[mask_zero_z]["r"]
        net_irr_g = self.ed[mask_zero_z]["g"] - self.eu[mask_zero_z]["g"]
        net_irr_b = self.ed[mask_zero_z]["b"] - self.eu[mask_zero_z]["b"]

        mu_a = np.zeros(len(net_irr_r), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))

        mu_a["depth"] = self.ed["depth"][mask_zero_z]

        mu_a["r"] = attenuation_coefficient(net_irr_r, self.ed[mask_zero_z]["depth"]) * (net_irr_r / self.eo[mask_zero_z]["r"])
        mu_a["g"] = attenuation_coefficient(net_irr_g, self.ed[mask_zero_z]["depth"]) * (net_irr_g / self.eo[mask_zero_z]["g"])
        mu_a["b"] = attenuation_coefficient(net_irr_b, self.ed[mask_zero_z]["depth"]) * (net_irr_b / self.eo[mask_zero_z]["b"])

        return mu_a

    def calculate_average_cosines(self):
        """
        Method to calculate the average cosines.
        :return:
        """

        if "baiedeschaleurs" in self.data_path:
            mask_zero_z = np.where(self.ed["depth"] >= self.freeboard)
        else:
            mask_zero_z = np.where(self.ed["depth"] >= 0)

        mu_d = np.zeros(mask_zero_z[0].shape[0], dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))
        mu_u = mu_d.copy()
        mu = mu_d.copy()

        band_name = ["r", "g", "b"]
        for b in band_name:
            mu_d[b] = self.ed[mask_zero_z][b] / self.edo[mask_zero_z][b]
            mu_u[b] = self.eu[mask_zero_z][b] / self.euo[mask_zero_z][b]
            mu[b] = (self.ed[mask_zero_z][b] - self.eu[mask_zero_z][b]) / self.eo[mask_zero_z][b]

        mu_d["depth"] = self.ed["depth"][mask_zero_z]
        mu_u["depth"] = self.ed["depth"][mask_zero_z]
        mu["depth"] = self.ed["depth"][mask_zero_z]

        return mu_d, mu_u, mu

    def get_radiance_avg_at_depth_wl(self, depth, wl, smooth=False):
        """

        :param depth:
        :param wl:
        :return:
        """

        zen = self.zenith_meshgrid[:, 0]
        wl_dct = {484: 2, 544: 1, 603: 0}
        lc = self.legendre_coeff[self.keys_from_depth[depth]]
        if smooth and np.any(lc):
            radiance = self.compute_legendre_polynomials(zen * np.pi / 180, lc[:, wl_dct[wl]])
        else:
            rad_az_avg = self.azimuthal_average(self.radiance_profile[self.keys_from_depth[depth]])
            radiance = rad_az_avg[:, wl_dct[wl]]

        return zen, radiance

    def show_smoothed_radiance_curves(self, raw=True):
        """
        Method that shows smoothed radiance curves
        :param raw:
        :return:
        """

        if self.legendre_coeff:
            fig, ax = plt.subplots(1, 3, sharey=True, figsize=(6.4, 3.3))

            # Angular variables
            zen = np.arange(0, 181, 1)  # 1 deg angular resolution
            zen_rad = zen * np.pi / 180

            depth_color = self.dct_depth.values()
            colo_reds = self.build_cmap_2cond_color("Reds", depth_color)
            colo_greens = self.build_cmap_2cond_color("Greens", depth_color)
            colo_blues = self.build_cmap_2cond_color("Blues", depth_color)

            cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
            cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
            cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

            for ke in self.ordered_keys:

                lc = self.legendre_coeff[ke]

                # Color increment
                col_r = next(cm_it_r)
                col_g = next(cm_it_g)
                col_b = next(cm_it_b)

                for b in range(3):

                    if b == 0:
                        curr_col = col_r
                    elif b == 1:
                        curr_col = col_g
                    else:
                        curr_col = col_b

                    if raw:
                        rad_az_avg = self.azimuthal_average(self.radiance_profile[ke])
                        ax[b].plot(zen, rad_az_avg[:, b], color=curr_col, linestyle="-")

                    if np.any(lc):
                        radi_fit = self.compute_legendre_polynomials(zen_rad, lc[:, b])
                        ax[b].plot(zen, radi_fit, color=curr_col, linestyle="--", linewidth=0.9)

            ax[0].set_yscale("log")

            ax[0].set_xlabel("Zenith [˚]")
            ax[1].set_xlabel("Zenith [˚]")
            ax[2].set_xlabel("Zenith [˚]")
            ax[0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")

            fig.tight_layout()
        else:
            print("Smoothed radiance curves not calculated.")

    def show_irradiance_curves(self):
        """

        :param irradiance_dort:
        :param irradiance_meas:
        :return:
        """

        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(6.136, 3.784))

        band_name = ["r", "g", "b"]
        lstyle = ["-", "--", ":", "-."]

        for b, band in enumerate(band_name):

            ax[b].plot(self.ed[band], self.ed["depth"], linewidth=0.8, color="#a6cee3", linestyle=lstyle[0], label="$E_{d}$")
            ax[b].plot(self.eu[band], self.eu["depth"], linewidth=0.8, color="#1f78b4", linestyle=lstyle[0], label="$E_{u}$")
            ax[b].plot(self.eo[band], self.eo["depth"], linewidth=0.8, color="#b2df8a", linestyle=lstyle[0], label="$E_{0}$")

            ax[b].set_xscale("log")
            ax[b].invert_yaxis()

            ax[b].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
            ax[b].text(-0.05, 1.05, "(" + string.ascii_lowercase[b] + ")", transform=ax[b].transAxes, size=11,
                           weight='bold')
            ax[b].legend(loc="best", frameon=False, fontsize=6)

        ax[0].set_ylabel("Depth [cm]")
        fig.tight_layout()

        return fig, ax

    def show_mean_cosines(self):
        """
        Method to plot mean cosines AOPs.
        :return:
        """

        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(6.136, 3.784))

        band_name = ["r", "g", "b"]
        lstyle = ["-", "--", ":", "-."]
        cl = ["#a6cee3", "#1f78b4", "#b2df8a"]
        xlabel = ["$\mu_{d}$", "$\mu_{u}$", "$\mu$"]
        leg_lab = ["red band: 603 nm", "green band: 544 nm", "blue band: 484 nm"]

        if "oden" in self.data_path:
            mask_zero_z = np.where(self.ed["depth"] >= 0)
        else:
            mask_zero_z = np.where(self.ed["depth"] >= self.freeboard)

        for b, band in enumerate(band_name):

            mu_d = self.ed[mask_zero_z][band] / self.edo[mask_zero_z][band]
            mu_u = self.eu[mask_zero_z][band] / self.euo[mask_zero_z][band]
            mu = (self.ed[mask_zero_z][band] - self.eu[mask_zero_z][band]) / self.eo[mask_zero_z][band]

            ax[0].plot(mu_d, self.ed[mask_zero_z]["depth"], linewidth=0.8, color=cl[b], linestyle=lstyle[b], label=leg_lab[b])
            ax[1].plot(mu_u, self.eu[mask_zero_z]["depth"], linewidth=0.8, color=cl[b], linestyle=lstyle[b], label=leg_lab[b])
            ax[2].plot(mu, self.eo[mask_zero_z]["depth"], linewidth=0.8, color=cl[b], linestyle=lstyle[b], label=leg_lab[b])

            ax[b].set_xlabel(xlabel[b])

            ax[b].text(-0.05, 1.05, "(" + string.ascii_lowercase[b] + ")", transform=ax[b].transAxes, size=11, weight='bold')

        ax[0].legend(loc="best", frameon=False, fontsize=6)
        ax[1].legend(loc="best", frameon=False, fontsize=6)
        ax[2].legend(loc="best", frameon=False, fontsize=6)

        ax[0].invert_yaxis()

        ax[0].set_ylabel("Depth [cm]")
        fig.suptitle(self.station)
        fig.tight_layout()

        return fig, ax

    def show_absorption_coefficient(self):
        """

        :return:
        """

        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(6.136, 3.784))

        band_name = ["r", "g", "b"]
        lstyle = ["-", "--", ":", "-."]
        cl = ["#a6cee3", "#1f78b4", "#b2df8a"]
        xlabel = ["$E_{net}~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$",
                  "$k_{d}~\mathrm{[m^{-1}]}$",
                  "$a~\mathrm{[m^{-1}]}$"]
        leg_lab = ["red band: 630 nm", "green band: 544 nm", "blue band: 484 nm"]

        if "oden" in self.data_path:
            mask_zero_z = np.where(self.ed["depth"] >= 0)
        else:
            mask_zero_z = np.where(self.ed["depth"] >= self.freeboard)

        for b, band in enumerate(band_name):

            ednet = self.ed[mask_zero_z][band] - self.eu[mask_zero_z][band]
            kd = attenuation_coefficient(self.ed[mask_zero_z][band], self.ed[mask_zero_z]["depth"])

            ax[0].plot(ednet, self.ed[mask_zero_z]["depth"], linewidth=0.8, color=cl[b], linestyle=lstyle[b], label=leg_lab[b])
            ax[1].plot(kd, self.eu[mask_zero_z]["depth"], linewidth=0.8, color=cl[b], linestyle=lstyle[b], label=leg_lab[b])
            ax[2].plot(self.mu_a[mask_zero_z][band], self.mu_a[mask_zero_z]["depth"], linewidth=0.8, color=cl[b], linestyle=lstyle[b], label=leg_lab[b])

            ax[b].set_xlabel(xlabel[b])

            ax[b].text(-0.05, 1.05, "(" + string.ascii_lowercase[b] + ")", transform=ax[b].transAxes, size=11, weight='bold')

        ax[0].legend(loc="best", frameon=False, fontsize=6)
        ax[1].legend(loc="best", frameon=False, fontsize=6)
        ax[2].legend(loc="best", frameon=False, fontsize=6)

        ax[0].set_xscale("log")
        #ax[1].set_xscale("log")
        #ax[2].set_xscale("log")

        ax[0].invert_yaxis()

        ax[0].set_ylabel("Depth [cm]")
        fig.suptitle(self.station)
        fig.tight_layout()

        return fig, ax

    def save_radiance_curves_csv(self, path_filename="data/r-curves-oden-08312018.csv"):
        """
        Function that create a panda Dataframe with the radiance curves and save it to csv file.
        :return:
        """

        zenith = self.zenith_meshgrid[:, 0].copy()

        df_dc = pandas.DataFrame({"Zenith angle (°)": zenith})
        band_wl = {0: "603 nm", 1: "544 nm", 2: "484 nm"}

        str_pd = "Radiance {0}, {1} cm, {2} (W sr-1 m-2 nm-1)"

        for ke in self.ordered_keys:

            # Radiance raw
            radi_az_avg = self.azimuthal_average(self.radiance_profile[ke])

            # Radiance smoothed
            lc = self.legendre_coeff[ke]

            for b in [2, 1, 0]:

                radi_az_avg_raw = radi_az_avg[:, b]
                str_raw = str_pd.format("raw", self.dct_depth[ke], band_wl[b])

                if np.any(lc):
                    radi_az_avg_fit = self.compute_legendre_polynomials(zenith * np.pi/180, lc[:, b])
                    str_fit = str_pd.format("fit", self.dct_depth[ke], band_wl[b])
                    df_current = pandas.DataFrame({str_raw: radi_az_avg_raw,
                                                   str_fit: radi_az_avg_fit})
                else:
                    df_current = pandas.DataFrame({str_raw: radi_az_avg_raw})

                df_dc = pandas.concat([df_dc, df_current], axis=1)

        # Save to csv
        df_dc.to_csv(path_filename, sep=',')

        return df_dc

    @staticmethod
    def build_cmap_2cond_color(cmap_name, d):
        """

        :param cmap_name:
        :return:
        """

        CMA = matplotlib.cm.get_cmap(cmap_name, len(d) + 1)
        colooor = CMA(np.arange(1, CMA.N))
        custom_cmap = matplotlib.colors.ListedColormap(colooor[::-1])
        return custom_cmap

    @staticmethod
    def legendre_fit(theta, values, deg):
        """

        :param zenith: zenith angle (in radians)
        :param radiance_zenith: radiance as a function of zenith angle (W sr-1 m-2 nm-1)
        :return:
        """

        mu = np.cos(theta)  # cos(theta)
        leg_fit = np.polynomial.legendre.Legendre.fit(mu, values, deg, domain=[-1., 1.])
        return leg_fit.convert().coef

    @staticmethod
    def compute_legendre_polynomials(theta, coeff):
        """

        :param zenith:
        :param coeff:
        :return:
        """

        return np.polynomial.legendre.legval(np.cos(theta), coeff)

    @staticmethod
    def azimuthal_average(rad):
        """
        Average of radiance in azimuth direction.

        :return:
        """
        condzero = rad == 0
        rad2 = rad.copy()
        rad2[condzero] = np.nan
        return np.nanmean(rad2, axis=1)