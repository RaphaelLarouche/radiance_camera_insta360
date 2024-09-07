# -*- coding: utf-8 -*-
"""
Super class to analyze Insta360 X3 radiance data

Coded by Raphaël Larouche
"""

import string
import pandas
import matplotlib
import numpy as np
from scipy import integrate
from openpyxl import load_workbook
import matplotlib.pyplot as plt


# Function and classes
class RadClassX3:

    def __init__(self, data_path="data/QI0310_radiance.xlsx", wl_dct=None):
        """
        Class for radiance analysis for QikIce2023 mission.
        """

        # Save attributes
        if wl_dct is None:
            self.wl_dct = {480: 2, 540: 1, 600: 0} # not good wavelength... to be changed

        self.data_path = data_path

        # Radiance data
        self.rad_data = pandas.read_excel(self.data_path, sheet_name=None)
        self.depths = np.unique(self.rad_data["Radiance green"]["Depth"].to_numpy())

        #self.freeboard = freeboard  # cm
        #self.station = station
        #self.wl_dct = wl_dct

        self.zenith_meshgrid, self.azimuth_meshgrid, self.radiance_profile = self.format_radiance_data()

        # Legendre extrapolation
        self.legendre_coeff = self.fit_radiance_curves(leg_deg=7)

        # Irradiance data
        self.ed, self.eu, self.eo, self.edo, self.euo = self.create_irradiance_data()

        # Absorption coefficient
        self.mu_a = self.calculate_mua()

        # Diffuse attenuation coefficient
        self.K_d = self.calculate_Kd()

        # Average cosines
        self.u_d, self.u_u, self.u = self.calculate_average_cosines()

    def format_radiance_data(self):
        """

        :return:
        :rtype:
        """
        data_gen = self.rad_data["Radiance green"][self.rad_data["Radiance green"]["Depth"] == self.depths[0]].copy()
        azimuth_mesh, zenith_mesh = np.meshgrid(data_gen.columns[9:].to_numpy().astype(float),
                                                data_gen["Zenith"].to_numpy().astype(float))

        radiance_profile = {}
        band_orders = ["Radiance red", "Radiance green", "Radiance blue"]
        for i, d in enumerate(self.depths):
            rad_i = np.zeros((azimuth_mesh.shape[0], azimuth_mesh.shape[1], 3))
            for b, rkeys in enumerate(band_orders):
                df_c = self.rad_data[rkeys]
                rad_c = df_c[df_c["Depth"] == d].loc[:, 0:360].values
                mask_n = np.isnan(rad_c)
                rad_c[mask_n] = 0.0
                rad_i[:, :, b] = rad_c.copy()

            radiance_profile[d] = rad_i

        return zenith_mesh, azimuth_mesh, radiance_profile

    def fit_radiance_curves(self, leg_deg=5):
        """
        Method that fits radiance curves using Legendre Polynomials of degree 5.

        :return: dictionary of legendre polynomials array for each depths
        :rtype: dct
        """

        dct_legendre_coeff = {}
        # LOOP
        for i, ke in enumerate(self.depths):

            # Radiance at depth
            rad = self.radiance_profile[ke]

            # Cond for extrapolation
            if np.sum(rad == 0.0) >= 1000:

                radiance_az_average = self.azimuthal_average(rad)  # Azimuthal average
                zenith = self.zenith_meshgrid[:, 0].copy()

                coeff_array = np.zeros((leg_deg + 1, radiance_az_average.shape[1]))

                # Loop for each band
                for b in range(radiance_az_average.shape[1]):

                    # Get not NaN values
                    curr_radiance_az_avg = radiance_az_average[:, b]
                    mask_co = ~np.isnan(curr_radiance_az_avg)  # not NaN values
                    radiance_val = curr_radiance_az_avg[mask_co]
                    zenith_val = zenith[mask_co]

                    # Further mask for value over 25 degrees (because of camera drastic drop at the edges)
                    mask_deg = zenith_val >= 25.0
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
        ed = np.zeros(self.depths.shape[0], dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))
        eu, eo = ed.copy(), ed.copy()
        edo, euo = ed.copy(), ed.copy()

        # LOOP
        for i, ke in enumerate(self.depths):

            print(ke)
            rad = self.radiance_profile[ke].copy()  # radiance angular disttribution

            lc = self.legendre_coeff[ke]  # legendre polynomials

            if np.any(lc):
                for b in range(rad.shape[2]):
                    curr_radiance = rad[:, :, b]
                    cond_zero = curr_radiance == 0

                    curr_radiance[cond_zero] = self.compute_legendre_polynomials(self.zenith_meshgrid[cond_zero]
                                                                                 * np.pi/180,
                                                                                 lc[:, b])
                    rad[:, :, b] = curr_radiance

            ed[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 0, 90)) + (ke, )
            edo[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 0, 90, planar=False)) + (ke, )
            eu[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 90, 180)) + (ke, )
            euo[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 90, 180, planar=False)) + (ke, )
            eo[i] = tuple(irradiance(self.zenith_meshgrid, self.azimuth_meshgrid, rad, 0, 180, planar=False)) + (ke, )

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

    def calculate_Kd(self):
        """
        Method to calculate the diffuse attenuation coefficient [m-1].
        :return:
        """

        mask_zero_z = np.where(self.ed["depth"] >= 0)

        Kd = np.zeros(mask_zero_z[0].shape[0], dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))

        Kd["depth"] = self.ed["depth"][mask_zero_z]

        Kd["r"] = attenuation_coefficient(self.ed[mask_zero_z]["r"], self.ed[mask_zero_z]["depth"])
        Kd["g"] = attenuation_coefficient(self.ed[mask_zero_z]["g"], self.ed[mask_zero_z]["depth"])
        Kd["b"] = attenuation_coefficient(self.ed[mask_zero_z]["b"], self.ed[mask_zero_z]["depth"])

        return Kd

    def calculate_average_cosines(self):
        """
        Method to calculate the average cosines.
        :return:
        """

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
        lc = self.legendre_coeff[depth]
        if smooth and np.any(lc):
            radiance = self.compute_legendre_polynomials(zen * np.pi / 180, lc[:, self.wl_dct[wl]])
        else:
            rad_az_avg = self.azimuthal_average(self.radiance_profile[depth])
            radiance = rad_az_avg[:, self.wl_dct[wl]]

        return zen, radiance

    def get_radiance_dist_at_depth_wl(self, depth, wl):
        """
        Extract radiance distribution at certain depth and wavelength

        :param depth: depth [cm]
        :type depth: float
        :param wl: wavelength [nm]
        :type wl: float
        :return: rgb 3d array of angular radiance distribution
        :rtype: narray
        """
        rad_depth = self.radiance_profile[depth].copy()
        return rad_depth[:, :, self.wl_dct[wl]]

    def get_aops_average(self, min_depth:float, max_depth:float, aops_key="Kd", verbose=True):
        """
        Method to average an apparent optical property between two depth values.
        :param min_depth: min depth value - float
        :param max_depth: max depth value - float
        :param aops_key:
        :return:
        """
        if aops_key == "Kd":
            aops = self.K_d.copy()
        elif aops_key == "uu":
            aops = self.u_u.copy()
        elif aops_key == "ud":
            aops = self.u_d.copy()
        else:
            raise ValueError("Invalid name for AOPs.")

        depths = aops["depth"]  # depths array
        mask_depths = (min_depth <= depths) & (depths <= max_depth)

        avg_r = aops["r"][mask_depths].mean()
        avg_g = aops["g"][mask_depths].mean()
        avg_b = aops["b"][mask_depths].mean()

        if verbose:
            print(f"AOP - {aops_key} - {min_depth} cm <= depths <= {max_depth} cm"
                  f"\nAverage blue = {avg_b:.5f}"
                  f"\nAverage green = {avg_g:.5f}"
                  f"\nAverage red = {avg_r:.5f}")
        else:
            return (avg_r, avg_g, avg_b)

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

            depth_color = self.depths.copy()
            colo_reds = self.build_cmap_2cond_color("Reds", depth_color)
            colo_greens = self.build_cmap_2cond_color("Greens", depth_color)
            colo_blues = self.build_cmap_2cond_color("Blues", depth_color)

            cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
            cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
            cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

            for ke in self.depths:

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

       # if "oden" in self.data_path:
        mask_zero_z = np.where(self.ed["depth"] >= 0)
        #else:
        #mask_zero_z = np.where(self.ed["depth"] >= self.freeboard)

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
        #fig.suptitle(self.station)
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

        #if "oden" in self.data_path:
        mask_zero_z = np.where(self.ed["depth"] >= 0)
        #else:
            #mask_zero_z = np.where(self.ed["depth"] >= self.freeboard)

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

        ax[0].invert_yaxis()

        ax[0].set_ylabel("Depth [cm]")
        #fig.suptitle(self.station)
        fig.tight_layout()

        return fig, ax

    def show_polar_plot(self, which_depth, ncontour, fig_ax=(None, None), wl_order=None):
        """
        Function that plot contour filled figure.
        :param ncontour: number of levels
        :return: (figure, axe) - tuple
        """

        mappedradiance = self.radiance_profile[which_depth].copy()

        depth_label = f"{which_depth:.1f} cm"

        if fig_ax == (None, None):
            fig, ax = plt.subplots(1, 3, figsize=(6.6929, 6.6929 * 0.7), subplot_kw={'projection': 'polar'})
        else:
            fig, ax = fig_ax

        if wl_order is None:
            wl_order = ["red", "green", "blue"]
        if len(mappedradiance.shape) == 3:
            # wl_lab = {"red": "603 nm", "green": "544 nm", "blue": "484 nm"}
            wl_lab = {"red": "600 nm", "green": "540 nm", "blue": "480 nm"}
            wl_corr = {"red": 0, "green": 1, "blue": 2}

            for n, b in enumerate(wl_order):
                zeni = self.zenith_meshgrid.copy()
                azi = self.azimuth_meshgrid.copy() * np.pi/180

                im = mappedradiance[:, :, wl_corr[b]].copy()
                insideFOV = np.where(mappedradiance > 0)
                mini, maxi = np.nanmin(mappedradiance[insideFOV]), np.nanmax(mappedradiance[insideFOV])

                cax = ax[n].contourf(azi, zeni, np.clip(im, 0, None), np.linspace(mini, maxi, ncontour), cmap="coolwarm")

                # Tick parameters
                ax[n].tick_params(axis='x', which='major', labelsize=7.5, pad=-3)
                # ytick
                ytik = np.arange(0, 200, 40)
                ax[n].set_yticks(ytik)
                ax[n].set_yticklabels(["{}˚".format(i) for i in ytik], fontsize=7)

                # Ticks azimuth
                ax[n].grid(linestyle="-.")
                cl = fig.colorbar(cax, ax=ax[n], orientation="horizontal", format='%.1e', pad=0.19)
                cl.ax.set_title("$L_{{{0}}}({1})$ [$\mathrm{{W \cdot sr^{{-1}} \cdot m^{{-2}} \cdot nm^{{-1}}}}$]".format(wl_lab[b], depth_label), fontsize=6.5)
                ls_xtick = cl.ax.get_xticks()
                cl.ax.set_xticks(ls_xtick)
                cl.ax.set_xticklabels([format(xt, '.2e') for xt in ls_xtick], rotation=20, fontsize=7)

                # Letter at corner of image
                ax[n].text(-0.1, 1.1, "(" + string.ascii_lowercase[n] + ")", transform=ax[n].transAxes, size=10, weight='bold')
            fig.tight_layout()
            return fig, ax
        else:
            raise ValueError("Radiance map should be build before.")

    def load_excel_columns(self, sheet_name, columns):
        # Load the Excel workbook
        workbook = load_workbook(filename=self.data_path, read_only=True)

        # Select the specified sheet
        sheet = workbook[sheet_name]

        # Get the values from the selected columns
        selected_columns = []
        all_data = np.array([])
        for column in columns:
            data = []
            for ro in sheet.rows:
                data.append(ro[column].value)

            all_data = np.column_stack((all_data, data))

        #for column in columns:
            #column_values = [cell.value for cell in sheet[column]]
            #selected_columns.append(column_values)

        # Stack the selected columns horizontally to create a new array
        #selected_data = np.column_stack(selected_columns)

        return all_data

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
    def azimuthal_average(rad):
        """
        Average of radiance in azimuth direction.

        :return:
        """
        condzero = rad == 0
        rad2 = rad.copy()
        rad2[condzero] = np.nan
        return np.nanmean(rad2, axis=1)


# Other function accessible
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


def attenuation_coefficient(E, d):
    """
    Computation of attenuation coefficient of down-welling irradiance.

    :param E: irradiance in W m-2 nm-1 (array)
    :param d: depths in cm (array)
    :return: attenuation coefficient
    """
    return -np.gradient(E, d/100, edge_order=2) * (1/E)


if __name__ == "__main__":

    station = 5
    site = 3
    num = 0
    norm_name = f"QI{station:02}{site:1}{num:1}"

    path_to_data = f"/Volumes/MYBOOK/QikIce2023/Qik2023/QI{station:02}/{norm_name}_radiance_raw/processed_data/{norm_name}_radiance.xlsx"

    #path_to_data = "/Users/raphaellarouche/Desktop/QI0320_radiance.xlsx"
    rdx3 = RadClassX3(data_path=path_to_data)

    ncontour = 20
    rdx3.show_irradiance_curves()
    rdx3.show_mean_cosines()
    rdx3.show_absorption_coefficient()
    rdx3.show_polar_plot(15, ncontour)

    plt.show()
