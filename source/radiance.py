# -*- coding: utf-8 -*-
"""
Principal file containing the ImageRadiancei360 class to transform raw insta360 One digital values
(red, green, blue bands) into radiance angular distribution.

"""

# Module importation
import os
import glob
import h5py
import string
import deepdish
import numpy as np
from scipy import integrate
import scipy.interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Other module
from source.processing import ProcessImage
from source.geometric_rolloff import MatlabGeometricMengine


# Classes
class ImageRadiancei360(ProcessImage):
    """
    Class to build radiance map from Insta360 ONE images
    """

    def __init__(self, image_path, medium):

        # Calibration files
        self.base_path = os.path.dirname(os.path.dirname(__file__))

        # FoV
        self.medium = medium.lower()
        self.fov = self.define_field_of_view()

        # Geometric calibration
        self.geometric_close, self.geometric_far = self.open_geometric_calibration()

        # Absolute radiance coefficients
        self.cl_close = self.open_calibrations("lens-close/20200909/cal-coefficients", calibration="absolute")
        self.cl_far = self.open_calibrations("lens-far/20200909/cal-coefficients", calibration="absolute")

        # Immersion factor
        self.ifactor_close = self.open_calibrations("lens-close/20200911/immersion", calibration="immersion")

        # Roll-off
        # self.rolloff_close = self.open_rolloff_calibration("lens-close/20190501/fit-coefficients")
        # self.rolloff_far = self.open_rolloff_calibration("lens-close/20190501/fit-coefficients")  # !!!!!
        self.rolloff_close = self.open_rolloff_calibration()
        self.rolloff_far = self.open_rolloff_calibration()

        # Attributes
        self.im_original, self.metadata = self._readDNG_np(image_path)  # From ProcessImage class
        self.im = self.im_original.copy().astype(float)

        self.rad_c, self.zen_c, self.az_c = self.get_band_angular_coord(self.geometric_close)
        self.rad_f, self.zen_f, self.az_f = self.get_band_angular_coord(self.geometric_far)

        # Radiance map (attributes to be defined later)
        self.zenith_mesh = np.array([])
        self.azimuth_mesh = np.array([])
        self.mappedradiance = np.array([])
        self.mapped_radiance_4pi = np.array([])

    def open_geometric_calibration(self):
        """
        Function that opens geometric calibration.
        :return: tuple (geo_close, geo_far)
        """
        p = self.base_path + "/calibrations/geometric-calibration/calibrationfiles/"

        if self.medium.lower() == "air":
            geocalib_close = deepdish.io.load(p + "geometric-calibration-air.h5", "/lens-close/20190104_192404/")
            geocalib_far = deepdish.io.load(p + "geometric-calibration-air.h5", "/lens-far/20190104_214037/")

        elif self.medium.lower() == "water":
            geocalib_close = deepdish.io.load(p + "geometric-calibration-water.h5", "/lens-close/20200730_112353/")
            geocalib_far = deepdish.io.load(p + "geometric-calibration-water.h5", "/lens-far/20200730_143716/")
        else:
            raise ValueError("Invalid name for medium. Should be 'air' or 'water'.")

        # Building dictionary
        geometric_close = {}
        geometric_far = {}
        for k in geocalib_close["fp"].keys():
            geometric_close[k] = MatlabGeometricMengine(geocalib_close["fp"][k], geocalib_close["ierror"][k])
            geometric_far[k] = MatlabGeometricMengine(geocalib_far["fp"][k], geocalib_far["ierror"][k])

        return geometric_close, geometric_far

    def open_rolloff_calibration(self):
        """
        Open roll-off calibration stored inside hdf5 file format.
        :return:
        """

        if self.medium.lower() == "air":
            path_tf = self.base_path + "/calibrations/roll-off/calibrationfiles/rolloff_a.h5"
            tag = "lens-close/20170102/fit-coefficients"
        elif self.medium.lower() == "water":
            path_tf = self.base_path + "/calibrations/roll-off/calibrationfiles/rolloff_w.h5"
            tag = "lens-close/20190501/fit-coefficients"
        else:
            raise ValueError("Invalid name for medium. Should be 'air' or 'water'.")

        with h5py.File(path_tf) as hfrel:
            cal = hfrel[tag][:]
        return cal

    def open_calibrations(self, tag, calibration="absolute"):
        """
        Open all other calibration. So absolute radiance calibration, immersion factor and roll-off.
        :return:
        """

        if calibration.lower() == "absolute":
            ptf = self.base_path + "/calibrations/absolute-spectral-radiance/calibrationfiles/absolute_radiance.h5"
        elif calibration.lower() == "immersion":
            ptf = self.base_path + "/calibrations/immersion-factor/calibrationfiles/immersion_factor.h5"
        else:
            raise ValueError("Invalid entry calibration. Only value permitted are 'absolute' or 'immersion'.")
        with h5py.File(ptf) as hfrel:
            cal = hfrel[tag][:]
        return cal

    def get_radiance(self, dark_metadata=True):
        """
        Processing steps to transform raw image in spectral radiance image.

        :param dark_metadata:
        :return:
        """

        # Downsampling
        self.im = self.dwnsampling(self.im, "RGGB")  # From ProcessImage class

        # Dark correction
        if dark_metadata:
            self.dark_correction()
        else:
            self.dark_correction_image_plane()

        # Normalization by integration time and gain
        self.normalisation()

        # Roll-off
        self.rolloff_correction()

        # Absolute Coefficient
        self.apply_absolute_radiance_calibration()

        # Immersion factor
        if self.medium == "water":
            self.immersion_correction()

    def map_radiance(self, angular_resolution=1.0):
        """

        :param angular_resolution:
        :return:
        """

        if len(self.im.shape) == 3:

            azi, zen = self.angle_grid(angular_resolution)

            # Reference coordinate rotation
            px, py, pz = self.points_3d(zen, azi)

            # Rotation for lens far (no rotation for lens close)
            rotation_far = self.rz(np.pi)
            npx_f, npy_f, npz_f = self.rotation(px, py, pz, rotation_far)

            theta_c, phi_c = np.arccos(py), np.arctan2(pz, px)  # angular coordinates (zenith, azimuth) lens close
            theta_f, phi_f = np.arccos(npy_f), np.arctan2(npz_f, -npx_f)  # ... lens far

            cond_c = theta_c < self.fov * np.pi / 180
            cond_f = theta_f < self.fov * np.pi / 180

            # Dewarping
            dewarp = np.zeros((theta_c.shape[0], theta_c.shape[1], 3))

            im_c = self.getimage("close")
            im_f = self.getimage("far")

            for b, k in enumerate(self.geometric_close.keys()):

                de = dewarp[:, :, b].copy()

                de[cond_c] = self.dewarpband(im_c[:, :, b], theta_c[cond_c], phi_c[cond_c], self.rad_c[:, :, b], self.zen_c[:, :, b], self.geometric_close[k])
                de[cond_f] = self.dewarpband(im_f[:, :, b], theta_f[cond_f], phi_f[cond_f], self.rad_f[:, :, b], self.zen_f[:, :, b], self.geometric_far[k])

                dewarp[:, :, b] = de

            self.zenith_mesh = zen
            self.azimuth_mesh = azi
            self.mappedradiance = dewarp

            return self.zenith_mesh, self.azimuth_mesh, self.mappedradiance

        else:
            raise ValueError("Downsampling or demosaic must be done before!")

    def dark_correction(self):
        """
        Method to remove dark noise using the stored value in the tif metadata.
        :return:
        """

        self.im -= float(str(self.metadata["Image BlackLevel"]))
        return self.im

    def dark_correction_image_plane(self):
        """

        :return:
        """

        if self.im.shape[2] == 3:
            ima = self.im_original.copy().astype(float)
            height = ima.shape[0]
            half_height = int(height // 2)

            im_c = ima[half_height:height:1, :]
            im_f = ima[0:half_height:1, :]

            ima_c_dws = self.dwnsampling(im_c, "RGGB")
            ima_f_dws = self.dwnsampling(im_f, "RGGB")

            for i in range(ima_c_dws.shape[2]):

                # Fov + 15??
                cond_c = self.zen_c[:, :, i] >= self.fov + 20
                cond_f = self.zen_f[:, :, i] >= self.fov + 20

                # BlackLevel estimations
                bl_c = ima_c_dws[cond_c].mean()
                bl_f = ima_f_dws[cond_f].mean()

                print("Lens-close blacklevel estimation: {0:.1f}\n"
                      "Lens-far blacklevel estimation: {1:.1f}".format(bl_c, bl_f))

                self.im[half_height:height:1, :, i] -= bl_c
                self.im[0:half_height:1, :, i] -= bl_f

            return self.im
        else:
            raise ValueError("Mosaic down-sampling not done.")

    def normalisation(self):
        """
        Method for normalisation for gain and exposure time
        :return:
        """

        self.im /= (self.extract_integrationtime(self.metadata) * (self.extract_iso(self.metadata) / 100))
        return self.im

    def rolloff_correction(self):
        """
        Roll-off correction for each spectral band.
        :return:
        """
        if len(self.im.shape) == 3:

            imsize = self.geometric_close["red"].imsize.astype(int)

            rollclose = np.zeros((imsize[0], imsize[1], 3))
            rollfar = np.zeros((imsize[0], imsize[1], 3))

            for band, k in enumerate(self.geometric_close.keys()):

                zen_close_dws = self.zen_c[:, :, band]
                zen_far_dws = self.zen_f[:, :, band]

                rollc = self.rolloff_polynomial(zen_close_dws, *self.rolloff_close[band, :])
                rollf = self.rolloff_polynomial(zen_far_dws, *self.rolloff_far[band, :])

                rollc[zen_close_dws > self.fov] = 1.0
                rollf[zen_far_dws > self.fov] = 1.0

                rollclose[:, :, band] = rollc
                rollfar[:, :, band] = rollf

            # Assemble both roll-off for each lenses
            rolloff = np.concatenate((rollfar, rollclose), axis=0)

            # Roll-off correction
            self.im /= rolloff

            return self.im

        else:
            raise ValueError("Downsampling or demosaic must be done before!")

    def apply_absolute_radiance_calibration(self):
        """
        Apply absolute spectral radiance calibration coefficient to the digital numbers of each spectral band.
        :return:
        """

        if len(self.im.shape) == 3:

            height = self.im.shape[0]
            for n in range(self.im.shape[2]):
                self.im[0:int(height // 2):1, :, n] *= self.cl_far[n]
                self.im[int(height // 2):height:1, :, n] *= self.cl_close[n]

            return self.im
        else:
            raise ValueError("Downsampling or demosaic must be done before!")

    def immersion_correction(self):
        """
        Immersion correction for when the camera is in-water.
        :return:
        """

        if len(self.im.shape) == 3:
            self.im *= self.ifactor_close
            return np.clip(self.im, 0, None)
        else:
            raise ValueError("Downsampling or demosaic must be done before!")

    def dewarp(self, image, theta, phi, which):
        """

        :param image:
        :param theta:
        :param phi:
        :return:
        """
        if which == "close":
            center = np.round(self.geometric_close.center / 2).astype(int)
            inv_mapping_fit = self.geometric_close.popt_inv
        elif which == "far":
            center = np.round(self.geometric_far.center / 2).astype(int)
            inv_mapping_fit = self.geometric_far.popt_inv
        else:
            raise ValueError("Not a valid value for which variable. Should be either close or far.")

        # Inverse mapping to get radial position in function of theta
        rho = self.polynomial_fit_forcedzero(theta * 180/np.pi, *inv_mapping_fit) * 0.5  # Division by two (dws)

        # Center coordinate of image
        cdx, cdy = center[0] - 1, center[1] - 1

        # x and y pos
        xcam, ycam = rho * np.cos(phi), rho * np.sin(phi)
        xcam, ycam = xcam.astype(int) + cdx, ycam.astype(int) + cdy

        return image[ycam, xcam]

    def dewarpband(self, image, theta, phi, rho, zen, geo):
        """
        Basic dewap process using geometric calibration specific to each spectral band. X and Y position on image
        matrix is found by inverse mapping of the zenithal and azimuthal world coordinates.

        :param image:
        :param theta:
        :param phi:
        :param geo:
        :return:
        """

        # Get rho and zenith values of each pixel
        cond = zen < 90
        rho = rho[cond]
        zen = zen[cond]
        argso = np.argsort(rho)

        # Interpolation
        zen *= np.pi / 180
        rho_interpol = np.interp(theta, zen[argso], rho[argso])

        # Find undistorted coordinates
        x, y = rho_interpol * np.cos(phi), rho_interpol * np.sin(phi)

        # Real coordinates from affine transformation
        xprim, yprim = geo.affine_transfo(x, y)

        return image[yprim.astype(int), xprim.astype(int)]

    def exposuretime_metadata(self):
        """

        :return:
        """
        exp = str(self.metadata["Image ExposureTime"]).split("/")
        if len(exp) > 1:
            return float(exp[0])/float(exp[1])
        else:
            return float(exp[0])

    def getimage(self, which):
        """
        Select image according to the specified lens.

        :param which:
        :return:
        """

        height = self.im.shape[0]
        half_height = int(height // 2)

        if which == "close":
            im_s = self.im[half_height:height:1, :]
        elif which == "far":
            im_s = self.im[0:half_height:1, :]
        else:
            raise ValueError("Argument which image must be either close of far.")
        return im_s

    def azimuthal_integration(self):
        """
        Integration of radiance angular distribution for azimuth angles.

        :return:
        """

        if len(self.mappedradiance.shape) > 1:
            return np.trapz(self.mappedradiance, x=self.azimuth_mesh[0, :], axis=1)
        else:
            raise ValueError("Build radiance map before any integration.")

    def azimuthal_average(self):
        """
        Average of radiance in azimuth direction.

        :return:
        """
        if len(self.mappedradiance.shape) > 1:
            maprad = self.mappedradiance.copy()
            condzero = maprad == 0

            maprad[condzero] = np.nan

            return np.nanmean(maprad, axis=1)
        else:
            raise ValueError("Build radiance map before any integration.")

    def irradiance(self, zenimin, zenimax, planar=True, interpolation=False):
        """
        Estimate irradiance from the radiance angular distribution. By default, it calculates the planar irradiance.
        By setting the parameter planar to false, the scalar irradiance is computed. Zenimin = 0?? and Zenimax = 90?? gives
        the downwelling irradiance, while Zenimin = 90?? and Zenimax = 180?? gives the upwelling irradiance.

        :param zenimin:
        :param zenimax:
        :param planar:
        :return:
        """
        if interpolation:
            radm = self.mapped_radiance_4pi.copy()
        else:
            radm = self.mappedradiance.copy()
        if np.any(radm):

            zeni = self.zenith_mesh.copy()
            azi = self.azimuth_mesh.copy()

            mask = (zenimin * np.pi / 180 <= zeni) & (zeni <= zenimax * np.pi / 180)

            irr = np.array([])
            for b in range(radm.shape[2]):

                # Current radiance
                rad = radm[:, :, b]

                # Integrand
                if planar:
                    integrand = rad[mask] * np.absolute(np.cos(zeni[mask])) * np.sin(zeni[mask])
                else:
                    integrand = rad[mask] * np.sin(zeni[mask])

                azimuth_inte = integrate.simps(integrand.reshape((-1, azi.shape[1])), azi[mask].reshape((-1, azi.shape[1])), axis=1)
                e = integrate.simps(azimuth_inte, zeni[mask].reshape((-1, azi.shape[1]))[:, 0], axis=0)

                irr = np.append(irr, e)

            return irr

        else:
            print("No radiance map in regular angular grid found. Method map_radiance() must be done. ")

    def interpolation_3dpoints(self):
        """
        Not best method....
        :return:
        """

        if np.any(self.mappedradiance):
            radm = self.mappedradiance.copy()

            x, y, z = self.points_3d(self.zenith_mesh.copy(), self.azimuth_mesh.copy())

            rad_interp = np.empty(radm.shape)
            for b in range(radm.shape[2]):

                curr_rad = radm[:, :, b].copy()

                cond_zero = curr_rad == 0

                ref_coord = (x[~cond_zero], y[~cond_zero], z[~cond_zero])
                wanted_coord = (x[cond_zero], y[cond_zero], z[cond_zero])

                curr_rad[cond_zero] = scipy.interpolate.griddata(ref_coord, curr_rad[~cond_zero], wanted_coord, method="nearest")
                rad_interp[:, :, b] = curr_rad

            self.mapped_radiance_4pi = rad_interp
            return self.mapped_radiance_4pi
        else:
            print("No radiance map in regular angular grid found. Method map_radiance() must be done. ")

    def interpolation_gaussian_function(self):
        """
        Extrapolation of missing angles (over 4pi sr) using a gaussian fit on the data. See function
        self.general_gaussian().

        :return:
        """
        if np.any(self.mappedradiance):
            radm = self.mappedradiance.copy()
            az_avg = self.azimuthal_average()

            rad_interp = np.empty(radm.shape)
            for b in range(radm.shape[2]):
                co = ~np.isnan(az_avg[:, b])
                norm_val = np.mean(az_avg[:, b][co][:5])  # 5 first values
                az_avg_norm = az_avg[:, b][co] / norm_val
                zen = self.zenith_mesh[:, 0][co]

                popt, pcov = curve_fit(self.general_gaussian, zen, az_avg_norm, p0=[-0.7, 0, 0.1])

                curr_rad = radm[:, :, b].copy()
                cond_zero = curr_rad == 0
                curr_rad[cond_zero] = self.general_gaussian(self.zenith_mesh[cond_zero], *popt) * norm_val

                rad_interp[:, :, b] = curr_rad

            self.mapped_radiance_4pi = rad_interp

            return self.mapped_radiance_4pi
        else:
            print("No radiance map in regular angular grid found. Method map_radiance() must be done. ")

    @staticmethod
    def general_gaussian(x, a, b, c):
        """
        Gaussian function to extrapolate missing angles. 
        :param x: independent val
        :param a: a parameter
        :param b: b parameter
        :param c: c parameter
        :param d: d parameter
        :return: gaussian function apply to in val.
        """""
        return np.exp(-(x * a - b) ** 2) + c

    def polar_plot_contourf(self, fig, ax, ncontour):
        """
        Filled contour plot.

        :param fig: figure matplotlib
        :param ax: axe matplotlib
        :param ncontour: number of contour region.
        :return: tuple (fig, ax)
        """

        if len(self.mappedradiance.shape) == 3:
            lab = ["red", "green", "blue"]

            for n, a in enumerate(ax):
                zeni = self.zenith_mesh.copy() * 180 / np.pi
                azi = self.azimuth_mesh.copy()

                im = self.mappedradiance[:, :, n].copy()

                insideFOV = np.where(im > 0)
                mini, maxi = np.nanmin(im[insideFOV]), np.nanmax(im[insideFOV])

                cax = a.contourf(azi, zeni, np.clip(im, 0, 2 ** 14), np.linspace(mini, maxi, ncontour))

                ytik = np.arange(0, 200, 40)
                a.set_yticks(ytik)
                a.set_yticklabels(["{}??".format(i) for i in ytik], fontsize=6)

                a.grid(linestyle="--")

                cl = fig.colorbar(cax, ax=a, orientation="horizontal", format='%.1e')
                cl.ax.set_title("$L_{0}$".format(lab[n][0]), fontsize=7)
                cl.ax.set_xticklabels(cl.ax.get_xticklabels(), rotation=70)

                a.text(-0.1, 1.1, string.ascii_lowercase[n] + ")", transform=a.transAxes, size=11, weight='bold')

            return fig, ax
        else:
            raise ValueError("Radiance map should be build before.")

    def show_mapped_radiance(self):
        """
        Rapid plot of each spectral band radiance mapped on a regular degree spaced grid.

        :return:
        """

        if np.any(self.mappedradiance):
            f, a = plt.subplots(3, 1, sharex=True)

            radiance_copy = self.mappedradiance.copy()
            titless = ["red band", "green band", "blue band"]

            for n, aa in enumerate(a):

                aa.imshow(radiance_copy[:, :, n])
                aa.set_title(titless[n], fontsize=7)
                aa.set_ylabel("Zenith [??]")

            a[2].set_xlabel("Azimuth [??]")
            f.tight_layout()
        else:
            print("Noting to show. Method map_radiance() must be done. ")

    def angle_from_axis(self, axis="x"):
        """

        :param axis:
        :return:
        """
        if self.zenith_mesh.any() and self.azimuth_mesh.any():
            xp, yp, zp = self.points_3d(self.zenith_mesh, self.azimuth_mesh)

            if axis == "x":
                angl = np.arctan2((yp ** 2 + zp ** 2) ** (1 / 2), xp)
            elif axis == "y":
                angl = np.arctan2((xp ** 2 + zp ** 2) ** (1 / 2), yp)
            else:
                raise ValueError("Invalid parameter axis.")
            return angl
        else:
            raise ValueError("Dewarping should be done before.")

    def define_field_of_view(self):
        """
        Define field-of-view according to medium info.

        :return: fov (float)
        """
        if self.medium == "air":
            _fov = 90.0
        elif self.medium == "water":
            _fov = 72.0
        else:
            raise ValueError("Medium attribute seems to be invalid.")
        return _fov

    @staticmethod
    def get_band_angular_coord(geometric):
        """

        :param geometric:
        :return:
        """
        dict_cor = {"red": 0, "green": 1, "blue": 2}
        rad = np.empty((int(geometric["red"].imsize[1]), int(geometric["red"].imsize[0]), 3))
        zen = rad.copy()
        az = rad.copy()

        for k in geometric.keys():
            r, z, a = geometric[k].angular_coordinates()
            rad[:, :, dict_cor[k]] = r
            zen[:, :, dict_cor[k]] = z
            az[:, :, dict_cor[k]] = a

        return rad, zen, az

    @staticmethod
    def angle_grid(resolution):
        """
        Meshgrid of azimuth and zenith on a complete sphere according to angular resolution.

        :param resolution:
        :return: tuple (azimuth, zenith) in radians
        """

        zenith_lim = np.array([0.0, 180.0])
        azimuth_lim = np.array([0.0, 360.0])

        # Zenith and azimuth meshgrid
        n_zen = np.round(abs(zenith_lim[1] - zenith_lim[0]) / resolution) + 1
        n_azi = np.round(abs(azimuth_lim[1] - azimuth_lim[0]) / resolution) + 1

        azi, zen = np.meshgrid(np.linspace(azimuth_lim[0].astype(int), azimuth_lim[1].astype(int), n_azi.astype(int)),
                               np.linspace(zenith_lim[0].astype(int), zenith_lim[1].astype(int), n_zen.astype(int)))

        azi *= np.pi / 180
        zen *= np.pi / 180

        return azi, zen

    @staticmethod
    def rotation(PX, PY, PZ, rmat):
        """
        Function that applies a specified 3D rotation matrix on camera coordinates.

        :param PX:
        :param PY:
        :param PZ:
        :param rotation_matrix:
        :return:
        """
        # Application of rotation matrix
        rotation = rmat.dot(np.array([PX.flatten(), PY.flatten(), PZ.flatten()]))

        return rotation[0, :].reshape(PX.shape), rotation[1, :].reshape(PY.shape), rotation[2, :].reshape(PZ.shape)

    @staticmethod
    def points_3d(zeni, azi):
        """

        :param zeni: Zenith 2D array in radian.
        :param azi: Azimuth 2D array in radian.
        :return:
        """

        return np.sin(zeni) * np.cos(azi), np.sin(zeni) * np.sin(azi), np.cos(zeni)

    @staticmethod
    def rx(roll):
        """
        Rotation matrix around x axis.

        :param roll:
        :return:
        """
        return np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    @staticmethod
    def ry(pitch):
        """
        Rotation matrix around y axis.

        :param pitch:
        :return:
        """
        return np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

    @staticmethod
    def rz(yaw):
        """
        Rotation matrix around z axis.

        :param yaw:
        :return:
        """
        return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])


# Other function accessible
def irradiance(zeni, azi, radm, zenimin, zenimax, planar=True):
    """
    Estimate irradiance from the radiance angular distribution. By default, it calculates the planar irradiance.
    By setting the parameter planar to false, the scalar irradiance is computed. Zenimin = 0?? and Zenimax = 90?? gives
    the downwelling irradiance, while Zenimin = 90?? and Zenimax = 180?? gives the upwelling irradiance.

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


def attenuation_coefficient(E, d):
    """
    Computation of attenuation coefficient of down-welling irradiance.

    :param E: irradiance in W m-2 nm-1 (array)
    :param d: depths in cm (array)
    :return: attenuation coefficient
    """
    return -np.gradient(E, d/100, edge_order=2) * (1/E)


def azimuthal_average(rad):
    """
    Average of radiance in azimuth direction.

    :return:
    """
    condzero = rad == 0
    rad2 = rad.copy()
    rad2[condzero] = np.nan
    return np.nanmean(rad2, axis=1)


if __name__ == "__main__":

    # Test
    oden_data_list = glob.glob("/Volumes/MYBOOK/data-i360/field/oden-08312018/IMG*.dng")
    im_rad = ImageRadiancei360(oden_data_list[10], "water")

    im_rad.get_radiance(dark_metadata=True)
    im_rad.map_radiance(angular_resolution=1.0)
    im_rad.show_mapped_radiance()

    # Interpolation for the missing angles
    A = im_rad.interpolation_3dpoints()
    B = im_rad.interpolation_gaussian_function()

    # Irradiances
    ed = im_rad.irradiance(0, 90, planar=True)
    eu = im_rad.irradiance(90, 180)
    e0 = im_rad.irradiance(0, 180, planar=False)

    plt.figure()
    plt.imshow(A[:, :, 1])

    plt.figure()
    plt.imshow(B[:, :, 1])

    plt.show()
