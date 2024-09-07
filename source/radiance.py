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
import numpy as np
import pandas
from openpyxl import load_workbook
from scipy import integrate
import scipy.interpolate
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

# Other module
import calibrations.calibrations_info as cinfo
from source.processing import ProcessImage
from source.geometric_rolloff import MatlabGeometricMengine


# Classes
class ImageRadiancei360x3(ProcessImage):
    """

    """
    def __init__(self, image_path, medium="air", cam_sn="2BW7X7", cover="cover"):

        # ** Save init attributes **
        self.medium = medium.lower()
        self.cam_sn = cam_sn
        self.cover = cover.lower()

        self.fov = self.cam_fov()

        # ** Open calibration files **
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        # Geometric
        self.geometric_front, self.geometric_back = self.open_geometric_calibration()
        # Roll-off
        self.roll_off_front, self.roll_off_back = self.open_roll_off()
        # Absolute radiance coefficients
        self.abs_co_front, self.abs_co_back = self.open_radiance_calibration_coefficients()
        # Immersion factor coefficients
        self.ifactor_front, self.ifactor_back = self.open_immersion_factor()

        # ** Class attributes **
        self.im_original, self.metadata = self._readDNG_rawpy(image_path)  # From ProcessImage class
        self.im = self.im_original.copy().astype(float)

        self.rho_front, self.zen_front, self.az_front = self.get_band_angular_coord(self.geometric_front)
        self.rho_back, self.zen_back, self.az_back = self.get_band_angular_coord(self.geometric_back)

        # Radiance map (attributes to be defined later)
        self.zenith_mesh = np.array([])
        self.azimuth_mesh = np.array([])
        self.mapped_radiance = np.array([])
        self.mapped_radiance_4pi = np.array([])
        self.legendre_coefficients = np.zeros(4)

    def open_geometric_calibration(self):
        """
        Method that opens geometric calibration according to camera info.
        :return:
        :rtype: tuple
        """
        pa = self.base_path + "/calibrations/geometric-calibration/calibrationfiles"
        geodata = h5py.File(pa + f"/geometric-calibration-{self.cam_sn}.h5", "r")

        # Front
        geo_id_front = cinfo.geometric[f"{self.cam_sn}"][f"{self.cover}"][f"{self.medium}"]["front"]
        group_name_geo_front = f"{self.medium}/{self.cover}/front/{geo_id_front}"
        geodata_front = geodata[group_name_geo_front]
        print(geodata_front)

        # Back
        geo_id_back = cinfo.geometric[f"{self.cam_sn}"][f"{self.cover}"][f"{self.medium}"]["back"]
        group_name_geo_back = f"{self.medium}/{self.cover}/back/{geo_id_back}"
        geodata_back = geodata[group_name_geo_back]
        print(geodata_back)

        geometric_front = {}
        geometric_back = {}
        for k in ["red", "green", "blue"]:
            geometric_front[k] = MatlabGeometricMengine(geodata_front["fp"][k], geodata_front["ierror"][k])
            geometric_back[k] = MatlabGeometricMengine(geodata_back["fp"][k], geodata_back["ierror"][k])

        return geometric_front, geometric_back

    def open_roll_off(self):
        """
        Method to open roll-off calibrations (front and back).
        :return:
        :rtype:
        """
        pa = self.base_path + "/calibrations/roll-off/calibrationfiles"
        rfdata = h5py.File(pa + f"/roll-off-{self.cam_sn}.h5", "r")

        # Front
        rf_id_front = cinfo.rf[f"{self.cam_sn}"][f"{self.cover}"][f"{self.medium}"]["front"]
        rf_front = rfdata[f"{self.cover}/{self.medium}/front/{rf_id_front}/fit-coefficients"][:]

        # Back
        rf_id_back = cinfo.rf[f"{self.cam_sn}"][f"{self.cover}"][f"{self.medium}"]["back"]
        rf_back = rfdata[f"{self.cover}/{self.medium}/back/{rf_id_back}/fit-coefficients"][:]

        return rf_front, rf_back

    def open_radiance_calibration_coefficients(self):
        """
        Method that extracts absolute spectral radiance calibration coefficients (front and back).
        :return: front rgb calib coefficients [W sr-1 m-2 nm-1 ADU-1], back rgb calib coefficients [W sr-1 m-2 nm-1 ADU-1]
        :rtype: tuple
        """

        pa = self.base_path + "/calibrations/absolute-spectral-radiance/calibrationfiles"
        radcal_file = h5py.File(pa + f"/absolute-radiance-{self.cam_sn}.h5", "r")

        # Front
        id_front = cinfo.abs_rad[f"{self.cam_sn}"][f"{self.cover}"]["front"]
        coeff_front = radcal_file[f"{self.cover}/front/{id_front}/cal-coefficients"][:]

        # Back
        id_back = cinfo.abs_rad[f"{self.cam_sn}"][f"{self.cover}"]["back"]
        coeff_back = radcal_file[f"{self.cover}/back/{id_back}/cal-coefficients"][:]

        return coeff_front, coeff_back

    def open_immersion_factor(self):
        """
        Method that extracts calibrated immersion factors when camera is in-water.
        :return: immersion factors front ndarray, immersion factor back ndarray
        :rtype: tuple
        """

        pa = self.base_path + "/calibrations/immersion-factor/calibrationfiles"
        imf_file = h5py.File(pa + f"/immersion-factor-{self.cam_sn}.h5", "r")

        # Front
        id_front = cinfo.imf[f"{self.cam_sn}"][f"{self.cover}"]["front"]
        imf_front = imf_file[f"{self.cover}/front/{id_front}/immersion-factor"][:]

        # Back
        id_back = cinfo.imf[f"{self.cam_sn}"][f"{self.cover}"]["back"]
        imf_back = imf_file[f"{self.cover}/back/{id_back}/immersion-factor"][:]

        return imf_front, imf_back

    def get_radiance(self):
        """

        :return:
        :rtype:
        """
        # Dark correction
        self.dark_correction()

        # Down-sampling
        self.im = self.dwnsampling(self.im, "GBRG")

        # Normalization
        self.normalization()

        # Roll-off
        self.roll_off_correct()

        # Absolute radiance
        self.apply_absolute_radiance_coefficients()

        # Immersion factor
        if self.medium == "water":
            self.apply_immersion_factor()

        # Clip
        self.im = np.clip(self.im, 0, None)

    def map_radiance(self, angular_resolution=1.0):
        """
        Main method to transpose fish-eye circular projection to equirectangular image.
        :param angular_resolution:
        :type angular_resolution:
        :return:
        :rtype:
        """

        if len(self.im.shape) == 3:

            # Meshgrid
            azi, zen = self.angle_grid(angular_resolution)

            # Reference coordinate rotation
            px, py, pz = self.points_3d(zen, azi)

            # Rotation for lens back
            rotation_back = self.rz(np.pi)
            npx_b, npy_b, npz_b = self.rotation(px, py, pz, rotation_back)

            theta_f, phi_f = np.arccos(py), np.arctan2(pz, px)  # angular coordinates (zenith, azimuth) lens front
            #theta_b, phi_b = np.arccos(npy_b), np.arctan2(npz_b, -npx_b)  # lens back
            theta_b, phi_b = np.arccos(npy_b), np.arctan2(npz_b, npx_b)  # lens back

            cond_f = theta_f < self.fov * np.pi / 180
            cond_b = theta_b < self.fov * np.pi / 180

            # Dewarping
            dewarp = np.zeros((theta_f.shape[0], theta_f.shape[1], 3))

            #im_c = self.getimage("close")
            #im_f = self.getimage("far")
            height = self.im.shape[0]
            im_front = self.im[0:int(height // 2):1, :, :]
            im_back = self.im[int(height // 2): height: 1, :, :]

            for b, k in enumerate(["red", "green", "blue"]):

                de = dewarp[:, :, b].copy()


                # Dewarping camera 2 (f)
                de[cond_b] = self.dewarpband(im_back[:, :, b],
                                             theta_b[cond_b], phi_b[cond_b],
                                             self.rho_back[:, :, b], self.zen_back[:, :, b], self.geometric_back[k])

                # Dawarping camera 1 (c)
                de[cond_f] = self.dewarpband(im_front[:, :, b],
                                             theta_f[cond_f], phi_f[cond_f],
                                             self.rho_front[:, :, b], self.zen_front[:, :, b], self.geometric_front[k])

                dewarp[:, :, b] = de

            self.zenith_mesh = zen
            self.azimuth_mesh = azi
            self.mapped_radiance = dewarp

            return self.zenith_mesh, self.azimuth_mesh, self.mapped_radiance

    def get_radiance_angular_distribution(self):
        """
        High level class to get the radiance values and perform the mapping.

        1. Get radiance using a dark estimate from the CMOS (dark_metadata=False)
        2. Map radiance into a meshgrid of 1 degree in angular resolution (for zenith and azimuth)
        3. Perform smoothing and extrapolation by fitting a 3 order Legendre Polynomial to the radiance azimuthal average.

        :return:
        """

        self.get_radiance()
        self.map_radiance(angular_resolution=1.0)  # 1 deg in angular resolution (zenith and azimuth)

        if self.medium == "water":
            self.extrapolation_legendre_polynomials()  # Fit radiance and extract missing values

    def dark_correction(self):
        """

        :return:
        :rtype:
        """
        rind, gind, bind = self.dws_pattern("GBRG")  # Todo Extract from metadata
        darkimage = np.ones(self.im_original.shape)
        dark_list_metadata = np.array(self.metadata["Image BlackLevel"].values).astype(float)

        darkimage[bind[0]::2, bind[1]::2] *= dark_list_metadata[1]
        darkimage[gind[0, 0]::2, gind[0, 1]::2] *= dark_list_metadata[0]
        darkimage[gind[1, 0]::2, gind[1, 1]::2] *= dark_list_metadata[3]
        darkimage[rind[0]::2, rind[1]::2] *= dark_list_metadata[2]

        self.im -= darkimage

        return self.im

    def normalization(self):
        """
        Normilzation by exposure time and gain.
        :return:
        :rtype:
        """
        self.im /= (self.extract_integrationtime(self.metadata) * self.extract_iso(self.metadata) / 100)
        return self.im

    def roll_off_correct(self):
        """
        Method that performs roll-off correction in each spectral band.
        :return:
        :rtype:
        """
        if len(self.im.shape) == 3:

            im_size = self.geometric_front["red"].imsize.astype(int)

            rf_im_front = np.zeros((im_size[0], im_size[1], 3))
            rf_im_back = np.zeros((im_size[0], im_size[1], 3))

            for band, k in enumerate(["red", "green", "blue"]):

                rf_im_front[:, :, band] = self.rolloff_polynomial(self.zen_front[:, :, band], *self.roll_off_front[band, :])
                rf_im_back[:, :, band] = self.rolloff_polynomial(self.zen_back[:, :, band], *self.roll_off_back[band, :])

                rf_im_front[:, :, band][self.zen_front[:, :, band] > self.fov] = 1.0
                rf_im_back[:, :, band][self.zen_back[:, :, band] > self.fov] = 1.0

            # Assembling roll-off of each lens
            rf_total = np.concatenate((rf_im_front, rf_im_back), axis=0)
            #plt.figure()
            #plt.imshow(rf_total[:, :, 0])

            # Roll-off correction
            self.im /= rf_total

            return self.im

        else:
            raise ValueError("Down-sampling has to be done before")

    def apply_absolute_radiance_coefficients(self):
        """
        Method applying absolute spectral radiance calibration coefficient to the image.
        :return: absolute spectral radiance 3D image
        :rtype: ndarray
        """

        if len(self.im.shape) == 3:
            height = self.im.shape[0]

            # Apply absolute coefficient to each band
            for n in range(self.im.shape[2]):
                self.im[0:int(height // 2):1, :, n] *= self.abs_co_front[n]
                self.im[int(height // 2):height:1, :, n] *= self.abs_co_back[n]

            return self.im
        else:
            raise ValueError("Downsalpling or demosaic must be done before.")

    def apply_immersion_factor(self):
        """

        :return:
        :rtype:
        """
        if len(self.im.shape) == 3:
            height = self.im.shape[0]

            # Apply absolute coefficient to each band
            for n in range(self.im.shape[2]):
                self.im[0:int(height // 2):1, :, n] *= self.ifactor_front[0][n]
                self.im[int(height // 2):height:1, :, n] *= self.ifactor_back[0][n]
            return self.im
        else:
            raise ValueError("Downsalpling or demosaic must be done before.")

    def cam_fov(self):
        """
        Define field-of-view according to medium and cam covered or not.

        :return: field of view
        :rtype: float
        """
        if self.medium == "air":
            #_fov = 91.0
            _fov = 90.5
        elif self.medium == "water":
            if self.cover == "nocover":
                _fov = 75.0
            elif self.cover == "cover":
                _fov = 80.0
            else:
                raise ValueError("Not valid cover attribute.")
        else:
            raise ValueError("Not valid medium attribute.")
        return _fov

    def extrapolation_legendre_polynomials(self):
        """
        Extrapolation of missing angles using Legendre Polynomials to fit the radiance azimuthaly averaged. The fourth
        order (n=4) is used in this case. We also discard angle below 25 degrees in (zenith) as they are affected by
        hole effects and camera drastic fall-off in irradiance.

        :return: the mapped radiance over 4pi sr
        """

        # Check if there is any mapped radiance
        if np.any(self.mapped_radiance):
            radiance_m = self.mapped_radiance.copy()  # Radiance mapped
            radiance_ex = np.empty(radiance_m.shape)  # New radiance mapped with extrapolated values
            radiance_az_average = self.azimuthal_average()  # Azimuthal average

            zenith = self.zenith_mesh[:, 0].copy()

            # Loop for each band
            for b in range(radiance_m.shape[2]):

                # Get not NaN values
                curr_radiance_az_avg = radiance_az_average[:, b]
                mask_co = ~np.isnan(curr_radiance_az_avg)  # not NaN values
                radiance_val = curr_radiance_az_avg[mask_co]
                zenith_val = zenith[mask_co]

                # Further mask for value over 20 degrees (because of camera drastic drop at the edges)
                mask_23 = zenith_val >= (20 * np.pi/180)
                radiance_val = radiance_val[mask_23]
                zenith_val = zenith_val[mask_23]

                # Legendre fit
                mu = np.cos(zenith_val)  # cos(theta)
                leg_fit = np.polynomial.legendre.Legendre.fit(mu, radiance_val, 4, domain=[-1., 1.])
                coeff = leg_fit.convert().coef
                self.legendre_coefficients = coeff

                # Replace value in new matrix
                curr_radiance = radiance_m[:, :, b]
                cond_zero = curr_radiance == 0

                curr_radiance[cond_zero] = np.polynomial.legendre.legval(np.cos(self.zenith_mesh[cond_zero]), coeff)

                radiance_ex[:, :, b] = curr_radiance

            self.mapped_radiance_4pi = radiance_ex

            return self.mapped_radiance_4pi
        else:
            print("No radiance map found. Method self.map_radiance() must be done first.")

    def azimuthal_average(self):
        """
        Average of radiance in azimuth direction.

        :return:
        """
        if len(self.mapped_radiance.shape) > 1:
            maprad = self.mapped_radiance.copy()
            condzero = maprad == 0
            maprad[condzero] = np.nan

            return np.nanmean(maprad, axis=1)
        else:
            raise ValueError("Build radiance map before any integration.")

    def irradiance(self, zenimin, zenimax, planar=True, extrapolation=False):
        """
        Estimate irradiance from the radiance angular distribution. By default, it calculates the planar irradiance.
        By setting the parameter planar to false, the scalar irradiance is computed. Zenimin = 0˚ and Zenimax = 90˚ gives
        the downwelling irradiance, while Zenimin = 90° and Zenimax = 180˚ gives the upwelling irradiance.

        :param zenimin:
        :param zenimax:
        :param planar:
        :return:
        """
        # Get radiance angular distribution
        if extrapolation:
            radm = self.mapped_radiance_4pi.copy()
        else:
            radm = self.mapped_radiance.copy()

        # Integration
        if np.any(radm):

            zeni = self.zenith_mesh.copy()
            azi = self.azimuth_mesh.copy()

            mask = (zenimin * np.pi / 180 <= zeni) & (zeni <= zenimax * np.pi / 180)  # Mask for radiance direction

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

    @staticmethod
    def dewarpband(image, theta, phi, rho, zen, geo):
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
        cond = zen <= 90
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

    @staticmethod
    def get_band_angular_coord(geometric):
        """
        Create angular coordinate of camera for each spectral band given the MatlabGeometricMengine passed in entry.

        :param geometric:
        :return:
        """
        dict_cor = {"red": 0, "green": 1, "blue": 2}
        rad = np.empty((int(geometric["red"].imsize[0]), int(geometric["red"].imsize[1]), 3))  # TODO verfify good matrix size ?
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

    def show_mapped_radiance(self):
        """
        Rapid plot of each spectral band radiance mapped on a regular degree spaced grid.

        :return:
        """

        if np.any(self.mapped_radiance):
            f, a = plt.subplots(3, 1, sharex=True)

            radiance_copy = self.mapped_radiance.copy()
            titless = ["red band", "green band", "blue band"]

            for n, aa in enumerate(a):

                aa.imshow(radiance_copy[:, :, n])
                aa.set_title(titless[n], fontsize=7)
                aa.set_ylabel("Zenith [˚]")

            a[2].set_xlabel("Azimuth [˚]")
            f.tight_layout()
        else:
            print("Noting to show. Method map_radiance() must be done. ")


class ImageRadiancei360(ProcessImage):
    """
    Class to build radiance map from Insta360 ONE images.
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
        #self.cl_close = self.open_calibrations("lens-close/20200909/cal-coefficients", calibration="absolute")
        self.cl_close = self.open_calibrations("lens-close/20200908/cal-coefficients", calibration="absolute")
        #self.cl_far = self.open_calibrations("lens-far/20200909/cal-coefficients", calibration="absolute")
        self.cl_far = self.open_calibrations("lens-far/20200908/cal-coefficients", calibration="absolute")

        # Immersion factor
        #self.ifactor_close = self.open_calibrations("lens-close/20200911/immersion", calibration="immersion")
        self.ifactor_close = self.open_calibrations("lens-close/20200910/immersion", calibration="immersion")

        # Roll-off
        # self.rolloff_close = self.open_rolloff_calibration("lens-close/20190501/fit-coefficients")
        # self.rolloff_far = self.open_rolloff_calibration("lens-close/20190501/fit-coefficients")  # !!!!!
        #self.rolloff_close = self.open_rolloff_calibration()
        #self.rolloff_far = self.open_rolloff_calibration()
        self.rolloff_close = self.open_rolloff_calibration()
        self.rolloff_far = self.rolloff_close.copy()

        # Attributes
        self.im_original, self.metadata = self._readDNG_np(image_path)  # From ProcessImage class
        self.im = self.im_original.copy().astype(float)

        self.rad_c, self.zen_c, self.az_c = self.get_band_angular_coord(self.geometric_close)  # time; approx 1.08 second
        self.rad_f, self.zen_f, self.az_f = self.get_band_angular_coord(self.geometric_far)

        # Orientation (likely)
        self.ori = self.get_orientation()

        # Radiance map (attributes to be defined later)
        self.zenith_mesh = np.array([])
        self.azimuth_mesh = np.array([])
        self.mappedradiance = np.array([])
        self.mapped_radiance_4pi = np.array([])
        self.legendre_coefficients = np.zeros(4)

    def open_geometric_calibration(self):
        """
        Function that opens geometric calibration.
        :return: tuple (geo_close, geo_far)
        """
        p = self.base_path + "/calibrations/geometric-calibration/calibrationfiles/"

        if self.medium.lower() == "air":
            geo_air = h5py.File(p + "geometric-calibration-air.h5", "r")
            geocalib_c = geo_air["/lens-close/20190104_192404/"]
            geocalib_f = geo_air["/lens-far/20190104_214037/"]

        elif self.medium.lower() == "water":
            geo_water = h5py.File(p + "geometric-calibration-water.h5")
            geocalib_c = geo_water["/lens-close/20200730_112353/"]
            geocalib_f = geo_water["/lens-far/20200730_143716/"]
        else:
            raise ValueError("Invalid name for medium. Should be 'air' or 'water'.")

        # Build dictionary
        geometric_close = {}
        geometric_far = {}
        for k in geocalib_c["fp"].keys():
            geometric_close[k] = MatlabGeometricMengine(geocalib_c["fp"][k], geocalib_c["ierror"][k])
            geometric_far[k] = MatlabGeometricMengine(geocalib_f["fp"][k], geocalib_f["ierror"][k])

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
            #ptf = self.base_path + "/calibrations/absolute-spectral-radiance/calibrationfiles/absolute_radiance.h5"
            #ptf = self.base_path + "/calibrations/absolute-spectral-radiance/calibrationfiles/absolute_radiance_fluorolog.h5"
            ptf = self.base_path + "/calibrations/absolute-spectral-radiance/calibrationfiles/absolute_radiance_imf_fluorolog.h5"
        elif calibration.lower() == "immersion":
            #ptf = self.base_path + "/calibrations/immersion-factor/calibrationfiles/immersion_factor.h5"
            ptf = self.base_path + "/calibrations/immersion-factor/calibrationfiles/immersion_factor_fluorolog.h5"
        else:
            raise ValueError("Invalid entry calibration. Only value permitted are 'absolute' or 'immersion'.")
        with h5py.File(ptf) as hfrel:
            cal = hfrel[tag][:]
        return cal

    def get_radiance_angular_distribution(self):
        """
        High level class to get the radiance values and perform the mapping.

        1. Get radiance using a dark estimate from the CMOS (dark_metadata=False)
        2. Map radiance into a meshgrid of 1 degree in angular resolution (for zenith and azimuth)
        3. Perform smoothing and extrapolation by fitting a 3 order Legendre Polynomial to the radiance azimuthal average.

        :return:
        """

        self.get_radiance(dark_metadata=False)
        self.map_radiance(angular_resolution=1.0)  # 1 deg in angular resolution (zenith and azimuth)

        if self.medium == "water":
            self.extrapolation_legendre_polynomials()  # Fit radiance and extract missing values

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
            self.dark_correction_image_plane()  # Dark correction using average of CMOS array outside image circle

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

                # Dawarping camera 1 (c)
                de[cond_c] = self.dewarpband(im_c[:, :, b], theta_c[cond_c], phi_c[cond_c],
                                             self.rad_c[:, :, b], self.zen_c[:, :, b], self.geometric_close[k])

                # Dewarping camera 2 (f)
                de[cond_f] = self.dewarpband(im_f[:, :, b], theta_f[cond_f], phi_f[cond_f],
                                             self.rad_f[:, :, b], self.zen_f[:, :, b], self.geometric_far[k])

                dewarp[:, :, b] = de

            self.zenith_mesh = zen
            self.azimuth_mesh = azi
            self.mappedradiance = dewarp

            return self.zenith_mesh, self.azimuth_mesh, self.mappedradiance

        else:
            raise ValueError("Downsampling or demosaic must be done before!")

    def dark_correction(self):
        """
        Method to remove dark noise using the stored value in the metadata.
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

                # Fov + 20˚
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
        Depreciated function of dewarping.

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

    @staticmethod
    def dewarpband(image, theta, phi, rho, zen, geo):
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
        cond = zen <= 90
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

    def irradiance(self, zenimin, zenimax, planar=True, extrapolation=False):
        """
        Estimate irradiance from the radiance angular distribution. By default, it calculates the planar irradiance.
        By setting the parameter planar to false, the scalar irradiance is computed. Zenimin = 0˚ and Zenimax = 90˚ gives
        the downwelling irradiance, while Zenimin = 90° and Zenimax = 180˚ gives the upwelling irradiance.

        :param zenimin:
        :param zenimax:
        :param planar:
        :return:
        """
        # Get radiance angular distribution
        if extrapolation:
            radm = self.mapped_radiance_4pi.copy()
        else:
            radm = self.mappedradiance.copy()

        # Integration
        if np.any(radm):

            zeni = self.zenith_mesh.copy()
            azi = self.azimuth_mesh.copy()

            mask = (zenimin * np.pi / 180 <= zeni) & (zeni <= zenimax * np.pi / 180)  # Mask for radiance direction

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

    def extrapolation_3dpoints(self):
        """
        Extrapolation of missing data using scipy.interpolate.griddata fonction and nearest method in 3D coordinates.
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
            print("No radiance map in regular angular grid found. Method map_radiance() must be done.")

    def extrapolation_rbf(self):
        """
        Extrapolation based on radial basis function. Time consuming....
        :return:
        """
        if np.any(self.mappedradiance):

            radiance_map = self.mappedradiance.copy()[::2, ::2]
            radiance_map_4pi = np.empty(radiance_map.shape)
            thetas = self.zenith_mesh.copy()
            phis = self.azimuth_mesh.copy()

            for band in range(3):
                print(band)
                f, a = plt.subplots(2, 1)

                radiance_current = radiance_map[:, :, band]
                a[0].imshow(radiance_current)
                print(radiance_current.shape)
                mask_zeros = radiance_current == 0

                # Interpolator
                interpolat = scipy.interpolate.Rbf(thetas[::2, ::2][~mask_zeros], phis[::2, ::2][~mask_zeros],
                                                   radiance_current[~mask_zeros], smooth=2.0, epsilon=0.005,
                                                   function="linear")
                print(interpolat.epsilon)

                # Interpolation
                radiance_current[mask_zeros] = interpolat(thetas[::2, ::2][mask_zeros], phis[::2, ::2][mask_zeros])
                radiance_map_4pi[:, :, band] = radiance_current.copy()
                a[1].imshow(radiance_current)

        else:
            print("No radiance map in regular angular grid found. Method map_radiance() must be done.")

        return radiance_map_4pi

    def extrapolation_gaussian_function(self):
        """
        Extrapolation (!) of missing angles (over 4pi sr) using a gaussian fit on the data. See function
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

                # Curve fitting with a gaussian function
                popt, pcov = curve_fit(self.general_gaussian, zen, az_avg_norm, p0=[-0.7, 0, 0.1])

                curr_rad = radm[:, :, b].copy()
                cond_zero = curr_rad == 0
                curr_rad[cond_zero] = self.general_gaussian(self.zenith_mesh[cond_zero], *popt) * norm_val

                rad_interp[:, :, b] = curr_rad

            self.mapped_radiance_4pi = rad_interp

            return self.mapped_radiance_4pi
        else:
            print("No radiance map in regular angular grid found. Method map_radiance() must be done. ")

    def extrapolation_legendre_polynomials(self):
        """
        Extrapolation of missing angles using Legendre Polynomials to fit the radiance azimuthaly averaged. The fourth
        order (n=4) is used in this case. We also discard angle below 25 degrees in (zenith) as they are affected by
        hole effects and camera drastic fall-off in irradiance.

        :return: the mapped radiance over 4pi sr
        """

        # Check if there is any mapped radiance
        if np.any(self.mappedradiance):
            radiance_m = self.mappedradiance.copy()  # Radiance mapped
            radiance_ex = np.empty(radiance_m.shape)  # New radiance mapped with extrapolated values
            radiance_az_average = self.azimuthal_average()  # Azimuthal average

            zenith = self.zenith_mesh[:, 0].copy()

            # Loop for each band
            for b in range(radiance_m.shape[2]):

                # Get not NaN values
                curr_radiance_az_avg = radiance_az_average[:, b]
                mask_co = ~np.isnan(curr_radiance_az_avg)  # not NaN values
                radiance_val = curr_radiance_az_avg[mask_co]
                zenith_val = zenith[mask_co]

                # Further mask for value over 23 degrees (because of camera drastic drop at the edges)
                mask_23 = zenith_val >= (25.0 * np.pi/180)
                radiance_val = radiance_val[mask_23]
                zenith_val = zenith_val[mask_23]

                # Legendre fit
                mu = np.cos(zenith_val)  # cos(theta)
                leg_fit = np.polynomial.legendre.Legendre.fit(mu, radiance_val, 4, domain=[-1., 1.])
                coeff = leg_fit.convert().coef
                self.legendre_coefficients = coeff

                # Replace value in new matrix
                curr_radiance = radiance_m[:, :, b]
                cond_zero = curr_radiance == 0

                curr_radiance[cond_zero] = np.polynomial.legendre.legval(np.cos(self.zenith_mesh[cond_zero]), coeff)

                radiance_ex[:, :, b] = curr_radiance

            self.mapped_radiance_4pi = radiance_ex

            return self.mapped_radiance_4pi
        else:
            print("No radiance map found. Method self.map_radiance() must be done first.")

        return

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
                a.set_yticklabels(["{}˚".format(i) for i in ytik], fontsize=6)

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
                aa.set_ylabel("Zenith [˚]")

            a[2].set_xlabel("Azimuth [˚]")
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

    def get_orientation(self):
        """"
        Kind of obscure Image UserComment in metadata with 6 floats changing at each image (likely pitch, yaw, roll)
        """

        string_ = str(self.metadata["Image UserComment"]).split(" ")[1]
        numb_str = string_.split("_")
        numb = [float(n) for n in numb_str]
        numb = np.array(numb) * 180 / np.pi
        #print(numb)
        return numb

    @staticmethod
    def get_band_angular_coord(geometric):
        """
        Create angular coordinate of camera for each spectral band given the MatlabGeometricMengine passed in entry.

        :param geometric:
        :return:
        """
        dict_cor = {"red": 0, "green": 1, "blue": 2}
        rad = np.empty((int(geometric["red"].imsize[1]), int(geometric["red"].imsize[0]), 3))
        zen = rad.copy()
        az = rad.copy()

        for k in geometric.keys():
            r, z, a = geometric[k].angular_coordinates()
            band_number = dict_cor[k]

            rad[:, :, band_number] = r
            zen[:, :, band_number] = z
            az[:, :, band_number] = a

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


# Class usable with saved radiance profiles
class RadClass:

    def __init__(self, data_path="data/oden-08312018_v02.h5", station="station_1", data_type="camera", freeboard=20.0,
                 wl_dct={480: 2, 540: 1, 600: 0}):
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
        self.wl_dct = wl_dct

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

        # Diffuse attenuation coefficient
        #self.K_d = self.calculate_Kd()
        self.K_d = self.calculate_kd_layers()

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

        print(radiance_profile.keys())

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
        ed = np.zeros(self.ordered_keys.shape[0], dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))
        eu, eo = ed.copy(), ed.copy()
        edo, euo = ed.copy(), ed.copy()

        # LOOP
        for i, ke in enumerate(self.ordered_keys):

            print(ke)
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

    def calculate_kd_layers(self):
        """
        Method to calculate the diffuse attenuation coefficient [m-1].
        :return:
        """

        mask_zero_z = np.where(self.ed["depth"] >= 0)

        Kd = np.zeros(mask_zero_z[0].shape[0] - 1, dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))

        #Kd["depth"] = self.ed["depth"][mask_zero_z]

        Kd["r"], _ = attenuation_coefficients_layers(self.ed[mask_zero_z]["r"], self.ed[mask_zero_z]["depth"])
        Kd["g"], _ = attenuation_coefficients_layers(self.ed[mask_zero_z]["g"], self.ed[mask_zero_z]["depth"])
        Kd["b"], depths = attenuation_coefficients_layers(self.ed[mask_zero_z]["b"], self.ed[mask_zero_z]["depth"])

        #mask_zero_z = np.where(depths >= 0)

        Kd['depth'] = depths

        return Kd

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
        #wl_dct = {484: 2, 544: 1, 603: 0}
        #wl_dct = {480: 2, 540: 1, 600: 0}
        lc = self.legendre_coeff[self.keys_from_depth[depth]]
        if smooth and np.any(lc):
            radiance = self.compute_legendre_polynomials(zen * np.pi / 180, lc[:, self.wl_dct[wl]])
        else:
            rad_az_avg = self.azimuthal_average(self.radiance_profile[self.keys_from_depth[depth]])
            radiance = rad_az_avg[:, self.wl_dct[wl]]

        return zen, radiance

    def get_radiance_dist_at_depth_wl(self, depth, wl):
        """

        :param depth:
        :param wl:
        :return:
        """
        #wl_dct = {484: 2, 544: 1, 603: 0}
        #wl_dct = {480: 2, 540: 1, 600: 0}
        rad_depth = self.radiance_profile[self.keys_from_depth[depth]].copy()
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

    def show_reflectance(self):
        """

        :return:
        :rtype:
        """

        fig, ax = plt.subplots(1, 1, sharey=True)

        band_name = ["r", "g", "b"]
        lstyle = ["-", "--", ":", "-."]
        cl = ["#a6cee3", "#1f78b4", "#b2df8a"]
        xlabel = ["$R~[-]$"]
        leg_lab = ["red band: 630 nm", "green band: 544 nm", "blue band: 484 nm"]

        for b, band in enumerate(band_name):

            refl = self.eu[band] / self.ed[band]
            ax.plot(refl[1:], self.ed["depth"][1:], linewidth=0.8, color=cl[b], linestyle=lstyle[b], label=leg_lab[b])

        ax.set_xlabel(xlabel[0])
        ax.legend(loc="best", frameon=False, fontsize=6)
        ax.invert_yaxis()
        #ax[b].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax[b].transAxes, size=11, weight='bold')
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


class RadClassX3:

    def __init__(self, data_path="data/QI0310_radiance.xlsx", wl_dct=None):
        """
        """

        # Save attributes
        if wl_dct is None:
            self.wl_dct = {480: 2, 540: 1, 600: 0}

        self.data_path = data_path

        # Radiance data
        self.rad_data = pandas.read_excel(self.data_path, sheet_name=None)

        self.depths = np.unique(self.rad_data["Radiance green"]["Depth"].to_numpy())
        #self.freeboard = freeboard  # cm
        #self.station = station
        #self.wl_dct = wl_dct

        self.zenith_meshgrid, self.azimuth_meshgrid, self.radiance_profile = self.format_radiance_data()

        self.legendre_coeff = self.fit_radiance_curves()

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

    def fit_radiance_curves(self):
        """

        :return:
        """

        dct_legendre_coeff = {}
        leg_deg = 5  # Degree of Legendre Polynomials
        # LOOP
        for i, ke in enumerate(self.depths):

            # Radiance at depth
            rad = self.radiance_profile[ke]

            # Cond for extrapolation
            if np.sum(rad == 0.0) >= 1000:
            #if ke >= self.freeboard:
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

        :param depth:
        :param wl:
        :return:
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
        #ax[1].set_xscale("log")
        #ax[2].set_xscale("log")

        ax[0].invert_yaxis()

        ax[0].set_ylabel("Depth [cm]")
        #fig.suptitle(self.station)
        fig.tight_layout()

        return fig, ax

    def load_excel_columns(self, sheet_name, columns):
        # Load the Excel workbook
        workbook = load_workbook(filename=self.data_path, read_only=True)

        # Select the specified sheet
        sheet = workbook[sheet_name]

        # Get the values from the selected columns
        selected_columns = []
        for column in columns:
            column_values = [cell.value for cell in sheet[column]]
            selected_columns.append(column_values)

        # Stack the selected columns horizontally to create a new array
        selected_data = np.column_stack(selected_columns)

        return selected_data

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


def attenuation_coefficients_layers(E, d):
    """

    :param E:
    :type E:
    :param d:
    :type d:
    :return:
    :rtype:
    """

    z_inter = 0.5 * (d[:-1] + d[1:])
    E_inter = 0.5 * (E[:-1] + E[1:])
    dz = (d[1:] - d[:-1]) / 100
    return -(1/E_inter) * ((E[1:] - E[:-1]) / dz), z_inter


def azimuthal_average(rad):
    """
    Average of radiance in azimuth direction.

    :return:
    """
    condzero = rad == 0
    rad2 = rad.copy()
    rad2[condzero] = np.nan
    return np.nanmean(rad2, axis=1)


def fit_f1f3(th, a1, a2, a3, eta1, eta2, eta3, mu1, mu2, mu3):
    """
    $ BEST FUNCTION
    :param th:
    :param epsilon:
    :param eta:
    :return:
    """
    mu = np.cos(th * np.pi/180)
    f1 = mu1 / ((1 - (a1 * mu)) ** eta1)
    f2 = mu2 / ((1 - (a2 * mu)) ** eta2)
    f3 = mu3 / ((1 - (a3 * mu)) ** eta3)
    return f1 + f2 + f3


def fit_f1f2(th, a1, a2, eta1, eta2, mu1, mu2):
    """
    $ BEST FUNCTION
    :param th:
    :param epsilon:
    :param eta:
    :return:
    """
    mu = np.cos(th * np.pi/180)
    f1 = mu1 / ((1 - (a1 * mu)) ** eta1)
    f2 = mu2 / ((1 - (a2 * mu)) ** eta2)
    return f1 + f2


def fit_f1(th, a1, eta1, mu1):
    """

    :param th:
    :param epsilon:
    :param eta:
    :return:
    """
    mu = np.cos(th * np.pi/180)
    f1 = mu1 / ((1 - (a1 * mu)) ** eta1)
    return f1


if __name__ == "__main__":

    # Test RadClassX3

    p = "/Users/raphaellarouche/Desktop/QI0320_radiance.xlsx"
    rdx3 = RadClassX3(data_path=p)


    # Test ImageRadiancei360x3
    #p = "/Users/raphaellarouche/Desktop/stitch_test/IMG_20230413_140157_00_002.dng"
    #p = "/Users/raphaellarouche/Desktop/IMG_20230316_112747_00_003.dng"
    #imrad = ImageRadiancei360x3(p, cam_sn="2BW7X7", medium="air", cover="cover")
    #imrad.get_radiance()
    #imrad.map_radiance(angular_resolution=1.0)

    #plt.figure()
    #plt.imshow(imrad.im_original.copy())

    #A = imrad.mapped_radiance.copy()[:, :, 1]
    #plt.figure()
    #plt.imshow(A, vmin=A.min() * 2, vmax=A.max() * 0.1)

    #az_avg = imrad.azimuthal_average(imrad.mapped_radiance.copy())

    #plt.figure()
    #plt.plot(az_avg)


    #plt.show()

    # Test
   #oden_data_list = glob.glob("D:/data-i360/field/oden-08312018/IMG*.dng")
   #im_rad = ImageRadiancei360(oden_data_list[12], "water")
   #
   ##im_rad.get_radiance(dark_metadata=True)
   ##im_rad.map_radiance(angular_resolution=1.0)
   #im_rad.get_radiance_angular_distribution()
   #im_rad.show_mapped_radiance()
   #
   ## Interpolation for the missing angles
   #B = im_rad.extrapolation_gaussian_function()
   #C = im_rad.extrapolation_legendre_polynomials()
   #
   ## Raw azimuthal average
   #D = im_rad.azimuthal_average()
   #angl = np.arange(0, 181, 1)
   #mu = np.cos(angl * np.pi/180)
   #maskzero = np.logical_not(np.isnan(D))
   #
   ## Legendre fit
   #legfit = np.polynomial.legendre.Legendre.fit(mu[maskzero[:, 0]][3:], D[maskzero[:, 0], 0][3:], 3, domain=[-1.,  1.])
   #
   ## Tyler equation
   #lnorm = D[maskzero[:, 0], 0]/D[maskzero[:, 0], 0][71]
   #t = angl[maskzero[:, 0]]
   #res_tyler_mod = curve_fit(fit_f1, t[5:], lnorm[5:], maxfev=10000)
   #
   ## plot methods
   #fig1, ax1 = plt.subplots(1, 1)
   #
   #ax1.plot(angl, D[:, 0], linewidth=1.5, label="Raw")
   #ax1.plot(np.arange(0, 181, 1), B[:, :, 0].mean(axis=1), label="Gaussian extrapolation")
   #ax1.plot(np.arange(0, 181, 1), C[:, :, 0].mean(axis=1), label="Legendre extrapolation")
   #
   #ax1.legend(loc="best")
   #
   #fig2, ax2 = plt.subplots(1, 1)
   #
   #ax2.plot(angl, D[:, 0], linewidth=1.5, label="Raw")
   #ax2.plot(angl, fit_f1(angl, *res_tyler_mod[0]) * D[maskzero[:, 0], 0][71], label="Tyler function")
   #ax2.plot(np.arccos(legfit.linspace()[0]) * 180/np.pi, legfit.linspace()[1], label="Legendre polynomials")
   #
   #ax2.legend(loc="best")
   #ax2.set_xlabel(r"$\theta$ [°]")
   #ax2.set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
   #
   #fig2.tight_layout()
   #
   ## Irradiances
   #ed = im_rad.irradiance(0, 90, planar=True, extrapolation=True)
   #eu = im_rad.irradiance(90, 180)
   #e0 = im_rad.irradiance(0, 180, planar=False)
   #
   #plt.figure()
   #plt.imshow(B[:, :, 1])
   #
   #plt.show()
