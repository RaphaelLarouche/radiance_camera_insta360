# -*- coding: utf-8 -*-
"""
Classes for geometric calibration methods of insta360 ONE.
"""

# Module importation
import numpy as np
import matlab.engine
import matplotlib.cm
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Other modules
from source.processing import ProcessImage


# Classes
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


class MatlabGeometric(OpenMatlabFiles):
    """
    Class with methods to open and process files using Matlab fisheye calibration OcamCalib object.
    """

    def __init__(self, fisheye_parameters_path):

        # Opening fisheye params
        fisheye_params = self.loadmat(fisheye_parameters_path)  # Same structure as matlab calibration object
        k = list(fisheye_params.keys())
        self.fisheye_params = fisheye_params[k[3]]

        # Intrinsics wanted variables
        self.intrinsics = self.fisheye_params["Intrinsics"]
        self.imsize = self.intrinsics["ImageSize"].astype(float)
        self.center = self.intrinsics["DistortionCenter"].astype(float)
        self.mapping_coefficients = self.intrinsics["MappingCoefficients"].astype(float)

        # Inverse mapping
        self.up, self.vp, self.radial = self.euclidean_distance()

    def recursive_matlab2array(self, dic):
        """
        Function to transpose matlab.double to numpy array.

        :param dic:
        :return:
        """
        for k in dic.keys():
            if isinstance(dic[k], dict):
                dic[k] = self.recursive_matlab2array(dic[k])
            elif isinstance(dic[k], matlab.double):
                dic[k] = np.squeeze(np.array(dic[k]))
        return dic

    @staticmethod
    def imagingfunction(rdistance, MapC):
        """
        Imaging function as implemented in Scaramuzza et al.

        :param MapC: Mapping Coefficient (sorted in increase order, no first order)
        :param radistance: Radial distance
        :return:
        """
        return MapC[0] + (MapC[1] * rdistance ** 2) + (MapC[2] * rdistance ** 3) + (MapC[3] * rdistance ** 4)

    def affine_transfo(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """

        distortion_matrix = self.intrinsics["StretchMatrix"]
        contxy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1).T

        res = np.dot(distortion_matrix, contxy)

        return res[0, :].reshape(x.shape) + self.center[0], res[1, :].reshape(y.shape) + self.center[1]

    def inverse_affine_transfo(self, xprim, yprim):
        """

        :param xprim:
        :param yprim:
        :return:
        """

        xprim -= self.center[0]
        yprim -= self.center[1]

        inv_distortion_matrix = np.linalg.inv(self.intrinsics["StretchMatrix"])

        contxy = np.concatenate((xprim.reshape(-1, 1), yprim.reshape(-1, 1)), axis=1).T

        res = np.dot(inv_distortion_matrix, contxy)

        return res[0, :].reshape(xprim.shape), res[1, :].reshape(yprim.shape)

    def angular_coordinates(self):
        """
        Function to calculate the angular coordinates of each pixels.

        :param intrinsics: Fisheye calibration intrinsics parameters.
        :return: radial distance, zenith, azimuth (tuple)
        """

        rdistance = self.radial

        g = self.imagingfunction(rdistance, self.mapping_coefficients)

        zen = np.arctan2(rdistance, g) * 180 / np.pi
        zen[zen < 0] = zen[zen < 0] + 180

        az = np.arctan2(self.up, self.vp) * 180 / np.pi
        az[az < 0] = az[az < 0] + 360

        return rdistance, zen, az

    def euclidean_distance(self):
        """

        :return:
        """

        x, y = np.meshgrid(np.arange(1, self.imsize[1] + 1, 1), np.arange(1, self.imsize[0] + 1, 1))  # Meshgrid
        uprim, vprim = self.inverse_affine_transfo(x, y)  # Inverse affine transformation

        return uprim, vprim, np.sqrt((uprim ** 2) + (vprim ** 2))

    @staticmethod
    def reprojection_errors(fisheyeParams):
        """
        Function to outputted the radial distance of chessboard points (relative to center) detected and reprojected.

        :param fisheyeParams: Data which is a Matlab Structure containing all paramters related to the fisheye calibration.
        :return: (radial_distance - reprojection points, radial_distance_fitted_points - corner detection algorithm)
        """

        # Finding radial distance from center of each reprojected points
        reprojection_points = fisheyeParams["ReprojectedPoints"]
        intrinsics = fisheyeParams["Intrinsics"]
        xcenter, ycenter = intrinsics["DistortionCenter"][0], intrinsics["DistortionCenter"][1]

        radial_distance = np.sqrt((reprojection_points[:, 0, :] - xcenter) ** 2 +
                                  (reprojection_points[:, 1, :] - ycenter) ** 2)

        # Reprojection mean error per image (eucledian distance between corners detected and reprojections)
        reprojection_error = fisheyeParams["ReprojectionErrors"]

        eucledian_error = np.sqrt(reprojection_error[:, 0, :] ** 2 + reprojection_error[:, 1, :] ** 2)

        mean_x = np.mean(reprojection_error[:, 0, :], axis=0)  # Mean x
        mean_y = np.mean(reprojection_error[:, 1, :], axis=0)  # Mean y
        mean_e = np.mean(eucledian_error, axis=0)

        fitted_points = reprojection_points + reprojection_error

        radial_distance_fitted_points = np.sqrt((fitted_points[:, 0, :] - xcenter) ** 2 +
                                                (fitted_points[:, 1, :] - ycenter) ** 2)

        return radial_distance, radial_distance_fitted_points, mean_x, mean_y, mean_e, eucledian_error

    def get_results(self):
        """
        Output useful results.

        :return: (r: radial distance, zen: zenith angle, rmap: reprojected points in radial distance,
                 residuals: residuals in degrees, n_im: number of images) - tuple
        """
        r, zen, _ = self.angular_coordinates()

        cond = zen < 120
        r = r[cond]
        zen = zen[cond]

        rmap, rfitted, _, _, _, _ = self.reprojection_errors(self.fisheye_params)

        # Residuals in degrees
        ang_map = np.arctan2(rmap, self.imagingfunction(rmap, self.mapping_coefficients)) * 180 / np.pi
        ang_fitted = np.arctan2(rfitted, self.imagingfunction(rfitted, self.mapping_coefficients)) * 180 / np.pi
        residuals = abs(ang_map - ang_fitted)

        # Sorting
        argsort_r = np.argsort(r)

        # Number of images
        n_im = self.fisheye_params["ReprojectionErrors"].shape[2]

        return r[argsort_r], zen[argsort_r], rmap, residuals, n_im

    def plot_results(self):
        """

        :return:
        """
        # Calibration results
        r, zen, rmap, residuals, Nim = self.get_results()

        # Interpolation
        r_reduced = np.linspace(0, rmap.max() * 1.1, 100)
        z_reduced = np.interp(r_reduced, r, zen)

        # Plots
        fig1 = plt.figure(figsize=(12,  3.57))
        ax1 = [fig1.add_subplot(1, 3, 2), fig1.add_subplot(1, 3, 3)]
        ax1.append(fig1.add_subplot(1, 3, 1, projection="3d"))

        # Axe 1
        ax1[0].plot(r_reduced, z_reduced, linewidth=1.5, linestyle="-", color="black")
        ax1[0].set_xlabel(r"Radial distance [px]")
        ax1[0].set_ylabel(r"Scene angle [$\degree$]")

        # Axe 2
        text_med = "Residuals median: {0:.3f}˚".format(np.median(residuals))
        ax1[1].scatter(rmap.ravel(), residuals.ravel(), marker="o", s=8, edgecolor="black", facecolor="none", label="corners")
        ax1[1].set_yscale("log")

        ax1[1].set_xlabel(r"Radial distance [px]")
        ax1[1].set_ylabel(r"Residuals [˚]")
        ax1[1].legend(loc=4, fontsize=11)

        # Axe 3
        self.draw_targets(ax1[2])

        return fig1, ax1

    def plot_projection_residual(self):
        """

        :return:
        """
        r, zen, rmap, residuals, Nim = self.get_results()

        # Figure
        fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

        # Interpolation
        r_dws = np.linspace(0, rmap.max() * 1.1, 100)
        z_dws = np.interp(r_dws, r, zen)

        # Ax1
        ax[0].plot(r_dws, z_dws, "k")
        ax[0].set_xlabel("$r$ [px]")
        ax[0].set_ylabel(r"$\theta$ [˚]")

        # Ax2
        text_med = "Residuals median: {0:.3f}˚".format(np.median(residuals))
        ax[1].scatter(rmap.ravel(), residuals.ravel(), marker="o", s=8, edgecolor="black", facecolor="none")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("$r$ [px]")
        ax[1].set_ylabel(r"Residuals [˚]")

        return fig, ax

    def plot_results_axes(self, ax1, ax2, cl="black", mark="o", lab=""):
        """

        :param ax1:
        :param ax2:
        :param cl:
        :param mark:
        :param lab:
        :return:
        """
        # Calibration results
        r, zen, rmap, residuals, Nim = self.get_results()

        # Interpolation
        r_reduced = np.linspace(0, rmap.max() * 1.1, 100)
        z_reduced = np.interp(r_reduced, r, zen)

        # Axe 1
        ax1.plot(r_reduced, z_reduced, linewidth=1.5, linestyle="-", color=cl, label=lab)
        ax1.set_xlabel(r"Radial distance $r$ [px]")
        ax1.set_ylabel(r"Scene angle $\theta$ [$\degree$]")
        ax1.legend(loc="best")

        # Axe 2
        ax2.scatter(rmap.ravel(), residuals.ravel(), marker=mark, s=8, edgecolor=cl, facecolor="none", label="{0} acquisitions, residuals median: {1:.3f}˚".format(Nim, np.median(residuals)))
        ax2.set_yscale("log")

        ax2.set_xlabel(r"Radial distance $r$ [px]")
        ax2.set_ylabel(r"$\theta$ residuals [˚]")
        ax2.legend(loc="best")

        return ax1, ax2

    def draw_targets(self, ax):
        """

        :param ax:
        :return:
        """
        R = self.fisheye_params["RotationMatrices"]
        T = self.fisheye_params["TranslationVectors"]
        WP = self.fisheye_params["WorldPoints"]
        WP = np.c_[WP, np.zeros(WP.shape[0])]

        A = WP.reshape((np.unique(WP[:, 0]).shape[0], -1, 3))

        cm = matplotlib.cm.bone(np.linspace(0, 1, R.shape[2]))
        ch = cm.shape[0]

        for i in range(R.shape[2]):
            C = WP.dot(R[:, :, i]) + T[i, :]
            C = C.reshape(A.shape)
            vertex = np.array([C[0, 0, :], C[0, -1, :], C[-1, -1, :], C[-1, 0, :]])
            ax.add_collection3d(Poly3DCollection([list(zip(vertex[:, 0], vertex[:, 2], vertex[:, 1]))], facecolors=cm[int(ch/2), :], edgecolors="k", alpha=0.5))

        ax.scatter(0, 0, 0, marker="o", c="k")

        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        ax.set_zlabel("$y$ [mm]")

        return ax

    def geometric_curvefit(self, radial, angles):
        """
        Curve fit for geometric calibration using polynomial_fit_forcedzero.

        :param radial:
        :param angles:
        :return:
        """
        return curve_fit(self.polynomial_fit_forcedzero, radial, angles)

    def inverse_mapping(self):
        """

        :return:
        """

        r, zen, _ = self.angular_coordinates()

        popt, pcov = self.geometric_curvefit(zen.ravel()[::100000], r.ravel()[::100000])

        rsquared = self.rsquare(self.polynomial_fit_forcedzero, popt, pcov, zen.ravel()[::100000], r.ravel()[::100000])

        return popt, pcov, rsquared

    def rsquare(self, func, popt, covmat, x, y):
        """

        :param func:
        :param popt:
        :param covmat:
        :param x:
        :param y:
        :return:
        """
        # Std of coefficient parameters
        perr = np.sqrt(np.diag(covmat))

        # Rsquare
        residuals = y - func(x, *popt)
        rsquared = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)

        return rsquared

    @staticmethod
    def points_3d(zeni, azi):
        """

        :param zeni: Zenith 2D array in radian.
        :param azi: Azimuth 2D array in radian.
        :return:
        """
        return np.sin(zeni) * np.cos(azi), np.sin(zeni) * np.sin(azi), np.cos(zeni)

    @staticmethod
    def polynomial_fit_forcedzero(x, a1, a2, a3, a4):
        """
        Polynomial fit with a0 forced to zero for geometric calibration.

        :param x:
        :param a1:
        :param a2:
        :param a3:
        :param a4:
        :return:
        """
        return a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4


class MatlabGeometricMengine(MatlabGeometric):
    """

    """
    def __init__(self, fisheye_parameters, fisheye_intrinsic_error):
        # Intrinsics
        self.fisheye_params = self.recursive_matlab2array(fisheye_parameters)
        self.intrinsic_errors = self.recursive_matlab2array(fisheye_intrinsic_error)

        # Intrinsics wanted variables
        self.intrinsics = self.fisheye_params["Intrinsics"]
        self.imsize = self.intrinsics["ImageSize"].astype(float)
        self.center = self.intrinsics["DistortionCenter"].astype(float)
        self.mapping_coefficients = self.intrinsics["MappingCoefficients"].astype(float)

        # Inverse mapping
        self.up, self.vp, self.radial = self.euclidean_distance()

    def print_results(self):
        _, _, rmap, residuals, Nim = self.get_results()

        print("Image number: {0}\n".format(int(Nim)))
        print("Mapping coefficients: {}".format(str(self.mapping_coefficients)))
        print("Mapping coefficient standard error: {}\n".format(str(self.intrinsic_errors["MappingCoefficientsError"])))
        print("Distortion center: {}".format(str(self.center)))
        print("Distortion center error: {}\n".format(str(self.intrinsic_errors["DistortionCenterError"])))
        print("Mean reprojection error: {0:.3f} ˚".format(residuals.mean()))
        print("Mean reprojection error: {0:.3f} px".format(self.fisheye_params["MeanReprojectionError"]))


class RolloffFunctions(ProcessImage):
    """

    """
    def __init__(self, fp, ierror, lens):

        # Geometric calibration
        self.geo = {}

        for ke in fp.keys():
            self.geo[ke] = MatlabGeometricMengine(fp[ke], ierror[ke])

        # Which lens
        self.which_lens = lens

    def rolloff_centroid_water(self, imlist, nremove, npixel=15, azimuth="0"):
        """

        :param imlist:
        :param nremove:
        :param npixel:
        :param azimuth:
        :return:
        """
        if nremove > 0:
            imlist = imlist[nremove:-nremove]

        # Pre-allocation
        ims = np.array(np.round(self.geo["red"].imsize)).astype(int)
        imtotal = np.zeros((ims[0], ims[1], 3))

        centroid = np.empty((len(imlist), 3), dtype=[("y", "float32"), ("x", "float32")])
        rolloff = np.empty((len(imlist), 3), dtype=[("a", "float32"), ("DN_avg", "float32"), ("DN_std", "float32")])

        centroid.fill(np.nan)
        rolloff.fill(np.nan)

        for n, path in enumerate(imlist):
            print("Processing image number {0}".format(n))

            im_dws, metadata = self.initial_process_i360(path)

            # Gain and integration time
            print(self.extract_iso(metadata))
            print(self.extract_integrationtime(metadata))

            # Region properties for the centroids
            bin, region_properties = self.region_properties(im_dws[:, :, 0], 1E3, 1E4)  # Red image

            # Image total
            imtotal += im_dws

            # Filtering regionproperties using centroid position
            if azimuth == "0":
                region_properties = [reg for reg in region_properties if reg.centroid[0] > (ims[0] // 2 - 30)]
            elif azimuth == "90":
                region_properties = [reg for reg in region_properties if (ims[1] // 2 + 200) > reg.centroid[1] > (ims[1] // 2 - 200)]

            if region_properties:
                ke = {0: "red", 1: "green", 2: "blue"}

                for j, k in enumerate(self.geo.keys()):

                    _, zen_dwsa, _ = self.geo[ke[j]].angular_coordinates()

                    yc, xc = region_properties[0].centroid

                    # To be changed for other type of roll-off processing
                    _, data = self.values_around_centroid(im_dws[:, :, j], (yc, xc), npixel)  # Using 15 pixels

                    centroid[n, j] = yc, xc  # storing centroid
                    rolloff[n, j] = zen_dwsa[int(round(yc)), int(round(xc))], np.mean(data), np.std(data)

        return imtotal, rolloff, centroid

    def initial_process_i360(self, path):
        """

        :param path:
        :return:
        """
        # Reading data
        im_op, metadata = self.readDNG_insta360_np(path, self.which_lens)
        im_op = im_op.astype(float)

        # Read noise removal
        im_op -= float(str(metadata["Image BlackLevel"]))

        # Downsampling
        im_dws = self.dwnsampling(im_op, "RGGB", ave=True)

        return im_dws, metadata

    @staticmethod
    def values_around_centroid(image, centroid, radius):
        """
        Taking the image, a centroid from a connected region in the threshold image (region properties), the function
        return the data around the centroid according to a given radius.

        Can be done for image with stack RGB data? To be tested

        :param image: Image
        :param centroid: Centroid from region properties with scikit-image
        :param radius: Radius around centroid
        :return:
        """
        # Rounding centroid
        centroid_y, centroid_x = round(centroid[0]), round(centroid[1])
        imshape = image.shape
        # Pixel coordinates
        grid_x, grid_y = np.meshgrid(np.arange(0, imshape[1], 1), np.arange(0, imshape[0], 1))
        # Subtraction of centroid to pixel coordinates
        ngrid_x, ngrid_y = grid_x - centroid_x, grid_y - centroid_y

        # Norm calculation
        norm = np.sqrt(ngrid_x ** 2 + ngrid_y ** 2)
        # Binary image of norm below or equal to radius
        bin = norm <= radius

        return bin, image[bin]

    def rolloff_curvefit(self, angles, rolloff):
        """
        Curve fit for roll-off.

        :param angles:
        :param rolloff:
        :return:
        """

        popt, pcov = curve_fit(self.rolloff_polynomial, angles, rolloff)
        rsquared, perr = self.rsquare(self.rolloff_polynomial, popt, pcov, angles, rolloff)

        return popt, pcov, rsquared, perr

    def rsquare(self, func, popt, covmat, x, y):
        """

        :param func:
        :param popt:
        :param covmat:
        :param x:
        :param y:
        :return:
        """
        # Std of coefficient parameters
        perr = np.sqrt(np.diag(covmat))

        # Rsquare
        residuals = y - func(x, *popt)
        rsquared = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Display results
        print("rsquared = {0:.8f}".format(rsquared))
        res = ""
        param = func.__code__.co_varnames
        for i in zip(param[1:], popt, perr):
            res += "%s: %.4E (%.4E)\n" % i
        print(res)

        return rsquared, perr


if __name__ == "__main__":

    plt.show()
