# -*- coding: utf-8 -*-
"""
Random classes.
"""

# Module importation
import os
import cv2
import time
import h5py
import glob
import pandas
import exifread
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


# Classes
class ProcessImage:
    """
     Class to perform some processing function on raw imagery.
    """

    def readDNG_insta360_np(self, path_name, which_image):
        """
        Function to read and separate both image of Insta360 ONE.

        :param path_name: absolute path to image in DNG format
        :param which_image: string "close" or "far"
        :return: raw image, metadata
        """

        image, metadata = self._readDNG_np(path_name)
        height = int(metadata["Image ImageLength"].values[0])
        half_height = int(height/2)

        if which_image == "close":
            im_c = image[half_height:height:1, :]
        elif which_image == "far":
            im_c = image[0:half_height:1, :]
        else:
            raise ValueError("Argument which image must be either close of far.")

        return im_c, metadata

    @staticmethod
    def _readDNGmetadata(path):
        """
        Reading metadata in DNG files.

        :param path: absolute path to file.
        :return:
        """
        with open(path, "rb") as file:
            met = exifread.process_file(file)
        return met

    def _readDNG_np(self, path):
        """
        Read raw DNG image using numpy.

        :param path: absolute path to image
        :return: tuple (raw image, metadata)
        """
        metadata = self._readDNGmetadata(path)
        rows, cols = 6912, 3456
        img = np.fromfile(path, dtype=np.uint16, count=rows * cols)
        return img.reshape((rows, cols)), metadata

    @staticmethod
    def folder_choice(initialpath="/Volumes/KINGSTON/"):
        """
        Function that open a dialog box to go trough files.

        :param initialpath: Initial path to search.
        :return: absolute path to the folder
        """
        root = tk.Tk()
        root.withdraw()
        impath = filedialog.askdirectory(parent=root, initialdir=initialpath, title="Please select a directory")
        root.destroy()
        return impath

    @staticmethod
    def imageslist(path):
        """
        Creating a list of all image starting with IMG.

        :param path: absolute path to repository
        :return:
        """

        imlist = glob.glob(path + "/IMG_*")
        imlist.sort()

        return imlist

    @staticmethod
    def imageslist_dark(path, prefix="DARK"):
        """
        Creating a list of all image starting with the prefix DARK.

        :param path: absolute path to repository
        :param prefix:
        :return:
        """

        imlist = glob.glob(path + "/" + prefix + "*")
        imlist.sort()

        return imlist

    def imagestack(self, imagelist, whichim):
        """
        Stacking each image given in the list of absolute paths (one path to every image).

        :param imagepath: list of absolute paths
        :param whichim: string "close" or "far"
        :return: tuple -- (stack of every images in 3rd dimension, exposure time, iso gain, black level)
        """

        imstack = np.empty((3456, 3456, len(imagelist)))
        iso = np.array([])
        exp = np.array([])
        blevel = np.array([])

        for n, impath in enumerate(imagelist):
            # Image number
            print("Opening image {0}".format(n+1))

            # Opening image
            im, met = self.readDNG_insta360_np(impath, whichim)
            imstack[:, :, n] = im

            exp = np.append(exp, self.extract_integrationtime(met))
            iso = np.append(iso, self.extract_iso(met))
            blevel = np.append(blevel, self.extract_blevel(met))

        return imstack, exp, iso, blevel

    def dwnsampling(self, image_mosaic, pattern, ave=True):
        """
        Function to downsample each band in the raw image (Bayer Mosaic) according to the basic pattern of the first four
        pixels. Possible values are RGGB, BGGR, GRBG or GBRG.

        :param image_mosaic: Raw image (array)
        :param pattern: Bayer pattern of the first four pixels (str)
        :param ave: averaging two consecutive row of green pixels (bool)
        :return: 3 dimension array of ave=True, tuple (r, g, b) if ave=False
        """

        if len(image_mosaic.shape) == 2:
            rind, gind, bind = self.dws_pattern(pattern)

            r = image_mosaic[rind[0]::2, rind[1]::2]
            b = image_mosaic[bind[0]::2, bind[1]::2]

            if ave:
                g = image_mosaic[gind[0, 0]::2, gind[0, 1]::2]/2 + image_mosaic[gind[1, 0]::2, gind[1, 1]::2]/2
                return np.dstack([r, g, b])  # Stack
            else:
                g = np.zeros((int(image_mosaic.shape[0]), int(image_mosaic.shape[1] / 2)))
                g[0::2, :] = image_mosaic[gind[0, 0]::2, gind[0, 1]::2]
                g[1::2, :] = image_mosaic[gind[1, 0]::2, gind[1, 1]::2]

                return r, g, b
        else:
            raise Exception("Not a valid Bayer pattern.")

    @staticmethod
    def dws_pattern(pattern):
        """
        Indexes of the first pixel for each band according to Bayer Pattern given in entry.

        :param pattern: Bayer pattern (string of capital letters R-red, G-green, B-blue)
        :return: tuples of arrays of the indexes (red, green, blue)
        """
        if pattern == "RGGB":
            rind = np.array([0, 0])
            bind = np.array([1, 1])
            gind = np.array([[0, 1], [1, 0]])
        elif pattern == "BGGR":
            rind = np.array([1, 1])
            bind = np.array([0, 0])
            gind = np.array([[0, 1], [1, 0]])
        elif pattern == "GRBG":
            rind = np.array([0, 1])
            bind = np.array([1, 0])
            gind = np.array([[0, 0], [1, 1]])
        elif pattern == "GBRG":
            rind = np.array([1, 0])
            bind = np.array([0, 1])
            gind = np.array([[0, 0], [1, 1]])
        else:
            raise Exception("Not a valid Bayer pattern.")

        return rind, gind, bind

    @staticmethod
    def avg2row(array):
        """
        Fonction to average data every two row of the green pixels.
        :param array: initial numpy array
        :return: new average array
        """
        return 0.5 * (array[0::2] + array[1::2])

    @staticmethod
    def interpolation(wl, data):
        """
        Simple 1D linear interpolation.

        :param wl: Wavelength for interpolation
        :param data: Tuple of measured wavelength and quantum efficiency  (wavelength, QE)
        :return:
        """
        wl_m, qe_m = data
        return np.interp(wl, wl_m, qe_m)

    @staticmethod
    def ratio2float(array):
        """
        Some insta360 ONE comes as a string in format : A / B. This function computes the float values from the ratio of
        A over B

        :param array:
        :return:
        """
        if isinstance(array[0], exifread.utils.Ratio):
            spl = [str(j).split(sep="/") for j in array]
            return [float(i[0])/float(i[1]) if len(i) == 2 else float(i[0]) for i in spl]
        else:
            raise TypeError("Wrong data type inside array.")

    @staticmethod
    def show_metadata_insta360(met):
        """
        Function to print metadata of insta360 ONE camera.

        :param met: metadata in dict format
        :return:
        """
        for i, j in zip(met.keys(), met.values()):
            print("{0} : {1}".format(i, j))

    @staticmethod
    def CFApattern_insta360(array):
        """
        Returning CFA pattern array of camera insta360 ONE from metadata.

        :param array: Array corresponding to exif tage of CFA pattern.
        :return: Matrix of bayer CFA pattern.
        """
        cfastr = ""
        exifvalues = {0: "R", 1: "G", 2: "B"}
        for i in array:
            cfastr += exifvalues[i]
        return cfastr

    @staticmethod
    def extract_integrationtime(metadata):
        """
        Extracting time of exposure from metadata.

        :param metadata: insta360 metadata
        :return: exposure time [s]
        """
        exptime = str(metadata['Image ExposureTime']).split("/")
        if len(exptime) == 2:
            exptime = float(exptime[0]) / float(exptime[1])
        else:
            exptime = float(exptime[0])
        return exptime

    @staticmethod
    def extract_iso(metadata):
        """
        Outputs iso gain speed from metadata.

        :param metadata: metadata dictionary
        :return:
        """
        return float(str(metadata["Image ISOSpeedRatings"]))

    @staticmethod
    def extract_blevel(metadata):
        """
        Outputs black level from metadata.

        :param metadata: metadata dictionary
        :return:
        """
        # return float(str(metadata["Image Tag 0xC61A"]))
        return float(str(metadata["Image BlackLevel"]))

    @staticmethod
    def gain_linear(gain_db):
        """
        Function to return the gain in linear value.

        :param gain_db: gain [db]
        :return: linear gain
        """
        return 10**(gain_db/20)

    @staticmethod
    def exposure_second(exposure_us):
        """
        Function transforming exposure time in us to exposure time in s.

        :param exposure_us: exposure time [us]
        :return: exposure time [s]
        """
        return exposure_us*1E-6

    @staticmethod
    def raw2gray(image, metadata, brightnessfactor):
        """
        Compute gray image from array [x, y, band] of raw red, green blue values.

        :param image: raw image array
        :param metadata: metadata
        :param brightnessfactor: brightness adjustment factor
        :return: stack of grey image
        """
        saturation = float(str(metadata["Image Tag 0xC61D"]))
        #blacklevel = float(str(metadata["Image Tag 0xC61A"]))
        blacklevel = float(str(metadata["Image BlackLevel"]))

        im_norm = (image - blacklevel) / (saturation - blacklevel)
        im_norm *= brightnessfactor
        im_norm = np.clip(im_norm, 0, 1)
        im_norm *= 255

        return im_norm.astype(np.uint8)

    def imageshape(self, path, which):
        """
        Get the shape of the image given by absolute path.

        :param path: absolute path (str)
        :param which: which optic ('close' or 'far' as str)
        :return: shape (array)
        """
        im_sh, met_sh = self.readDNG_insta360_np(path, which)
        gray = self.raw2gray(im_sh, met_sh, 3.0)
        gray_dws = self.dwnsampling(gray, "RGGB").astype(np.uint8)

        return gray_dws.shape

    @staticmethod
    def polynomial_fit_forcedzero(x, a1, a2, a3, a4):
        """
        Polynomial fit with a0 forced to zero. This is mostly used for geometric calibration.

        :param x:
        :param a1:
        :param a2:
        :param a3:
        :param a4:
        :return:
        """
        return a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4

    @staticmethod
    def rolloff_polynomial(x, a0, a2, a4, a6, a8):
        """
        Polynomial fit with even coefficients (degree 0 to 8) for roll-off fitting.

        :param x:
        :param a0:
        :param a2:
        :param a4:
        :param a6:
        :param a8:
        :return:
        """
        return a0 + a2*x**2 + a4*x**4 + a6*x**6 + a8*x**8

    @staticmethod
    def detect_corners(img, vis=False):

        height, width = img.shape  # Shape of image

        # Refinement criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # Resizing for speed optimization
        resizefactor = 2
        img_dwnsca = cv2.resize(img, (int(width / resizefactor), int(height / resizefactor)))  # Resize image

        ret, corners_dwnsca = cv2.findChessboardCorners(img_dwnsca, (8, 6), None)

        print(ret)

        corners_refine = False
        if ret:
            corners = corners_dwnsca
            corners[:, 0][:, 0] = corners_dwnsca[:, 0][:, 0] * resizefactor
            corners[:, 0][:, 1] = corners_dwnsca[:, 0][:, 1] * resizefactor

            # Refinement
            corners_refine = cv2.cornerSubPix(img, np.float32(corners), (5, 5), (-1, -1), criteria)
            corners_refine = corners_refine[:, 0]

            # Visualisation
            if vis:
                img = cv2.drawChessboardCorners(img, (8, 6), corners_refine, ret)
                cv2.imshow("image", cv2.resize(img, (int(width / 4), int(height / 4))))
                cv2.waitKey(1)

        return corners_refine

    @staticmethod
    def region_properties(image, minimum, *maximum):
        """

        :param image:
        :param minimum:
        :param maximum:
        :return:
        """
        binary = image > minimum
        if maximum:
            binary = (image > minimum) & (image < maximum)

        # Region properties of binary image
        labels = label(binary.astype(int))
        region_properties = regionprops(labels)

        # Sorting
        sorted_region_properties = sorted(region_properties, key=lambda region: region.area)

        return binary, sorted_region_properties[::-1]

    @staticmethod
    def save_results(text="Do you want to save this calibration?"):
        """
        Input from user to prompt if he wants to save data.

        :return: answer (y:yes or n: no), (str)
        """
        ans = ""
        while ans not in ["y", "n"]:
            ans = input("{0} (yes: y, no: n)".format(text))

        return ans.lower()

    @staticmethod
    def create_hdf5_dataset(path, group, dataname, dat):
        """
        Function to save calibration data in hdf5 files. The absolute spectral radiance and the immersion factor are
        saved using this function.

        :param path: relative path to saving directory (str)
        :param group: group name (str)
        :param dataname: data name (str)
        :param dat: data
        :return: ...
        """
        datapath = group + "/" + dataname
        with h5py.File(path) as hf:
            if datapath in hf:
                d = hf[datapath]  # load the data
                d[...] = dat
            else:
                hf.create_dataset(group + "/" + dataname, data=dat)

    @staticmethod
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


class FlameSpectrometer:
    """

    """

    def __init__(self, general_path):

        self.path_calibration = glob.glob(general_path + "/" + "spectrometer/calibration*.hdf5")[0]
        self.path_sources = glob.glob(general_path + "/" + "spectrometer/sources*.hdf5")[0]
        self.path_cops = general_path + "/" + "cops-acquisitions"

        # Calibration lamp
        self.oo_lampdata = pandas.read_csv(glob.glob(os.path.dirname(__file__) + "/data/*.lmp")[0], names=["wl", "flux"], header=None, sep="\t")
        self.oo_lamp_uncertainty = np.array([[400, 0.091], [500, 0.074], [600, 0.070], [800, 0.069], [1000, 0.069]])
        # ** The lamp uncertainties are the ones given by the manufacturer

        self.info = self.readinfo(general_path + "/")

        # C-OPS uncertainty
        self.cops_unc = np.array([0.025])  # 2.5 %

        # Spectrometer calibration data
        self.spectro_calibration_co = np.array([])
        self.spectro_calibration_wl = np.array([])
        self.spectro_calibration_unc = np.array([])

    def open_hdf5files(self, path, groupname, dname="light", verbose=True):
        """

        :param path:
        :param groupname:
        :param dname:
        :param verbose:
        :return:
        """

        with h5py.File(path, "r") as fi:

            if groupname + "/" + "dark" in fi:
                d_d = fi[groupname + "/" + "dark"][:].mean(axis=1)
            else:
                d_d = fi[groupname + "/" + dname].attrs["Dark"]

            d_c = fi[groupname + "/" + dname][:]
            wave = fi[groupname + "/" + dname].attrs["Wavelengths"]
            t = fi[groupname + "/" + dname].attrs["Integration time us"]
            t *= 1e-6  # us to s
            epoch = fi[groupname + "/" + dname].attrs["Date"]

            if verbose:
                self.verbose_spectro(fi[groupname + "/" + dname].attrs.items())

        return wave, d_d, d_c, t, epoch

    def calibration_coefficient(self, datname):
        """

        :param datname:
        :return:
        """
        w, d_d, d_l, t, _ = self.open_hdf5files(self.path_calibration, self.info["Calibration"], dname=datname)
        d_d_dsubs = d_l - d_d[:, None]

        finterpo = np.interp(w, self.oo_lampdata["wl"], self.oo_lampdata["flux"])
        s_cal_c = (finterpo * t) / d_d_dsubs.mean(axis=1)

        cond = w <= self.oo_lampdata["wl"].min()
        s_cal_c[cond] = 0

        # Uncertainty
        flux_unc = np.interp(w, self.oo_lamp_uncertainty[:, 0], self.oo_lamp_uncertainty[:, 1])
        dark_unc = d_d_dsubs.std(axis=1) / d_d_dsubs.mean(axis=1)
        s_cal_c_unc = self.multiplication_division_unc_propagation((flux_unc, dark_unc))

        self.spectro_calibration_co = s_cal_c
        self.spectro_calibration_wl = w
        self.spectro_calibration_unc = s_cal_c_unc

        return s_cal_c, s_cal_c_unc

    def source_spectral_radiance(self, sourcename, copswl, copsfile):
        """

        :param sourcename:
        :param copswl:
        :param copsfile:
        :return:
        """

        w_s, dark_s, counts_s, t_s, _ = self.open_hdf5files(self.path_sources, self.info["Sources"], dname=sourcename)

        count_s_ds = np.mean((counts_s - dark_s[:, None]) / t_s, axis=1)
        count_s_calib = count_s_ds * self.spectro_calibration_co

        cops_data = self.create_cops_radiance(self.path_cops, copswl)
        argwl = np.argmin(np.absolute(w_s - cops_data["wl"][copsfile]))
        c_cal_cops = cops_data["DN_avg"][copsfile] / count_s_calib[argwl]

        # Uncertainty
        count_s_unc = np.std((counts_s - dark_s[:, None]) / t_s, axis=1) / count_s_ds
        count_s_calib_unc = self.multiplication_division_unc_propagation((count_s_unc, self.spectro_calibration_unc))
        c_cal_cops_unc = self.multiplication_division_unc_propagation((self.cops_unc, count_s_calib_unc[argwl]))

        return w_s, count_s_calib * c_cal_cops, self.multiplication_division_unc_propagation(
            (count_s_calib_unc, c_cal_cops_unc)), copswl[copsfile], cops_data["DN_avg"][copsfile]

    @staticmethod
    def create_cops_radiance(path, wl_list):
        """
        Function to recreate spectral radiance from C-OPS data.

        :param path: absolute path to data (string)
        :param wl_list: wanted wavelength (list)
        :return:
        """
        dict_header = {380: 'LuZ380 (µW/(sr cm² nm))',
                       395: 'LuZ395 (µW/(sr cm² nm))',
                       412: 'LuZ412 (µW/(sr cm² nm))',
                       443: 'LuZ443 (µW/(sr cm² nm))',
                       465: 'LuZ465 (µW/(sr cm² nm))',
                       490: 'LuZ490 (µW/(sr cm² nm))',
                       510: 'LuZ510 (µW/(sr cm² nm))',
                       532: 'LuZ532 (µW/(sr cm² nm))',
                       555: 'LuZ555 (µW/(sr cm² nm))',
                       560: 'LuZ560 (µW/(sr cm² nm))',
                       589: 'LuZ589 (µW/(sr cm² nm))',
                       625: 'LuZ625 (µW/(sr cm² nm))',
                       665: 'LuZ665 (µW/(sr cm² nm))',
                       683: 'LuZ683 (µW/(sr cm² nm))',
                       694: 'LuZ694 (µW/(sr cm² nm))',
                       710: 'LuZ710 (µW/(sr cm² nm))',
                       765: 'LuZ765 (µW/(sr cm² nm))',
                       780: 'LuZ780 (µW/(sr cm² nm))',
                       875: 'LuZ875 (µW/(sr cm² nm))'}

        path_copsfiles = glob.glob(path + "/*.tsv")
        path_copsfiles.sort()

        data = np.empty((len(wl_list), 1),
                        dtype=[("fname", "U27"), ("wl", "float32"), ("DN_avg", "float32"), ("DN_std", "float32")])

        for n, val in enumerate(zip(path_copsfiles, wl_list)):
            fpath, wl = val
            filename = os.path.basename(fpath)
            df1 = pandas.read_csv(fpath, sep="\t", header=0, encoding="ISO-8859-1")
            curr_wl_data = df1[dict_header[wl]]

            curr_wl_data /= 100.0  # uW sr-1 cm-2 nm-1 to W sr-1 m-2 nm-1

            data[n, :]["fname"] = filename
            data[n, :]["wl"] = wl
            data[n, :]["DN_avg"] = curr_wl_data.mean()
            data[n, :]["DN_std"] = curr_wl_data.std()

        return np.sort(data[:, 0], order="wl")

    @staticmethod
    def verbose_spectro(attrs):
        """

        :param attrs:
        :return:
        """
        for n, v in attrs:
            if n == "Date":
                print(n + ": " + time.ctime(v))
            elif n == "Wavelengths":
                continue
            else:
                print(n + ": " + str(v))

    @staticmethod
    def readinfo(path):
        """
        Reading info.txt file with data group name.

        :param path: absolute path to data (str)
        :return:
        """
        d = {}
        with open(path + "info.txt", "r") as f:
            for line in f:
                k, v = line.strip("\n").split(":")
                if k == "Sources name":
                    d[k] = v.split(",")
                else:
                    d[k] = v
        return d

    @staticmethod
    def multiplication_division_unc_propagation(values):
        """

        :param values:
        :return:
        """
        tot_unc = np.zeros(values[0].shape)

        for v in values:
            tot_unc += v ** 2

        return tot_unc ** (1 / 2)


class FigureFunctions:
    """
    Functions to configure figures.
    """
    @staticmethod
    def set_size(width=443.86319, fraction=1, subplots=(1, 1), height_ratio=0.6180339887498949):
        """
        Set figure dimensions to avoid scaling in LaTeX.

        :param width: float, Document textwidth or columnwidth in pts
        :param fraction: float, optional
        :param subplots: subplots configuration of the figure
        :param height_ratio: ratio height over width

        :return: tuple, Dimensions of figure in inches
        """
        # Width of figure (in pts)
        fig_width_pt = width * fraction

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt

        # Figure height in inches
        if subplots[0] == 1:
            fig_height_in = fig_width_in * height_ratio
        else:
            fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])

        return fig_width_in, fig_height_in

    @staticmethod
    def set_size_subplot(width, subp=(1, 1)):
        """Set figure dimensions to avoid scaling in LaTeX.

        Parameters
        ----------
        width: float
               Document textwidth or columnwidth in pts
        fraction: float, optional
                  Fraction of the width which you wish the figure to occupy

        Returns
        -------
        fig_dim: tuple
                 Dimensions of figure in inches
        """
        # Width of figure (in pts)
        fig_width_pt = width

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5 ** .5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt * subp[1]

        # Figure height in inches
        fig_height_in = plt.rcParams.get('figure.figsize')[1] * subp[0]
        # fig_height_in = fig_width_in * golden_ratio * subp[0]

        return fig_width_in, fig_height_in


if __name__ == "__main__":

    pim = ProcessImage()
    f_spectr = FlameSpectrometer("/Volumes/MYBOOK/data-i360-tests/calibrations/absolute-radiance/09082020")

    plt.show()

