# -*- coding: utf-8 -*-
"""
"""

# Module importation
import os
import glob
import rawpy
import numpy as np
import matlab.engine
import time
import timeit
import deepdish
import matplotlib.pyplot as plt

# Other module importation
import source.processing as processing
from source.geometric_rolloff import MatlabGeometricMengine


def raw2gray(image, metadata, brightnessfactor):
    """
    Compute gray image from array [x, y, band] of raw red, green blue values.

    :param image: raw image array
    :param metadata: metadata
    :param brightnessfactor: brightness adjustment factor
    :return: stack of grey image
    """
    saturation = float(metadata["Image Tag 0xC61D"].values[0])
    blacklevel = float(metadata["Image BlackLevel"].values[0])

    im_norm = (image - blacklevel) / (saturation - blacklevel)
    im_norm *= brightnessfactor
    im_norm = np.clip(im_norm, 0, 1)
    im_norm *= 255

    return im_norm.astype(np.uint8)


def recursive_matlab_to_array(ori_dct):
    """

    :param ori_dct:
    :type ori_dct:
    :return:
    :rtype:
    """
    dic = {}
    if type(ori_dct) == dict:  # High level
        for k in ori_dct.keys():
                dic[k] = recursive_matlab_to_array(ori_dct[k])
    elif type(ori_dct) in [matlab.double, matlab.single, matlab.int8, matlab.int16, matlab.int32,
                           matlab.int64, matlab.uint8, matlab.uint16, matlab.uint32, matlab.uint8, matlab.logical]:
        return np.squeeze(np.array(ori_dct))
    else:
        return ori_dct
    return dic


if __name__ == "__main__":

    # Objects initialization
    pr = processing.ProcessImage()

    #path = "/Users/raphaellarouche/Desktop/insta360x3_02072023/2W7X7/back"
    #path = "/Users/raphaellarouche/Desktop/test_geo"
    #path = pr.folder_choice()

    # Which calibrabration ? Serial number, cover/no-cover, which lens, date, medium
    sn = "2C9JCA"
    cover = "nocover"
    wlen = "back"
    date = "20230404"
    med = "water"

    # Medium square size
    if med.lower() == "air":
        square_size = 40.0  # mm
        grayscale_factor = 3.0
    elif med.lower() == "water":
        square_size = 20.0  # mm
        grayscale_factor = 6.0
    else:
        raise ValueError("Not a valid natural medium.")

    # Path of images
    path = f"/Volumes/MYBOOK/data-i360X3/calibrations/geometric/{sn}/{cover}/{med}/{wlen}/{date}"
    list_path = glob.glob(path + "/*.dng")


    # RGGB = [0, 1, 1, 2], 0 = R, 1 = G, 2 = B
    # GBRG = [1, 2, 0, 1]

    # Loop
    data = {"i": 0, "point_x": np.array([]), "point_y": np.array([]), "image_tot": np.array([])}
    band_data = {"red": data, "green": data.copy(), "blue": data.copy()}

    for a, p in enumerate(list_path):

        print(p)
        met = pr._readDNGmetadata(p)

        rawimg = rawpy.imread(p)
        npimg = rawimg.raw_image

        if wlen.lower() == "front":
            im_wlen = npimg[:npimg.shape[0] // 2, :]
        elif wlen.lower() == "back":
            im_wlen = npimg[npimg.shape[0] // 2:, :]

        #gray = raw2gray(im_wlen, met, 3.0)
        gray = raw2gray(im_wlen, met, grayscale_factor)
        gray_dws = pr.dwnsampling(gray, "GBRG").astype(np.uint8)

        for n, k in enumerate(band_data.keys()):
            corners = pr.detect_corners_x3(gray_dws[:, :, n], vis=True)

            if np.any(corners):

                band_data[k]["i"] += 1
                band_data[k]["point_x"] = np.append(band_data[k]["point_x"], corners[:, 0])
                band_data[k]["point_y"] = np.append(band_data[k]["point_y"], corners[:, 1])
                band_data[k]["image_tot"] = np.append(band_data[k]["image_tot"], gray_dws[:, :, n].flatten())

    band_matlab_data = {"red": np.array([]), "green": np.array([]), "blue": np.array([])}
    imshape = (gray_dws.shape[0], gray_dws.shape[1])

    for ke in band_data.keys():
        band_matlab_data[ke] = np.empty((49, 2, band_data[ke]["i"]))
        band_matlab_data[ke][:, 0, :] = band_data[ke]["point_x"].reshape((band_data[ke]["i"], 49)).T
        band_matlab_data[ke][:, 1, :] = band_data[ke]["point_y"].reshape((band_data[ke]["i"], 49)).T

        #band_matlab_data[ke] = np.delete(band_matlab_data[ke], np.s_[::8], 0)
        #band_matlab_data[ke] = np.delete(band_matlab_data[ke], np.s_[6::7], 0)
        band_matlab_data[ke] = matlab.double(band_matlab_data[ke].tolist())

        band_data[ke]["image_tot"] = band_data[ke]["image_tot"].reshape((-1, imshape[0], imshape[1]))

    fp = {}
    ierror = {}
    geo = {}
    starttime = timeit.default_timer()
    eng = matlab.engine.start_matlab()
    print(f"Opening Matlab engine took : {timeit.default_timer() - starttime:.1f}")

    for n, k in enumerate(band_matlab_data.keys()):

        print("Band number {}".format(n))

        eng.cd(os.path.dirname(__file__))

        fp[k], ierror[k] = eng.scaramuzza_calibration_x3(matlab.double([imshape[0]]), matlab.double([imshape[1]]), band_matlab_data[k], matlab.double([square_size]), nargout=2)

        fp[k] = recursive_matlab_to_array(fp[k])
        ierror[k] = recursive_matlab_to_array(ierror[k])

        geo[k] = MatlabGeometricMengine(fp[k], ierror[k])
        geo[k].print_results()
        res = geo[k].get_results()

    # Save data
    save_answer = pr.save_results()
    acq_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.stat(list_path[0])[-2]))
    file_path_name = f"calibrationfiles/geometric-calibration-{sn}.h5"

    if save_answer.lower() == "y":
        if os.path.exists(file_path_name):
            savd = deepdish.io.load(file_path_name)
            tbs = {acq_time: {"fp": fp, "ierror": ierror}}
            if med in savd.keys():
                if cover in savd[med].keys():
                    if wlen in savd[med][cover].keys():
                       savd[med][cover][wlen] = tbs
                    else:
                        savd[med][cover][wlen] = tbs
                else:
                    savd[med][cover] = {f"{wlen}": tbs}
            else:
                savd[med] = {f"{cover}": {f"{wlen}": tbs}}
        else:
            savd = {f"{med}": {f"{cover}": {f"{wlen}": {acq_time: {"fp": fp, "ierror": ierror}}}}}

        deepdish.io.save(file_path_name, savd)

    plt.show()
