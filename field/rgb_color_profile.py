"""

"""

# Module importation
import glob
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from field.baiedeschaleurs2022.bdc_process_stations import get_ice_freeboard
from source.radiance import RadClass


def get_iso_exp(pil_obj):
    """

    :param pil_obj:
    :return:
    """

    data_exif = pil_obj._getexif()

    exif_exp = data_exif[33434]  # Exposure
    exif_iso = data_exif[34867]

    if isinstance(exif_exp, bytes):
        exif_exp = exif_exp.decode()
    if isinstance(exif_iso, bytes):
        exif_iso = exif_iso.decode()

    return float(exif_exp), int(exif_iso)


def process_jpg(images_list):
    """

    :param images_list:
    :return:
    """

    center = np.array([3456 // 2, 6912 // 2])
    num_px = 51  # 51 x 51

    depth_image = np.zeros((len(images_list), 1, 3))
    depth = []

    exp = []
    iso = []

    for i, im in enumerate(images_list):

        # Depth
        depth.append(str(int(float(im.split("_")[-1].strip(".jpg")))))
        # Image
        ima = Image.open(im)
        current_image = np.array(ima).astype(float)[::-1, :, :]
        roi = current_image[center[0] - num_px // 2: center[0] + num_px // 2, center[1] - num_px // 2: center[1] + num_px // 2,:]

        depth_image[i, :, 0] = roi[:, :, 0].mean()
        depth_image[i, :, 1] = roi[:, :, 1].mean()
        depth_image[i, :, 2] = roi[:, :, 2].mean()

        # ISO and exposure time
        exposure, isospeed = get_iso_exp(ima)
        exp.append(exposure)
        iso.append(isospeed)

    return exp, iso, depth, depth_image


def process_rgb_images(path, radclass, start=0, interpo=True):
    """

    :param path:
    :param radclass:
    :return:
    """

    image_list = glob.glob(path + "/*.jpg")
    image_list = image_list[start:]

    _, _, d, d_image = process_jpg(image_list)

    # Check_depth
    defl = np.array([float(i) for i in d])
    mask = np.in1d(radclass.K_d["depth"], defl)

    kdown = radclass.K_d[mask]
    kd_average = np.mean((kdown["r"], kdown["g"], kdown["b"]), axis=0)

    rgb_ima = d_image / kd_average.reshape(-1, 1)[None, None, 1]
    rgb_ima /= rgb_ima.max()

    if interpo:
        di = np.diff(defl)
        if not np.all(di[0] == di):
            steps = di.min()
            new_z = np.arange(defl.min(), defl.max() + steps, steps)

            # interpolators
            new_rgb_ima = np.zeros((new_z.shape[0], rgb_ima.shape[1], rgb_ima.shape[2]))

            for b in range(new_rgb_ima.shape[2]):
                f = scipy.interpolate.interp1d(defl, rgb_ima[:, 0, b], kind='nearest')
                new_rgb_ima[:, 0, b] = f(new_z)

            rgb_ima = new_rgb_ima.copy()
            defl = new_z.copy()

    return rgb_ima, defl


if __name__ == "__main__":

    # -------------------------- BDC --------------------------
    path2im = "baiedeschaleurs2022/jpeg_img"
    imlist = glob.glob(path2im + "/*.jpg")

    # Freeboard
    ifb_st2 = get_ice_freeboard("baiedeschaleurs2022/data/station_2_data.txt")
    rc_st2 = RadClass(data_path="baiedeschaleurs2022/data/baiedeschaleurs-03232022-fluo.h5", station="station_2", data_type="camera", freeboard=ifb_st2)

    exp, iso, depth, depth_image = process_jpg(imlist)

    # Processing
    #exposure_param = np.array(exp) * (np.array(iso) / 100)
    #rgb_image = depth_image / exposure_param.reshape(-1, 1)[None, None, 1]

    rgb_image = depth_image.copy()

    kd = rc_st2.K_d[2:-1]
    avg_kd = np.mean((kd["r"], kd["g"], kd["b"]), axis=0)

    #rgb_image[:, :, 0] /= kd["r"].reshape(-1, 1)
    #rgb_image[:, :, 1] /= kd["g"].reshape(-1, 1)
    #rgb_image[:, :, 2] /= kd["b"].reshape(-1, 1)
    rgb_image /= avg_kd.reshape(-1, 1)[None, None, 1]
    rgb_image /= rgb_image.max()

    # Taking irradiance instead
    #irr_st2 = rc_st2.eo.copy()
    #irr_st2_arr = np.array(irr_st2.tolist())
    #min_z, max_z = 5.0, 80.0
    #mask_depth = np.where((min_z <= irr_st2_arr[:, 3]) & (irr_st2_arr[:, 3] <= max_z))
    #irr_st2_arr = irr_st2_arr[mask_depth]
    #rgb_bdc_irr = irr_st2_arr[:, :3]
    #rgb_bdc_irr /= rgb_bdc_irr.max()
    #rgb_bdc_irr[:, 0] /= rgb_bdc_irr[:, 0].max()
    #rgb_bdc_irr[:, 1] /= rgb_bdc_irr[:, 1].max()
    #rgb_bdc_irr[:, 2] /= rgb_bdc_irr[:, 2].max()
    #rgb_bdc_irr = np.stack((rgb_bdc_irr[:, 0].reshape(-1, 1), rgb_bdc_irr[:, 1].reshape(-1, 1), rgb_bdc_irr[:, 2].reshape(-1, 1)), axis=2)

    # -------------------------- Oden --------------------------

    path2im = "oden2018/jpeg_img"
    imlist_oden = glob.glob(path2im + "/*.jpg")[1:]
    rc_oden = RadClass(data_path="oden2018/data/oden-08312018-fluo.h5")

    _, _, depth_oden, depth_image_oden = process_jpg(imlist_oden)

    kd_oden = rc_oden.K_d[1:]
    avg_kd_oden = np.mean((kd_oden["r"], kd_oden["g"], kd_oden["b"]), axis=0)

    rgb_image_oden = depth_image_oden.copy() / avg_kd_oden.reshape(-1, 1)[None, None, 1]
    rgb_image_oden /= rgb_image_oden.max()

    # Figures
    fig1, ax1 = plt.subplots(1, 2, figsize=(2, 4.8))

    ax1[0].imshow(rgb_image)

    ax1[0].set_xticks([])
    ax1[0].set_yticks(np.arange(0, len(imlist), 1))

    ax1[0].set_yticklabels(depth)
    ax1[0].set_ylabel("Depth [cm]")

    ax1[1].imshow(rgb_image_oden)

    ax1[1].set_xticks([])
    ax1[1].set_yticks(np.arange(0, len(imlist_oden), 1))

    ax1[1].set_yticklabels(depth_oden)
    ax1[1].set_ylabel("Depth [cm]")

    fig1.tight_layout()

    # Fig 2 - new fonction

    image_RGB_oden, de_od = process_rgb_images("oden2018/jpeg_img", rc_oden, start=1)
    image_RGB_bdc, de_bcd = process_rgb_images("baiedeschaleurs2022/jpeg_img", rc_st2, start=0)

    fig2, ax2 = plt.subplots(1, 2, figsize=(2, 4.8))

    ax2[0].imshow(image_RGB_bdc)
    ax2[1].imshow(image_RGB_oden)

    ax2[0].set_xticks([])
    ax2[0].set_yticks(np.arange(0, de_bcd.shape[0], 1))
    ax2[0].set_yticklabels(de_bcd.astype(str).tolist())
    ax2[0].set_ylabel("Depth [cm]")

    ax2[1].set_xticks([])
    ax2[1].set_yticks(np.arange(0,  de_od.shape[0], 1))

    ax2[1].set_yticklabels(de_od.astype(str).tolist())
    ax2[1].set_ylabel("Depth [cm]")

#ax2[1].plot(avg_kd, np.array(depth).astype(float))

    #ax2[0].invert_yaxis()

    plt.show()
