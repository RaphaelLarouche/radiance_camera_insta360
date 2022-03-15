from processing import ProcessImage, ProcessImage
import radiance as r
from geometric_rolloff import OpenMatlabFiles
from processing import ProcessImage, FigureFunctions
import pandas
import matplotlib.pyplot as plt
import numpy as np

from HE60PY.ac9simulation import AC9Simulation
from HE60PY.Tools.environmentbuilder import create_irrad_file


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


if __name__ == "__main__":
    path_to_irrad_file = '/Applications/HE60.app/Contents/data/HE60DORT_irrad_trios_.txt'
    # process = ProcessImage()
    # ze_mesh, az_mesh, rad_profile = process.open_radiance_data(path="data/oden-08312018.h5")
    # print(az_mesh, '\n', ze_mesh)
    time_stamp, radiometer_wl, irr_incom, irr_below = open_TriOS_data()
    mean_irrad = np.mean(irr_incom, axis=1)
    create_irrad_file(wavelength_Ed=np.array((radiometer_wl, mean_irrad)).T, total_path=path_to_irrad_file)

    print(radiometer_wl.shape, irr_incom.shape)
    for i in range(9):
        plt.plot(radiometer_wl, irr_incom[:,i])
    plt.show()