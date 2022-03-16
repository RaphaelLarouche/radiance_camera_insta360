from processing import ProcessImage, ProcessImage
import radiance as r
from geometric_rolloff import OpenMatlabFiles
from processing import ProcessImage, FigureFunctions
import pandas

from HE60PY.ac9simulation import AC9Simulation
from HE60PY.data_manager import DataFinder



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
    # process = ProcessImage()
    # ze_mesh, az_mesh, rad_profile = process.open_radiance_data(path="data/oden-08312018.h5")
    # print(az_mesh, '\n', ze_mesh)
    # time_stamp, radiometer_wl, irr_incom, irr_below = open_TriOS_data()
    windspeeds = np.arange(30)
    for windspeed in winspeeds:
        windspeed = int(windspeed)
        path_to_user_files = r'/Users/braulier/Documents/HE60/run'
        oden_simulation = AC9Simulation(path=path_to_user_files,
                                                 run_title=f'He60_dort_oden_comp_windspeed_{windspeed}',
                                                 root_name=f'he60_comp_windspeed_{windspeed}',
                                                 mode='HE60DORT')
        oden_simulation.set_z_grid(z_max=3.0, wavelength_list=[484, 544, 602])
        oden_simulation.add_layer(z1=0.0, z2=0.10, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=2277, bb=0.0) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
        oden_simulation.add_layer(z1=0.10, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=303, bb=0.0)
        oden_simulation.add_layer(z1=0.80, z2=2.0, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=79, bb=0.0)
        oden_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 0.01, '544': 0.01, '602': 0.01}, scat=0.1, bb=0.0)
        oden_simulation.run_simulation(printoutput=True)

        oden_analysis = DataFinder(oden_simulation.hermes)
        results = oden_analysis.get_Eudos_lambda()
        results.to_csv(f'data/wind_speed_experiment/HE60_results_{windspeed}.txt')

