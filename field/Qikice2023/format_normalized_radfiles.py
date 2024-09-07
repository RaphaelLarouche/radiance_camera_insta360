"""
Format radiance to standardize.
"""

# Module importation
import time
import os
import h5py
import datetime
import pandas
import numpy as np

from process_sites import create_label


# Function and classes
def sort_h5_keys(ke):
    """

    :param ke:
    :type ke:
    :return:
    :rtype:
    """
    depth = []
    for i in ke:
        if (i == "azimuth") or (i == "zenith"):
            continue
        else:
            depth.append(int(float(i.split(" ")[0])))
    aso = np.argsort(np.array(depth))
    return aso, np.array(depth)[aso]


if __name__ == "__main__":

    # Current profile info
    station = 2
    site = 1
    mes = 0

    lat = 67.475655  # TO CHANGE
    lon = -63.968977  # TO CHANGE

    dt = datetime.datetime(2023, 4, 22)  # TO CHANGE

    data_id = f"QI{station:02}{site:1}{mes:1}"  # unique id of the profile (TO CHANGE)
    # gen_path: path where the .h5 file is located  (TO CHANGE)
    gen_path = f"/Volumes/MYBOOK/QikIce2023/Qik2023/QI{station:02}/{data_id}_radiance_raw/processed_data/"
    path_to_h5 = gen_path + f"{data_id}_radiance.h5"

    data_h5 = h5py.File(path_to_h5, "r")
    data_h5 = data_h5[data_id]
    #data_h5 = data_h5['QI0732']

    # Filename
    fn_dct, _, _, _, _ = create_label(os.path.dirname(os.path.dirname(gen_path)) + "/README.txt")

    uskeys = list(data_h5.keys())
    skeys = sort_h5_keys(uskeys)
    skeylist = [uskeys[j] for j in skeys[0]]

    # Loop
    for b, k in enumerate(skeylist):

        d = skeys[1][b]
        str_time = time.strftime("%H:%M:%S", time.strptime(fn_dct[d].split("_")[2], "%H%M%S"))
        print(str_time)
        df_dc = {"Data": data_id,
                 "Lat": lat,
                 "Long": lon,
                 "Julian sampling day": dt.timetuple().tm_yday,
                 "Year": dt.year,
                 "Month": dt.month,
                 "Day": dt.day,
                 "Local time": str_time,
                 "Depth": d,
                 "Zenith": data_h5["zenith"][:, 0] * 180/np.pi}  # RAVEL in row-major

        data_red = data_h5[k][:][:, :, 0]
        data_red[data_red == 0] = np.nan

        data_green = data_h5[k][:][:, :, 1]
        data_green[data_green == 0] = np.nan

        data_blue = data_h5[k][:][:, :, 2]
        data_blue[data_blue == 0] = np.nan

        #pd_rad_red = pandas.DataFrame(data_h5[k][:][:, :, 0], columns=[f"Azimuth {i:.5f} rad" for i in data_h5["azimuth"][0, :]])
        #pd_rad_green = pandas.DataFrame(data_h5[k][:][:, :, 1], columns=[f"Azimuth {i:.5f} rad" for i in data_h5["azimuth"][0, :]])
        #pd_rad_blue = pandas.DataFrame(data_h5[k][:][:, :, 2], columns=[f"Azimuth {i:.5f} rad" for i in data_h5["azimuth"][0, :]])
        pd_rad_red = pandas.DataFrame(data_red, columns=data_h5["azimuth"][0, :] * 180/np.pi)
        pd_rad_green = pandas.DataFrame(data_green, columns=data_h5["azimuth"][0, :] * 180/np.pi)
        pd_rad_blue = pandas.DataFrame(data_blue, columns=data_h5["azimuth"][0, :] * 180/np.pi)

        df_gen = pandas.DataFrame(df_dc)
        if b == 0:
            # red
            #df_gen = pandas.DataFrame(df_dc)
            df_tot_red = pandas.concat([df_gen, pd_rad_red], axis=1)

            # Green
            df_tot_green = pandas.concat([df_gen, pd_rad_green], axis=1)

            # Blue
            df_tot_blue = pandas.concat([df_gen, pd_rad_blue], axis=1)

        else:
            #df_gen = pandas.DataFrame(df_dc)

            df_tot_red = pandas.concat([df_tot_red, pandas.concat([df_gen, pd_rad_red], axis=1)])
            df_tot_green = pandas.concat([df_tot_green, pandas.concat([df_gen, pd_rad_green], axis=1)])
            df_tot_blue = pandas.concat([df_tot_blue, pandas.concat([df_gen, pd_rad_blue], axis=1)])

    # Saving the radiance data in excel format
    filenamepath = gen_path + data_id + f"_radiance.xlsx"

    writer = pandas.ExcelWriter(filenamepath, engine='openpyxl')
    df_tot_red.to_excel(writer, sheet_name="Radiance red", index=False)
    df_tot_green.to_excel(writer, sheet_name="Radiance green", index=False)
    df_tot_blue.to_excel(writer, sheet_name="Radiance blue", index=False)
    writer.close()



