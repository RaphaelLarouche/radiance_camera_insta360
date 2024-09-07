"""
Format radiance data.
"""

# Module importation
import os
import datetime
import numpy as np
import pandas

from source.radiance import ImageRadiancei360x3


if __name__ == "__main__":

    # Rad data
    p = "/Users/raphaellarouche/Desktop/IMG_20230316_112747_00_003.dng"
    imrad = ImageRadiancei360x3(p, cam_sn="2BW7X7", medium="air", cover="cover")
    imrad.get_radiance()
    imrad.map_radiance(angular_resolution=1.0)

    rad = imrad.mapped_radiance.copy()
    zenith = imrad.zenith_mesh.copy()
    azimuth = imrad.azimuth_mesh.copy()

    # Path to logsheet_TS_template
    data_ = pandas.read_excel("/Users/raphaellarouche/Desktop/Logsheet_TS_template.xlsx")

    # Information
    dt = datetime.datetime(2023, 4, 19)
    julian_day = 146
    station = 1
    site = 1
    num = 1

    lat = 67.478933
    lon = -63.79015
    depth = 0  # cm

    # Filename
    data = f"QI{station:02}{site:1}{num:1}"
    filename = "data/" + data + f"_radiance.xlsx"

    # Create Dataframe
    df_dc = {"Data": data,
             "Lat": lat,
             "Long": lon,
             "Julian sampling day": julian_day,
             "Year": dt.year,
             "Month": dt.month,
             "Day": dt.day,
             "Depth": depth,
             "Zenith": zenith.ravel(),
             "Azimuth": azimuth.ravel(),
             "Rad blue": rad[:, :, 2].ravel(),
             "Rad green": rad[:, :, 1].ravel(),
             "Rad red": rad[:, :, 0].ravel()}  # RAVEL in row-major

    df = pandas.DataFrame(df_dc)

    print(df)

    #df.to_csv(filename, index=False, mode="a")  # convert to excel

