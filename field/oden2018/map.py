# -*- coding: utf-8 -*-
"""
Map of the station where radiance profile were taken.
"""

# Module importation
import numpy as np
import matplotlib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

LAND = cfeature.NaturalEarthFeature(
    category='physical',
    name='coastline',
    scale='10m',
)

from source.processing import FigureFunctions


# Functions
def dms_to_decimal(degrees, minutes, northeast=True):
    """
    Function that transforms degrees-minutes coordinate (lon or lat) to decimal coordinates (lon or lat).

    :param degrees: degree
    :param minutes: minute of degree
    :return: decimal longitude or latitude
    """
    c = degrees + float(minutes) / 60
    if not northeast:
        c*=-1
    return c


if __name__ == "__main__":

    # Initialize figure
    ff = FigureFunctions()

    fig1 = plt.figure(figsize=ff.set_size())
    ax1 = fig1.add_subplot(111, projection=ccrs.NorthPolarStereo())
    ax1.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())

    # Adding coastlines and features
    ax1.coastlines()
    #ax1.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.LAND)

    ax1.gridlines(draw_labels=True, color="k", linestyle="--", alpha=0.6)

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = matplotlib.path.Path(verts * radius + center)

    # Plot
    lon = dms_to_decimal(63.0, 8.76)
    lat = dms_to_decimal(89.0, 25.21)

    ax1.plot(lon, lat, marker="*", markersize=10, markeredgecolor="k", markerfacecolor="yellow")
    ax1.annotate('Site', xy=(lon, lat), xytext=(-2, 10), textcoords='offset points',
                 xycoords=ccrs.NorthPolarStereo()._as_mpl_transform(ax1), fontsize=7,
                 arrowprops=dict(arrowstyle="->"))

    plt.show()
