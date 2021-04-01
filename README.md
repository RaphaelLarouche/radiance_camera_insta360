# radiance_camera_insta360 
## Description and structure
This repository contains the scripts to convert the images taken with the omnidirectional camera Insta360 ONE 
(in Adobe Digital Negative DNG format) in their spectral radiance equivalent. As for normal rgb sensor, this camera
has three spectral bands covering the visible spectrum so that radiance is computed at the red, green and blue channel. 
 
The project is organized in three main directories:
1. **calibrations**
2. **source**
3. **field**

***
##### 1. calibrations
**calibrations** directory contains all the routines used to perform the camera radiometric calibrations. 
Those are given :

 - Absolute spectral radiance calibration (calibrations/absolute-spectral-radiance)
 - Roll-off calibration (calibrations/roll-off)
 - Immersion factor characterization (calibrations/immersion-factor)
 - Geometric calibration (calibrations/geometric-calibration)
 - Relative spectral response calibration (calibrations/relative-spectral-response)
 - Linearity characterization (calibrations/linearity)

The description of those methodologies are detailed in the following paper https://www.the-cryosphere.net...

***
##### 2. source
**source** directory contains all the necessary classes with method to process and treat the images. Those functions
are used for the calibration scripts. 

In the file `radiance.py`, the class `ImageRadiancei360` enclosed all the methods
to retrieve the spectral radiance for each pixels and build regularly angular spaced grid of radiance. This class uses
the calibration results saved in hdf5 format (in folders /calibrationfiles of each calibration folder).

***
##### 3. field
**field** folder is used to placed any script which performs calculation on field measurements. For instance, radiance 
angular distributions were computed for a profile taken inside arctic sea ice during a campaign on Oden icebreaker near 
the geographic North Pole (89˚ 25.21'N, 63˚ 08.67'E). More details on `oden2018` results are described in 
https://www.the-cryosphere.net...


The folder `oden2018` contains the script to plot the results and to perform radiative transfer using Matlab DORT2002 
model.
 
- Paper on DORT2002: https://doi.org/10.1137/S0036144503438718
- Installation of matlab engine for python: [Install MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

## Packages installation
The environment requirements are specified in `environment.yml` and `requirements.txt`. For Anaconda users, recreate the 
virtual environment using:
```
$ conda env create -f environment.yml
``` 

To use with [Virtualenv](https://virtualenv.pypa.io/en/latest/):

```
$ python3 -m venv env
$ source env/bin/activate
$ python3 -m pip3 install -r requirements.txt
``` 

## Examples
To analyze new data taken with the camera, it is recommended to create a new directory in the folder `field`. To compute 
radiance from an image `image.dng` placed in the same directory as the one containing the script, you can do:
```
import source.radiance as radiance

path = "image.dng"

rad_im = radiance.ImageRadiancei360(path, "water") # object

rad_im.get_radiance(dark_metadata=True)
rad_im.map_radiance(angular_resolution=1.0)
rad_im.show_mapped_radiance()
``` 

For irradiance calculations:

```
ed = im_rad.irradiance(0, 90, planar=True) # planar downwelling irradiance
eu = im_rad.irradiance(90, 180, planar=True) # planar upwelling irradiance
e0 = im_rad.irradiance(0, 180, planar=False) # scalar irradiance
``` 