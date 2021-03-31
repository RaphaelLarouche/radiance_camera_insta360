# Radiance camera Insta360 

This repository contains the script applied to images of
the commercial omnidirectional camera Insta360 ONE to use as a scientific radiometer.

Our specific usage is related to measurement of 
radiance angular distributions internally in Arctic sea ice. but it can be used for multiple usage related to environmental 
passive radiometry. 

The environment requirements are specified in `binder/environment.yml`. To recreate the virtual environment use with:
```
$ conda env create -f binder/environment.yml
``` 

There are three main directories: **calibrations**, **field** and **utilscode** corresponding respectively to the script
for the different calibrations made on the camera, the field work results/analysis and the multiple functions and classes
shared by the scripts.