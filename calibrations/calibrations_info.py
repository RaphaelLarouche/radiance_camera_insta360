# -*- coding: utf-8 -*-
"""
File with info regarding latest/best calibrations for each cam.
"""

# Geometric calibration
geometric = {"2BW7X7": {"cover": {"air": {"front": '20230406_114342', "back": '20230406_120016'},
                                  "water": {"front": '20230411_171230', "back": '20230411_172456'}},
                        "nocover": {"air": {"front": '20230212_115034', "back": '20230212_121752'},
                                    "water": {"front": '20230404_122521', "back": '20230404_124245'}}},
             "2C9JCA": {"nocover": {"air": {"front": '20230322_150036', "back": '20230322_153728'}, "water": {"front": '20230404_114030', "back": '20230404_115610'}}}}

# Relative spectral response
rsr = {"2BW7X7": {"cover": {"front": '20230407', "back": '20230407'},
                  "nocover": {"front": '20230222', "back": '20230313'}},
       "2C9JCA": {"nocover": {"front": "20230330", "back": "20230330"}}}

# Roll-off
rf = {"2BW7X7": {"cover": {"air": {"front": "20230406", "back": "20230406"},
                           "water": {"front": "20230413", "back": "20230411"}},
                 "nocover": {"air": {}, "water": {}}},
      "2C9JCA": {"nocover": {"air": {"front": "20230329", "back": "20230329"},
                             "water": {"front": "20230404", "back": "20230404"}}}}

# Absolute radiance
abs_rad = {"2BW7X7": {"cover": {"front": '20230407', "back": '20230407'}, "nocover": {}},
           "2C9JCA": {"nocover": {"front": "20230323", "back": "20230323"}}}

# Immersion factor
imf = {"2BW7X7": {"cover": {"front": '20230412', "back": '20230412'}, "nocover": {}},
       "2C9JCA": {"nocover": {"front": "20230412", "back": "20230412"}}}

if __name__ == "__main__":

    print("Calibrations info")
