# -*- coding: utf-8 -*-
"""
File with info regarding latest/best calibrations for each cam.
"""


geometric = {"2BW7X7": {"cover": {"air": {"front": '20230406_114342', "back": '20230406_120016'},
                                  "water": {"front": '20230411_171230', "back": '20230411_172456'}},
                        "nocover": {"air": {"front": '20230212_115034', "back": '20230212_121752'},
                                    "water": {"front": '20230404_122521', "back": '20230404_124245'}}},
             "2C9JCA": {"nocover": {"air": {"front": '20230322_150036'}, "water": {"front": '20230404_114030'}}}}

rsr = {"2BW7X7": {"cover": {"front": '20230407', "back": '20230407'},
                  "nocover": {"front": '20230222', "back": '20230313'}}}

if __name__ == "__main__":

    print("Calibrations info")
