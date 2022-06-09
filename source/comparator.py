import numpy as np
import matplotlib.pyplot as plt


from radclass import RadClass
from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *

class HFComparator:
    def __init__(self, RadClass, HEDataViewer):
        self.radclass = RadClass
        self.HEViewer = HEDataViewer
        self.HEdepths = self.HEViewer.Eudos_IOPs_df['depths']

        # Field measurements
        self.FEd600, self.FEd540, self.FEd480, self.Fdepths = np.array(self.radclass.ed.tolist()).T # depths in [cm]
        self.FEu600, self.FEu540, self.FEu480, self.Fdepths = np.array(self.radclass.eu.tolist()).T
        self.FEo600, self.FEo540, self.FEo480, self.Fdepths = np.array(self.radclass.eo.tolist()).T

        self.Fdepths_si = self.Fdepths/100 # For god's sake, please use the international system of units! [m]

        # Hydro Light results

    def compare_irradiances(self, wavelengths):
        plt.rcParams['lines.linewidth'] = 3.0
        plt.rc('legend', fontsize=12)
        fig , ax = plt.subplots(1,3, figsize=(12, 6))
        color_field = "navy"
        color_h = "darkorange"
        for wavelength in wavelengths:
            if wavelength == 480:
                self.HEu480, self.HEd480, self.HEo480 = self.HEViewer.get_Eudos_at_depths(self.Fdepths_si, 480.)

                ax[0].plot(self.FEu480, self.Fdepths_si, linestyle="--", color=color_field)
                ax[1].plot(self.FEd480, self.Fdepths_si, linestyle="--", color=color_field)
                ax[2].plot(self.FEo480, self.Fdepths_si, linestyle="--", color=color_field, label="Field $\lambda$=480nm")

                ax[0].plot(self.HEu480, self.Fdepths_si, linestyle="-", color=color_h)
                ax[1].plot(self.HEd480, self.Fdepths_si, linestyle="-", color=color_h)
                ax[2].plot(self.HEo480, self.Fdepths_si, linestyle="-", color=color_h, label="Simulation $\lambda$=480nm")

            elif wavelength == 540:
                self.HEu540, self.HEd540, self.HEo540 = self.HEViewer.get_Eudos_at_depths(self.Fdepths_si, 540.)

                ax[0].plot(self.FEu540, self.Fdepths_si, linestyle="--", color=color_field)
                ax[1].plot(self.FEd540, self.Fdepths_si, linestyle="--", color=color_field)
                ax[2].plot(self.FEo540, self.Fdepths_si, linestyle="--", color=color_field, label="Field $\lambda$=540nm")

                ax[0].plot(self.HEu540, self.Fdepths_si, linestyle="-", color=color_h)
                ax[1].plot(self.HEd540, self.Fdepths_si, linestyle="-", color=color_h)
                ax[2].plot(self.HEo540, self.Fdepths_si, linestyle="-", color=color_h, label="Simulation $\lambda$=540nm")

            elif wavelength == 600:
                self.HEu600, self.HEd600, self.HEo600 = self.HEViewer.get_Eudos_at_depths(self.Fdepths_si, 600.)

                ax[0].plot(self.FEu600, self.Fdepths_si, linestyle="--", color=color_field)
                ax[1].plot(self.FEd600, self.Fdepths_si, linestyle="--", color=color_field)
                ax[2].plot(self.FEo600, self.Fdepths_si, linestyle="--", color=color_field, label="Field $\lambda$=600nm")

                ax[0].plot(self.HEu600, self.Fdepths_si, linestyle="-", color=color_h)
                ax[1].plot(self.HEd600, self.Fdepths_si, linestyle="-", color=color_h)
                ax[2].plot(self.HEo600, self.Fdepths_si, linestyle="-", color=color_h, label="Simulation $\lambda$=600nm")
            ylims = (self.Fdepths_si[0], self.Fdepths_si[-1])
            titles = ["E$_u$", "E$_d$", "E$_o$"]
            [self.HEViewer.format_profile_plot(ax, ylims, str(title)) for ax, title in zip(ax, titles)]
        return 0


class HHComparator:
    def __init__(self, SimulationDataviewer, FieldDataViewer):
        self.radclass = RadClass
        self.HEViewer = HEDataViewer
        self.HEdepths = self.HEViewer.Eudos_IOPs_df['depths']

        # Field measurements
        self.FEd600, self.FEd540, self.FEd480, self.Fdepths = np.array(self.radclass.ed.tolist()).T # depths in [cm]
        self.FEu600, self.FEu540, self.FEu480, self.Fdepths = np.array(self.radclass.eu.tolist()).T
        self.FEo600, self.FEo540, self.FEo480, self.Fdepths = np.array(self.radclass.eo.tolist()).T

        self.Fdepths_si = self.Fdepths/100 # For god's sake, please use the international system of units! [m]

        # Hydro Light results

    def compare_irradiances(self, wavelengths):
        plt.rcParams['lines.linewidth'] = 3.0
        plt.rc('legend', fontsize=12)
        fig , ax = plt.subplots(1,3, figsize=(12, 6))
        color_field = "navy"
        color_h = "darkorange"
        for wavelength in wavelengths:
            if wavelength == 480:
                self.HEu480, self.HEd480, self.HEo480 = self.HEViewer.get_Eudos_at_depths(self.Fdepths_si, 480.)

                ax[0].plot(self.FEu480, self.Fdepths_si, linestyle="--", color=color_field)
                ax[1].plot(self.FEd480, self.Fdepths_si, linestyle="--", color=color_field)
                ax[2].plot(self.FEo480, self.Fdepths_si, linestyle="--", color=color_field, label="Field $\lambda$=480nm")

                ax[0].plot(self.HEu480, self.Fdepths_si, linestyle="-", color=color_h)
                ax[1].plot(self.HEd480, self.Fdepths_si, linestyle="-", color=color_h)
                ax[2].plot(self.HEo480, self.Fdepths_si, linestyle="-", color=color_h, label="Simulation $\lambda$=480nm")

            elif wavelength == 540:
                self.HEu540, self.HEd540, self.HEo540 = self.HEViewer.get_Eudos_at_depths(self.Fdepths_si, 540.)

                ax[0].plot(self.FEu540, self.Fdepths_si, linestyle="--", color=color_field)
                ax[1].plot(self.FEd540, self.Fdepths_si, linestyle="--", color=color_field)
                ax[2].plot(self.FEo540, self.Fdepths_si, linestyle="--", color=color_field, label="Field $\lambda$=540nm")

                ax[0].plot(self.HEu540, self.Fdepths_si, linestyle="-", color=color_h)
                ax[1].plot(self.HEd540, self.Fdepths_si, linestyle="-", color=color_h)
                ax[2].plot(self.HEo540, self.Fdepths_si, linestyle="-", color=color_h, label="Simulation $\lambda$=540nm")

            elif wavelength == 600:
                self.HEu600, self.HEd600, self.HEo600 = self.HEViewer.get_Eudos_at_depths(self.Fdepths_si, 600.)

                ax[0].plot(self.FEu600, self.Fdepths_si, linestyle="--", color=color_field)
                ax[1].plot(self.FEd600, self.Fdepths_si, linestyle="--", color=color_field)
                ax[2].plot(self.FEo600, self.Fdepths_si, linestyle="--", color=color_field, label="Field $\lambda$=600nm")

                ax[0].plot(self.HEu600, self.Fdepths_si, linestyle="-", color=color_h)
                ax[1].plot(self.HEd600, self.Fdepths_si, linestyle="-", color=color_h)
                ax[2].plot(self.HEo600, self.Fdepths_si, linestyle="-", color=color_h, label="Simulation $\lambda$=600nm")
            ylims = (self.Fdepths_si[0], self.Fdepths_si[-1])
            titles = ["E$_u$", "E$_d$", "E$_o$"]
            [self.HEViewer.format_profile_plot(ax, ylims, str(title)) for ax, title in zip(ax, titles)]
        return 0

if __name__ == "__main__":

    test_comp = Comparator(RadClass=RadClass(data_path="oden-08312018.h5"),
                           HEDataViewer=DataViewer(root_name='manual_opt_no_interface'))
    test_comp.compare_irradiances(wavelengths=[600])
    plt.show()

