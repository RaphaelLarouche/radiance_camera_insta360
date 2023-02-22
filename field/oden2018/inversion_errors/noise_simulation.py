"""
Secret IOPs with random errors added to the curves.
"""

# Module importation
import os
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt


# Other module
import pandas

from source.radiance import RadClass
from field.oden2018.oden_dort_vs_hl import load_zenith_radiance, get_zenith_radiance_profile_at_depth
from field.oden2018.oden_2018_aops_iops import get_Eudos_at_depth


# Functions
def irradiance_calculation(zeni, azi, radm, zenimin, zenimax, planar=True):
    """
    Estimate irradiance from the radiance angular distribution. By default, it calculates the planar irradiance.
    By setting the parameter planar to false, the scalar irradiance is computed. Zenimin = 0˚ and Zenimax = 90˚ gives
    the downwelling irradiance, while Zenimin = 90° and Zenimax = 180˚ gives the upwelling irradiance.

    :param zeni: zenith meshgrid in degrees
    :param azi: azimuth meshgrid in degrees
    :param radm: radiance angular distribution
    :param zenimin: min zenith in degrees
    :param zenimax: max zenith in degrees
    :param planar: if True - planar radiance, if false - scalar (bool)
    :return:
    """

    mask = (zenimin <= zeni) & (zeni <= zenimax)
    zeni_rad = zeni * np.pi / 180
    azi_rad = azi * np.pi / 180

    # Integrand
    if planar:
        integrand = radm[mask] * np.absolute(np.cos(zeni_rad[mask])) * np.sin(zeni_rad[mask])
    else:
        integrand = radm[mask] * np.sin(zeni_rad[mask])

    # Azimuthal integration
    azimuth_inte = simps(integrand.reshape((-1, azi_rad.shape[1])), azi_rad[mask].reshape((-1, azi_rad.shape[1])), axis=1)

    # Zenith integration
    e = simps(azimuth_inte, zeni_rad[mask].reshape((-1, azi_rad.shape[1]))[:, 0], axis=0)

    return e


def calculate_irradiance(zenith_meshgrid, azimuth_meshgrid, radiance):
    ed = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 0, 90)
    edo = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 0, 90, planar=False)
    eu = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 90, 180)
    euo = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 90, 180, planar=False)
    eo = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 0, 180, planar=False)

    return eu, ed, eo, euo, edo


def rad_rmse(rad_true, rad_error):
    """
    """
    return np.sqrt(np.nanmean(np.square((rad_error - rad_true) / rad_true))) * 100


def get_noisy_radiance_curve(data, amplitude_distribution, phase_distribution, depth=40.0, wave=480, seed=0, show=True):
    """

    :param radiance:
    :param amplitude_distribution:
    :param phase_distribution:
    :return:
    """

    wl_dct = {480: 2, 540: 1, 600: 0}

    zen_simu, rad_simu = get_zenith_radiance_profile_at_depth(data, depth=depth / 100, wavelength=wave, interpolate=True)

    # Weight
    weight = np.ones(amplitude_distribution.shape[1])
    weight[:6] = 0.1
    weight[6:] = 0.01

    #weight[:7] = 0.1
    #weight[7:] = 0.01

    # Amplitude sampling
    mean_amplitude = amplitude_distribution[:, :, wl_dct[wave]].mean(axis=0)
    std_amplitude = amplitude_distribution[:, :, wl_dct[wave]].std(axis=0)
    low_bound_ampl = mean_amplitude - weight * std_amplitude
    high_bound_ampl = mean_amplitude + weight * std_amplitude

    np.random.seed(seed)  # seed for reproducibility
    ampl_random = np.random.uniform(low=low_bound_ampl, high=high_bound_ampl)
    if ampl_random.shape[0] % 2 == 1:
        ampl_random = np.append(ampl_random, ampl_random[::-1][:-1])
    else:
        ampl_random = np.append(ampl_random, ampl_random[::-1])

    # Phase sampling
    mean_phase = phase_distribution[:, :, wl_dct[wave]].mean(axis=0)
    std_phase = phase_distribution[:, :, wl_dct[wave]].std(axis=0)
    low_bound_phase = mean_phase - weight * std_phase
    high_bound_phase = mean_phase + weight * std_phase

    phase_random = np.random.uniform(low=low_bound_phase, high=high_bound_phase)
    if phase_random.shape[0] % 2 == 1:
        phase_random = np.append(phase_random, phase_random[::-1][:-1])
    else:
        phase_random = np.append(phase_random, phase_random[::-1])

    # FFT addition
    fft_rad_curve = np.fft.fft(rad_simu)

    # Absolute calculations
    #new_ampl = abs(fft_rad_curve) + ampl_random
    #new_ph = np.arctan2(fft_rad_curve.imag, fft_rad_curve.real) + phase_random

    # Relative calculations
    abs_ampl = ampl_random * abs(fft_rad_curve)
    new_ampl = abs(fft_rad_curve) + abs_ampl

    abs_ph = phase_random * np.arctan2(fft_rad_curve.imag, fft_rad_curve.real)
    new_ph = np.arctan2(fft_rad_curve.imag, fft_rad_curve.real) + abs_ph

    fft_noisy = np.zeros(len(ampl_random), dtype=np.complex64)

    fft_noisy.real = new_ampl * np.cos(new_ph)
    fft_noisy.imag = new_ampl * np.sin(new_ph)
    new_rad_curve = np.fft.ifft(fft_noisy)

    if show:
        rmse_ = rad_rmse(rad_simu, abs(new_rad_curve))
        xval = np.arange(mean_amplitude.shape[0])

        fig, ax = plt.subplots(1, 3, figsize=(8, 8 * 0.6))

        # Axe 0
        ax[0].plot(xval, mean_amplitude, color="k", linestyle="--", label="Average")
        ax[0].plot(xval, ampl_random[:mean_amplitude.shape[0]], color="k", linestyle="-", label="Sampled")
        ax[0].fill_between(xval, mean_amplitude - weight * std_amplitude,
                           mean_amplitude + weight * std_amplitude, color="grey", alpha=0.6, label="Possible values")

        ax[0].set_xlabel("Zenith [˚]")
        ax[0].legend(loc="best")
        ax[0].set_title(f"{depth} cm, {wave} nm", fontsize=8)

        # Axe 1
        ax[1].plot(xval, mean_phase, color="k", linestyle="--", label="Average")
        ax[1].plot(xval, phase_random[:mean_phase.shape[0]], color="k", linestyle="-", label="Sampled")
        ax[1].fill_between(xval, mean_phase - weight * std_phase,
                           mean_phase + weight * std_phase, color="grey", alpha=0.6, label="Possible values")

        ax[1].set_xlabel("Zenith [˚]")
        ax[1].legend(loc="best")
        ax[1].set_title(f"{depth} cm, {wave} nm", fontsize=8)

        # Axe 2
        ax[2].plot(zen_simu, rad_simu, label="Original")
        ax[2].plot(zen_simu, abs(new_rad_curve), label="Noisy")

        ax[2].annotate(f"rmse = {rmse_:.2f} %", (0.1, 0.1), xycoords="axes fraction")
        ax[2].set_xlabel("Zenith [˚]")
        ax[2].legend(loc="best")
        ax[2].set_title(f"{depth} cm, {wave} nm", fontsize=8)

        fig.tight_layout()

    return abs(new_rad_curve)


def get_fourier_error(rc_obj, sim_data, depths, show=True):
    """
    Fourier analysis (amplitude and phase) of the error between field measurements and simulations (HydroLight).

    :param rc_obj:
    :param sim_data:
    :param depths:
    :return:
    """
    wl_hl = np.array([600, 540, 480])
    wl_cam = np.array([603, 544, 484])

    fig, ax = plt.subplots(3, 3, sharex=True)

    diff_ampl_depth = np.zeros((len(depths), 91, 3))  # 91 values
    diff_ph_depth = diff_ampl_depth.copy()

    for i, de in enumerate(depths):
        for b, wave in enumerate(zip(wl_hl, wl_cam)):
            wave_hl, wave_cam = wave
            zen_sim, rad_sim = get_zenith_radiance_profile_at_depth(sim_data, depth=de / 100, wavelength=wave_hl, interpolate=True)
            zen_cam, rad_cam = rc_obj.get_radiance_avg_at_depth_wl(depth=de, wl=wave_cam, smooth=True)
            # Smooth true for now so values up to 180 deg
            # TODO: Maybe change for unsmooth (for simul closer to reality) ?

            # FFT
            # Remove nan values
            mask = ~np.isnan(rad_cam)
            rad_cam = rad_cam[mask]
            rad_sim = rad_sim[mask]

            # Spatial frequency
            dx = np.diff(zen_cam)
            fs = 1 / dx[0]
            freq = np.linspace(-fs / 2, fs / 2, num=rad_cam.shape[0], endpoint=True)
            mask_freq = freq >= 0

            # Fourier transform
            fft_cam = np.fft.fftshift(np.fft.fft(rad_cam))[mask_freq]
            fft_sim = np.fft.fftshift(np.fft.fft(rad_sim))[mask_freq]

            # Phase
            ph_cam = np.arctan2(fft_cam.imag, fft_cam.real)
            ph_sim = np.arctan2(fft_sim.imag, fft_sim.real)

            # Differentials relative
            diff_ampl_depth[i, :, b] = (abs(fft_cam) - abs(fft_sim)) / abs(fft_sim)
            diff_ph_depth[i, :, b] = (ph_cam - ph_sim) / ph_sim

            # Differentials absolute
            #diff_ampl_depth[i, :, b] = (abs(fft_cam) - abs(fft_sim))
            #diff_ph_depth[i, :, b] = (ph_cam - ph_sim)

            ax[0, b].plot(abs(fft_cam), label=f"{de} cm")
            ax[1, b].plot(abs(fft_sim))
            #ax[2, b].plot(diff_ampl_depth[i, :, b])
            ax[2, b].plot(abs(fft_cam) - abs(fft_sim))

            ax[0, b].set_yscale("log")
            ax[1, b].set_yscale("log")
            ax[0, b].set_title("{0} nm".format(wave_cam), fontsize=8)

    ax[0, 0].set_ylabel("$FFT[L_{cam}]$")
    ax[1, 0].set_ylabel("$FFT[L_{sim}]$")

    ax[0, 0].legend(loc="best", fontsize=9, frameon=False)

    ax[2, 0].set_xlabel("Angular freq [1/°]", fontsize=8)
    ax[2, 1].set_xlabel("Angular freq [1/°]", fontsize=8)
    ax[2, 2].set_xlabel("Angular freq [1/°]", fontsize=8)

    fig.tight_layout()

    # Show figure
    if show:
        fig_avg, ax_avg = plt.subplots(2, 3, sharex=True)

        for k in range(ax_avg.shape[1]):

            ax_avg[0, k].plot(diff_ampl_depth[:, :, k].mean(axis=0))
            ax_avg[0, k].fill_between(np.arange(91),
                                     diff_ampl_depth[:, :, k].mean(axis=0)-diff_ampl_depth[:, :, k].std(axis=0),
                                     diff_ampl_depth[:, :, k].mean(axis=0)+diff_ampl_depth[:, :, k].std(axis=0),
                                     color="grey", alpha=0.6)

            ax_avg[1, k].plot(diff_ph_depth[:, :, k].mean(axis=0))
            ax_avg[1, k].fill_between(np.arange(91),
                                      diff_ph_depth[:, :, k].mean(axis=0)-diff_ph_depth[:, :, k].std(axis=0),
                                      diff_ph_depth[:, :, k].mean(axis=0)+diff_ph_depth[:, :, k].std(axis=0),
                                      color="grey", alpha=0.6)

            ax_avg[0, k].set_title(f"{wl_cam[k]} nm", fontsize=8)
            ax_avg[1, k].set_xlabel("Angular freq [1/°]", fontsize=8)

        #ax_avg[0, 0].set_ylabel("$(FFT[L_{cam}] - FFT[L_{sim}])/ FFT[L_{sim}]$")
        fig_avg.tight_layout()

    return diff_ampl_depth, diff_ph_depth


if __name__ == '__main__':

    # Open data
    rc = RadClass("../data/oden-08312018.h5")
    zen_dist, radiance = rc.get_radiance_avg_at_depth_wl(depth=40.0, wl=544.0)

    mask = ~np.isnan(radiance)
    zen_dist = zen_dist[mask]
    radiance = radiance[mask]

    # Noise analysis in fourier space
    zd = load_zenith_radiance(path=r"../data/oden_fit")

    # *** METHOD 1 *** : Calculate the average delta amplitude (sim - cam) and delta phase (sim - cam) in fourier space,
    # then sample an average amplitude and phase from the error analysis in Fourier space.

    # Delta amplitude, delta phase
    diff_depth, diff_phase_depth = get_fourier_error(rc, zd, np.arange(20, 200, 20).astype(float))
    diff_ampl_base, diff_phase_base = get_fourier_error(rc, zd, np.arange(120, 180, 20).astype(float))

    # Random amplitude sampling
    low_bound_amp = diff_depth[:, :, 0].mean(axis=0) - diff_depth[:, :, 0].std(axis=0)
    high_bound_amp = diff_depth[:, :, 0].mean(axis=0) + diff_depth[:, :, 0].std(axis=0)
    amp_rnd = np.random.uniform(low=low_bound_amp, high=high_bound_amp)
    amp_rnd = np.append(amp_rnd, amp_rnd[::-1][:-1])

    # Random phase sampling
    low_bound_ph = diff_phase_depth[:, :, 0].mean(axis=0) - diff_phase_depth[:, :, 0].std(axis=0)
    high_bound_ph = diff_phase_depth[:, :, 0].mean(axis=0) + diff_phase_depth[:, :, 0].std(axis=0)
    phase_rnd = np.random.uniform(low=low_bound_ph, high=high_bound_ph)
    phase_rnd = np.append(phase_rnd, phase_rnd[::-1][:-1])

    #plt.figure()
    #plt.plot(amp_rnd)

    #zen_meas1, rad_meas1 = rc.get_radiance_avg_at_depth_wl(depth=40.0, wl=603, smooth=True)
    zen_meas1, rad_meas1 = get_zenith_radiance_profile_at_depth(zd, depth=40.0 / 100, wavelength=600.0, interpolate=True)
    fft_meas1 = np.fft.fft(rad_meas1)

    ifft_meas1_noise = np.zeros(len(amp_rnd), dtype=np.complex64)
    new_amplitude = abs(fft_meas1) + amp_rnd
    new_phase = np.arctan2(fft_meas1.imag, fft_meas1.real) + phase_rnd

    ifft_meas1_noise.real = new_amplitude * np.cos(new_phase)
    ifft_meas1_noise.imag = new_amplitude * np.sin(new_phase)
    ifft_meas1_noise = np.fft.ifft(ifft_meas1_noise)

    rmse_m1 = rad_rmse(rad_meas1, abs(ifft_meas1_noise))

    #plt.figure()
    #plt.plot(zen_meas1, ifft_meas1_noise.real)
    #plt.plot(zen_meas1, abs(ifft_meas1_noise), label=f"rmse = {rmse_m1:.3f} %")
    #plt.plot(zen_meas1, rad_meas1)
    #plt.gca().legend(loc="best")

    # *** METHOD 2 *** : add noise in fourier space using the three first frequencies
    # Add noise
    rad_fft = np.fft.fft(radiance)
    dx = np.diff(zen_dist)
    fs = 1 / dx[0]
    freq = np.linspace(-fs/2, fs/2, num=radiance.shape[0], endpoint=True)

    numa_1 = 1
    numa_2 = 2
    numa_3 = 3

    noise_1 = np.zeros(rad_fft.shape[0], dtype=np.complex64)
    noise_1[numa_1] = complex(rad_fft.max().real * 0.07, 0)

    noise_2 = np.zeros(rad_fft.shape[0], dtype=np.complex64)
    noise_2[numa_2] = complex(noise_1.max().real * np.random.random(1), 0)

    noise_3 = np.zeros(rad_fft.shape[0], dtype=np.complex64)
    noise_3[numa_3] = complex(noise_1.max().real * np.random.random(1), 0)

    rad_noisy = np.fft.ifft(rad_fft + noise_1 + noise_2 + noise_3)

    rad_rmse(radiance, rad_noisy.real)

    # Error
    ref_values = 0.5 * (radiance + rad_noisy.real)
    rel_err = 100 * (rad_noisy.real - radiance) / ref_values

    #

    #fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    #for a, d in enumerate(np.arange(0, 200, 20)):
        #_, orad = get_zenith_radiance_profile_at_depth(zd, depth=d / 100, wavelength=600.0, interpolate=True)
        #nrad = get_noisy_radiance_curve(zd, diff_ampl_base, diff_phase_base, seed=a, wave=600, depth=d, show=True)
        #ax1[0].plot(np.arange(0, 181, 1), orad, label=f"depth = {d}")
        #ax1[1].plot(np.arange(0, 181, 1), nrad, label=f"depth = {d}")
    #get_noisy_radiance_curve(zd, diff_depth, diff_phase_depth, seed=3, wave=600, depth=40.0, show=True)

    # Test with new set of data
    original_path = "C:\\Users\\Raphaël Larouche\\PycharmProjects\\radiance_camera_insta360\\field\\oden2018\\data\\inversion_errors\\fit_noise_errors\\original_files"
    secrets_iops_data = load_zenith_radiance(original_path)
    sios_df = pandas.read_csv(os.path.join(original_path, "eudos_iops.csv"))

    dtest = 120.0
    zen_siops, rad_siops = get_zenith_radiance_profile_at_depth(secrets_iops_data, depth=dtest / 100, wavelength=480.0, interpolate=True)
    new_rad = get_noisy_radiance_curve(secrets_iops_data, diff_ampl_base, diff_phase_base, seed=100, wave=480.0, depth=dtest, show=True)

    rad_dist_test = np.tile(rad_siops.reshape(-1, 1), (1, 361))
    azi_mesh, zen_mesh = np.meshgrid(np.arange(0, 361, 1), zen_siops)

    irr_hl = get_Eudos_at_depth(sios_df, dtest/100, 480.0)
    irr_noise = calculate_irradiance(zen_mesh, azi_mesh, rad_dist_test)

    print(f"Eu = {irr_noise[0]:.6f} W/m2, Eu_hl = {irr_hl[1]:.6f} W/m2\n"
          f"Ed = {irr_noise[1]:.6f} W/m2, Ed_hl = {irr_hl[0]:.6f} W/m2\n"
          f"Eo = {irr_noise[2]:.6f} W/m2, Eo_hl = {irr_hl[2]:.6f} W/m2\n"
          f"Euo = {irr_noise[3]:.6f} W/m2, Euo_hl = {irr_hl[4]:.6f} W/m2\n"
          f"Edo = {irr_noise[4]:.6f} W/m2, Edo_hl = {irr_hl[3]:.6f} W/m2\n")


    # Figure
    #fig1, ax1 = plt.subplots(3, 1)

    #ax1[0].plot(np.fft.fftshift(rad_fft).real[rad_fft.shape[0]//2:])
    #ax1[0].plot(np.fft.fftshift(rad_fft_n).real[rad_fft_n.shape[0]//2:])
    #ax1[0].set_yscale("log")
    #ax1[0].set_xscale("log")show=True

    #ax1[1].plot(zen_dist, radiance)
    #ax1[1].plot(zen_dist, rad_noisy.real)

    #ax1[2].plot(rel_err)

    plt.show()
