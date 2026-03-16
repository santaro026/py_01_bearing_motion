"""
Created on Tue Feb 17 14:07:10 2026
@author: honda-shin



"""

import numpy as np
from scipy import fft
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

import myplotter

def generate_sampledata(duration, fs, elements, envs):
    N = int(duration * fs)
    t = np.linspace(0, duration, N)
    sig = np.zeros(N)
    for f, a, p in (elements):
        sig = sig +  a * np.sin(2*np.pi * f * t + p)
    # sig = sig * (1 + envs[1] * np.sin(2*np.pi * envs[0] * t + envs[2]))
    # sig = sig * (0 + envs[1] * np.sin(2*np.pi * envs[0] * t))
    return t, sig


def butterworth_test():
    order = 4
    duration = 1
    fs = 10000
    N = int(duration * fs)
    t = np.linspace(0, duration, N)
    elements = [
        (200, 1, 0),
        (300, 1, 0),
        (400, 1, 0),
        (800, 1, 0),
        (850, 1, 0),
        (900, 1, 0),
        (2000, 1, 0)
    ]
    envs = None
    t, sig = generate_sampledata(duration, fs, elements, envs)
    elements1 = [
        (200, 1, 0),
        (300, 1, 0),
    ]
    t, sig1_ideal = generate_sampledata(duration, fs, elements1, envs)
    b1, a1 = signal.butter(order, [150/(fs/2), 350/(fs/2)], btype="band")
    sig1 = signal.filtfilt(b1, a1, sig)
    elements2 = [
        (800, 1, 0),
        (850, 1, 0),
        (900, 1, 0),
    ]
    t, sig2_ideal = generate_sampledata(duration, fs, elements2, envs)
    b2, a2 = signal.butter(order, [700/(fs/2), 1000/(fs/2)], btype="band")
    sig2 = signal.filtfilt(b2, a2, sig)
    freqs = fft.rfftfreq(N, 1/fs)
    fft_sig = np.abs(fft.rfft(sig))
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_31)
    fig, axs = plotter.myfig(ylabel=["sound pressure", "", "psd"], xlabel=["time", "frequency", "frequency"], xsigf=[2, 0, 0], xtick=[0.1, 100, 100], xrange=[(0, 0.2), (0, 0.02), (0, 0.02)])
    axs[0].plot(t, sig, lw=0.4, c='k', alpha=1)
    axs[1].plot(t, sig1_ideal, lw=1, c='k', alpha=1)
    axs[1].plot(t, sig1, lw=4, c='g', alpha=0.4)
    axs[2].plot(t, sig2_ideal, lw=1, c='k', alpha=1)
    axs[2].plot(t, sig2, lw=4, c='g', alpha=0.4)
    plt.show()


if __name__ == "__main__":
    print("---- test ----")

    order = 4

    duration = 1
    fs = 10000
    N = int(duration * fs)
    t = np.linspace(0, duration, N)

    elements = [
        (200, 1, 0),
        (300, 1, 0),
    ]
    envs = None
    t, sig = generate_sampledata(duration, fs, elements, envs)
    sig = np.random.random(N) * (1 + 0.5 * np.sin(2*np.pi*20*t))

    analytic = signal.hilbert(sig)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    carrier = np.exp(1j * phase)
    baseband = analytic / carrier

    b1, a1 = signal.butter(order, 20000/60/(fs/2), btype="low")
    env_low = signal.filtfilt(b1, a1, envelope)



    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_31)
    fig, axs = plotter.myfig(ylabel=["sound pressure", "", "psd"], xlabel=["time", "frequency", "frequency"], xsigf=[2, 0, 0], xtick=[0.1, 100, 100], xrange=[(0, 0.2), (0, 0.2), (0, 0.2)])

    axs[0].plot(t, sig, lw=0.4, c='k', alpha=1)
    # axs[0].plot(t, envelope, lw=1, c='g', alpha=0.4)
    axs[0].plot(t, env_low, lw=1, c='r', alpha=1)

    axs[1].plot(t, baseband, lw=1, c='k', alpha=1)

    # axs[2].plot(t, sig2_ideal, lw=1, c='k', alpha=1)
    # axs[2].plot(t, sig2, lw=4, c='g', alpha=0.4)

    plt.show()




