"""
Created on Fri Mar 06 18:40:44 2026
@author: santaro


this module helps format frequently used plots to improve redability and consistency.

"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from scipy import signal

import re
from pathlib import Path
import json

from mymods import myplotter, myfft, mylogger
from mymods.myplotter import MyPlotter, PlotSizeCode, MyAnimator

import config
import data_handler
import data_processor

def extract_clearance_testinfo(testinfo):
    if testinfo["dp_measured"]:
        dp = testinfo["dp_measured"]
        note_dp = 'dp is measured.'
    elif testinfo["dp_drawing"]:
        dp = testinfo["dp_drawing"]
        note_dp = 'dp is nominal.'
    else:
        dp = np.nan
        note_dp = 'dp is none'
    if testinfo["dl_measured"]:
        dl = testinfo["dl_measured"]
        note_dl = 'dl is measured.'
    elif testinfo["dl_drawing"]:
        dl = testinfo["dl_drawing"]
        note_dl = 'dl is nominal.'
    else:
        dl = np.nan
        note_dl = 'dl is none.'
    if isinstance(dp, list):
        dp = np.asarray(dp)
    if isinstance(dl, list):
        dl = np.asarray(dl)
    clearance = {
        "dp": dp,
        "note_dp": note_dp,
        "dl": dl,
        "note_dl": dl
    }
    return clearance

class PlotterForCageVisualization:
    def __init__(self, testinfo, notell="", notelr=""):
        self.testinfo = testinfo
        if self.testinfo is not None:
            clearance = extract_clearance_testinfo(self.testinfo)
            self.dp = clearance["dp"]
            self.note_dp = clearance["note_dp"]
            self.dl = clearance["dl"]
            self.note_dl = clearance["note_dl"]
        elif self.testinfo is None:
            self.dp = 0
            self.note_dp = ""
            self.dl = 0
            self.note_dl = ""
        self.note_dpdl = f'{self.note_dp} {self.note_dl}'
        self.dp_list = PlotterForCageVisualization.cvt_var2list(self.dp)
        self.dl_list = PlotterForCageVisualization.cvt_var2list(self.dl)
        num_dp, num_dl = len(self.dp_list), len(self.dl_list)
        self.auxiliary_circles_radii = [d/2 for d in (self.dp_list + self.dl_list)]
        self.auxiliary_circles_colors = ['b']*num_dp + ['r']*num_dl
        self.notell = notell
        self.notelr = notelr
        print(repr(self))
    @staticmethod
    def cvt_var2list(*args):
        res = []
        for var in args:
            if isinstance(var, (list, np.ndarray)):
                for var2 in var:
                    res.append(var2)
                continue
            res.append(var)
        return res
    @staticmethod
    def add_auxiliary_circles(fig, ax, radii, colors='k', lw=1, alpha=1, zorder=1, ls="--"):
        for r, c in zip(radii, colors):
            circle = plt.Circle((0, 0), r, color=c, fill=False, lw=lw, alpha=alpha, zorder=zorder, ls=ls)
            ax.add_artist(circle)
        return fig, ax

    def slice_data(self, data, frange=None, trange=None, skip=1, fps=1):
        def trange2frange(trange, fps):
            if not isinstance(trange, (list, np.ndarray)):
                raise ValueError(f"trange must be list or ndarray: you passed {type(trange)}")
            if np.ndim(np.asarray(trange)) != 1:
                raise ValueError(f"trange must be 1-d list or ndarray: you passed {np.ndim(np.asarray(trange))}")
            st, et = trange
            sf, ef = int(st*fps), int(et * fps)
            return [sf, ef]
        def frange2trange(frange, fps):
            if not isinstance(frange, (list, np.ndarray)):
                raise ValueError(f"frange must be list or ndarray: you passed {type(frange)}")
            if np.ndim(np.asarray(frange)) != 1:
                raise ValueError(f"frange must be 1-d list or ndarray: you passed {np.ndim(np.asarray(frange))}")
            sf, ef = frange
            st, et = sf/fps, ef/fps
            return [st, et]
        if frange is not None:
            trange = frange2trange(frange, fps)
            sf, ef = frange
            st, et = sf / fps, ef / fps
        if trange is not None:
            if frange is not None:
                print(f"[WANR] both frame range and time range was designated, now frame range ({frange}) was used")
                trange = frange2trange(frange, fps)
            elif frange is None:
                frange = trange2frange(trange, fps)
        if frange is None and trange is None:
            frange = [0, len(data)]
            trange = [0, len(data)/fps]
            sf, ef = frange
        plotduration = trange[1] - trange[0]
        # print(f"plot duration / cage period: {plotduration / self.cage_period:.1f} [rotation] | {plotduration:.3f} [sec] | fps: {fps}")
        data = data[sf:ef:skip]
        return data

    def trajectory(self, y, z, frange=None, trange=None, skip=1, fps=1, xyrange=0.35, slide=False):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(xlabel="y [mm]", ylabel="z [mm]", xrange=(-xyrange, xyrange), yrange=(-xyrange, xyrange), slide=slide)
        ax = axs[0]
        ax.set_aspect(1)
        y = self.slice_data(y, frange=frange, trange=trange, skip=skip, fps=fps)
        z = self.slice_data(z, frange=frange, trange=trange, skip=skip, fps=fps)
        ax.plot(y, z, lw=2, c='k', alpha=1)
        fig, ax = PlotterForCageVisualization.add_auxiliary_circles(fig, ax, radii=self.auxiliary_circles_radii, colors=self.auxiliary_circles_colors, lw=2, alpha=1, zorder=1, ls="--")
        return fig, ax

    def timeseries2(self, times, signals, frange=None, trange=None, skip=1, fps=1, yrange=[(-0.35, 0.35), (-0.35, 0.35)], ytick=[0.1, 0.1], xlabel=["", "time [sec]"], ylabel=[], slide=False):
        st = min([times[i][0] for i in range(2)])
        et = max([times[i][0] for i in range(2)])
        margin = (et -st) * 0.05
        xrange = (-margin + t[0], et - st + t[0] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[2, 2, 0], sharex=[0, 0, 0], slide=slide)
        for i in range(2):
            t = times[i]
            s = signals[i]
            fps = 1 / (t[1] - t[0])
            t = self.slice_data(t, frange=frange, trange=trange, skip=skip, fps=fps)
            s = self.slice_data(s, frange=frange, trange=trange, skip=skip, fps=fps)
            axs[i].plot(t, s, lw=1, c='k', alpha=1)
        return fig, axs

    def timeseries3(self, times, signals, frange=None, trange=None, skip=1, fps=1, yrange=[(-0.35, 0.35), (-0.35, 0.35), (0, 24)], ytick=[0.1, 0.1, 5], xlabel=["", "", "time [sec"], ylabel=[], slide=False):
        st = min([times[i][0] for i in range(3)])
        et = max([times[i][0] for i in range(3)])
        margin = (et -st) * 0.05
        xrange = (-margin + t[0], et - st + t[0] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_31)
        fig, axs = plotter.myfig(xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[2, 2, 0], sharex=[0, 0, 0], slide=slide)
        for i in range(3):
            t = times[i]
            s = signals[i]
            fps = 1 / (t[1] - t[0])
            t = self.slice_data(t, frange=frange, trange=trange, skip=skip, fps=fps)
            s = self.slice_data(s, fps, frange=frange, trange=trange, skip=skip, fps=fps)
            axs[i].plot(t, s, lw=1, c='k', alpha=1)
        return fig, axs

    def cagecoord(self, frange=None, trange=None, yrange=[(-0.35, 0.35), (-0.35, 0.35)], ytick=[0.1, 0.1, 5], offset_time=False, slide=False):
        frange_camera, trange_camera = self.get_ftrange(frange=frange, trange=trange)
        sf, ef = frange_camera
        st, et = trange_camera
        t = self.t_camera[sf:ef]
        if offset_time: t = t - t[0]
        y = self.cage[:, 0, 0][sf:ef]
        z = self.cage[:, 0, 1][sf:ef]
        margin = (et -st) * 0.05
        xrange = (-margin + t[0], et - st + t[0] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(xlabel=["", "time [sec]"], ylabel=["y [mm]", "z [mm]"], xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[2, 2], sharex=[0, 0], slide=slide)
        axs[0].plot(t, y, lw=2, c='k', alpha=1)
        axs[1].plot(t, z, lw=2, c='k', alpha=1)
        # for i in range(2):
        #     axs[i].axhline(y=0, lw=0.4, c='k')
        log = (
            f"frange: {frange_camera}\n"
            f"trange: {trange_camera}\n"
        )
        return fig, axs, log

    def spectrogram(self, t, signal, frange=None, trange=None, skip=1, fps=1, nperseg=2**9, noverlap=0.5, scaling="density", spectromode="psd", window_func="hann", vrange=(-60, -20),
                    yrange=[(-50, 50), (0, 24)], ytick=[20, 5], xlabel=["", "time [sec]"], ylabel=["",  "frequency [Hz]"], markt=None, offset_time=False, slide=False):
        t = self.slice_data(t, frange=frange, trange=trange, skip=skip, fps=fps)
        s = self.slice_data(s, frange=frange, trange=trange, skip=skip, fps=fps)
        margin = (t[1] -t[0]) * 0.05
        xrange = (-margin + t[0], t[1] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[0, 0], sharex=[0, 0], slide=slide)
        #### signal
        axs[0].plot(t, s, lw=1, c='k', alpha=1)
        #### rms
        # timeprocessor = data_processor.TimeSeriesDataProcessor(self.sound, fps=48000)
        # window_time = 0.02
        # rms = timeprocessor.calc_rms(window_time=window_time)
        # db = timeprocessor.cvt_pa2db(rms)
        # axs[0].plot(t, db[sf:ef], lw=1, c='k', alpha=1)
        # ymin, ymax, ytick = 60, 120, 20
        # axs[0].set_ylim(ymin, ymax)
        # axs[0].set_yticks(np.arange(ymin, ymax + (ymax-ymin)*0.05, ytick))
        # axs[0].set_ylabel("SPL [dB]")
        #### spectrogram
        fft = myfft.Myfft(self.t_sound, self.sound, self.sample_rate)
        freq, t_segment, sxx = fft.compute_spectrogram(nperseg=nperseg, noverlap=noverlap, is_log=True, scaling=scaling, mode=spectromode, window=window_func)
        pcm = axs[1].pcolormesh(t_segment, freq/1000, sxx, cmap="viridis", shading="auto")
        if vrange:
            pcm.set_clim(vmin=vrange[0], vmax=vrange[1])
        if markt:
            for e in markt:
                x1, x2 = e["trange"]
                color = e["color"]
                alpha = e["alpha"]
                axs[0].fill_betweenx(y=np.array([-1000, 1000]), x1=x1, x2=x2, color=color, alpha=alpha)
        log = (
            f"trange: {trange}\n"
            f"frange: {frange}\n"
            f"nperseg: {nperseg}\n"
            f"noverlap: {noverlap}\n"
            f"window_func: {window_func}\n"
            f"scaling: {scaling}\n"
            f"spectrogram display mode: {spectromode}\n"
            f"vrange: {vrange}\n"
        )
        return fig, axs, log

    def fft(self, datalist, slide=False):
        log = ""
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.RECTANGLE_FIG)
        fig, axs = plotter.myfig(xlabel="frequency [kHz]", ylabel="psd [dB]", xrange=(-1, 26), yrange=(-80, -10), ytick_0center=False, slide=slide, xsigf=0, ysigf=0, xtick=5, ytick=10)
        for e in datalist:
            note = e["note"]
            frange = e["frange"]
            trange = e["trange"]
            color = e["color"]
            frange, trange = self.get_ftrange(frange=frange, trange=trange, fps="audio")
            sf, ef = frange
            st, et = trange
            t = self.t_sound[sf:ef]
            ax = axs[0]
            sf, ef = frange
            st, et = trange
            fft_size = 2**12
            overlap = 0.5
            lastseg = "cut"
            window_func = "hann"
            mode = "psd"
            fft = myfft.Myfft(self.t_sound, self.sound, self.sample_rate)
            freq, sp = fft.compute_segmented_fft(mode=mode, tranges=[trange], fft_size=fft_size, overlap=overlap, lastseg=lastseg, window_func=window_func, is_log=True)
            ax.plot(freq/1000, sp, lw=2, c=color, alpha=1)
            _log = (
                f"note: {note}\n"
                f"trange: {trange}\n"
                f"frange: {frange}\n"
                f"fftsize {fft_size}\n"
                f"overlap: {overlap}\n"
                f"window_func: {window_func}\n"
                f"lastseg: {lastseg}\n"
                f"mode: {mode}\n"
            )
            log = log + _log
        return fig, ax, log






class PlotterForCageVisualization_old:
    def __init__(self, t_camera=None, cage=None, markers=None, rotspeed=None, t_sound=None, sound=None, testinfo=None, notell="", notelr=""):
        self.t_camera = t_camera
        self.fps = int(1 / (t_camera[1] - t_camera[0]))
        self.cage = cage
        self.markers = markers
        self.rotspeed = rotspeed
        self.rotspeed_avg = np.nanmean(self.rotspeed)
        self.cage_period = 1 / abs(self.rotspeed_avg / (2*np.pi))
        self.t_sound = t_sound
        self.sample_rate = int(1 / (t_sound[1] - t_sound[0])) if t_sound is not None else None
        self.sound = sound
        self.testinfo = testinfo
        if self.testinfo is not None:
            clearance = extract_clearance_testinfo(self.testinfo)
            self.dp = clearance["dp"]
            self.note_dp = clearance["note_dp"]
            self.dl = clearance["dl"]
            self.note_dl = clearance["note_dl"]
        elif self.testinfo is None:
            self.dp = 0
            self.note_dp = ""
            self.dl = 0
            self.note_dl = ""
        self.note_dpdl = f'{self.note_dp} {self.note_dl}'
        self.dp_list = PlotterForCageVisualization.cvt_var2list(self.dp)
        self.dl_list = PlotterForCageVisualization.cvt_var2list(self.dl)
        num_dp, num_dl = len(self.dp_list), len(self.dl_list)
        self.auxiliary_circles_radii = [d/2 for d in (self.dp_list + self.dl_list)]
        self.auxiliary_circles_colors = ['b']*num_dp + ['r']*num_dl
        self.notell = notell
        self.notelr = notelr
        print(repr(self))
    @staticmethod
    def cvt_var2list(*args):
        res = []
        for var in args:
            if isinstance(var, (list, np.ndarray)):
                for var2 in var:
                    res.append(var2)
                continue
            res.append(var)
        return res
    @staticmethod
    def add_auxiliary_circles(fig, ax, radii, colors='k', lw=1, alpha=1, zorder=1, ls="--"):
        for r, c in zip(radii, colors):
            circle = plt.Circle((0, 0), r, color=c, fill=False, lw=lw, alpha=alpha, zorder=zorder, ls=ls)
            ax.add_artist(circle)
        return fig, ax

    def __repr__(self):
        num_frames = len(self.t_camera) if self.t_camera is not None else 0
        num_samples = len(self.t_sound) if self.t_sound is not None else 0
        return (
            f"---- camera ----\n"
            f"N: {num_frames}\n"
            f"fps: {self.fps} [frame/sec]\n"
            f"rot speed: {self.rotspeed_avg:.2f} [rad/sec]\n"
            f"cage rev period: {self.cage_period:.3f} [sec]\n"
            f"---- audio ----\n"
            f"N: {num_samples}\n"
            f"sample_rage: {self.sample_rate} [sample/sec]\n"
        )

    def get_ftrange(self, frange=None, trange=None, fpsmode="camera"):
        def trange2frange(trange, fps):
            if not isinstance(trange, (list, np.ndarray)):
                raise ValueError(f"trange must be list or ndarray: you passed {type(trange)}")
            if np.ndim(np.asarray(trange)) != 1:
                raise ValueError(f"trange must be 1-d list or ndarray: you passed {np.ndim(np.asarray(trange))}")
            st, et = trange
            sf, ef = int(st*fps), int(et * fps)
            return [sf, ef]
        def frange2trange(frange, fps):
            if not isinstance(frange, (list, np.ndarray)):
                raise ValueError(f"frange must be list or ndarray: you passed {type(frange)}")
            if np.ndim(np.asarray(frange)) != 1:
                raise ValueError(f"frange must be 1-d list or ndarray: you passed {np.ndim(np.asarray(frange))}")
            sf, ef = frange
            st, et = sf/fps, ef/fps
            return [st, et]
        sf = 0
        if fpsmode == "camera":
            fps = self.fps
            ef = len(self.t_camera)
        elif fpsmode == "audio":
            fps = self.sample_rate
            ef = len(self.t_sound)
        st, et = sf / fps, ef / fps
        if frange is not None:
            trange = frange2trange(frange, fps)
            sf, ef = frange
            st, et = sf / fps, ef / fps
        if trange is not None:
            if frange is not None:
                print(f"[WANR] both frame range and time range was designated, now frame range ({frange}) was used")
                trange = frange2trange(frange, fps)
            elif frange is None:
                frange = trange2frange(trange, fps)
        if frange is None and trange is None:
            frange = [sf, ef]
            trange = [st, et]
        plotduration = trange[1] - trange[0]
        print(f"plot duration / cage period: {plotduration / self.cage_period:.1f} [rotation] | {plotduration:.3f} [sec] | fps: {fps}")
        return frange, trange

    def trajectory(self, frange=None, trange=None, xyrange=0.35, slide=False):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(xlabel="y [mm]", ylabel="z [mm]", xrange=(-xyrange, xyrange), yrange=(-xyrange, xyrange), slide=slide)
        ax = axs[0]
        ax.set_aspect(1)
        frange_camera, trange_camera = self.get_ftrange(frange=frange, trange=trange)
        sf, ef = frange_camera
        y = self.cage[:, 0, 0][sf:ef]
        z = self.cage[:, 0, 1][sf:ef]
        ax.plot(y, z, lw=2, c='k', alpha=1)
        fig, ax = PlotterForCageVisualization.add_auxiliary_circles(fig, ax, radii=self.auxiliary_circles_radii, colors=self.auxiliary_circles_colors, lw=2, alpha=1, zorder=1, ls="--")
        log = (
            f"frange: {frange_camera}\n"
            f"trange: {trange_camera}\n"
        )
        return fig, ax, log

    def cagecoord_sound(self, refdata="sound_rms", frange=None, trange=None, markt=None, yrange=[(-0.35, 0.35), (-0.35, 0.35), (0, 24)], ytick=[0.1, 0.1, 5], offset_time=False, slide=False):
        frange_camera, trange_camera = self.get_ftrange(frange=frange, trange=trange)
        sf, ef = frange_camera
        st, et = trange_camera
        t = self.t_camera[sf:ef]
        if offset_time: t = t - t[0]
        y = self.cage[:, 0, 0][sf:ef]
        z = self.cage[:, 0, 1][sf:ef]
        margin = (et -st) * 0.05
        xrange = (-margin + t[0], et - st + t[0] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_31)
        fig, axs = plotter.myfig(xlabel=["", "", "time [sec]"], ylabel=["y [mm]", "z [mm]", ""], xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[2, 2, 0], sharex=[0, 0, 0], slide=slide)
        axs[0].plot(t, y, lw=1, c='k', alpha=1)
        axs[1].plot(t, z, lw=1, c='k', alpha=1)
        log = (
            f"frange_camera: {frange_camera}\n"
            f"trange_camera: {trange_camera}\n"
        )
        _log = ""
        if self.t_sound is not None:
            frange_sound, trange_sound = self.get_ftrange(frange=frange, trange=trange, fpsmode="audio")
            sf2, ef2 = frange_sound
            t2 = self.t_sound[sf2:ef2]
            if offset_time: t2 = t2 - t2[0]
            #### sound pressure
            if refdata == "sound_pressure":
                axs[2].plot(t2, self.sound[sf2:ef2], lw=1, c='k', alpha=1)
                axs[2].set_ylim(-50, 50)
                axs[2].set_yticks(np.arange(-50, 51, 10))
                axs[2].set_ylabel("sound pressure [Pa]")
                _log = (
                    f"frange_sound: {frange_sound}\n"
                )
            #### rms
            elif refdata == "sound_rms":
                timeprocessor = data_processor.TimeSeriesProcessor(self.sound, fs=48000)
                window_time = 0.02
                rms = timeprocessor.calc_rms(window_time=window_time)
                db = timeprocessor.pa2db(rms)
                axs[2].plot(t2, db[sf2:ef2], lw=1, c='k', alpha=1)
                ymin, ymax, ytick = 60, 120, 20
                axs[2].set_ylim(ymin, ymax)
                axs[2].set_yticks(np.arange(ymin, ymax + (ymax-ymin)*0.05, ytick))
                axs[2].set_ylabel("SPL [dB]")
                _log = (
                    f"frange_sound: {frange_sound}\n"
                    f"window_time: {window_time}\n"
                )
            #### spectrogram
            elif refdata == "spectrogram":
                vrange = (-60, -20)
                nperseg = 2**9
                noverlap = 0.5
                scaling = "density"
                mode = "psd"
                fft = myfft.Myfft(self.t_sound, self.sound, self.sample_rate)
                freq, t_segment, sxx = fft.compute_spectrogram(nperseg=nperseg, noverlap=noverlap, is_log=True, scaling=scaling, mode=mode)
                pcm = axs[2].pcolormesh(t_segment, freq/1000, sxx, cmap="viridis", shading="auto")
                axs[2].set_ylabel("frequency [Hz]")
                if vrange:
                    pcm.set_clim(vmin=vrange[0], vmax=vrange[1])
                _log = (
                    f"frange_sound: {frange_sound}\n"
                    f"vrange: {vrange}\n"
                    f"nperseg: {nperseg}\n"
                    f"noverlap: {noverlap}\n"
                    f"scaling: {scaling}\n"
                    f"mode: {mode}\n"
                )
        if markt:
            for e in markt:
                x1, x2 = e["trange"]
                color = e["color"]
                alpha = e["alpha"]
                for i in range(3):
                    axs[i].fill_betweenx(y=np.array([-1000, 1000]), x1=x1, x2=x2, color=color, alpha=alpha)
        log = log + _log
        return fig, axs, log

    def cagecoord(self, frange=None, trange=None, yrange=[(-0.35, 0.35), (-0.35, 0.35)], ytick=[0.1, 0.1, 5], offset_time=False, slide=False):
        frange_camera, trange_camera = self.get_ftrange(frange=frange, trange=trange)
        sf, ef = frange_camera
        st, et = trange_camera
        t = self.t_camera[sf:ef]
        if offset_time: t = t - t[0]
        y = self.cage[:, 0, 0][sf:ef]
        z = self.cage[:, 0, 1][sf:ef]
        margin = (et -st) * 0.05
        xrange = (-margin + t[0], et - st + t[0] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(xlabel=["", "time [sec]"], ylabel=["y [mm]", "z [mm]"], xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[2, 2], sharex=[0, 0], slide=slide)
        axs[0].plot(t, y, lw=2, c='k', alpha=1)
        axs[1].plot(t, z, lw=2, c='k', alpha=1)
        # for i in range(2):
        #     axs[i].axhline(y=0, lw=0.4, c='k')
        log = (
            f"frange: {frange_camera}\n"
            f"trange: {trange_camera}\n"
        )
        return fig, axs, log

    def spectrogram(self, datamode="camera", frange=None, trange=None, yrange=[(-50, 50), (0, 24)], ytick=[20, 5], markt=None, offset_time=False, slide=False):
        frange, trange = self.get_ftrange(frange=frange, trange=trange, fpsmode="audio")
        sf, ef = frange
        st, et = trange
        t = self.t_sound[sf:ef]
        if offset_time: t = t - t[0]
        margin = (et -st) * 0.05
        xrange = (-margin + t[0], et - st + t[0] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(xlabel=["", "time [sec]"], ylabel=["sound pressure [Pa]", "frequecy [Hz]"], xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[0, 0], sharex=[0, 0], slide=slide)
        #### sound pressure
        axs[0].plot(t, self.sound[sf:ef], lw=1, c='k', alpha=1)
        axs[0].set_ylabel("sound pressure [Pa]")
        #### rms
        # timeprocessor = data_processor.TimeSeriesDataProcessor(self.sound, fps=48000)
        # window_time = 0.02
        # rms = timeprocessor.calc_rms(window_time=window_time)
        # db = timeprocessor.cvt_pa2db(rms)
        # axs[0].plot(t, db[sf:ef], lw=1, c='k', alpha=1)
        # ymin, ymax, ytick = 60, 120, 20
        # axs[0].set_ylim(ymin, ymax)
        # axs[0].set_yticks(np.arange(ymin, ymax + (ymax-ymin)*0.05, ytick))
        # axs[0].set_ylabel("SPL [dB]")
        #### spectrogram
        vrange = (-60, -20)
        nperseg = 2**9
        noverlap = 0.5
        scaling = "density"
        mode = "psd"
        window_func = "hann"
        fft = myfft.Myfft(self.t_sound, self.sound, self.sample_rate)
        freq, t_segment, sxx = fft.compute_spectrogram(nperseg=nperseg, noverlap=noverlap, is_log=True, scaling=scaling, mode=mode, window=window_func)
        pcm = axs[1].pcolormesh(t_segment, freq/1000, sxx, cmap="viridis", shading="auto")
        if vrange:
            pcm.set_clim(vmin=vrange[0], vmax=vrange[1])
        if markt:
            for e in markt:
                x1, x2 = e["trange"]
                color = e["color"]
                alpha = e["alpha"]
                axs[0].fill_betweenx(y=np.array([-1000, 1000]), x1=x1, x2=x2, color=color, alpha=alpha)
        log = (
            f"trange: {trange}\n"
            f"frange: {frange}\n"
            f"nperseg: {nperseg}\n"
            f"noverlap: {noverlap}\n"
            f"window_func: {window_func}\n"
            f"scaling: {scaling}\n"
            f"mode: {mode}\n"
            f"vrange: {vrange}\n"
        )
        return fig, axs, log

    def fft(self, datalist, slide=False):
        log = ""
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.RECTANGLE_FIG)
        fig, axs = plotter.myfig(xlabel="frequency [kHz]", ylabel="psd [dB]", xrange=(-1, 26), yrange=(-80, -10), ytick_0center=False, slide=slide, xsigf=0, ysigf=0, xtick=5, ytick=10)
        for e in datalist:
            note = e["note"]
            frange = e["frange"]
            trange = e["trange"]
            color = e["color"]
            frange, trange = self.get_ftrange(frange=frange, trange=trange, fps="audio")
            sf, ef = frange
            st, et = trange
            t = self.t_sound[sf:ef]
            ax = axs[0]
            sf, ef = frange
            st, et = trange
            fft_size = 2**12
            overlap = 0.5
            lastseg = "cut"
            window_func = "hann"
            mode = "psd"
            fft = myfft.Myfft(self.t_sound, self.sound, self.sample_rate)
            freq, sp = fft.compute_segmented_fft(mode=mode, tranges=[trange], fft_size=fft_size, overlap=overlap, lastseg=lastseg, window_func=window_func, is_log=True)
            ax.plot(freq/1000, sp, lw=2, c=color, alpha=1)
            _log = (
                f"note: {note}\n"
                f"trange: {trange}\n"
                f"frange: {frange}\n"
                f"fftsize {fft_size}\n"
                f"overlap: {overlap}\n"
                f"window_func: {window_func}\n"
                f"lastseg: {lastseg}\n"
                f"mode: {mode}\n"
            )
            log = log + _log
        return fig, ax, log


def get_result_file(datadir, glob=""):
    datafile = list(datadir.glob(glob))
    if len(datafile) == 1:
        datafile = datafile[0]
    else:
        raise ValueError(f"file must be 1, {len(datadir)} file was found by pattern {glob}.")
    print(f"datafile: {str(datafile)}")
    return datafile

def save_fig_log(fig, log, outfilepath):
    outfilepath = check_existfile_get_newfilepath(outfilepath)
    outfilename = outfilepath.stem
    outdir = outfilepath.parent
    fig.savefig(outdir / f"{outfilename}.png", dpi=300)
    text = str(log)
    with open(outdir / f"{outfilename}.log", "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == '__main__':
    print('---- test ----')

    # tc = 23, 3 # TYN
    # tc, sc = 4, 5 # v2
    tc, sc = 35, 1
    trange = [0, 0.05]
    trange2 = [5, 5.05]
    # trange = None
    markt = [
        {"trange": trange, "color": 'm', "alpha": 0.2},
        {"trange": trange2, "color": 'g', "alpha": 0.2},
    ]
    note = "note"
    save = 0

    print("\n---- load datamap ----\n")
    datamaploader = data_handler.DataMapLoader(r"D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    testinfo = datamaploader.extract_testinfo(tc)
    datainfo = datamaploader.extract_info_from_tcsc(tc, sc)
    rec = datainfo["recording_number"]
    print(f"test information:\n{testinfo}")
    datadir_camera = config.ROOT / "results" / "260327_cage_visualization_v_1_1_10"
    print("\n---- load acoustic data ----\n")
    datadir_audio = Path(r"D:/100_data")
    audiofile = get_result_file(datadir_audio, glob=f"**/???{rec}.csv")
    audiodataloader = data_handler.AudioDataLoader(audiofile)
    print("\n---- load coordinate data ----\n")
    cagefile = get_result_file(datadir_camera, f"**/tc{tc:02}_sc{sc:02}_res_cage.csv")
    cage = pl.read_csv(cagefile, has_header=True, infer_schema_length=1000)
    cage_p = np.column_stack([cage["cy [mm]"], cage["cz [mm]"]])
    print("\n---- rot speed data ----\n")
    rotfile = get_result_file(datadir_camera, f"**/tc{tc:02}_sc{sc:02}_res_rotation_speed.csv")
    rotspeed = pl.read_csv(rotfile, has_header=True, infer_schema_length=1000)
    rotspeed = rotspeed["angular velosity [rad/sec]"]

    fftdatalist = [
        {"note": "noisy", "frange": None, "trange": (0.1, 1.1), "color": "m"},
        {"note": "silent", "frange": None, "trange": (3, 4), "color": "g"},
    ]
    fftmarkt = [
        {"trange": fftdatalist[0]["trange"], "color": 'm', "alpha": 0.2},
        {"trange": fftdatalist[1]["trange"], "color": 'g', "alpha": 0.2},
    ]

    plotter = PlotterForCageVisualization(t_camera=cage["time [sec]"], cage=cage_p, rotspeed=rotspeed, t_sound=audiodataloader.t, sound=audiodataloader.sound, testinfo=testinfo)
    fig_coord_sound, ax_coord_sound, log_coord_sound = plotter.cagecoord_sound(trange=None, slide=True, markt=markt)
    fig_trj, ax_trj, log_trj = plotter.trajectory(trange=trange, slide=True)
    fig_trj2, ax_trj2, log_trj2 = plotter.trajectory(trange=trange2, slide=True)
    fig_coord, ax_coord, log_coord = plotter.cagecoord(trange=trange, slide=True)
    fig_coord2, ax_coord2, log_coord2 = plotter.cagecoord(trange=trange2, slide=True)
    fig_spectrogram, ax_spectrogram, log_spectrogram = plotter.spectrogram(markt=fftmarkt, slide=True)
    fig_fft, ax_fft, log_fft = plotter.fft(datalist=fftdatalist, slide=True)
    plt.show()

    outdir = Path(r"D:/1005_tyn/01_thematic_materials/251201_achivement_report/images")
    outdir = outdir / "visualization_test" / "latest"
    outdir.mkdir(exist_ok=True, parents=True)
    head = f"tc{tc}_sc{sc}_rec{rec}_{note}"
    outdatalist = [
        {"outfilepath": outdir / f"{head}_trj_noisy.png", "fig": fig_trj, "log": log_trj},
        {"outfilepath": outdir / f"{head}_trj_silent.png", "fig": fig_trj2, "log": log_trj2},
        {"outfilepath": outdir / f"{head}_coordsound.png", "fig": fig_coord_sound, "log": log_coord_sound},
        {"outfilepath": outdir / f"{head}_coord_noisy.png", "fig": fig_coord, "log": log_coord},
        {"outfilepath": outdir / f"{head}_coord_silent.png", "fig": fig_coord2, "log": log_coord2},
        {"outfilepath": outdir / f"{head}_spectrogram.png", "fig": fig_spectrogram, "log": log_spectrogram},
        {"outfilepath": outdir / f"{head}_fft.png", "fig": fig_fft, "log": log_fft},
    ]

    log = (
        f"tc, sc: {tc}, {sc}\n"
        f"trange: {trange}\n"
        f"trange2: {trange2}\n"
        f"test information:\n{testinfo}\n"
    )

    if save:
        for e in outdatalist:
            fig = e["fig"]
            log = e["log"]
            outfilepath = e["outfilepath"]
            print(outfilepath)
            print("saving")
            save_fig_log(fig, log, outfilepath)
            print("saved")
        with open(outdir/f"{head}.log", 'w') as f:
            f.write(log)


