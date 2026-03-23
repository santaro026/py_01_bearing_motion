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

import config
import data_handler
import data_processor

class MotionPlotter:
    @staticmethod
    def add_auxiliary_cicles(fig, ax, radii, colors='k', lw=1, alpha=1):
        for r, c in zip(radii, colors):
            circle = plt.Circle((0, 0), r, color=c, fill=False, lw=lw, alpha=alpha)
            ax.add_artist(circle)
        return fig, ax
    def __init__(self, name=""):
        self.name = name

    def plot_trajectory(self, xs, ys, colors=['r', 'b', 'g', 'm', 'c', 'y']*100, lws=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lw=2, auxiliary_circles_alpha=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False, slide=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(slide=slide, title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        axs[0].set_aspect(1)
        for i in range(len(xs)):
            axs[0].plot(xs[i], ys[i], c=colors[i], lw=lws[i])
        axs[0].axhline(y=0, lw=0.4, c='k')
        axs[0].axvline(x=0, lw=0.4, c='k')
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MotionPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lw=auxiliary_circles_lw, alpha=auxiliary_circles_alpha)
        return fig, axs

    def plot_vstime2(self, ts, fts, colors=['k']*2, lws=[0.4]*2, alphas=[1]*2, xlabel='time [sec]', ylabel=None, xrange=None, yrange=None, xsigf=2, ysigf=2, xtick=None, ytick=None, xtick_0center=True, ytick_0center=True, title='', notell='', notelr='', plottype=['plot']*2, slide=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(sharex=[0, 0], sharey=False, title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, grid=0, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, slide=slide)
        for i in range(len(axs)):
            if plottype[i] == 'plot':
                axs[i].plot(ts[i], fts[i], lw=lws[i], c=colors[i], alpha=alphas[i])
            elif plottype[i] == 'scatter':
                axs[i].scatter(ts[i], fts[i], s=lws[i], c=colors[i], alpha=alphas[i])
        return fig, axs

    def plot_vstime3(self, ts, fts, colors=['k']*3, lws=[0.4]*3, alphas=[1]*3, xlabel=['', '', 'time [sec]'], ylabel=None, xrange=None, yrange=None, xsigf=2, ysigf=2, xtick=None, ytick=None, xtick_0center=True, ytick_0center=True, title='', notell='', notelr='', plottype=['plot']*3, slide=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.LANDSCAPE_FIG_31)
        fig, axs = plotter.myfig(sharex=[0, 0, 0], sharey=False, title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, grid=0, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, slide=slide)
        for i in range(len(axs)):
            if plottype[i] == 'plot':
                axs[i].plot(ts[i], fts[i], lw=lws[i], c=colors[i], alpha=alphas[i])
            elif plottype[i] == 'scatter':
                axs[i].scatter(ts[i], fts[i], s=lws[i], c=colors[i], alpha=alphas[i])
        return fig, axs

    def plot_probability(self, probability_map, bins=100, cmap='viridis', vrange=(None, None), auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=1, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', slide=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, grid=0, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, slide=slide)
        axs[0].set_aspect(1)
        axs[0].imshow(probability_map.T, origin='lower', cmap=cmap, extent=([xrange[0], xrange[1], yrange[0], yrange[1]]), vmin=vrange[0], vmax=vrange[1])
        axs[0].axhline(y=0, lw=0.4, c='k')
        axs[0].axvline(x=0, lw=0.4, c='k')
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MotionPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        return fig, axs[0]

    def animate_trajectory(self, xs, ys, trjcolors=['k', 'r', 'b'] + ['k']*97, trjdispmax=[100, 100] + [1]*98, trjlws=[0.4]*100, trjlalphas=[0.4]*100, trjmarkersizes=[8, 20, 20] + [8]*97, trjmarkeralphas=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        axs[0].set_aspect(1)
        axs[0].axhline(y=0, lw=0.4, c='k')
        axs[0].axvline(x=0, lw=0.4, c='k')
        axs[1].axis("off")
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MyPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        data_list = []
        for i in range(len(xs)):
            _data = {"id": 0, "data": [xs[i], ys[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            data_list.append(_data)
        vct_list = None
        vline_list = None
        hline_list = None
        offset = MyPlotter.offsetpx2axAxes(fig, axs[1], text=f"frame: 10000", fontsize=10, fontfamily="monospace", xem=1.2, yem=1.2)
        note_list = [
            {"id": 1, "prefix": "frame: ", "data": np.arange(len(t)), "sigf": 0, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]), "fontsize": 10, "fontfamily": "monospace"},
            {"id": 1, "prefix": "time:  ", "data": np.round(t, 3), "sigf": 3, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]*2), "fontsize": 10, "fontfamily": "monospace"},
            ]
        animator = MyAnimator(fig, axs, data_list=data_list, vct_list=vct_list, vline_list=vline_list, hline_list=hline_list, note_list=note_list)
        ani = animator.make_func_ani(skip=2, interval=10)
        return fig, axs, ani

    def animate_trajectory2(self, xs, ys, ys1, zs1, gravity_angle, time, trjcolors=['k', 'r', 'b'] + ['k']*97, trjdispmax=[100, 100] + [1]*98, trjlws=[0.4]*100, trjlalphas=[0.4]*100, trjmarkersizes=[8, 20, 20] + [8]*97, trjmarkeralphas=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY_2)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        for i in range(2):
            fig, axs[i] = MyPlotter.draw_center_line(fig, axs[i])
            axs[i].set_aspect(1)
            if auxiliary_circles_radii is not None:
                fig, axs[i] = MyPlotter.add_auxiliary_cicles(fig, axs[i], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        axs[2].axis("off")
        data_list = []
        for i in range(len(xs)):
            _data0 = {"id": 0, "data": [xs[i], ys[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            _data1 = {"id": 1, "data": [ys1[i], zs1[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            data_list.append(_data0)
            data_list.append(_data1)
        vct_list = [
            {"mode": "force", "id": 0, "data": [np.zeros_like(xs[0]), np.zeros_like(ys[0]), np.zeros_like(xs[0]), -np.ones_like(ys[0])], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
            {"mode": "force", "id": 1, "data": [np.zeros_like(xs[0]), np.zeros_like(ys[0]), np.cos(gravity_angle), np.sin(gravity_angle)], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
        ]
        vline_list = None
        hline_list = None
        offset = MyPlotter.offsetpx2axAxes(fig, axs[2], text=f"frame: 10000", fontsize=10, fontfamily="monospace", xem=1.2, yem=1.2)
        note_list = [
            {"id": 2, "prefix": "frame: ", "data": np.arange(len(time)), "sigf": 0, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]), "fontsize": 10, "fontfamily": "monospace"},
            {"id": 2, "prefix": "time:  ", "data": np.round(time, 3), "sigf": 3, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]*2), "fontsize": 10, "fontfamily": "monospace"},
            ]
        animator = MyAnimator(fig, axs, data_list=data_list, vct_list=vct_list, vline_list=vline_list, hline_list=hline_list, note_list=note_list)
        ani = animator.make_func_ani(skip=2, interval=10)
        return fig, axs, ani

    def animate_trajectory3(self, xs, ys, ys1, zs1, time, ts, fts, gravity_angle, trjcolors=['k', 'r', 'b'] + ['k']*97, trjdispmax=[100, 100] + [1]*98, trjlws=[0.4]*100, trjlalphas=[0.4]*100, trjmarkersizes=[8, 20, 20] + [8]*97, trjmarkeralphas=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, ftcolors=['k']*100, ftlws=[0.4]*100, ftalphas=[1]*100, title='', xlabel=["y [mm]", "y [mm]", "time [sec]", None], ylabel=["y [mm]", "y [mm]", "time [sec]", None], xrange=[(-0.5, 0.5), (-0.5, 0.5), (0, 1), None], yrange=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), None], xtick=[0.1, 0.1, 0.2, None], ytick=[0.1, 0.1, 0.1, None], xtick_0center=True, ytick_0center=True, xsigf=[2, 2, 1, None], ysigf=[2, 2, 2, None], notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY_WITH_TIMESERIES)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        for i in range(2):
            fig, axs[i] = MyPlotter.draw_center_line(fig, axs[i])
            axs[i].set_aspect(1)
            if auxiliary_circles_radii is not None:
                fig, axs[i] = MyPlotter.add_auxiliary_cicles(fig, axs[i], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        for i in range(len(ts)):
            axs[2].plot(ts[i], fts[i], color=ftcolors[i], lw=ftlws[i], alpha=ftalphas[i])
        axs[2].axhline(y=0, xmin=-10000, xmax=10000, lw=0.8, alpha=1, color='k')
        axs[3].axis("off")
        data_list = []
        for i in range(len(xs)):
            _data0 = {"id": 0, "data": [xs[i], ys[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            _data1 = {"id": 1, "data": [ys1[i], zs1[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            data_list.append(_data0)
            data_list.append(_data1)
        vct_list = [
            {"mode": "force", "id": 0, "data": [np.zeros_like(xs[0]), np.zeros_like(ys[0]), np.zeros_like(xs[0]), -np.ones_like(ys[0])], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
            {"mode": "force", "id": 1, "data": [np.zeros_like(xs[0]), np.zeros_like(ys[0]), np.cos(gravity_angle), np.sin(gravity_angle)], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
        ]
        vline_list = [
            {"id": 2, "data": ts[0], "color": 'k', "lw": 0.4, "alpha": 1, "ymin": -10000, "ymax": 10000},
        ]
        hline_list = [
            {"id": 2, "data": fts[0], "color": 'k', "lw": 0.4, "alpha": 1, "xmin": -10000, "xmax": 10000},
        ]
        offset = MyPlotter.offsetpx2axAxes(fig, axs[3], text=f"frame: 10000", fontsize=10, fontfamily="monospace", xem=1.2, yem=1.2)
        note_list = [
            {"id": 3, "prefix": "frame: ", "data": np.arange(len(time)), "sigf": 0, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]), "fontsize": 10, "fontfamily": "monospace"},
            {"id": 3, "prefix": "time:  ", "data": np.round(time, 3), "sigf": 3, "disp_width": 5, "suffix": " [sec]", "position": (0, 1-offset[1]*2), "fontsize": 10, "fontfamily": "monospace"},
            ]
        animator = MyAnimator(fig, axs, data_list=data_list, vct_list=vct_list, vline_list=vline_list, hline_list=hline_list, note_list=note_list)
        ani = animator.make_func_ani(skip=2, interval=10)
        return fig, axs, ani


class ResultLoader:
    def __inti__(self, cage, markers, rotspeed, sound):
        self.cage = cage
        self.markers = markers
        self.rotspeed = rotspeed
        self.sound = sound


class PlotterForCageVisualization:
    @staticmethod
    def cvt_var2list(*args):
        res = []
        for var in args:
            if isinstance(var, (list, np.ndarray)):
                for var2 in var:
                    print(var2)
                    res.append(var2)
                continue
            res.append(var)
        return res
    @staticmethod
    def add_auxiliary_circles(fig, ax, radii, colors='k', lw=1, alpha=1, zorder=1):
        for r, c in zip(radii, colors):
            circle = plt.Circle((0, 0), r, color=c, fill=False, lw=lw, alpha=alpha, zorder=zorder)
            ax.add_artist(circle)
        return fig, ax
    def __init__(self, t_camera=None, cage=None, markers=None, rotspeed=None, t_sound=None, sound=None, testinfo=None, notell="", notelr=""):
        self.t_camera = t_camera
        self.fps = int(1 / (t_camera[1] - t_camera[0]))
        self.cage = cage # cage coordinate
        self.markers = markers # markers coordinate
        self.rotspeed = rotspeed # rotation speed
        self.rotspeed_avg = np.nanmean(self.rotspeed)
        self.cage_period = 1 / abs(self.rotspeed_avg / (2*np.pi))
        self.t_sound = t_sound
        self.sample_rate = int(1 / (t_sound[1] - t_sound[0]))
        self.sound = sound
        self.testinfo = testinfo
        if self.testinfo["dp_measured"]:
            self.dp = self.testinfo["dp_measured"]
            self.note_dp = 'dp is measured.'
        elif self.testinfo["dp_drawing"]:
            self.dp = self.testinfo["dp_drawing"]
            self.note_dp = 'dp is nominal.'
        else:
            self.dp = np.nan
            self.note_dp = 'dp is none'
        if self.testinfo["dl_measured"]:
            self.dl = self.testinfo["dl_measured"]
            self.note_dl = 'dl is measured.'
        elif self.testinfo["dl_drawing"]:
            self.dl = self.testinfo["dl_drawing"]
            self.note_dl = 'dl is nominal.'
        else:
            self.dl = np.nan
            self.note_dl = 'dl is none.'
        if isinstance(self.dp, list):
            self.dp = np.array(self.dp)
        if isinstance(self.dl, list):
            self.dl = np.array(self.dl)
        self.note_dpdl = f'{self.note_dp} {self.note_dl}'
        self.dp_list = PlotterForCageVisualization.cvt_var2list(self.dp)
        self.dl_list = PlotterForCageVisualization.cvt_var2list(self.dl)
        num_dp, num_dl = len(self.dp_list), len(self.dl_list)
        self.auxiliary_circles_radii = [d/2 for d in (self.dp_list + self.dl_list)]
        self.auxiliary_circles_colors = ['b']*num_dp + ['r']*num_dl
        self.notell = notell
        self.notelr = notelr
        print(repr(self))
    def __repr__(self):
        return (
            f"---- camera ----\n"
            f"N: {len(self.t_camera)}\n"
            f"fps: {self.fps} [frame/sec]\n"
            f"rot speed: {self.rotspeed_avg:.2f} [rad/sec]\n"
            f"cage rev period: {self.cage_period:.3f} [sec]\n"
            f"---- audio ----\n"
            f"N: {len(self.t_sound)}\n"
            f"sample_rage: {self.sample_rate} [sample/sec]\n"
        )
    def get_ftrange(self, frange, trange, fps="camera"):
        sf = 0
        if fps == "camera":
            fps = self.fps
            ef = len(self.t_camera)
        elif fps == "audio":
            fps = self.sample_rate
            ef = len(self.t_sound)
        st, et = sf / fps, ef / fps
        if frange:
            sf, ef = frange
            st, et = sf / fps, ef / fps
        if trange:
            if frange:
                print(f"[WANR] both frame and time range was designated, using time range {trange}")
            st, et = trange
            sf, ef = int(fps * st), int(fps * et)
        plottime = (ef - sf) / fps
        print(f"plot time / cage period: {plottime / self.cage_period:.1f} [rotation] | {plottime:.3f} [sec] | fps: {fps}")
        frange = (sf, ef)
        trange = (st, et)
        return frange, trange
        # return sf, ef, st, et

    def trajectory(self, frange=None, trange=None, xyrange=0.4, slide=False):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(xlabel="y [mm]", ylabel="z [mm]", xrange=(-xyrange, xyrange), yrange=(-xyrange, xyrange), slide=slide)
        ax = axs[0]
        ax.set_aspect(1)
        frange_camera, trange_camera = self.get_ftrange(frange, trange)
        sf, ef = frange_camera
        y = self.cage[:, 0][sf:ef]
        z = self.cage[:, 1][sf:ef]
        ax.plot(y, z, lw=2, c='k', alpha=1)
        fig, ax = PlotterForCageVisualization.add_auxiliary_circles(fig, ax, radii=self.auxiliary_circles_radii, colors=self.auxiliary_circles_colors, lw=2, alpha=1, zorder=1)
        log = (
            f"frange: {frange_camera}\n"
            f"trange: {trange_camera}\n"
        )
        return fig, ax, log

    def cagecoord_sound(self, refdata="sound_rms", frange=None, trange=None, markt=None, yrange=[(-0.4, 0.4), (-0.4, 0.4), (0, 24)], ytick=[0.2, 0.2, 5], offset_time=False, slide=False):
        frange_camera, trange_camera = self.get_ftrange(frange, trange)
        sf, ef = frange_camera
        st, et = trange_camera
        t = self.t_camera[sf:ef]
        if offset_time: t = t - t[0]
        y = self.cage[:, 0][sf:ef]
        z = self.cage[:, 1][sf:ef]
        margin = (et -st) * 0.05
        xrange = (-margin + t[0], et - st + t[0] + margin)
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_31)
        fig, axs = plotter.myfig(xlabel=["", "", "time [sec]"], ylabel=["y [mm]", "z [mm]", ""], xrange=xrange, yrange=yrange, ytick=ytick, ysigf=[2, 2, 0], sharex=[0, 0, 0], slide=slide)
        axs[0].plot(t, y, lw=1, c='k', alpha=1)
        axs[1].plot(t, z, lw=1, c='k', alpha=1)
        # for i in range(2):
        #     axs[i].axhline(y=0, lw=0.4, c='k')
        log = (
            f"frange_camera: {frange_camera}\n"
            f"trange_camera: {trange_camera}\n"
        )

        frange_sound, trange_sound = self.get_ftrange(frange, trange, fps="audio")
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
            timeprocessor = data_processor.TimeSeriesDataProcessor(self.sound, fps=48000)
            window_time = 0.02
            rms = timeprocessor.calc_rms(window_time=window_time)
            db = timeprocessor.cvt_pa2db(rms)
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

    def cagecoord(self, frange=None, trange=None, yrange=[(-0.4, 0.4), (-0.4, 0.4)], ytick=[0.2, 0.2, 5], offset_time=False, slide=False):
        frange_camera, trange_camera = self.get_ftrange(frange, trange)
        sf, ef = frange_camera
        st, et = trange_camera
        t = self.t_camera[sf:ef]
        if offset_time: t = t - t[0]
        y = self.cage[:, 0][sf:ef]
        z = self.cage[:, 1][sf:ef]
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

    def spectrogram(self, frange=None, trange=None, yrange=[(-50, 50), (0, 24)], ytick=[20, 5], markt=None, offset_time=False, slide=False):
        frange, trange = self.get_ftrange(frange, trange, fps="audio")
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
            frange, trange = self.get_ftrange(frange, trange, fps="audio")
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

def check_existfile_get_newfilepath(filepath, max_num=100):
    # curdir = Path.cwd()
    filepath = Path(filepath).resolve()
    targetdir = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix
    match = re.match(r'^(.*)_(\d+)$', stem) # basname_n.ext
    if match:
        base = match.group(1)
        n = int(match.group(2))
    else:
        base = stem
        n = 0
    same_suffix_files = [f.name for f  in targetdir.iterdir() if f.is_file() and f.suffix == suffix]
    new_name = filepath.name
    while new_name in same_suffix_files:
        n += 1
        new_name = f"{base}_{n}{suffix}"
        if (n >= max_num):
            raise ValueError('too many files with the same name exist. change output file name, or change max permitted number.')
    return targetdir / new_name

if __name__ == '__main__':
    print('---- test ----')

    # tc = 23, 3 # TYN
    tc, sc = 4, 5 # v2
    # trange = [2.5, 3]
    trange = [0.2, 0.25]
    trange2 = [5, 5.05]
    # trange = None
    markt = [
        {"trange": trange, "color": 'm', "alpha": 0.2},
        {"trange": trange2, "color": 'g', "alpha": 0.2},
    ]
    note = "note"
    save = 1

    print("\n---- load datamap ----\n")
    datamaploader = data_handler.DataMapLoader(r"D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    testinfo = datamaploader.extract_testinfo(tc)
    datainfo = datamaploader.extract_info_from_tcsc(tc, sc)
    rec = datainfo["recording_number"]
    print(f"test information:\n{testinfo}")
    datadir_camera = config.ROOT / "results" / "260123_cage_visualization_v_1_1_9"
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
    outdir = outdir / "visualization_test"
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


