# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:02:28 2025
@author: santaro



"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


from enum import Enum, auto
from pathlib import Path
import json

import config

# plt.rcParams["font.family"] = "DejaVu Sans"
# plt.rcParams["font.family"] = "monospace"
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"

class PlotSizeCode(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()
    SQUARE_ILLUST = auto()
    SQUARE_FIG = auto()
    RECTANGLE_FIG = auto()
    LANDSCAPE_FIG_21 = auto()
    LANDSCAPE_FIG_31 = auto()
    TRAJECTORY = auto()
    TRAJECTORY_2 = auto()
    TRAJECTORY_22 = auto()
    TRAJECTORY_WITH_TIMESERIES = auto()
    TRAJECTORY_WITH_TIMESERIES2 = auto()

class MyPlotter:
    @staticmethod
    def cnvt_val2list(N, *args):
        vals = []
        for val in args:
            if not isinstance(val, list):
                val = [val] * N
            # elif len(val) == 1:
                # val = [val[0]] * N
            elif len(val) < N:
                for i in range(N-len(val)):
                    val.append(None)
            vals.append(val)
        return vals
    @staticmethod
    def make_formatter(decimal_places, hide0=False):
        def formatter(x, pos=None):
            if x == 0:
                if hide0:
                    return ""
                else:
                    return '0'
            else:
                return f"{x:.{decimal_places}f}"
        return ticker.FuncFormatter(formatter)
    @staticmethod
    def get_axsfromfig(fig):
        axs_load = []
        for _ax in fig.axes:
            l_list = []
            for _l in _ax.get_lines():
                _l_dict = {
                    'xdata': _l.get_xdata(),
                    'ydata': _l.get_ydata(),
                    'color': _l.get_color(),
                    'linestyle': _l.get_linestyle(),
                    'marker': _l.get_marker(),
                    'label': _l.get_label()
                    }
                l_list.append(_l_dict)
            coll_list = []
            for _coll in _ax.collections:
                xydata = _coll.get_offsets().T
                _coll_dict ={
                    'xdata': xydata[0],
                    'ydata': xydata[1],
                    'size': _coll.get_sizes(),
                    'facecolors': _coll.get_facecolors(),
                    'edgecolors': _coll.get_edgecolors()
                    }
                coll_list.append(_coll_dict)
            _ax_dict = {
                'ax': _ax,
                'xlim': _ax.get_xlim(),
                'ylim': _ax.get_ylim(),
                'xticks': _ax.get_xticks(),
                'yticks': _ax.get_yticks(),
                'lines': l_list,
                'collections': coll_list
                }
            axs_load.append(_ax_dict)
        return axs_load
    @staticmethod
    def measure_text_size_px(fig, text, fontsize, fontfamily):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        t = fig.text(0, 0, text, fontsize=fontsize, fontfamily=fontfamily, transform=fig.transFigure)
        bbox = t.get_window_extent(renderer=renderer)
        width_px, height_px = bbox.width, bbox.height
        t.remove()
        return width_px, height_px
    @staticmethod
    def calc_text_offset_prcisely(fig, ax, text, fontsize, fontfamily, xem=1, yem=0):
        w_px, h_px = MyPlotter.measure_text_size_px(fig, text, fontsize, fontfamily)
        dx_inch = w_px / fig.dpi * xem
        dy_inch = h_px / fig.dpi * yem
        return mtransforms.ScaledTranslation(dx_inch, dy_inch, fig.dpi_scale_trans)
    @staticmethod
    def measure_text_size_pt(text, fontsize, fontfamily):
        tp = TextPath((0, 0), text, size=fontsize, prop=fm.FontProperties(family=fontfamily))
        bbox = tp.get_extents()
        width_pt, height_pt = bbox.width, bbox.height
        return width_pt, height_pt
    @staticmethod
    def calc_text_offset(fig, ax, text, fontsize, fontfamily, xem=1, yem=0):
        string_width_pt, char_height_pt = MyPlotter.measure_text_size_pt(text, fontsize, fontfamily)
        char_width_pt = string_width_pt / len(text)
        dx_inch = char_width_pt * xem / 72
        dy_inch = char_height_pt * yem / 72
        return mtransforms.ScaledTranslation(dx_inch, dy_inch, fig.dpi_scale_trans)
    @staticmethod
    def offsetpx2axAxes(fig, ax, text, fontsize, fontfamily, xem=1, yem=1):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_w_inch, ax_h_inch = bbox.width, bbox.height
        ax_w_px, ax_h_px = ax_w_inch * fig.dpi, ax_h_inch * fig.dpi
        offset_px = MyPlotter.calc_text_offset(fig=fig, ax=ax, text=text, fontsize=fontsize, fontfamily=fontfamily, xem=xem, yem=yem).transform((0, 0))
        offset_Axes = offset_px / np.array([ax_w_px, ax_h_px])
        return offset_Axes
    @staticmethod
    def offset_em(fig, fontsize, xem=1, yem=0):
        xpt = fontsize * xem
        ypt = fontsize * yem
        dx_inch = xpt / 72
        dy_inch = ypt / 72
        return mtransforms.ScaledTranslation(dx_inch, dy_inch, fig.dpi_scale_trans)
    def __init__(self, sizecode, json_path=config.ROOT/"config"/"plot_settings.json"):
        self.json_path = json_path
        with open(json_path, "r") as f:
            self.fig_settings = json.load(f)
        self.sizecode = sizecode
        self.load_settings()
    def _apply_settings(self, settings, suffix):
        keys_std = ["num_axes", "labelsize", "ticklength", "tickwidth", "major_grid_lw", "minor_grid_lw"]
        keys_tuple = ["figsize", "marge_size", "notell_pos", "notelr_pos",
                        "gs_shape", "gs_width_ratios", "gs_height_ratios", "gs_whspace",
                        "gsub_shape", "gsub_height_ratios", "gsub_width_ratios", "gsub_whspace"
                        ]
        for k in keys_std:
            if not k in settings:
                settings[k] = None
                # raise KeyError(k)
            setattr(self, f"{k}{suffix}", settings[k])
        for k in keys_tuple:
            if not k in settings:
                # raise KeyError(k)
                setattr(self, f"{k}{suffix}", None)
            else:
                setattr(self, f"{k}{suffix}", tuple(settings[k]))
    def _get_settings(self, *args, **kwargs):
        isslide = kwargs["slide"]
        attrs = []
        for attr_name in args:
            suffix = "_slide" if isslide else ""
            attr = getattr(self, f"{attr_name}{suffix}")
            attrs.append(attr)
        return attrs
    def load_settings(self):
        try:
            settings = self.fig_settings[self.sizecode.value]["analysis"]
            settings_slide = self.fig_settings[self.sizecode.value]["slide"]
        except KeyError as e:
            raise ValueError(f"{self.sizecode.value} cannot be found in self.fig_settings.") from e
        self._apply_settings(settings, suffix="")
        self._apply_settings(settings_slide, suffix="_slide")

    def myfig(self, sharex=False, sharey=False, title=None, notell=None, notelr=None, xlabel=None, ylabel=None, xrange=None, yrange=None, xtick=None, ytick=None, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, grid=False, slide=False):
        figsize, num_axes, marge_size = self._get_settings("figsize", "num_axes", "marge_size", slide=slide)
        gs_shape, gs_height_ratios, gs_width_ratios, gs_whspace = self._get_settings("gs_shape", "gs_height_ratios", "gs_width_ratios", "gs_whspace", slide=slide)
        gsub_shape, gsub_height_ratios, gsub_width_ratios, gsub_whspace, labelsize, ticklength, tickwidth, major_grid_lw, minor_grid_lw, marge_size, notell_pos, notelr_pos = self._get_settings("gsub_shape", "gsub_height_ratios", "gsub_width_ratios", "gsub_whspace", "labelsize", "ticklength", "tickwidth", "major_grid_lw", "minor_grid_lw", "marge_size", "notell_pos", "notelr_pos", slide=slide)
        labelsize, ticklength, tickwidth, major_grid_lw, minor_grid_lw = self._get_settings("labelsize", "ticklength", "tickwidth", "major_grid_lw", "minor_grid_lw", slide=slide)
        notell_pos, notelr_pos = self._get_settings("notell_pos", "notelr_pos", slide=slide)
        sharex, sharey, xlabel, ylabel, xrange, yrange, xtick, ytick, grid, xsigf, ysigf , xtick_0center, ytick_0center = MyPlotter.cnvt_val2list(num_axes, sharex, sharey, xlabel, ylabel, xrange, yrange, xtick, ytick, grid, xsigf, ysigf, xtick_0center, ytick_0center)
        fig = plt.figure(figsize=figsize)
        axs = np.empty(num_axes, dtype=object)
        self.gs_master = GridSpec(gs_shape[0], gs_shape[1], figure=fig,
                    left=marge_size[0], right=1-marge_size[1], bottom=marge_size[2], top=1-marge_size[3],
                    wspace=gs_whspace[0], hspace=gs_whspace[1], width_ratios=gs_width_ratios, height_ratios=gs_height_ratios)
        self.gs = np.array([self.gs_master[r, c] for r in range(self.gs_master.nrows) for c in range(self.gs_master.ncols)]).flatten()
        i = 0
        self.gsub_masters = []
        if gsub_shape is None: gsub_shape = [[1, 1] for i in range(len(self.gs))]
        if gsub_height_ratios is None: gsub_height_ratios = [[1] for i in range(len(self.gs))]
        if gsub_width_ratios is None: gsub_width_ratios = [[1] for i in range(len(self.gs))]
        if gsub_whspace is None: gsub_whspace = [[1, 1] for i in range(len(self.gs))]
        for j in range(len(self.gs)):
            gsub_master = self.gs[j].subgridspec(gsub_shape[j][0], gsub_shape[j][1], wspace=gsub_whspace[j][0], hspace=gsub_whspace[j][1], width_ratios=gsub_width_ratios[j], height_ratios=gsub_height_ratios[j])
            self.gsub_masters.append(gsub_master)
            gsub = np.array([gsub_master[r, c] for r in range(gsub_master.nrows) for c in range(gsub_master.ncols)]).flatten()
            for k in range(len(gsub)):
                ss = gsub[k]
                sx, sy = sharex[i], sharey[i]
                if (sx is False or sx == i) and (sy is False or sy == i):
                    axs[i] = fig.add_subplot(ss)
                elif (sx is not False and sx != i) and sy is False:
                    axs[i] = fig.add_subplot(ss, sharex=axs[sx])
                elif sx is False and (sy is not False and sy != i):
                    axs[i] = fig.add_subplot(ss, sharey=axs[sy])
                else:
                    axs[i] = fig.add_subplot(ss, sharex=axs[sx], sharey=axs[sy])
                i += 1
        fig.suptitle(title, fontsize=10)
        alpha = 0.05 if slide else 1
        color = 'r' if slide else 'k'
        fig.text(notell_pos[0], notell_pos[1], notell, ha='left', va='center', fontsize=8, alpha=alpha, color=color)
        fig.text(notelr_pos[0], notelr_pos[1], notelr, ha='left', va='center', fontsize=8, alpha=alpha, color=color)
        for i in range(num_axes):
            axs[i].set_xlabel(xlabel[i], fontsize=labelsize)
            axs[i].set_ylabel(ylabel[i], fontsize=labelsize)
            axs[i].set_xlim(xrange[i])
            axs[i].set_ylim(yrange[i])
            if xtick[i] is not None and xrange[i] is not None:
                if xtick_0center[i]:
                    axs[i].set_xticks(np.hstack([np.arange(0, xrange[i][0]-xtick[i]/10, -xtick[i])[::-1], np.arange(0, xrange[i][1]+xtick[i]/10, xtick[i])]))
                else:
                    axs[i].set_xticks(np.arange(xrange[i][0], xrange[i][1]+xtick[i]/10, xtick[i]))
            elif xrange[i] is None:
                axs[i].autoscale(enable=True, axis='x')
            if ytick[i] is not None and yrange[i] is not None:
                if ytick_0center[i]:
                    axs[i].set_yticks(np.hstack([np.arange(0, yrange[i][0]-ytick[i]/10, -ytick[i])[::-1], np.arange(0, yrange[i][1]+ytick[i]/10, ytick[i])]))
                else:
                    axs[i].set_yticks(np.arange(yrange[i][0], yrange[i][1]+ytick[i]/10, ytick[i]))
            elif yrange[i] == None:
                axs[i].autoscale(enable=True, axis='y')
            hide0 = True if axs[i].get_xlim()[0] == 0 and axs[i].get_ylim()[0] == 0 else False
            axs[i].xaxis.set_major_formatter(MyPlotter.make_formatter(xsigf[i], hide0=hide0))
            axs[i].yaxis.set_major_formatter(MyPlotter.make_formatter(ysigf[i], hide0=hide0))
            if hide0:
                offset = MyPlotter.calc_text_offset(fig, axs[i], text='0', fontsize=labelsize, fontfamily="DejaVu Sans" , xem=1.4, yem=1.4)
                axs[i].text(0, 0, "0", transform=axs[i].transAxes-offset, fontsize=labelsize, clip_on=False, zorder=10)
            axs[i].tick_params(axis='both', which="both", direction="in", length=ticklength, width=tickwidth, labelsize=labelsize)
            if grid[i]:
                axs[i].minorticks_on()
                axs[i].grid(which='major', lw=major_grid_lw)
                axs[i].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                axs[i].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                axs[i].grid(which='minor', lw=minor_grid_lw)
        return fig, axs

class MyAnimator:
    @staticmethod
    def make_point_and_line(fig, ax, x, y, frame, disp_max, color, markersize, malpha, lw, lalpha):
        if disp_max == 1:
            l, = ax.plot([], [])
            p = ax.scatter(x[frame], y[frame], c=color, s=markersize)
        elif disp_max > 1:
            startf = frame - disp_max if frame > disp_max else 0
            l, = ax.plot(x[startf: frame+1], y[startf: frame+1], c=color, lw=lw, alpha=lalpha)
            p = ax.scatter(x[frame], y[frame], c=color, s=markersize, alpha=malpha)
        return fig, ax, p, l
    @staticmethod
    def make_auxiliary_line_endpoint(x, y, scale1=100, scale2=100):
        angle = np.arctan2(y, x)
        x1 = -scale1 * x
        x2 = scale2 * x
        y1 = np.tan(angle) * x1
        y2 = np.tan(angle) * x2
        xs = np.vstack([x1, x2]).T
        ys = np.vstack([y1, y2]).T
        endpoints = [xs, ys]
        return endpoints
    def _get_max_frame_count(self):
        f_data = len(self.data_list_original[0]["data"][0]) if self.data_list_original is not None else 0
        f_vct = len(self.vct_list_original[0]["data"][0]) if self.vct_list_original is not None else 0
        f_vline = len(self.vline_list_original[0]["data"][0]) if self.vline_list_original is not None else 0
        f_hline = len(self.hline_list_original[0]["data"][0]) if self.hline_list_original is not None else 0
        f_fline = len(self.fline_list_original[0]["data"][0]) if self.fline_list_original is not None else 0
        f_note = len(self.note_list_original[0]["data"][0]) if self.note_list_original is not None else 0
        num_frames = max(f_data, f_vct, f_vline, f_hline, f_fline, f_note, 0)
        return num_frames

    def __init__(self, fig, axs, data_list=None, vct_list=None, vline_list=None, hline_list=None, fline_list=None, note_list=None):
        """
        Initialize the animator with figure, axes, and various data lists.
        Args:
            fig (matplotlib.figure.Figure): The main figure object.
            axs (list of matplotlib.axes.Axes): List of axes for subplots.
            data_list (list of dict): Point and trajectory data.
                - id (int): Axis index.
                - data (list): [x_array, y_array] for all frames.
                - color (str): Color for the point and line.
                - markersize (float): Size of the scatter point.
                - malpha (float): Alpha transparency for the point.
                - lw (float): Line width for the trajectory.
                - lalpha (float): Alpha transparency for the trajectory.
                - disp_max (int): Max number of previous points to display as a tail.
            vct_list (list of dict, optional): Vector (arrow) data.
                - mode (str): 'force' (moving single arrow) or 'field' (vector field).
                - data (list): [x, y, u, v] for 'force' or [X, Y, U, V, C] for 'field'.
                - color/cmap/clim/width/scale/alpha: Styling parameters.
            vline_list (list of dict, optional): Vertical lines (axvline).
                - data (array): X-position for each frame.
            hline_list (list of dict, optional): Horizontal lines (axhline).
                - data (array): Y-position for each frame.
            fline_list (list of dict, optional): Free lines (plot).
                - data (list): [[x_start, x_end], [y_start, y_end]] for each frame.
            note_list (list of dict, optional): Text annotations.
                - prefix/suffix (str): Text before/after the value.
                - data (array): Numerical data to display.
                - position (list): [x, y] in axes coordinates (0 to 1).
                - sigf/disp_width: Formatting for the number.
        """
        self.fig = fig
        self.axs = axs
        self.data_list_original = data_list # point and line for timeseries data like trajectory
        self.data_list = None
        self.vct_list_original = vct_list # vector
        self.vct_list = None
        self.vline_list_original = vline_list # vertical line with ax.axvline
        self.vline_list = None
        self.hline_list_original = hline_list # horizontal line with ax.axhline
        self.hline_list = None
        self.fline_list_original = fline_list # free line with ax.plot
        self.fline_list = None
        self.note_list_original = note_list # note with ax.text
        self.note_list = None
        self.num_frames_original = self._get_max_frame_count()

    def skip_frames(self, frange, skip):
        if frange:
            s, e = frange
        else:
            s, e = 0, self.num_frames_original
        self.data_list = []
        if self.data_list_original:
            for d in self.data_list_original:
                x, y = d["data"]
                new_entry = {**d,"data": [x[s:e:skip], y[s:e:skip]]}
                self.data_list.append(new_entry)
        if self.vct_list_original:
            self.vct_list = []
            for d in self.vct_list_original:
                if d["mode"] == "force":
                    x, y, u, v = d["data"]
                    new_entry = {**d,"data": [x[s:e:skip], y[s:e:skip], u[s:e:skip], v[s:e:skip]]}
                elif d["mode"] == "field":
                    x, y, u, v, c = d["data"]
                    new_entry = {**d,"data": [x, y, u[s:e:skip], v[s:e:skip], c[s:e:skip]]}
                self.vct_list.append(new_entry)
        if self.vline_list_original:
            self.vline_list = []
            for d in self.vline_list_original:
                data = d["data"]
                new_entry = {**d,"data": data[s:e:skip]}
                self.vline_list.append(new_entry)
        if self.hline_list_original:
            self.hline_list = []
            for d in self.hline_list_original:
                data = d["data"]
                new_entry = {**d,"data": data[s:e:skip]}
                self.hline_list.append(new_entry)
        if self.fline_list_original:
            self.fline_list = []
            for d in self.fline_list_original:
                data = d["data"]
                new_entry = {**d,"data": [data[0][s:e:skip], data[1][s:e:skip]]}
                self.fline_list.append(new_entry)
        if self.note_list_original:
            self.note_list = []
            for d in self.note_list_original:
                data = d["data"]
                new_entry = {**d,"data": data[s:e:skip]}
                self.note_list.append(new_entry)

    def make_artist_ani(self, disp_max=100, frange=None, skip=1, interval=100):
        self.skip_frames(frange, skip)
        num_frames = len(self.data_list[0]["data"][0])
        artists = []
        for f in range(num_frames):
            container = []
            for entry in self.data_list:
                i_ax = entry["id"]
                data = entry["data"]
                color = entry["color"]
                markersize = entry["markersize"]
                malpha = entry["malpha"]
                lw = entry["lw"]
                lalpha = entry["lalpha"]
                disp_max = entry["disp_max"]
                self.fig, self.axs[i_ax], p, l = MyAnimator.make_point_and_line(fig=self.fig, ax=self.axs[i_ax], x=data[0], y=data[1], frame=f, disp_max=disp_max, color=color, markersize=markersize, malpha=malpha, lw=lw, lalpha=lalpha)
                container.append(p)
                container.append(l)
            if self.vct_list:
                for entry in self.vct_list:
                    i_ax = entry["id"]
                    width = entry["width"]
                    scale = entry["scale"]
                    if entry["mode"] == "force":
                        color = entry["color"]
                        alpha = entry["alpha"]
                        x, y, u, v = entry["data"]
                        v = self.axs[i_ax].quiver(x[f], y[f], u[f], v[f], angles="xy", scale_units="xy", scale=scale, width=width, color=color, alpha=alpha)
                    elif entry["mode"] == "field":
                        cmap = entry["cmap"]
                        clim = entry["clim"]
                        x, y, u, v, c = entry["data"]
                        v = self.axs[i_ax].quiver(x, y, u[f], v[f], c[f], angles="xy", scale_units="xy", scale=scale, width=width, cmap=cmap, clim=clim)
                    container.append(v)
            if self.vline_list:
                for entry in self.vline_list:
                    i_ax = entry["id"]
                    x = entry["data"][f]
                    color = entry["color"]
                    lw = entry["lw"]
                    alpha = entry["alpha"]
                    ymin, ymax = entry["ymin"], entry["ymax"]
                    vl = self.axs[i_ax].axvline(x=x, ymin=ymin, ymax=ymax, c=color, lw=lw, alpha=alpha)
                    container.append(vl)
            if self.hline_list:
                for entry in self.hline_list:
                    i_ax = entry["id"]
                    y = entry["data"][f]
                    color = entry["color"]
                    lw = entry["lw"]
                    alpha = entry["alpha"]
                    xmin, xmax = entry["xmin"], entry["xmax"]
                    hl = self.axs[i_ax].axhline(y=y, xmin=xmin, xmax=xmax, c=color, lw=lw, alpha=alpha)
                    container.append(hl)
            if self.fline_list:
                for entry in self.fline_list:
                    i_ax = entry["id"]
                    x, y = entry["data"][f]
                    color = entry["color"]
                    lw = entry["lw"]
                    alpha = entry["alpha"]
                    fl, = self.axs[i_ax].plot(x, y, c=color, lw=lw, alpha=alpha)
                    container.append(fl)
            if self.note_list:
                for entry in self.note_list:
                    i_ax = entry["id"]
                    prefix = entry["prefix"]
                    data = entry["data"][f]
                    suffix = entry["suffix"]
                    note = f"{prefix}{data:.3f}{suffix}"
                    pos = entry["position"]
                    fontsize = entry["fontsize"]
                    fontfamily = entry["fontfamily"]
                    t = self.axs[i_ax].text(pos[0], pos[1], note, transform=self.axs[i_ax].transAxes, fontsize=fontsize, fontfamily=fontfamily)
                    container.append(t)
            artists.append(container)
        ani = ArtistAnimation(self.fig, artists, interval=interval)
        return ani

    def init_func_ani(self):
        self.scats = []
        self.lines = []
        self.vectors = []
        self.vlines = []
        self.hlines = []
        self.flines = []
        self.notes = []
        for entry in self.data_list:
            i_ax = entry["id"]
            color = entry["color"]
            markersize = entry["markersize"]
            malpha = entry["malpha"]
            lw = entry["lw"]
            lalpha = entry["lalpha"]
            p = self.axs[i_ax].scatter([], [], c=color, s=markersize, alpha=malpha)
            l, = self.axs[i_ax].plot([], [], c=color, lw=lw, alpha=lalpha)
            self.scats.append(p)
            self.lines.append(l)
        if self.vct_list:
            for entry in self.vct_list:
                i_ax = entry["id"]
                width = entry["width"]
                scale = entry["scale"]
                if entry["mode"] == "force":
                    color = entry["color"]
                    alpha = entry["alpha"]
                    x, y, u, v = entry["data"]
                    v = self.axs[i_ax].quiver(x[0], y[0], np.zeros_like(u[0]), np.zeros_like(v[0]), angles="xy", scale_units="xy", scale=scale, width=width, color=color, alpha=alpha)
                elif entry["mode"] == "field":
                    cmap = entry["cmap"]
                    clim = entry["clim"]
                    x, y, u, v, c = entry["data"]
                    v = self.axs[i_ax].quiver(x, y, np.zeros_like(u[0]), np.zeros_like(v[0]), c[0], angles="xy", scale_units="xy", scale=scale, width=width, cmap=cmap, clim=clim)
                self.vectors.append(v)
        if self.vline_list:
            for entry in self.vline_list:
                i_ax = entry["id"]
                color = entry["color"]
                lw = entry["lw"]
                alpha = entry["alpha"]
                ymin, ymax = entry["ymin"], entry["ymax"]
                vl = self.axs[i_ax].axvline(x=0, ymin=ymin, ymax=ymax, c=color, lw=lw, alpha=alpha)
                self.vlines.append(vl)
        if self.hline_list:
            for entry in self.hline_list:
                i_ax = entry["id"]
                color = entry["color"]
                lw = entry["lw"]
                alpha = entry["alpha"]
                xmin, xmax = entry["xmin"], entry["xmax"]
                hl = self.axs[i_ax].axhline(y=0, xmin=xmin, xmax=xmax, c=color, lw=lw, alpha=alpha)
                self.hlines.append(hl)
        if self.fline_list:
            for entry in self.fline_list:
                i_ax = entry["id"]
                color = entry["color"]
                lw = entry["lw"]
                alpha = entry["alpha"]
                fl, = self.axs[i_ax].plot([], [], c=color, lw=lw, alpha=alpha)
                self.flines.append(fl)
        if self.note_list:
            for entry in self.note_list:
                i_ax = entry["id"]
                pos = entry["position"]
                fontsize = entry["fontsize"]
                fontfamily = entry["fontfamily"]
                t = self.axs[i_ax].text(pos[0], pos[1], "", transform=self.axs[i_ax].transAxes, fontsize=fontsize, fontfamily=fontfamily)
                self.notes.append(t)
        return self.scats + self.lines + self.vectors + self.vlines + self.hlines + self.flines + self.notes

    def update(self, frame):
        for i, entry in enumerate(self.data_list):
            disp_max = entry["disp_max"]
            startf = frame - disp_max if frame > disp_max else 0
            data = entry["data"]
            self.scats[i].set_offsets([data[0][frame], data[1][frame]])
            if disp_max > 1:
                self.lines[i].set_data(data[0][startf:frame+1], data[1][startf:frame+1])
        if self.vct_list:
            for i, entry in enumerate(self.vct_list):
                if entry["mode"] == "force":
                    x, y, u, v = entry["data"]
                    self.vectors[i].set_UVC(u[frame], v[frame])
                    self.vectors[i].set_offsets([x[frame], y[frame]])
                elif entry["mode"] == "field":
                    x, y, u, v, c = entry["data"]
                    self.vectors[i].set_UVC(u[frame], v[frame], c[frame])
        if self.vline_list:
            for i, entry in enumerate(self.vline_list):
                x = entry["data"][frame]
                self.vlines[i].set_xdata([x])
        if self.hline_list:
            for i, entry in enumerate(self.hline_list):
                y = entry["data"][frame]
                self.hlines[i].set_ydata([y])
        if self.fline_list:
            for i, entry in enumerate(self.fline_list):
                x, y = entry["data"]
                self.flines[i].set_data(x[frame], y[frame])
        if self.note_list:
            for i, entry in enumerate(self.note_list):
                prefix = entry["prefix"]
                data = entry["data"][frame]
                sigf = entry["sigf"]
                width = entry["disp_width"]
                suffix = entry["suffix"]
                note = f"{prefix}{data:{width}.{sigf}f}{suffix}"
                self.notes[i].set_text(note)
        return self.scats + self.lines + self.vectors + self.vlines + self.hlines + self.flines + self.notes

    def make_func_ani(self, frange=None, skip=1, interval=100, blit=True):
        self.skip_frames(frange, skip)
        num_frames = self.num_frames_original // skip
        ani = FuncAnimation(self.fig, self.update, fargs=(), frames=num_frames, init_func=self.init_func_ani, interval=interval, blit=blit)
        return ani

def check_fonts():
    for f in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        # if "Times" in f or "Roman" in f:
        print(f)
    # print(fm.findfont("Times New Roman"))

def draw_3dcoordsys():
    fig = plt.figure(figsize=(12, 12), facecolor='white')
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_facecolor('white')
    ax1._axis3don = False
    axis_length = 2
    xyzlim = (-10, 10)
    # x-axis (up)
    ax1.quiver(0, 0, 0, 0, 0, axis_length, color='r', arrow_length_ratio=0.1)
    # y-axis (right)
    ax1.quiver(0, 0, 0, axis_length, 0, 0, color='g', arrow_length_ratio=0.1)
    # z-axis (into screen)
    ax1.quiver(0, 0, 0, 0, axis_length, 0, color='b', arrow_length_ratio=0.1)
    # Put labels "x", "y", "z" at arrow tips
    label_scale = 1.1
    ax1.text(0, 0, axis_length * label_scale, 'x', color='r', fontsize=14, ha='center', va='center')
    ax1.text(axis_length * label_scale, 0, 0, 'y', color='g', fontsize=14, ha='center', va='center')
    ax1.text(0, axis_length * label_scale, 0, 'z', color='b', fontsize=14, ha='center', va='center')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_zlabel('')
    ax1.set_xlim(xyzlim)
    ax1.set_ylim(xyzlim)
    ax1.set_zlim(xyzlim)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    # ======================================================
    # Thick short cylinder:
    #   - Axial direction: x (conceptual) -> matplotlib z
    #   - Axial thickness: ±axial_half_thickness
    #   - Radial thickness: ring_radius ± radial_half_thickness
    # ======================================================
    ring_radius = 5.0           # center radius of the ring
    radial_half_thickness = 0.1 # ±0.1 in radial direction
    axial_half_thickness = 1.0  # ±1.0 in axial (x) direction
    r_in = ring_radius - radial_half_thickness   # inner radius
    r_out = ring_radius + radial_half_thickness  # outer radius
    # --------- Side surfaces (inner and outer cylinders) ---------
    theta = np.linspace(0, 2 * np.pi, 200)                 # angle around
    h = np.linspace(-axial_half_thickness, axial_half_thickness, 50)              # axial coordinate (x)
    H, Theta = np.meshgrid(h, theta)
    # Outer cylinder in conceptual coordinates:
    #   y = r_out * cos(theta)
    #   z = r_out * sin(theta)
    #   x = h
    # Convert to matplotlib:
    #   X (matplotlib) = y
    #   Y (matplotlib) = z
    #   Z (matplotlib) = x
    X_out = r_out * np.cos(Theta)
    Y_out = r_out * np.sin(Theta)
    Z_out = H
    # Inner cylinder (same angle, same h, but with r_in)
    X_in = r_in * np.cos(Theta)
    Y_in = r_in * np.sin(Theta)
    Z_in = H
    # Plot outer and inner side surfaces
    ax1.plot_surface(
        X_out, Y_out, Z_out,
        color='gray', alpha=0.4, edgecolor='none'
    )
    ax1.plot_surface(
        X_in, Y_in, Z_in,
        color='gray', alpha=0.4, edgecolor='none'
    )
    # --------- Top and bottom annular surfaces (end caps) ---------
    # Use radius in [r_in, r_out]
    r_cap = np.linspace(r_in, r_out, 30)
    theta_cap = np.linspace(0, 2 * np.pi, 200)
    R_cap, Theta_cap = np.meshgrid(r_cap, theta_cap)
    # Conceptual y, z:
    Y_cap = R_cap * np.cos(Theta_cap)
    Z_cap = R_cap * np.sin(Theta_cap)
    # Bottom cap: x = -axial_half_thickness (-> matplotlib Z)
    X_bottom = Y_cap      # conceptual y -> X
    Y_bottom = Z_cap      # conceptual z -> Y
    Z_bottom = -axial_half_thickness * np.ones_like(R_cap)
    # Top cap: x = +axial_half_thickness
    X_top = Y_cap
    Y_top = Z_cap
    Z_top = axial_half_thickness * np.ones_like(R_cap)
    # Plot caps
    ax1.plot_surface(
        X_bottom, Y_bottom, Z_bottom,
        color='gray', alpha=0.4, edgecolor='none'
    )
    ax1.plot_surface(
        X_top, Y_top, Z_top,
        color='gray', alpha=0.4, edgecolor='none'
    )
    # Optional: draw outline circles on top and bottom
    theta_edge = np.linspace(0, 2 * np.pi, 400)
    # outer circle
    Y_edge_out = r_out * np.cos(theta_edge)
    Z_edge_out = r_out * np.sin(theta_edge)
    X_edge_out_bottom = Y_edge_out
    Y_edge_out_bottom = Z_edge_out
    Z_edge_out_bottom = -axial_half_thickness * np.ones_like(theta_edge)
    X_edge_out_top = Y_edge_out
    Y_edge_out_top = Z_edge_out
    Z_edge_out_top = axial_half_thickness * np.ones_like(theta_edge)
    # inner circle
    Y_edge_in = r_in * np.cos(theta_edge)
    Z_edge_in = r_in * np.sin(theta_edge)
    X_edge_in_bottom = Y_edge_in
    Y_edge_in_bottom = Z_edge_in
    Z_edge_in_bottom = -axial_half_thickness * np.ones_like(theta_edge)
    X_edge_in_top = Y_edge_in
    Y_edge_in_top = Z_edge_in
    Z_edge_in_top = axial_half_thickness * np.ones_like(theta_edge)
    ax1.plot3D(X_edge_out_bottom, Y_edge_out_bottom, Z_edge_out_bottom, color='k', linewidth=1)
    ax1.plot3D(X_edge_out_top, Y_edge_out_top, Z_edge_out_top, color='k', linewidth=1)
    ax1.plot3D(X_edge_in_bottom, Y_edge_in_bottom, Z_edge_in_bottom, color='k', linewidth=1)
    ax1.plot3D(X_edge_in_top, Y_edge_in_top, Z_edge_in_top, color='k', linewidth=1)
    ax1.set_title('3D coordinate with thick short ring (axial ±1, radial ±0.1)')
    # Adjust view angle (you can tweak this)
    ax1.view_init(elev=20, azim=-60)
    plt.show()

def draw_yzcoordsys():
    illustrator = MyPlotter(PlotSizeCode.SQUARE_ILLUST)
    fig, axs = illustrator.myfig()
    ax = axs[0]
    # ------------------------------------------------------------
    # Fig.2: 2D orthogonal coordinate system (z vs y)
    # horizontal = y, vertical = z
    # ------------------------------------------------------------
    # Draw y-axis (horizontal) and z-axis (vertical)
    axis_2d_limit = 3
    arrow_length = 2
    # Draw axes as arrows from origin
    # y-axis (horizontal)
    ax.arrow(0, 0, arrow_length, 0, head_width=0.1, head_length=0.2, length_includes_head=True, color='k')
    # z-axis (vertical)
    ax.arrow(0, 0, 0, arrow_length, head_width=0.1, head_length=0.2, length_includes_head=True, color='k')
    # Put labels near the arrow tips
    ax.text(arrow_length * 1.05, 0, 'y', ha='center', va='bottom')
    ax.text(0, arrow_length * 1.05, 'z', ha='left', va='center')
    ax.set_xlim(-axis_2d_limit, axis_2d_limit)
    ax.set_ylim(-axis_2d_limit, axis_2d_limit)
    ax.set_aspect('equal', 'box')  # make the scale same in y and z
    ax.set_title('Fig.2: 2D (z vs y)')
    # ax.axhline(y=0, lw=1, c='k')
    # ax.axvline(x=0, lw=1, c='k')
    ax.grid(False)
    plt.show()

def draw_localcoordsys():
    illustrator = MyPlotter(PlotSizeCode.SQUARE_ILLUST)
    fig, axs = illustrator.myfig()
    ax = axs[0]
    # ------------------------------------------------------------
    # Fig.3: 2D coordinate system + local rotated coordinate system
    # ------------------------------------------------------------
    axis_2d_limit = 3
    arrow_length = 2
    # 1) Draw the global axes (same as Fig.2)
    ax.arrow(0, 0, arrow_length, 0, head_width=0.1, head_length=0.2, length_includes_head=True, color='k')
    ax.arrow(0, 0, 0, arrow_length, head_width=0.1, head_length=0.2, length_includes_head=True, color='k')
    ax.text(arrow_length * 1.05, 0, 'y', ha='center', va='bottom')
    ax.text(0, arrow_length * 1.05, 'z', ha='left', va='center')
    # Draw the line from global origin to this point (shows r)
    r = 1.5
    theta = np.radians(30)
    t = np.linspace(0, 2*np.pi, 100)
    y = r * np.cos(t)
    z = r * np.sin(t)
    y0 = r * np.cos(theta)
    z0 = r * np.sin(theta)
    ax.plot([0, y0], [0, z0], color='k', linestyle='--')
    ax.plot(y, z, color='k', lw=1, ls='--')
    # 3) Draw the local coordinate system at this point
    u_dir_y = np.cos(theta)
    u_dir_z = np.sin(theta)
    # Direction of local "v" axis = u rotated by +90 degrees
    v_dir_y = -np.sin(theta)
    v_dir_z = np.cos(theta)
    local_axis_length = 1.0
    ax.arrow(y0, z0, local_axis_length * u_dir_y, local_axis_length * u_dir_z, head_width=0.08, head_length=0.15, length_includes_head=True, color='blue')
    ax.arrow(y0, z0, local_axis_length * v_dir_y, local_axis_length * v_dir_z, head_width=0.08, head_length=0.15, length_includes_head=True, color='blue')
    ax.text(y0 + local_axis_length * u_dir_y, z0 + local_axis_length * u_dir_z, "u'", color='blue', ha='left', va='bottom')
    ax.text(y0 + local_axis_length * v_dir_y, z0 + local_axis_length * v_dir_z, "v'", color='blue', ha='left', va='bottom')
    # Set view and style
    ax.set_xlim(-axis_2d_limit, axis_2d_limit)
    ax.set_ylim(-axis_2d_limit, axis_2d_limit)
    ax.set_aspect('equal', 'box')
    ax.set_title('Fig.3: Local system at P(r, θ)')
    # ax.axhline(y=0, lw=1, c='k')
    # ax.axvline(x=0, lw=1, c='k')
    ax.grid(False)
    plt.show()

class MySketcher:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
    @staticmethod
    def cvt_p2l(*args):
        x = []
        y = []
        for a in args:
            x.append(a[0])
            y.append(a[1])
        return np.asarray(x), np.asarray(y)
    def draw_circle(self, center, r, angle=np.array([0, 2*np.pi]), num_node=500, lw=1, ls='-', center_size=1, color='k', xmirror=False, ymirror=False):
        center = np.array(center, dtype=float).copy()
        angle = np.array(angle, dtype=float).copy()
        if xmirror:
            center[0] = -1 * center[0]
            angle = np.asarray(angle)[::-1] - np.pi
        if ymirror:
            center[1] = -1 * center[1]
            angle = -1 * np.asarray(angle)[::-1]
        node = np.linspace(*angle, num_node, endpoint=True)
        ball = r * np.array([np.cos(node), np.sin(node)]).T + center
        self.ax.plot(ball[:, 0], ball[:, 1], lw=lw, c=color, ls=ls)
        self.ax.scatter(*center, s=center_size, c=color)
        return self.fig, self.ax
    def draw_cline(self, center=np.zeros(2), length=1, lw=2, ls='-', color='k'):
        lx1, ly1 = self.cvt_p2l(center-np.array([length/2, 0]), center+np.array([length/2, 0]))
        lx2, ly2 = self.cvt_p2l(center-np.array([0, length/2]), center+np.array([0, length/2]))
        self.ax.plot(lx1, ly1, c=color, ls=ls, lw=lw)
        self.ax.plot(lx2, ly2, c=color, ls=ls, lw=lw)
        return self.fig, self.ax
    def draw_line(self, p1, p2, lw=1, ls='-', color='k', xmirror=False, ymirror=False):
        x, y = self.cvt_p2l(p1, p2)
        if xmirror: x = -1 * x
        if ymirror: y = -1 * y
        self.ax.plot(x, y, c=color, ls=ls, lw=lw)
        return self.fig, self.ax
    def draw_polyline(self, ps, lw=1, ls='-', color='k', xmirror=False, ymirror=False):
        x, y = self.cvt_p2l(*ps)
        if xmirror: x = -1 * x
        if ymirror: y = -1 * y
        self.ax.plot(x, y, c=color, ls=ls, lw=lw)
        return self.fig, self.ax
    def draw_angle(self, center=np.zeros(2), r=1, angle=[0, np.radians(45)], num_node=500, lw=1, color='k'):
        node = np.linspace(angle[0], angle[1], num_node, endpoint=True)
        x = r * np.cos(node) + center[0]
        y = r * np.sin(node) + center[1]
        self.ax.plot(x, y, lw=lw, c=color)
        c1 = 1.1 * r * np.array([np.cos(angle[0]), np.sin(angle[0])]) + center
        c2 = 1.1 * r * np.array([np.cos(angle[1]), np.sin(angle[1])]) + center
        lx1, ly1 = self.cvt_p2l(center, c1)
        lx2, ly2 = self.cvt_p2l(center, c2)
        self.ax.plot(lx1, ly1, lw=lw, c=color)
        self.ax.plot(lx2, ly2, lw=lw, c=color)
        return self.fig, self.ax
    def fill_ring(self, r_inner, r_outer, num_points=99, facecolor='r', edgecolor='none', alpha=0.2):
        theta = np.linspace(0, 2*np.pi, num_points)
        c_inner = np.array([r_inner * np.cos(theta), r_inner * np.sin(theta)]).T
        c_outer = np.array([r_outer * np.cos(theta), r_outer * np.sin(theta)]).T
        vertices = np.concatenate([c_outer, c_inner[::-1]])
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(vertices) - 1)
        ring_path = mpath.Path(vertices, codes)
        patch = patches.PathPatch(ring_path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        self.ax.add_patch(patch)
        return self.fig, self.ax

class MotionPlotter:
    @staticmethod
    def add_auxiliary_cicles(fig, ax, radii, colors='k', lws=1, alphas=1):
        for r, c, lw, a in zip(radii, colors, lws, alphas):
            circle = plt.Circle((0, 0), r, color=c, fill=False, lw=lw, alpha=a)
            ax.add_artist(circle)
        return fig, ax
    def __init__(self, name=""):
        self.name = name

    def plot_trajectory(self, xs, ys, colors=['r', 'b', 'g', 'm', 'c', 'y']*100, lws=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False, slide=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(slide=slide, title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        axs[0].set_aspect(1)
        for i in range(len(xs)):
            axs[0].plot(xs[i], ys[i], c=colors[i], lw=lws[i])
        fig, axs[0] = MyPlotter.draw_center_line(fig, axs[0])
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MotionPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
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
        fig, axs[0] = MyPlotter.draw_center_line(fig, axs[0])
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MotionPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        return fig, axs[0]

    def animate_trajectory(self, xs, ys, trjcolors=['k', 'r', 'b'] + ['k']*97, trjdispmax=[100, 100] + [1]*98, trjlws=[0.4]*100, trjlalphas=[0.4]*100, trjmarkersizes=[8, 20, 20] + [8]*97, trjmarkeralphas=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        axs[0].set_aspect(1)
        fig, axs[0] = MyPlotter.draw_center_line(fig, axs[0])
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



def calc_circle_center_from2pr(p1, p2, r):
    d_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    cx1 = (p1[0] + p2[0]) / 2 + (p1[1] - p2[1]) * np.sqrt((4*r**2 - d_sq) / (4*d_sq))
    cx2 = (p1[0] + p2[0]) / 2 - (p1[1] - p2[1]) * np.sqrt((4*r**2 - d_sq) / (4*d_sq))
    cy1 = (p1[1] + p2[1]) / 2 - (p1[0] - p2[0]) * np.sqrt((4*r**2 - d_sq) / (4*d_sq))
    cy2 = (p1[1] + p2[1]) / 2 + (p1[0] - p2[0]) * np.sqrt((4*r**2 - d_sq) / (4*d_sq))
    return np.array([[cx1, cy1], [cx2, cy2]])




if __name__ == '__main__':
    print('---- test ----')
    scrdir = Path(__file__).resolve().parent

    #### sample data
    t = np.linspace(0, 10, 10000)
    x = 0.2 * np.cos(2*np.pi*t)
    y = 0.2 * np.sin(2*np.pi*t)
    z = np.sin(2*np.pi*t*4)**2
    ft = 4*np.sin(2*np.pi*t) + 0.2*np.sin(2*np.pi*20*t)

    sizecode = PlotSizeCode.SQUARE_ILLUST
    sizecode = PlotSizeCode.SQUARE_FIG
    sizecode = PlotSizeCode.RECTANGLE_FIG
    sizecode = PlotSizeCode.LANDSCAPE_FIG_21
    sizecode = PlotSizeCode.LANDSCAPE_FIG_31
    sizecode = PlotSizeCode.TRAJECTORY_2
    sizecode = PlotSizeCode.TRAJECTORY_22
    # sizecode = PlotSizeCode.TRAJECTORY_WITH_TIMESERIES

    plotter = MyPlotter(sizecode=sizecode)
    fig, axs = plotter.myfig(notell="note lower left", slide=False)
    for i in range(4):
        axs[i].set_aspect(1)
    # axs[0].set_aspect(1)
    # axs[1].set_aspect(1)

    # axs[3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # axs[3].set_xlabel("")
    # axs[3].set_ylabel("")

    axs[0].plot(x, y, lw=1, c='b')

    # for field
    # xx = np.linspace(-10, 10, 21)
    # yy = np.linspace(-10, 10, 21)
    # X, Y = np.meshgrid(xx, yy)
    # U = np.cos((X[np.newaxis, :, :])*0.4 + t[:, np.newaxis, np.newaxis] * 6)
    # V = np.sin((Y[np.newaxis, :, :])*0.4 + t[:, np.newaxis, np.newaxis] * 6)
    # C = np.sqrt(U**2+V**2)

    # axs[3].axis("off")

    animator = MyAnimator(fig, axs, data_list=None, vct_list=None, vline_list=None, hline_list=None, fline_list=None, note_list=None)
    # help(MyAnimator)

    plt.show()
    print(vars(plotter))


