"""
Created on Wed Aug 27 08:42:28 2025
@author: honda-shin

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

from sintamods import mytools
from sintamods import myfitting
from sintamods import mydataclass
from sintamods import myplot

import config
from testcondition_loader import TestCondition

class PlotterForCageVisualization(myplot.PlotterForCage):
    @staticmethod
    def convert_2list(*args):
        res = []
        for var in args:
            try:
                var = list(var)
            except TypeError:
                var = [var]
            res.append(var)
        return res
    @staticmethod
    def mark_pocket(ax, num_pockets, r=1, colors=['k']*100, s=200):
        t = np.linspace(0, 2*np.pi, num_pockets, endpoint=False)
        x = r * np.cos(t+np.pi/2)
        y = r * np.sin(t+np.pi/2)
        for i in range(num_pockets):
            ax.scatter(x[i], y[i], c=colors[i], s=40, alpha=0.8, marker='+', zorder=10)
        return ax

    def __init__(self, testcond, json_path=None, xyrange_trj=(-0.6, 0.6), xyrange_markers=(-30, 30), notell="", notelr=""):
        json_path = config.ROOT / 'assets' / 'plot_settings.json'
        super().__init__(json_path)
        self.xyrange_trj = xyrange_trj
        self.xyrange_markers = xyrange_markers
        self.testcond = testcond
        self.notell = notell
        self.notelr = notelr
        if self.testcond.Dp_measured:
            self.Dp = self.testcond.Dp_measured
            self.note_Dp = 'Dp is measured.'
        elif self.testcond.Dp:
            self.Dp = self.testcond.Dp
            self.note_Dp = 'Dp is nominal.'
        else:
            self.Dp = np.nan
            self.note_Dp = 'Dp is none'
        if self.testcond.Dl_measured:
            self.Dl = self.testcond.Dl_measured
            self.note_Dl = 'Dl is measured.'
        elif self.testcond.Dl:
            self.Dl = self.testcond.Dl
            self.note_Dl = 'Dl is nominal.'
        else:
            self.Dl = np.nan
            self.note_Dl = 'Dl is none.'
        if isinstance(self.Dp, list):
            self.Dp = np.array(self.Dp)
        if isinstance(self.Dl, list):
            self.Dl = np.array(self.Dl)
        self.dp = self.Dp - self.testcond.Dw
        self.dl = self.testcond.PCD + self.testcond.Dw - self.Dl
        self.note_DpDl = f'{self.note_Dp} {self.note_Dl}'
        self.dp_list, self.dl_list = PlotterForCageVisualization.convert_2list(self.dp, self.dl)

        num_dp, num_dl = len(self.dp_list), len(self.dl_list)
        self.auxiliary_circles_radii = [d for d in (self.dp_list+self.dl_list)]
        self.auxiliary_circles_colors = ['b']*num_dp + ['r']*num_dl

    # def make_auxiliary_circles_list(self):
    #     num_dp, num_dl = len(self.dp_list), len(self.dl_list)
    #     auxiliary_circles_radii = [d for d in (self.dp_list+self.dl_list)]
    #     auxiliary_circles_colors = ['b']*num_dp + ['r']*num_dl
    #     return auxiliary_circles_radii, auxiliary_circles_colors

    def trajectory(self, p, title='', draw_auxiliary_circle=True, is_markpocket=False):
        y, z = p[0], p[1]
        if not draw_auxiliary_circle:
            auxiliary_circles_radii = None
        fig, ax = self.plot_trajectory(y, z, title=title, auxiliary_circles_radii=self.auxiliary_circles_radii, auxiliary_circles_colors=self.auxiliary_circles_colors, xrange=self.xyrange_trj, yrange=self.xyrange_trj, notell=self.notell, notelr=self.notelr)
        if is_markpocket:
            ax = PlotterForCageVisualization.mark_pocket(ax, self.testcond.num_pockets, r=0.5)
        return fig, ax

    def trajectories(self, ps, title='', markpocket=False):
        _px, _py = [], []
        for i in range(self.testcond.num_points):
            _px.append(ps[i*2])
            _py.append(ps[i*2+1])
        fig, ax = self.plot_trajectories(_px, _py, title=title, auxiliary_circles_radii=[self.testcond.PCD/2], auxiliary_circles_colors=['g'], xrange=self.xyrange_markers, yrange=self.xyrange_markers, xtick=5, ytick=5, xsigf=1, ysigf=1, notell=self.notell, notelr=self.notelr)
        ax = PlotterForCageVisualization.mark_pocket(ax, self.testcond.num_pockets, r=1.04*self.testcond.PCD/2)
        return fig, ax

    def vstime3(self, tlist, ylist, lws=[0.4]*3, alphas=[1]*3, ylabel='', xrange=(0, 6), yrange=None, ysigf=0, ytick=1, ytick_0center=True, title='', plottype=['plot', 'plot', 'plot']):
        fig, axs = self.plot_vstime3(tlist, ylist, colors=['k']*3, lws=lws, alphas=alphas,  ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=0.5, ysigf=ysigf, ytick=ytick, ytick_0center=ytick_0center, notell=self.notell, notelr=self.notelr, title=title, plottype=plottype)
        return fig, axs

    def probability(self, probability_map, vmin=None, vmax=None, title=''):
        auxiliary_circles_radii, auxiliary_circles_colors = self.make_auxiliary_circles_list()
        fig, ax = self.plot_probability(probability_map, cmap='binary', title=title, vmin=vmin, vmax=vmax, auxiliary_circles_radii=auxiliary_circles_radii, auxiliary_circles_colors=auxiliary_circles_colors, xlabel='y [mm]', ylabel='z [mm]', xrange=self.xyrange_trj, yrange=self.xyrange_trj, xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell=self.notell, notelr=self.notelr)
        return fig, ax

if __name__ == '__main__':
    print('---- test ----\n')

    p = np.vstack([
        np.cos(np.linspace(0, 2*np.pi, 100, endpoint=False)),
        np.sin(np.linspace(0, 2*np.pi, 100, endpoint=False))
    ]) * 0.2

    from testcondition_loader import TestCondLoader
    from testcondition_loader import TestEnum
    testcondloader = TestCondLoader()
    testcond = testcondloader.testcond_factory(test_enum=TestEnum.TEST8)
    print(testcond)

    plotter = PlotterForCageVisualization(testcond=testcond)

    fig, ax = plotter.trajectory(p)

    plt.show()
