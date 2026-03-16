"""
Created on Fri Mar 06 18:40:10 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import dill
import re

from mymods import myplot
from mymods import mytools

import helperfuncs
import config

def load_summary(tgtdir):
    df = pl.read_csv(tgtdir/'summary.csv', has_header=True, infer_schema_length=50000)
    return df

def load_test_information(tgtdir):
    tc = int(re.search(r'tc(\d+)', tgtdir.parent.name).group(1))
    sc = int(re.search(r'sc(\d+)', tgtdir.name).group(1))
    rpm = int(re.search(r'(\d+)rpm', tgtdir.name).group(1))
    fps = int(re.search(r'(\d+)fps', tgtdir.name).group(1))
    return tc, sc, rpm, fps

def load_csv(tgtdir, tgtfname):
    df = pl.read_csv(tgtfname, has_header=True, infer_schema_length=5000)
    return df

def load_pklfig(tgtfile):
    with open(tgtfile, 'rb') as f:
        fig = dill.load(f)
    axs = [fig.axes]
    print(axs)
    return fig, axs

def load_pklcsv(tgtfile):
    with open(tgtfile, 'rb') as f:
        data = dill.load(f)
    print(f'{tgtfile.name}: {data.shape}')
    return data

def see_headers(num_points=8):
    headers_cage, headers_cage2, headers_deformation, headers_rotation_speed = helperfuncs.make_headers(num_points)
    headers = [headers_cage, headers_deformation, headers_rotation_speed]
    headers_name = ['headers_cage', 'headers_deformation', 'headers_rotation_speed']
    for c, header in enumerate(headers):
        print('\n', headers_name[c])
        for c, h in enumerate(header):
            print(f'{c}: {h}')

def extract_frame(rpm, fps, num_rot=1, start_time=0):
    period = 1 / (rpm / 60 * 0.447) * num_rot
    num_frames = int(period * fps)
    start_frame = int(start_time * fps)
    end_frame = start_frame + num_frames
    frame_range = [start_frame, end_frame]
    print(num_frames)
    return frame_range


if __name__ == '__main__':
    print('----- main -----\n')
    # see_headers()

    _tc = 17
    _sc = 14

    resdir = config.ROOT / 'results'
    tgt = f'*v_1_1_9/tc{_tc}*/sc{_sc}*'
    tgtdir = list(resdir.glob(tgt))[0]

    tc, sc, rpm, fps = load_test_information(tgtdir)
    print(f'load data: tc{tc}, sc{sc}, {rpm}rpm, {fps}fps')

    testcond = helperfuncs.testcond_factory(helperfuncs.TestEnum[f'TEST{int(tc)}'])
    print(f'testcond: {testcond}')
    plotter = helperfuncs.PlotterForCageVisualization(config.ROOT/'assets'/'plot_settings.json', testcond)

    frame_range = extract_frame(rpm, fps, num_rot=1, start_time=0.1)#.82)
    st, et = frame_range
    print(f'frame range: {frame_range}')

    data = load_pklcsv(tgtdir/'pkl'/'res_cage_rotframe2.pkl')

    # st, et = 0, 40000

    t = data[1, st:et]
    cage = [data[2, st:et], data[3, st:et]]


    fig, ax = plotter.trajectory(cage)
    # fig, ax = plotter.vstime3([t, t, t], [cage[0], cage[1], cage[1]], ysigf=2)
    plt.show()


    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()



