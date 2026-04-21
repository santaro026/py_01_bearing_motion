"""
Created on Fri Mar 06 18:40:10 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import re
import pickle
import json
from dataclasses import dataclass

from mymods import mycoord, myfitting, myplotter

import config
import plot_drawer

def convert_unit(data, scale, shift):
    if isinstance(data, dict):
        _d = {}
        for k, v in data.items():
            if not "info" in k:
                _d[k] = convert_unit(v, scale, shift)
            else:
                _d[k] = v
        return _d
        # return {k: convert_unit(v, scale, shift) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        names = data.dtype.names
        if names is None:
            return data
        new_data = data.copy()
        for c, name in enumerate(names):
            if "coord_y" in name:
                new_data[name] = (new_data[name] - shift[0]) * scale
            elif "coord_z" in name:
                new_data[name] = (new_data[name] - shift[1]) * scale
            elif "length" in name:
                new_data[name] = new_data[name] * scale
        return new_data
    else:
        # raise ValueError(f"dtype was not found, data type: {type(data)}")
        print(f"[WARN]: fail to convert data, data type: {type(data)}")
        return data

def reset_view2float64(data):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = reset_view2float64(v)
    elif isinstance(data, np.ndarray):
        return data.view(np.float64)
    return data

def load_data(datapath, scale, shift):
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    data = convert_unit(data, scale, shift)
    data = reset_view2float64(data)
    return data

@dataclass
class Transformer:
    SI: mycoord.CoordTransformer2d
    SA: mycoord.CoordTransformer2d
    CI: mycoord.CoordTransformer2d
    CA: mycoord.CoordTransformer2d

def pixel2mm_coord(coord, center, pixel2mm):
    """
    coord: (N, N, 2)
    center: (2) -> (N, N, 2)

    """
    center = np.asarray(center)[np.newaxis, np.newaxis, :]
    arr_shift = coord - center
    arr_converted = pixel2mm * arr_shift
    return arr_converted
def pixel2mm_length(value, pixel2mm):
    return pixel2mm * value

def get_plotter():
    testinfo = {
        "dp_measured": 0.5,
        "dl_measured": 0.5,
        "dp_drawing": 0.5,
        "dl_drawing": 0.5,
    }
    # with open(datadir/"testinfo.json", "r") as f:
        # testinfo = json.load(f)
    # print(testinfo)
    plotter = plot_drawer.PlotterForCageVisualization(testinfo)
    return plotter

def main(datapath):
    data = load_data(datapath, scale=1, shift=(0, 0))
    print(data.keys())
    transformer_kasa = Transformer(
        SI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=np.zeros(2), theta=data["rotkinematics_kasa"]["cage_Rx"]),
        SA = mycoord.CoordTransformer2d(name="system_average", local_origin=np.zeros(2), theta=data["rotkinematics_kasa"]["cage_Rx_const"]),
        CI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=data["markers_fit"]["kasa"], theta=data["rotkinematics_kasa"]["cage_Rx"]),
        CA = mycoord.CoordTransformer2d(name="cage_average", local_origin=data["markers_fit"]["kasa"], theta=data["rotkinematics_kasa"]["cage_Rx"])
    )
    transformer_fitz = Transformer(
        SI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=np.zeros(2), theta=data["rotkinematics_fitz"]["cage_Rx"]),
        SA = mycoord.CoordTransformer2d(name="system_average", local_origin=np.zeros(2), theta=data["rotkinematics_fitz"]["cage_Rx_const"]),
        CI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=data["markers_fit"]["fitz"], theta=data["rotkinematics_fitz"]["cage_Rx"]),
        CA = mycoord.CoordTransformer2d(name="cage_average", local_origin=data["markers_fit"]["fitz"], theta=data["rotkinematics_fitz"]["cage_Rx_const"])
    )
    plotter = get_plotter()
    markers_fit = data["markers_fit"]

    plotlist = [
        {"axid": 0, "data": markers_fit["kasa"], "lw": 2, "color": 'k', "alpha": 0.5, "zorder": 2, "ls": '-'},
        {"axid": 0, "data": markers_fit["fitz"], "lw": 1, "color": 'g', "alpha": 0.5, "zorder": 1, "ls": '-'},
    ]

    plotter.trajectory(plotlist, frange=[0, 500], xyrange=3.2)
    plt.show()

    return 0


if __name__ == '__main__':
    print('----- main -----\n')

    code = "ROT_REV"
    datadir = config.ROOT / "results" / "test" / "tmp"
    datadir = config.ROOT / "results" / "test_noise" / "tmp"
    datapath = list(datadir.glob(f"*{code}_data.pkl"))[0]
    print(f"datapath: {datapath}")
    main(datapath)

