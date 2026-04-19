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

def load_data(datapath):
    with open(datapath, "rb") as f:
        data = pickle.load(f)

    dtype_map = {
        "marker": [("coord", np.float64, (2,))],
        "kasa": [("coord", np.float64, (2,)), ("length", np.float64)],
        "fitz": [("coord", np.float64, (2,)), ("length", np.float64, (2,)), ("angle", np.float64)],
    }

    #### markers
    print(f"data[markers]: {data["markers"].shape}")
    data["markers"] = np.ascontiguousarray(data["markers"]).view(dtype_map["marker"], np.recarray)
    #### markers_ref
    print(f"data[markers_ref]: {data["markers_ref"].shape}")
    data["markers_ref"] = np.ascontiguousarray(data["markers_ref"]).view(dtype_map["marker"], np.recarray)
    #### markers_fit
    print(f"data[markers_fit][kasa]: {data["markers_fit"]["kasa"].shape}")
    print(f"data[markers_fit][fitz]: {data["markers_fit"]["fitz"].shape}")
    data["markers_fit"]["kasa"] = np.ascontiguousarray(data["markers_fit"]["kasa"]).view(dtype_map["kasa"], np.recarray)
    data["markers_fit"]["fitz"] = np.ascontiguousarray(data["markers_fit"]["fitz"]).view(dtype_map["fitz"], np.recarray)
    #### trajectory_prop
    print(f"data[trajectory_prop][kasatrj_kasa]: {data["trajectory_prop"]["kasatrj_kasa"].shape}")
    print(f"data[trajectory_prop][fitztrj_kasa]: {data["trajectory_prop"]["fitztrj_kasa"].shape}")
    print(f"data[trajectory_prop][kasatrj_avg]: {data["trajectory_prop"]["kasatrj_avg"].shape}")
    print(f"data[trajectory_prop][kasatrj_avg]: {data["trajectory_prop"]["kasatrj_avg"]}")
    print(f"data[trajectory_prop][fitztrj_avg]: {data["trajectory_prop"]["fitztrj_avg"].shape}")
    data["trajectory_prop"]["kasatrj_kasa"] = np.ascontiguousarray(data["trajectory_prop"]["kasatrj_kasa"]).view(dtype_map["kasa"], np.recarray)
    data["trajectory_prop"]["fitztrj_kasa"] = np.ascontiguousarray(data["trajectory_prop"]["fitztrj_kasa"]).view(dtype_map["kasa"], np.recarray)
    # data["trajectory_prop"]["kasatrj_avg"] = np.ascontiguousarray(data["trajectory_prop"]["kasatrj_avg"]).view(dtype_map["marker"], np.recarray)
    #### deformation
    print(f"data[deformation][markers]: {data["deformation"]["deforomation_markers"].shape}")
    # data["deformation"]["markers"] = np.ascontiguousarray(data["deformation"]["markers"]).view(dtype_map["marker"], np.recarray)
    # data["deformation"]["centrifugal_expansion_kasa"] = np.ascontiguousarray(data["deformation"]["centrifugal_expansion_kasa"]).view(("length", np.float64), np.recarray)
    # data["deformation"]["roundness_fitz"] = np.ascontiguousarray(data["deformation"]["roundness_fitz"]).view(("length", np.float64), np.recarray)





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

to_convert_list = [
    "markers", "markers_ref", "markers_fit", "markers_ref_fit", "trajectory_prop", "deformation"
]

def convert_units(data, to_convert_list=to_convert_list):
    converted = {}
    converted["markers"] = pixel2mm_coord(data["markers"])
    converted["markers_ref"] = pixel2mm_coord(data["markers_ref"])
    converted["markers_fit"] = pixel2mm_coord(data["markers_fit"])



def main(datapath):
    data = load_data(datapath)
    print(data.keys())
    # transformer_kasa = Transformer(
    #     SI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=np.zeros(2), theta=data["rotkinematics_kasa"]["cage_Rx"]),
    #     SA = mycoord.CoordTransformer2d(name="system_average", local_origin=np.zeros(2), theta=data["rotkinematics_kasa"]["cage_Rx_const"]),
    #     CI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=data["markers_fit"]["kasa"], theta=data["rotkinematics_kasa"]["cage_Rx"]),
    #     CA = mycoord.CoordTransformer2d(name="cage_average", local_origin=data["markers_fit"]["kasa"], theta=data["rotkinematics_kasa"]["cage_Rx"])
    # )
    # transformer_fitz = Transformer(
    #     SI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=np.zeros(2), theta=data["rotkinematics_fitz"]["cage_Rx"]),
    #     SA = mycoord.CoordTransformer2d(name="system_average", local_origin=np.zeros(2), theta=data["rotkinematics_fitz"]["cage_Rx_const"]),
    #     CI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=data["markers_fit"]["fitz"], theta=data["rotkinematics_fitz"]["cage_Rx"]),
    #     CA = mycoord.CoordTransformer2d(name="cage_average", local_origin=data["markers_fit"]["fitz"], theta=data["rotkinematics_fitz"]["cage_Rx_const"])
    # )
    # plotter = get_plotter()

    # markers_fit = data["markers_fit"]

    # plotlist = [
    #     {"axid": 0, "data": markers_fit["kasa"], "lw": 2, "color": 'k', "alpha": 0.5, "zorder": 2, "ls": '-'},
    #     {"axid": 0, "data": markers_fit["fitz"], "lw": 1, "color": 'g', "alpha": 0.5, "zorder": 1, "ls": '-'},
    # ]

    # plotter.trajectory(plotlist, frange=[0, 500], xyrange=3.2)

    # plt.show()




    return 0


if __name__ == '__main__':
    print('----- main -----\n')

    code = "ROT_REV"
    datadir = config.ROOT / "results" / "test" / "tmp"
    # datadir = config.ROOT / "results" / "test_noise" / "tmp"
    datapath = list(datadir.glob(f"*{code}_data.pkl"))[0]
    print(f"datapath: {datapath}")
    main(datapath)



    # plotter = get_plotter()
    # fig, axs, log = plotter.trajectory(xyrange=(-10, 10))
    # fig, axs, log = plotter.cagecoord_sound(yrange=[(-10, 10), (-10, 10), (0, 24)])
    # fig, axs, log = plotter.cagecoord(yrange=[(-10, 10), (-10, 10)])
    # fig, axs, log = plotter.spectrogram()

    # fig, axs = plotter.trajectory(, xyrange=10)

    # plt.show()




