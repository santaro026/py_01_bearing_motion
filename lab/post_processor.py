"""
Created on Fri Mar 06 18:40:10 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json
from dataclasses import dataclass

from mymods import mycoord, myfitting, myplotter

import config
import plot_drawer

"""
rotframeSA: center: system, rotspeed: average
rotframeSI: center: system, rotspeed: instantaneous
rotframeCA: center: cage, rotspeed: average
rotframeCI: center: cage, rotspeed: instantaneous

"""
# transfomerSI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=system_center, theta=cage_Rx)
# transfomerSA = mycoord.CoordTransformer2d(name="system_average", local_origin=system_center, theta=cage_Rx_avg)
# transfomerCI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=cage_kasa, theta=cage_Rx)
# transfomerCA = mycoord.CoordTransformer2d(name="cage_average", local_origin=cage_kasa, theta=cage_Rx_avg)


datadir = config.ROOT / "results" / "test"

@dataclass
class Transformer:
    SI: mycoord.CoordTransformer2d
    SA: mycoord.CoordTransformer2d
    CI: mycoord.CoordTransformer2d
    CA: mycoord.CoordTransformer2d

@dataclass
class MainResult:
    fps: float
    num_frames: int
    duration: float
    pixel2mm: float
    system_center: float
    num_cage_markers: float
    num_ring_markers: float
    num_markers: float
    t: np.ndarray
    cage_kasa: np.ndarray
    cage_kasa_radius: np.ndarray
    cage_kasa_Rx: np.ndarray
    cage_fitz: np.ndarray
    cage_fitz_radii: np.ndarray
    cage_fitz_theta: np.ndarray
    cage_fitz_Rx: np.ndarray
    markers: np.ndarray
    markers_ref: np.ndarray

    def __repr__(self):
        return (
            f"\nResult list\n"
            f"fps: {self.fps}\n"
            f"num_frames: {self.num_frames}\n"
            f"duration: {self.duration}\n"
            f"pixel2mm: {self.pixel2mm}\n"
            f"system_center: {self.system_center}\n"
            f"t: {self.t.shape}\n"

            f"cage_kasa: {self.cage_kasa.shape}\n"
            f"cage_kasa_radius: {self.cage_kasa_radius.shape}\n"
            f"cage_kasa_Rx: {self.cage_kasa_Rx.shape}\n"

            f"cage_fitz: {self.cage_fitz.shape}\n"
            f"cage_fitz_radii: {self.cage_fitz_radii.shape}\n"
            f"cage_fitz_theta: {self.cage_fitz_theta.shape}\n"
            f"cage_fitz_Rx: {self.cage_fitz_Rx.shape}\n"

            f"markers: {self.markers.shape}\n"
            f"markers_ref: {self.markers_ref.shape}\n"
        )

def read_results(datadir):
    with open(datadir / "camera_info.json", 'r') as f:
        camera_info = json.load(f)
    df_cage_kasa = pl.read_csv(datadir / "cage_kasa.csv", has_header=True, infer_schema_length=5000)
    df_cage_fitz = pl.read_csv(datadir / "cage_fitz.csv", has_header=True, infer_schema_length=5000)
    df_markers = pl.read_csv(datadir / "markers.csv", has_header=True, infer_schema_length=5000)
    df_markers_ref = pl.read_csv(datadir / "markers_ref.csv", has_header=True)
    #### check the consistency
    if len(df_cage_kasa) != camera_info["num_frames"]:
        raise ValueError(f"num_frames of camera_info ({camera_info["num_frames"]}) doesnt match df_cage_kasa ({len(df_cage_kasa)})")
    if df_markers_ref.shape[-1] / 2 != camera_info["num_cage_markers"]:
        raise ValueError(f"num_cage_markers of camera_info ({camera_info["num_cage_markers"]}) doesnt match df_cage_kasa ({df_markers_ref.shape[-1]/2})")

    mainresult = MainResult(
        fps = camera_info["fps"],
        num_frames = camera_info["num_frames"],
        duration = camera_info["num_frames"]/camera_info["fps"],
        pixel2mm = camera_info["pixel2mm"],
        system_center = camera_info["system_center"],
        num_cage_markers = camera_info["num_cage_markers"],
        num_ring_markers = camera_info["num_ring_markers"],
        num_markers = camera_info["num_markers"],
        t = np.arange(camera_info["num_frames"]) / camera_info["fps"],
        cage_kasa = df_cage_kasa.select(pl.col("cy", "cz")).to_numpy()[:, np.newaxis, :],
        cage_kasa_radius = df_cage_kasa.select(pl.col("radius")).to_numpy().squeeze(),
        cage_kasa_Rx = df_cage_kasa.select(pl.col("Rx")).to_numpy().squeeze(),
        cage_fitz = df_cage_fitz.select(pl.col("cy", "cz")).to_numpy()[:, np.newaxis, :],
        cage_fitz_radii = df_cage_fitz.select(pl.col("major", "minor")).to_numpy(),
        cage_fitz_theta = df_cage_fitz.select(pl.col("theta")).to_numpy().squeeze(),
        cage_fitz_Rx = df_cage_fitz.select(pl.col("Rx")).to_numpy().squeeze(),
        markers = df_markers.to_numpy().reshape(camera_info["num_frames"], -1, 2),
        markers_ref = df_markers_ref.to_numpy().reshape(-1, 2)[np.newaxis, :, :]
    )

    transformer_kasa = Transformer(
        SI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=np.zeros(2), theta=mainresult.cage_kasa_Rx),
        SA = mycoord.CoordTransformer2d(name="system_average", local_origin=np.zeros(2), theta=np.nanmean(mainresult.cage_kasa_Rx)),
        CI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=mainresult.cage_kasa, theta=mainresult.cage_kasa_Rx),
        CA = mycoord.CoordTransformer2d(name="cage_average", local_origin=mainresult.cage_kasa, theta=np.nanmean(mainresult.cage_kasa_Rx))
    )
    transformer_fitz = Transformer(
        SI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=np.zeros(2), theta=mainresult.cage_fitz_Rx),
        SA = mycoord.CoordTransformer2d(name="system_average", local_origin=np.zeros(2), theta=np.nanmean(mainresult.cage_fitz_Rx)),
        CI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=mainresult.cage_fitz, theta=mainresult.cage_fitz_Rx),
        CA = mycoord.CoordTransformer2d(name="cage_average", local_origin=mainresult.cage_fitz, theta=np.nanmean(mainresult.cage_fitz_Rx))
    )

    results = {
        "main": mainresult,
        "transformer_kasa": transformer_kasa,
        "transformer_fitz": transformer_fitz,
    }

    return results

def calc_deformation(result):
    #### calculate deformation
    deformation_markers = myfitting.calc_elliptical_deformation(result.markers, result.markers_ref)
    deformation_fitz = {
        "roundness": result.cage_fitz_radii[:, 0] - result.cage_fitz_radii[:, 1],
        "delta_diameters": result.cage_fitz_radii - result.cage_kasa_radius[:, np.newaxis],
        "direction": np.column_stack([result.cage_fitz_theta, result.cage_fitz_theta + np.pi/2])
    }
    res = {
        "deformation_markers": deformation_markers,
        "deformation_fitz": deformation_fitz
    }
    return res

def transform_coord():
    pass

def pixel2mm_coord(coord, system_center, pixel2mm):
    system_center = np.asarray(system_center)[np.newaxis, :]
    arr_shift = coord - system_center
    arr_converted = pixel2mm * arr_shift
    return arr_converted
def pixel2mm_length(value, pixel2mm):
    return pixel2mm * value




def get_plotter(results):
    mainresult = results["main"]
    transformer_kasa = results["transformer_kasa"]
    transformer_fitz = results["transformer_fitz"]


    import json

    # testinfo = {
    #     "dp_measured": 0.5,
    #     "dl_measured": 0.5,
    #     "dp_drawing": 0.5,
    #     "dl_drawing": 0.5,
    # }
    with open(datadir/"testinfo.json", "r") as f:
        testinfo = json.load(f)
    print(testinfo)

    # plotter = plot_drawer.PlotterForCageVisualization_old(
    #     t_camera=mainresult.t,
    #     cage=mainresult.cage_kasa,
    #     markers=mainresult.markers,
    #     rotspeed=mainresult.cage_kasa_Rx,
    #     testinfo=testinfo
    #     )

    plotter = plot_drawer.PlotterForCageVisualization(testinfo)
    return plotter


if __name__ == '__main__':
    print('----- main -----\n')
    # see_headers()

    results = read_results(datadir)
    print(repr(results))
    mainresult = results["main"]
    transformer_kasa = results["transformer_kasa"]
    transformer_fitz = results["transformer_fitz"]

    # res_deform = calc_deformation(mainresult)
    # deform = res_deform["deformation_markers"]
    # deform_fitz = res_deform["deformation_fitz"]


    plotter = get_plotter(results)
    # fig, axs, log = plotter.trajectory(xyrange=(-10, 10))
    # fig, axs, log = plotter.cagecoord_sound(yrange=[(-10, 10), (-10, 10), (0, 24)])
    # fig, axs, log = plotter.cagecoord(yrange=[(-10, 10), (-10, 10)])
    # fig, axs, log = plotter.spectrogram()

    fig, axs = plotter.trajectory(mainresult.cage_kasa[:, :, 0], mainresult.cage_kasa[:, :, 1], xyrange=10)

    plt.show()




    # plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_31)
    # fig, axs = plotter.myfig()

    # axs[0].plot(mainresult.t, deform["roundness"], lw=1)
    # axs[0].plot(mainresult.t, deform["diameters_norm"][:, 0]/2, lw=1)
    # axs[0].plot(mainresult.t, deform_fitz["roundness"], lw=1)
    # axs[0].set(ylim=(-10, 30))


    # axs[1].plot(mainresult.t, deform_fitz["direction"][:, 0], lw=1, label="fitz")
    # axs[1].plot(mainresult.t, deform["direction"][:, 0], lw=1, label="node")
    # axs[1].plot(mainresult.t, deform["direction"][:, 1], lw=1, label="node")
    # axs[1].legend()

    # plt.show()


