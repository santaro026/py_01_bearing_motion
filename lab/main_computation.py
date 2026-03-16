"""
Created on Fri Mar 06 18:37:51 2026
@author: santaro

Notes:
main script for analyzing the data of high speed camera

modification
- revise the expression "theta_rotframe = t * angular_velocity_avg + (initial_phase_avg - np.pi/2)" about display phase setting of markers

"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal

import re
from pathlib import Path
from datetime import datetime

import config
import data_handler

from mymods import myplotter, mylogger, myfitting, mycoord

outdir = config.ROOT / "results" / "test"
outdir.mkdir(exist_ok=True, parents=True)

def main(dataseries: data_handler.Series, datamapinfo):
    logger_main = mylogger.MyLogger("main", outdir=outdir)
    logger_main.info(f"datamapinfo: {datamapinfo}")
    logger_main.info(f"dataseries: {repr(dataseries)}")

    logger_main.measure_time("main", 's')
    coord = dataseries.coord
    audio = dataseries.audio

    cir_cage_zero = myfitting.lsm_for_circle()

    logger_main.measure_time("main", 'e')


def analyze(markers, markers_ref, sound):
    pass



def analyze_cage(markers, markers_ref, fps, lsmmode="numpy"):
    """
    calculate cage motion form markers coordinates.
    - cage center, trajectory, rotation speed, probability
    - cage deformation: roundness, major and minor axis,

    markers: markers coordinates, (frames, N, 2)
    makers_ref: reference markers coordinates, (1, N, 2)
    fps: frame per second

    """

    # markers = mm_per_pixel * markers
    logger_cage = mylogger.MyLogger("logger_cage", outdir=outdir)
    logger_cage.measure_time("main", mode='s')
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]
    logger_cage.binfo(f"inputdata:markers: {markers.shape}, markers_ref: {markers_ref.shape}, fps: {fps}")
    t = np.arange(num_frames) / fps

    #### lsm fitting for circle
    logger_cage.measure_time("lsm_fitting_for_circle", mode='s')
    markers_ref_xyr, markers_ref_lsminfo = myfitting.lsm_for_circle(markers_ref)
    cage_radius_ref = markers_ref_xyr[2]
    markers_radii_ref = markers_ref_lsminfo["radii"]
    logger_cage.binfo(f"lsm fitting for markers_ref:\nmarkers_ref_xyr: {markers_ref_xyr}\nmarkers_ref_lsminfo: {markers_ref_lsminfo}")
    if lsmmode == "fast":
        markers_xyr, markers_lsminfo = myfitting.lsm_for_circles(markers)
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr: {markers_xyr.shape}")
        cage = markers_xyr[:, :2]
        cage_radius = markers_xyr[:, 2]
        markers_radii = markers_lsminfo["radii"]
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr: {np.nanmean(markers_xyr, axis=0)}, {markers_xyr.shape}\naverage of markers_radii: {np.nanmean(markers_radii, axis=0)}")
    elif lsmmode == "numpy":
        markers_xyr = np.zeros((num_frames, 3))
        markers_lsminfo = []
        markers_radii = np.zeros((num_frames, num_markers))
        markers_geom_error = np.zeros((num_frames, 3)) # mean, max, std
        for f in range(num_frames):
            markers_xyr[f], _lsminfo = myfitting.lsm_for_circle(markers[f])
            markers_lsminfo.append(_lsminfo)
            markers_radii[f] = _lsminfo["radii"]
            markers_geom_error[f] = _lsminfo["geom_error_mean"], _lsminfo["geom_error_max"], _lsminfo["geom_error_std"]
        cage = markers_xyr[:, :2]
        cage_radius = markers_xyr[:, 2]
        trj_xyr, trj_lsminfo = myfitting.lsm_for_circle(cage)
        trj_geom_error = [trj_lsminfo["geom_error_mean"], trj_lsminfo["geom_error_max"], trj_lsminfo["geom_error_std"]]
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr: {np.nanmean(markers_xyr, axis=0)}, {markers_xyr.shape}\naverage of markers_radii: {np.nanmean(markers_radii, axis=0)}\nmax of markers_geom_error (mean, max, std): {np.nanmax(markers_geom_error, axis=0)}")
        logger_cage.binfo(f"lsm fitting for trajectory:\ntrj_xyr: {trj_xyr}\ntrj_geom_error (mean, max, std): {trj_geom_error}")
    elif lsmmode == "compare":
        logger_cage.measure_time("numpy_lsm", mode='s')
        markers_xyr_fast, markers_lsminfo_fast = myfitting.lsm_for_circles(markers)
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr_fast: {markers_xyr_fast.shape}")
        cage_fast = markers_xyr_fast[:, :2]
        cage_radius_fast = markers_xyr_fast[:, 2]
        markers_radii_fast = markers_lsminfo_fast["radii"]
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr: {np.nanmean(markers_xyr_fast, axis=0)}, {markers_xyr_fast.shape}\naverage of markers_radii: {np.nanmean(markers_radii_fast, axis=0)}")
        logger_cage.measure_time("numpy_lsm", mode='e')
        logger_cage.measure_time("fast_lsm", mode='s')
        markers_xyr_numpy = np.zeros((num_frames, 3))
        markers_lsminfo_numpy = []
        markers_radii_numpy = np.zeros((num_frames, num_markers))
        markers_geom_error_numpy = np.zeros((num_frames, 3)) # mean, max, std
        for f in range(num_frames):
            markers_xyr_numpy[f], _lsminfo_numpy = myfitting.lsm_for_circle(markers[f])
            markers_lsminfo_numpy.append(_lsminfo_numpy)
            markers_radii_numpy[f] = _lsminfo_numpy["radii"]
            markers_geom_error_numpy[f] = _lsminfo_numpy["geom_error_mean"], _lsminfo_numpy["geom_error_max"], _lsminfo_numpy["geom_error_std"]
        cage_numpy = markers_xyr_numpy[:, :2]
        cage_radius_numpy = markers_xyr_numpy[:, 2]
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr: {np.nanmean(markers_xyr_numpy, axis=0)}, {markers_xyr_numpy.shape}\naverage of markers_radii: {np.nanmean(markers_radii_numpy, axis=0)}\nmax of markers_geom_error (mean, max, std): {np.nanmax(markers_geom_error_numpy, axis=0)}")
        logger_cage.measure_time("fast_lsm", mode='e')
        xyr_error = np.nansum(markers_xyr_fast - markers_xyr_numpy, axis=0)
        radii_error = np.nansum(markers_radii_fast - markers_radii_numpy, axis=0)
        logger_cage.binfo(f"difference between lsm mode numpy_linalg and fast for circle:\nxyr_error: {xyr_error}, markers_radii_error: {radii_error}")
        cage = cage_numpy
        cage_radius = cage_radius_numpy
        markers_radii = markers_radii_numpy
        trj_xyr, trj_lsminfo = myfitting.lsm_for_circle(cage)
        trj_geom_error = [trj_lsminfo["geom_error_mean"], trj_lsminfo["geom_error_max"], trj_lsminfo["geom_error_std"]]
        logger_cage.binfo(f"lsm fitting for trajectory:\ntrj_xyr: {trj_xyr}\ntrj_geom_error (mean, max, std): {trj_geom_error}")

    centrifugal_expansion = cage_radius - cage_radius_ref
    cage_xyr, cage_lsminfo = myfitting.lsm_for_circle(cage)
    trj_center = cage_xyr[:2]
    trj_radius = cage_xyr[2]
    trj_radii = cage_lsminfo["radii"]
    logger_cage.binfo(f"center of cage_trajectory: {trj_center}, radius of cage_trajectory: {trj_radius}, average of centerifugal_expansion: {np.nanmean(centrifugal_expansion):.3}")
    logger_cage.binfo(f"lsm fitting for cage_trajectory:\n{cage_lsminfo}")
    logger_cage.measure_time("lsm_fitting_for_circle", mode='e')

    #### fitzgibbon fitting for ellipse
    logger_cage.measure_time("fitzgibbon_fitting_for_ellipse", mode='s')
    markers_abcdef = np.zeros((num_frames, 6))
    markers_xyabtheta = np.zeros((num_frames, 5))
    for f in range(num_frames):
        markers_abcdef[f] = myfitting.fitzgibbon_ellipse(markers[f], allow_nan=False)
        _xyabtheta_fitz = myfitting.abcdef2xyabtheta(markers_abcdef[f])
        markers_xyabtheta[f] = np.array([
            _xyabtheta_fitz["center"][0],
            _xyabtheta_fitz["center"][1],
            _xyabtheta_fitz["axes"][0],
            _xyabtheta_fitz["axes"][1],
            _xyabtheta_fitz["angle"],
        ])
    cage_fitz = markers_xyabtheta[:, :2]
    cage_minor_radius = markers_xyabtheta[:, 2]
    cage_major_radius = markers_xyabtheta[:, 3]
    logger_cage.binfo(f"fitzgibbon fitting for markers:\nxyabtheta: {np.nanmean(markers_xyabtheta, axis=0)}, {markers_xyabtheta.shape}")
    fitztrj_xyr, fitztrj_lsminfo = myfitting.lsm_for_circle(cage_fitz)
    fitztrj_geom_error = [fitztrj_lsminfo["geom_error_mean"], fitztrj_lsminfo["geom_error_max"], fitztrj_lsminfo["geom_error_std"]]
    logger_cage.binfo(f"lsm fitting for trajectory:\ntrj_xyr: {fitztrj_xyr}\ntrj_geom_error (mean, max, std): {fitztrj_geom_error}")
    logger_cage.measure_time("fitzgibbon_fitting_for_ellipse", mode='e')

    logger_cage.measure_time("calc_angle", mode='s')
    markers_angles = np.zeros((num_frames, num_markers))
    for i in range(num_markers):
        markers_angles[:, i] = np.arctan2(markers[:, i, 1]-cage[:, 1], markers[:, i, 0]-cage[:, 0])
    markers_angles = np.unwrap(markers_angles, axis=0)
    initial_m0_angle = markers_angles[0, 0]
    markers_angular_displacement = markers_angles - markers_angles[0]
    cage_Rx = np.nanmean(markers_angular_displacement, axis=1)
    cage_Rvx = np.gradient(cage_Rx, t)
    cage_Rvx_avg = np.nanmean(cage_Rvx)
    logger_cage.binfo(f"average cage rotation speed: {cage_Rvx_avg/2/np.pi*60} [rpm]\ninitial angle of marker0: {np.degrees(initial_m0_angle)} [degree]")
    logger_cage.binfo(f"max difference of cage center between circle and fitzgibbon fitting: {np.nanmax(cage - cage_fitz, axis=0)}")
    logger_cage.measure_time("calc_angle", mode='e')


    logger_cage.measure_time("deformation", mode='s')
    system_center = np.zeros((num_frames, 2)) # defined based on inner ring center determined by image in static state
    """
    rotframeSA: center: system, rotspeed: average
    rotframeSI: center: system, rotspeed: instantaneous
    rotframeCA: center: cage, rotspeed: average
    rotframeCI: center: cage, rotspeed: instantaneous

    """
    rotframeSA_angle = cage_Rvx_avg * t
    rotframeSA_center = system_center
    rotframeSI_angle = cage_Rx
    rotframeSI_center = system_center
    rotframeCA_angle = cage_Rvx_avg * t
    rotframeCA_center = cage
    rotframeCI_angle = cage_Rx
    rotframeCI_center = cage
    deformation = myfitting.calc_elliptical_deformation(markers, markers_ref)

    logger_cage.measure_time("main", mode='e')

    if 0:
        data_list = [
            {"id": 0, "data": markers[:, 1, 0], "c": 'b'},
            {"id": 0, "data": markers[:, 1, 1], "c": 'r'},
            {"id": 0, "data": markers_radii[:, 1], "c": 'g'},

            # {"id": 1, "data": centrifugal_expansion, "c": 'r'},
            # {"id": 1, "data": np.degrees(markers_angles[:, 1]), "c": 'b'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 1]), "c": 'r'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 2]), "c": 'b'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 3]), "c": 'g'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 4]), "c": 'c'},
            # {"id": 1, "data": np.degrees(cage_Rx), "c": 'r'},
            {"id": 1, "data": cage_Rvx/2/np.pi, "c": 'r'},


            {"id": 2, "data": cage[:, 0], "c": 'b'},
            {"id": 2, "data": cage[:, 1], "c": 'r'},
            {"id": 2, "data": trj_radii, "c": 'b'},

        ]
        fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
        for i in range(len(data_list)):
            axs[data_list[i]["id"]].plot(t, data_list[i]["data"], lw=1, c=data_list[i]["c"])
        axs[1].set_ylim(0, 100)
        axs[2].set_ylim(-1, 1)
        axs[2].axhline(y=trj_radius, xmin=0, xmax=1, lw=4, alpha=0.2, c='g')
        plt.show()


if __name__ == '__main__':
    print("---- test ----")
    #### sample data
    import sampledata_generator
    fps = 10000
    cage = sampledata_generator.SimpleCage(name='', PCD=50, ID=48, OD=52, width=10, num_pockets=8, num_markers=8, num_mesh=100, Dp=6.25, Dw=5.953)
    cage.time_series_data2(fps=fps, duration=0.2, omega_rot=20*2*np.pi, omega_rev=20*2*np.pi, r_rev=0.4, a=cage.PCD/2, b=cage.PCD/2, omega_deform=0, noise_type="normal", noise_max=0.1*0.1)

    # datadir = config.ROOT / "data" / ""
    # dataseries_handler = data_handler.DataSeriesHandler()
    # dataseries_handler.scan_directory(datadir, datadir, datadir, "tc*", "REC*", "*.xlsx")

    # for k, s in dataseries_handler.seriesmap.items():
        # print(k, s)

    # datamap = dataseries_handler.datamap_loader.datamap
    # print(datamap)

    # dataseries, datamapinfo = dataseries_handler.select_series(23, 1)

    markers_ref = cage.p_markers_noise[0, :, 1:]
    makers = cage.p_markers_noise[:, :, 1:] * 1.01

    analyze_cage(makers, markers_ref, fps=fps, lsmmode="compare")



