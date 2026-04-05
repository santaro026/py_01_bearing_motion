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
import mycage
import data_processor

from mymods import myplotter, mylogger, myfitting, mycoord

outdir = config.ROOT / "results" / "test"
outdir.mkdir(exist_ok=True, parents=True)

def analyze_cage(markers, markers_ref, fps, lsmmode="numpy", check=False):
    """
    calculate cage motion form markers coordinates.
    - cage center, trajectory, rotation speed, probability
    - cage deformation: roundness, major and minor axis,

    input
    markers: markers coordinates, (frames, N, 2)
    makers_ref: reference markers coordinates, (1, N, 2)
    fps: frame per second

    """

    logger_cage = mylogger.MyLogger("logger_cage", outdir=outdir)
    logger_cage.measure_time("main", mode='s')
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]
    logger_cage.binfo(f"#### inputdata\nmarkers: {markers.shape}\nmarkers_ref: {markers_ref.shape}\nfps: {fps}")
    t = np.arange(num_frames) / fps

    #### lsm fitting for circle
    logger_cage.measure_time("lsm_fitting_for_circle", mode='s')
    markers_ref_xyr, markers_ref_lsminfo = myfitting.kasa_circle(markers_ref)
    cage_radius_ref = markers_ref_xyr[2]
    markers_radii_ref = markers_ref_lsminfo["radii"]
    logger_cage.binfo(f"#### lsm fitting for markers_ref\nmarkers_ref_xyr: {markers_ref_xyr}\nmarkers_ref_lsminfo: {markers_ref_lsminfo}")
    if lsmmode == "fast":
        markers_xyr, markers_lsminfo = myfitting.lsm_for_circles(markers)
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr: {markers_xyr.shape}")
        cage = markers_xyr[:, :2]
        cage_radius = markers_xyr[:, 2]
        markers_radii = markers_lsminfo["radii"]
        logger_cage.binfo(f"#### lsm fitting for markers\nmarkers_xyr: {np.nanmean(markers_xyr, axis=0)}, {markers_xyr.shape}\naverage of markers_radii: {np.nanmean(markers_radii, axis=0)}")
    elif lsmmode == "numpy":
        markers_xyr = np.zeros((num_frames, 3))
        markers_lsminfo = []
        markers_radii = np.zeros((num_frames, num_markers))
        markers_geom_error = np.zeros((num_frames, 3)) # mean, max, std
        for f in range(num_frames):
            markers_xyr[f], _lsminfo = myfitting.kasa_circle(markers[f])
            markers_lsminfo.append(_lsminfo)
            markers_radii[f] = _lsminfo["radii"]
            markers_geom_error[f] = _lsminfo["geom_error_mean"], _lsminfo["geom_error_max"], _lsminfo["geom_error_std"]
        cage = markers_xyr[:, :2]
        cage_radius = markers_xyr[:, 2]
        trj_xyr, trj_lsminfo = myfitting.kasa_circle(cage)
        trj_geom_error = [trj_lsminfo["geom_error_mean"], trj_lsminfo["geom_error_max"], trj_lsminfo["geom_error_std"]]
        logger_cage.binfo(f"#### lsm fitting for markers\nmarkers_xyr: {np.nanmean(markers_xyr, axis=0)}, {markers_xyr.shape}\naverage of markers_radii: {np.nanmean(markers_radii, axis=0)}\nmax of markers_geom_error (mean, max, std): {np.nanmax(markers_geom_error, axis=0)}")
        logger_cage.binfo(f"#### lsm fitting for trajectory\ntrj_xyr: {trj_xyr}\ntrj_geom_error (mean, max, std): {trj_geom_error}")
    elif lsmmode == "compare":
        logger_cage.measure_time("numpy_lsm", mode='s')
        markers_xyr_fast, markers_lsminfo_fast = myfitting.lsm_for_circles(markers)
        logger_cage.binfo(f"lsm fitting for markers:\nmarkers_xyr_fast: {markers_xyr_fast.shape}")
        cage_fast = markers_xyr_fast[:, :2]
        cage_radius_fast = markers_xyr_fast[:, 2]
        markers_radii_fast = markers_lsminfo_fast["radii"]
        logger_cage.binfo(f"#### lsm fitting for markers\nmarkers_xyr: {np.nanmean(markers_xyr_fast, axis=0)}, {markers_xyr_fast.shape}\naverage of markers_radii: {np.nanmean(markers_radii_fast, axis=0)}")
        logger_cage.measure_time("numpy_lsm", mode='e')
        logger_cage.measure_time("fast_lsm", mode='s')
        markers_xyr_numpy = np.zeros((num_frames, 3))
        markers_lsminfo_numpy = []
        markers_radii_numpy = np.zeros((num_frames, num_markers))
        markers_geom_error_numpy = np.zeros((num_frames, 3)) # mean, max, std
        for f in range(num_frames):
            markers_xyr_numpy[f], _lsminfo_numpy = myfitting.kasa_circle(markers[f])
            markers_lsminfo_numpy.append(_lsminfo_numpy)
            markers_radii_numpy[f] = _lsminfo_numpy["radii"]
            markers_geom_error_numpy[f] = _lsminfo_numpy["geom_error_mean"], _lsminfo_numpy["geom_error_max"], _lsminfo_numpy["geom_error_std"]
        cage_numpy = markers_xyr_numpy[:, :2]
        cage_radius_numpy = markers_xyr_numpy[:, 2]
        logger_cage.binfo(f"#### lsm fitting for markers\nmarkers_xyr: {np.nanmean(markers_xyr_numpy, axis=0)}, {markers_xyr_numpy.shape}\naverage of markers_radii: {np.nanmean(markers_radii_numpy, axis=0)}\nmax of markers_geom_error (mean, max, std): {np.nanmax(markers_geom_error_numpy, axis=0)}")
        logger_cage.measure_time("fast_lsm", mode='e')
        xyr_error = np.nansum(markers_xyr_fast - markers_xyr_numpy, axis=0)
        radii_error = np.nansum(markers_radii_fast - markers_radii_numpy, axis=0)
        logger_cage.binfo(f"#### difference between lsm mode numpy_linalg and fast for circle\nxyr_error: {xyr_error}, markers_radii_error: {radii_error}")
        cage = cage_numpy
        cage_radius = cage_radius_numpy
        markers_radii = markers_radii_numpy
        trj_xyr, trj_lsminfo = myfitting.kasa_circle(cage)
        trj_geom_error = [trj_lsminfo["geom_error_mean"], trj_lsminfo["geom_error_max"], trj_lsminfo["geom_error_std"]]
        logger_cage.binfo(f"#### lsm fitting for trajectory\ntrj_xyr: {trj_xyr}\ntrj_geom_error (mean, max, std): {trj_geom_error}")

    centrifugal_expansion = cage_radius - cage_radius_ref
    cagetrj_xyr, cagetrj_lsminfo = myfitting.kasa_circle(cage)
    trj_center = cagetrj_xyr[:2]
    trj_radius = cagetrj_xyr[2]
    trj_radii = cagetrj_lsminfo["radii"]
    logger_cage.binfo(f"#### lsm fitting for cage_trajectory:\n{cagetrj_lsminfo}")
    logger_cage.binfo(f"center of cage_trajectory: {trj_center}, radius of cage_trajectory: {trj_radius}, average of centerifugal_expansion: {np.nanmean(centrifugal_expansion):.3}")
    logger_cage.binfo(f"average of cage_trajectory:{np.nanmean(cage, axis=0)}")
    logger_cage.measure_time("lsm_fitting_for_circle", mode='e')

    #### fitzgibbon fitting for ellipse
    logger_cage.measure_time("fitzgibbon_fitting_for_ellipse", mode='s')
    markers_xyabtheta = np.zeros((num_frames, 5))
    for f in range(num_frames):
        markers_xyabtheta[f], _ = myfitting.fitzgibbon_ellipse(markers[f], allow_nan=False)
    cage_fitz = markers_xyabtheta[:, :2]
    cage_major_radius = markers_xyabtheta[:, 2]
    cage_minor_radius = markers_xyabtheta[:, 3]
    logger_cage.binfo(f"#### fitzgibbon fitting for markers\nxyabtheta: {np.nanmean(markers_xyabtheta, axis=0)}, {markers_xyabtheta.shape}")
    # logger_cage.binfo(f"max difference of cage center between circle and fitzgibbon fitting: {np.nanmax(cage - cage_fitz, axis=0)}")
    fitztrj_xyr, fitztrj_lsminfo = myfitting.kasa_circle(cage_fitz)
    fitztrj_geom_error = [fitztrj_lsminfo["geom_error_mean"], fitztrj_lsminfo["geom_error_max"], fitztrj_lsminfo["geom_error_std"]]
    logger_cage.binfo(f"#### lsm fitting for trajectory\ntrj_xyr: {fitztrj_xyr}\ntrj_geom_error (mean, max, std): {fitztrj_geom_error}")
    logger_cage.measure_time("fitzgibbon_fitting_for_ellipse", mode='e')

    #### calculate rotation speed
    logger_cage.measure_time("calc_rotspeed", mode='s')
    markers_angles = np.zeros((num_frames, num_markers))
    for i in range(num_markers):
        markers_angles[:, i] = np.arctan2(markers[:, i, 1]-cage[:, 1], markers[:, i, 0]-cage[:, 0])
    markers_angles = np.unwrap(markers_angles, axis=0)
    initial_m0_angle = markers_angles[0, 0]
    markers_angular_displacement = markers_angles - markers_angles[0]
    cage_Rx = np.nanmean(markers_angular_displacement, axis=1)
    cage_Rvx = np.gradient(cage_Rx, t)
    cage_Rvx_avg = np.nanmean(cage_Rvx)

    cage_polar = mycoord.CoordTransformer2d.cartesian2polar(cage)
    revolution_speed = np.gradient(np.unwrap(cage_polar[:, 1]), t)

    cage_vx = np.gradient(cage[:, 0], t)
    cage_vy = np.gradient(cage[:, 1], t)
    cage_v = np.vstack([cage_vx, cage_vy]).T
    cage_v_norm = np.linalg.norm(cage_v, axis=-1)
    revolution_speed2 = cage_v_norm / cage_polar[:, 0]

    logger_cage.binfo(f"average cage rotation speed: {cage_Rvx_avg/2/np.pi*60:.1f} [rpm]\ninitial angle of marker0: {np.degrees(initial_m0_angle)} [degree]")
    logger_cage.measure_time("calc_rotspeed", mode='e')

    #### calculate deformation
    logger_cage.measure_time("deformation", mode='s')
    deformation = myfitting.calc_elliptical_deformation(markers, markers_ref)
    logger_cage.measure_time("deformation", mode='e')
    system_center = np.zeros((num_frames, 2)) # defined based on inner ring center determined by image in static state


    """
    rotframeSA: center: system, rotspeed: average
    rotframeSI: center: system, rotspeed: instantaneous
    rotframeCA: center: cage, rotspeed: average
    rotframeCI: center: cage, rotspeed: instantaneous

    """
    cage_Rx_avg = cage_Rvx * t
    transfomerSI = mycoord.CoordTransformer2d(name="system_instantaneous", local_origin=system_center, theta=cage_Rx)
    transfomerSA = mycoord.CoordTransformer2d(name="system_average", local_origin=system_center, theta=cage_Rx_avg)
    transfomerCI = mycoord.CoordTransformer2d(name="cage_instantaneous", local_origin=cage, theta=cage_Rx)
    transfomerCA = mycoord.CoordTransformer2d(name="cage_average", local_origin=cage, theta=cage_Rx_avg)



    logger_cage.measure_time("main", mode='e')


    #### check
    if check:
        data_list = [
            # {"id": 0, "data": markers[:, 0, 0], "c": 'b'},
            # {"id": 0, "data": markers[:, 0, 1], "c": 'r'},
            # {"id": 0, "data": markers_radii[:, 0], "c": 'g'},

            # {"id": 1, "data": centrifugal_expansion, "c": 'r'},
            # {"id": 1, "data": np.degrees(markers_angles[:, 1]), "c": 'b'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 1]), "c": 'r'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 2]), "c": 'b'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 3]), "c": 'g'},
            # {"id": 1, "data": np.degrees(markers_angular_displacement[:, 4]), "c": 'c'},
            # {"id": 1, "data": np.degrees(cage_Rx), "c": 'r'},
            # {"id": 1, "data": cage_Rvx / (2 * np.pi), "c": 'r'},
            # {"id": 1, "data": cage_Rvx / (2 * np.pi), "c": 'r'},

            # {"id": 2, "data": trj_radii, "c": 'r'},

            {"id": 0, "data": cage[:, 0], "c": 'r'},
            {"id": 0, "data": cage[:, 1], "c": 'b'},
            {"id": 1, "data": cage_polar[:, 0], "c": 'r'},
            {"id": 1, "data": cage_polar[:, 1], "c": 'b'},
            {"id": 2, "data": revolution_speed, "c": 'r'},
            {"id": 2, "data": revolution_speed2, "c": 'b'},

        ]
        fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
        for i in range(len(data_list)):
            axs[data_list[i]["id"]].plot(t, data_list[i]["data"], lw=1, c=data_list[i]["c"])
        for i in range(3):
            axs[i].axhline(y=0, c='k', lw=0.4)
        # axs[1].set_ylim(0, 100)
        # axs[2].set_ylim(-1, 1)
        plt.show()

def analyze_sound(sound, fs):
    logger_sound = mylogger.MyLogger("logger_sound", outdir=outdir)
    logger_sound.measure_time("main", mode='s')
    N = len(sound)
    t = np.arange(N) / fs
    duration = N / fs
    logger_sound.binfo(f"#### inputdata\nsound: {sound.shape}\nsample_rate: {fs}\nduration: {duration}\nN: {N}")

    sound_processor = data_processor.TimeSeriesProcessor(sound=sound, fs=fs)
    window_time = 0.01
    noise_detection = sound_processor.detect_noise_rms(window_time=window_time)

    #### check
    rms = sound_processor.calc_rms(edge="pad", window_time=window_time)
    rms_db = data_processor.TimeSeriesProcessor.pa2db(rms)
    shift = rms_db[0] - rms[0]
    rms = rms + shift

    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
    fig, axs = plotter.myfig()

    axs[0].plot(t, sound, lw=1)
    axs[1].plot(t, rms, lw=1, c='b', label="rms")
    axs[1].plot(t, rms_db, lw=1, c='g', label="rms_dB")
    axs[1].plot(t, noise_detection["rms"], lw=1, c='r', label="rms for detection")
    axs[1].axhline(y=noise_detection["threshold"], lw=1, c='k', ls="--")


    noisy_period = []
    for _run in noise_detection["noisy_runs"]:
        st, et = _run[0] / sound_processor.fs, _run[1] / sound_processor.fs
        noisy_period.append([st, et])
        axs[1].axvspan(st, et, color='r', alpha=0.2)

    axs[1].legend()
    plt.show()

    logger_sound.binfo(f"noisy_period: {str(noisy_period)} [sec]")

    logger_sound.measure_time("main", mode='e')

if __name__ == '__main__':
    print("---- test ----")
    #### sample data

    datadir = config.ROOT / "sampledata" / "SIMPLE50" / "ROT_REV"
    print(f"datadir: {datadir.exists()}")

    def get_datapath(datadir):
        markers = list(datadir.glob("*markers.csv"))
        markers_noise = list(datadir.glob("*markers_noise.csv"))
        zero = list(datadir.glob("*zero.csv"))
        if len(markers) != 1:
            raise ValueError(f"makers data must be 1, {len(markers)}")
        if len(markers_noise) != 1:
            raise ValueError(f"makers_noise data must be 1, {len(markers_noise)}")
        if len(zero) != 1:
            raise ValueError(f"zero data must be 1, {len(zero)}")
        return markers[0], markers_noise[0], zero[0]

    markers, markers_noise, zero = get_datapath(datadir)
    coorddl = data_handler.CoordDataLoader(markers, zero, data_format="sample")
    print(repr(coorddl))

    # for k, s in dataseries_handler.seriesmap.items():
        # print(k, s)

    # datamap = dataseries_handler.datamap_loader.datamap
    # print(datamap)

    # dataseries, datamapinfo = dataseries_handler.select_series(23, 1)



    t = np.linspace(0, 1, 48000, endpoint=False)
    _sound = np.cos(400 * 2*np.pi*t)
    sound = np.where(t > 0.2, _sound*5, _sound)
    sound = np.where(t > 0.4, _sound*1.2, sound)
    sound = np.where(t > 0.6, _sound*1.8, sound)
    sound = np.where(t > 0.8, _sound*2, sound)

    # analyze_cage(makers, markers_ref, fps=fps, lsmmode="compare", check=True)
    # analyze_sound(sound, fs=48000)


