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

def analyze_cage(markers, markers_ref, fps, check=False):
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
    frame = np.arange(num_frames)
    t = frame / fps

    #### lsm fitting for circle
    logger_cage.measure_time("lsm_fitting_for_circle", mode='s')
    markers_ref_xyr, markers_ref_lsminfo = myfitting.kasa_circle(markers_ref)
    cage_radius_ref = markers_ref_xyr[2]
    markers_radii_ref = markers_ref_lsminfo["radii"]
    logger_cage.binfo(f"#### lsm fitting for markers_ref\nmarkers_ref_xyr: {markers_ref_xyr}\nmarkers_ref_lsminfo: {markers_ref_lsminfo}")

    markers_xyr = np.zeros((num_frames, 3))
    markers_lsminfo = []
    markers_radii = np.zeros((num_frames, num_markers))
    markers_geom_error = np.zeros((num_frames, 3)) # mean, max, std
    for f in range(num_frames):
        markers_xyr[f], _lsminfo = myfitting.kasa_circle(markers[f])
        markers_lsminfo.append(_lsminfo)
        markers_radii[f] = _lsminfo["radii"]
        markers_geom_error[f] = _lsminfo["geom_error_mean"], _lsminfo["geom_error_max"], _lsminfo["geom_error_std"]
    cage_kasa = markers_xyr[:, :2]
    cage_radius = markers_xyr[:, 2]
    logger_cage.binfo(f"#### lsm fitting for markers\nmarkers_xyr: {np.nanmean(markers_xyr, axis=0)}, {markers_xyr.shape}\naverage of markers_radii: {np.nanmean(markers_radii, axis=0)}\nmax of markers_geom_error (mean, max, std): {np.nanmax(markers_geom_error, axis=0)}")

    centrifugal_expansion = cage_radius - cage_radius_ref
    cagetrj_xyr, cagetrj_lsminfo = myfitting.kasa_circle(cage_kasa)
    trj_center = cagetrj_xyr[:2]
    trj_radius = cagetrj_xyr[2]
    trj_radii = cagetrj_lsminfo["radii"]
    logger_cage.binfo(f"#### lsm fitting for cage_trajectory:\n{cagetrj_lsminfo}")
    logger_cage.binfo(f"center of cage_trajectory: {trj_center}, radius of cage_trajectory: {trj_radius}, average of centerifugal_expansion: {np.nanmean(centrifugal_expansion):.3}")
    logger_cage.binfo(f"average of cage_trajectory:{np.nanmean(cage_kasa, axis=0)}")
    logger_cage.measure_time("lsm_fitting_for_circle", mode='e')

    #### fitzgibbon fitting for ellipse
    logger_cage.measure_time("fitzgibbon_fitting_for_ellipse", mode='s')
    markers_xyabtheta = np.zeros((num_frames, 5))
    for f in range(num_frames):
        markers_xyabtheta[f], _ = myfitting.fitzgibbon_ellipse(markers[f], allow_nan=False)
    cage_fitz = markers_xyabtheta[:, :2]
    cage_fitz_major = markers_xyabtheta[:, 2]
    cage_fitz_minor = markers_xyabtheta[:, 3]
    cage_fitz_theta = markers_xyabtheta[:, 4]
    logger_cage.binfo(f"#### fitzgibbon fitting for markers\nxyabtheta: {np.nanmean(markers_xyabtheta, axis=0)}, {markers_xyabtheta.shape}")
    # logger_cage.binfo(f"max difference of cage center between circle and fitzgibbon fitting: {np.nanmax(cage_kasa - cage_fitz, axis=0)}")
    fitztrj_xyr, fitztrj_lsminfo = myfitting.kasa_circle(cage_fitz)
    fitztrj_geom_error = [fitztrj_lsminfo["geom_error_mean"], fitztrj_lsminfo["geom_error_max"], fitztrj_lsminfo["geom_error_std"]]
    logger_cage.binfo(f"#### lsm fitting for trajectory\ntrj_xyr: {fitztrj_xyr}\ntrj_geom_error (mean, max, std): {fitztrj_geom_error}")
    logger_cage.measure_time("fitzgibbon_fitting_for_ellipse", mode='e')

    #### calculate rotation speed
    logger_cage.measure_time("calc_rotspeed", mode='s')

    def calc_rotkinematics(cage):
        markers_angles = np.zeros((num_frames, num_markers))
        for i in range(num_markers):
            markers_angles[:, i] = np.arctan2(markers[:, i, 1]-cage[:, 1], markers[:, i, 0]-cage[:, 0])
        markers_angles = np.unwrap(markers_angles, axis=0)
        initial_m0_angle = markers_angles[0, 0]
        markers_angular_displacement = markers_angles - markers_angles[0]
        cage_Rx = np.nanmean(markers_angular_displacement, axis=1)
        cage_Rvx = np.gradient(cage_Rx, t)
        cage_Rvx_avg = np.nanmean(cage_Rvx)
        cage_Rx_avg = cage_Rvx * t
        res = {
            "markers_angles": markers_angles,
            "markers_angular_displacement": markers_angular_displacement,
            "initial_m0_angle": initial_m0_angle,
            "cage_Rx": cage_Rx,
            "cage_Rx_avg": cage_Rx_avg,
            "cage_Rvx": cage_Rvx,
            "cage_Rvx_avg": cage_Rvx_avg
        }
        return res

    rotkinematics_kasa = calc_rotkinematics(cage_kasa)
    rotkinematics_fitz = calc_rotkinematics(cage_fitz)

    cage_kasa_polar = mycoord.CoordTransformer2d.cartesian2polar(cage_kasa)
    revolution_speed = np.gradient(np.unwrap(cage_kasa_polar[:, 1]), t)
    cage_fitz_polar = mycoord.CoordTransformer2d.cartesian2polar(cage_fitz)
    revolution_speed = np.gradient(np.unwrap(cage_fitz_polar[:, 1]), t)

    cage_vx = np.gradient(cage_kasa[:, 0], t)
    cage_vy = np.gradient(cage_kasa[:, 1], t)
    cage_v = np.vstack([cage_vx, cage_vy]).T
    cage_v_norm = np.linalg.norm(cage_v, axis=-1)
    revolution_speed2 = cage_v_norm / cage_kasa_polar[:, 0]

    logger_cage.binfo(f"average cage rotation speed: {rotkinematics_kasa["cage_Rvx_avg"]/2/np.pi*60:.1f} [rpm]\ninitial angle of marker0: {np.degrees(rotkinematics_kasa["initial_m0_angle"])} [degree]")
    logger_cage.measure_time("calc_rotspeed", mode='e')

    #### calculate deformation
    logger_cage.measure_time("deformation", mode='s')
    deformation = myfitting.calc_elliptical_deformation(markers, markers_ref)
    logger_cage.binfo(f"average roundness: {np.nanmean(deformation["roundness"])}")
    logger_cage.measure_time("deformation", mode='e')


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

    #### output
    logger_cage.measure_time("output", mode='s')
    header_cage_kasa = [
        "frame", "time", "cy", "cz", "radius", "Rx"
    ]
    header_cage_fitz = [
        "frame", "time", "cy", "cz", "major", "minor", "theta", "Rx"
    ]
    header_markers = ["frame", "time"]
    for i in range(num_markers):
        header_markers.append(f"p{i}y")
        header_markers.append(f"p{i}z")
    cage_kasa_arr = np.column_stack([
        frame,
        t,
        cage_kasa,
        cage_radius,
        rotkinematics_kasa["cage_Rx"],
    ])
    cage_kasa_df = pl.DataFrame(cage_kasa_arr, schema=header_cage_kasa)
    print(cage_kasa_df)
    cage_fitz_arr = np.column_stack([
        frame,
        t,
        cage_fitz,
        cage_fitz_major,
        cage_fitz_minor,
        cage_fitz_theta,
        rotkinematics_kasa["cage_Rx"],
    ])
    cage_fitz_df = pl.DataFrame(cage_fitz_arr, schema=header_cage_fitz)

    print(f"markers: {markers.shape}")
    print(f"markers_radii: {markers_radii.shape}")
    markers_arr = np.column_stack([
        frame,
        t,
        markers.reshape(num_frames, -1),
    ])
    markers_df = pl.DataFrame(markers_arr, schema=header_markers)
    cage_kasa_df.write_csv(outdir / "cage_kasa.csv")
    cage_fitz_df.write_csv(outdir / "cage_fitz.csv")
    markers_df.write_csv(outdir / "markers.csv")
    logger_cage.binfo(f"average roundness: {np.nanmean(deformation["roundness"])}")
    logger_cage.measure_time("output", mode='e')


    logger_cage.measure_time("main", mode='e')

    def compare_2array(a, b, atol=1e-8, equal_nan=False, name=""):
        print(f"compare {name}")
        same = np.allclose(a, b, atol=atol, equal_nan=equal_nan)
        print(f"same: {same}")
        diff = a - b
        print(f"max difference: {np.max(diff, axis=0)}")

    # compare_2array(cage_kasa, cage_fitz, name="kasa and fitz")

    #### check
    check = 0
    if check:
        data_list = [

            {"id": 0, "data": cage_kasa[:, 0], "color": 'r', "lw": 1},
            {"id": 0, "data": cage_kasa[:, 1], "color": 'b', "lw": 1},
            {"id": 0, "data": cage_fitz[:, 0], "color": 'm', "ls": "--", "lw": 2, "alpha": 0.8},
            {"id": 0, "data": cage_fitz[:, 1], "color": 'c', "ls": "--", "lw": 2, "alpha": 0.8},

            {"id": 1, "data": cage_kasa_polar[:, 0], "color": 'r', "lw": 1},
            {"id": 1, "data": cage_kasa_polar[:, 1], "color": 'b', "lw": 1},
            {"id": 1, "data": cage_fitz_polar[:, 0], "color": 'm', "ls": "--", "lw": 2, "alpha": 0.8},
            {"id": 1, "data": cage_fitz_polar[:, 1], "color": 'c', "ls": "--", "lw": 2, "lapha": 0.8},

            # {"id": 2, "data": revolution_speed, "color": 'r'},
            # {"id": 2, "data": revolution_speed2, "color": 'b'},

            # {"id": 2, "data": np.degrees(deformation["deformation_angle"]), "color": 'k', "lw": 1, "alpha": 0.4, "scatter": True},
            # {"id": 2, "data": deformation["roundness"], "color": 'r', "lw": 1, "alpha": 0.4, "scatter": True},
            # {"id": 2, "data": cage_fitz_major, "color": 'r', "lw": 1, "alpha": 0.4, "scatter": True},
            # {"id": 2, "data": cage_fitz_minor, "color": 'b', "lw": 1, "alpha": 0.4, "scatter": True},
            # {"id": 2, "data": np.degrees(cage_fitz_theta), "color": 'g', "lw": 1, "alpha": 0.4, "scatter": True},

            {"id": 2, "data": rotkinematics_kasa["cage_Rx"], "color": 'r', "lw": 1, "alpha": 1, "scatter": True},
            {"id": 2, "data": rotkinematics_fitz["cage_Rx"], "color": 'b', "lw": 1, "alpha": 1, "scater": True, "ls": "--"},


        ]
        fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
        for _d in data_list:
            idx = _d.get("id")
            data = _d.get("data")
            color = _d.get("color", 'k')
            lw = _d.get("lw", 1)
            alpha = _d.get("alpha", 1)
            ls = _d.get("ls", '-')
            scatter = _d.get("scatter", False)
            if not scatter:
                axs[idx].plot(t, data, lw=lw, ls=ls, color=color, alpha=alpha)
            elif scatter:
                axs[idx].scatter(t, data, s=lw, color=color, alpha=alpha)

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

    datamappath = Path(r"D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    datamapld = data_handler.DataMapLoader(datamappath)
    print(datamapld.summary)

    datadir = config.ROOT / "sampledata" / "SIMPLE50" / "ROT_REV_ELLIPSE"
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

    analyze_cage(coorddl.cage_markers, coorddl.cage_markers_zero.squeeze(), fps=coorddl.fps, check=False)




    # t = np.linspace(0, 1, 48000, endpoint=False)
    # _sound = np.cos(400 * 2*np.pi*t)
    # sound = np.where(t > 0.2, _sound*5, _sound)
    # sound = np.where(t > 0.4, _sound*1.2, sound)
    # sound = np.where(t > 0.6, _sound*1.8, sound)
    # sound = np.where(t > 0.8, _sound*2, sound)

    # analyze_cage(makers, markers_ref, fps=fps, lsmmode="compare", check=True)
    # analyze_sound(sound, fs=48000)


