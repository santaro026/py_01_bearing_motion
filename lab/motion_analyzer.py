"""
Created on Tue Apr 14 18:06:54 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal

import re
from pathlib import Path
from datetime import datetime
import json
import pickle

import config
import data_handler, data_processor, helper

from mymods import myplotter, mylogger, myfitting, mycoord

np.set_printoptions(linewidth=np.inf)

dtype_map = {
    "kasa": [("coord_y", np.float64), ("coord_z", np.float64), ("length_r", np.float64)],
    "fitz": [("coord_y", np.float64), ("coord_z", np.float64), ("length_a", np.float64), ("length_b", np.float64), ("angle", np.float64)],
    "coord": [("coord_y", np.float64), ("coord_z", np.float64)],
    "length": [("length", np.float64)],
    "angle": [("angle", np.float64)],

}

def reset_view2float64(data):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = reset_view2float64(v)
    elif isinstance(data, np.ndarray):
        return data.view(np.float64)
    return data

def perform_cage_fitting(markers):
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]
    #### kasa fitting for circle
    kasa = np.zeros((num_frames, 3))
    kasainfo = []
    kasa_geom_error = np.zeros((num_frames, 3)) # mean, max, std
    for f in range(num_frames):
        kasa[f], _kasainfo = myfitting.kasa_circle(markers[f])
        kasainfo.append(_kasainfo)
        if _kasainfo is None:
            kasa_geom_error[f] = np.full(3, np.nan)
        else:
            kasa_geom_error[f] = _kasainfo["geom_error_mean"], _kasainfo["geom_error_max"], _kasainfo["geom_error_std"]
    #### fitzgibbon fitting for ellipse
    fitz = np.zeros((num_frames, 5))
    fitzinfo = []
    fitz_geom_error = np.zeros((num_frames, 3)) # mean, max, std
    for f in range(num_frames):
        fitz[f], _fitzinfo = myfitting.fitzgibbon_ellipse(markers[f], allow_nan=False)
        fitzinfo.append(_fitzinfo)
        if _fitzinfo is None:
            fitz_geom_error[f] = np.full(3, np.nan)
        else:
            fitz_geom_error[f] = _fitzinfo["geom_error_mean"], _fitzinfo["geom_error_max"], _fitzinfo["geom_error_std"]
    results = {
        "kasa": np.ascontiguousarray(kasa).view(dtype_map["kasa"]),
        "kasainfo": kasainfo,
        "fitz": np.ascontiguousarray(fitz).view(dtype_map["fitz"]),
        "fitzinfo": fitzinfo
    }
    return results

def analyze_trajectory(fit):
    kasatrj_kasa, kasatrj_kasainfo = myfitting.kasa_circle(fit["kasa"].view(np.float64)[:, :2])
    kasatrj_avg = np.nanmean(fit["kasa"].view(np.float64)[:, :2], axis=0)
    fitztrj_kasa, fitztrj_kasainfo = myfitting.kasa_circle(fit["fitz"].view(np.float64)[:, :2])
    fitztrj_avg = np.nanmean(fit["fitz"].view(np.float64)[:, :2], axis=0)
    results = {
        "kasatrj_kasa": np.ascontiguousarray(kasatrj_kasa).view(dtype_map["kasa"]),
        "kasatrj_kasainfo": kasatrj_kasainfo,
        "kasatrj_avg": np.ascontiguousarray(kasatrj_avg).view(dtype_map["coord"]),
        "fitztrj_kasa": np.ascontiguousarray(fitztrj_kasa).view(dtype_map["kasa"]),
        "fitztrj_kasinfo": fitztrj_kasainfo,
        "fitztrj_avg": np.ascontiguousarray(fitztrj_avg).view(dtype_map["coord"]),
    }
    return results

def calc_rotational_kinematics(markers, cage, fps):
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]
    frame = np.arange(num_frames)
    t = frame / fps
    markers_azimuth = np.zeros((num_frames, num_markers))
    for i in range(num_markers):
        markers_azimuth[:, i] = np.arctan2(markers[:, i, 1]-cage[:, 1], markers[:, i, 0]-cage[:, 0])
    markers_azimuth = np.unwrap(markers_azimuth, axis=0)
    markers_angular_displacement = markers_azimuth - markers_azimuth[0]
    cage_Rx = np.nanmean(markers_angular_displacement, axis=1)
    cage_Rvx = np.gradient(cage_Rx, t)
    cage_Rvx_avg = np.nanmean(cage_Rvx)
    cage_Rx_const = cage_Rvx * t
    cage_polar = mycoord.CoordTransformer2d.cartesian2polar(cage)
    revolution_speed = np.gradient(np.unwrap(cage_polar[:, 1]), t)
    results = {
        "markers_azimuth": np.ascontiguousarray(markers_azimuth).view(dtype_map["angle"]),
        "cage_Rx": np.ascontiguousarray(cage_Rx).view(dtype_map["angle"]),
        "cage_Rx_const": np.ascontiguousarray(cage_Rx_const).view(dtype_map["angle"]),
        "cage_Rvx": np.ascontiguousarray(cage_Rvx).view(dtype_map["angle"]),
        "cage_Rvx_avg": np.ascontiguousarray(cage_Rvx_avg).view(dtype_map["angle"]),
        "cage_revolution_speed": np.ascontiguousarray(revolution_speed).view(dtype_map["angle"]),
    }
    return results

def calc_deformation(markers, markers_ref, fitting, fitting_ref):
    num_frames, num_markers, _ = markers.shape
    num_axes = num_markers // 2
    #### calclate diameters
    diameters_vct = markers[:, :num_axes, :] - markers[:, num_axes:, :]
    diameters_norm = np.linalg.norm(diameters_vct, axis=2)
    diameters_theta = np.arctan2(diameters_vct[:, :, 1], diameters_vct[:, :, 0])
    diameters_ref_vct = markers_ref[:, :num_axes, :] - markers_ref[:, num_axes:, :]
    diameters_ref_norm = np.linalg.norm(diameters_ref_vct, axis=2)
    delta_diameters = diameters_norm - diameters_ref_norm
    roundness = (np.amax(delta_diameters, axis=1) - np.amin(delta_diameters, axis=1)) / 2
    direction_id = np.array([np.argmax(delta_diameters, axis=1), np.argmin(delta_diameters, axis=1)]).T
    direction = np.array([diameters_theta[np.arange(num_frames), direction_id[:, 0]], diameters_theta[np.arange(num_frames), direction_id[:, 1]]]).T
    deformation_angle = np.abs(direction[:, 1] - direction[:, 0]) % np.pi
    centrifugal_expansion_kasa = fitting["kasa"].view(np.float64)[:, :2] - fitting_ref["kasa"].view(np.float64)[:, :2]
    roundness_fitz = fitting["fitz"].view(np.float64)[:, 3] - fitting["fitz"].view(np.float64)[:, 4]
    results = {
        "diameters_norm": np.ascontiguousarray(diameters_norm).view(dtype_map["length"]),
        "delta_diameters": np.ascontiguousarray(delta_diameters).view(dtype_map["length"]),
        "roundness": np.ascontiguousarray(roundness).view(dtype_map["length"]),
        "direction_id": direction_id,
        "direction": np.ascontiguousarray(direction).view(dtype_map["angle"]),
        "deformation_angle": np.ascontiguousarray(deformation_angle).view(dtype_map["angle"]),
        "centrifugal_expansion_kasa": np.ascontiguousarray(centrifugal_expansion_kasa).view(dtype_map["length"]),
        "roundness_fitz": np.ascontiguousarray(roundness_fitz).view(dtype_map["length"])}
    return results

def analyze_cage(markers, markers_ref, fps, outdir=config.ROOT/"results"/"test", prefix=""):
    loggaer_cage_fitting = mylogger.MyLogger(f"{prefix}_analyze_cage", outdir=outdir)
    loggaer_cage_fitting.measure_time("main", mode='s')
    num_frames, num_markers_cage, num_dimension = markers.shape
    frame = np.arange(num_frames, dtype=np.int64)
    t = frame / fps
    markers_fit = perform_cage_fitting(markers) # to convert
    markers_ref_fit = perform_cage_fitting(markers_ref) # to convert
    trajectory_prop = analyze_trajectory(markers_fit) # to convert
    rotkinematics_kasa = calc_rotational_kinematics(markers, markers_fit["kasa"].view(np.float64)[:, :2], fps) # not to convert
    rotkinematics_fitz = calc_rotational_kinematics(markers, markers_fit["fitz"].view(np.float64)[:, :2], fps) # not to convert
    deformation = calc_deformation(markers, markers_ref, markers_fit, markers_ref_fit) # to convert
    #### output
    tmpdir = outdir / "tmp"
    tmpdir.mkdir(exist_ok=True, parents=True)
    pkldict = {
        "frame": np.ascontiguousarray(frame).view([("frame", np.int64)]),
        "time": np.ascontiguousarray(t).view([("time", np.float64)]),
        "markers": np.ascontiguousarray(markers).view(dtype_map["coord"]),
        "markers_ref": np.ascontiguousarray(markers_ref).view(dtype_map["coord"]),
        "markers_fit": markers_fit,
        "markers_ref_fit": markers_ref_fit,
        "trajectory_prop": trajectory_prop,
        "rotkinematics_kasa": rotkinematics_kasa,
        "rotkinematics_fitz": rotkinematics_fitz,
        "deformation": deformation,
    }
    with open(tmpdir/f"{prefix}_data.pkl", "wb") as f:
        pickle.dump(pkldict, f)
    #### summarize result
    markers_fit = reset_view2float64(markers_fit)
    markers_ref_fit = reset_view2float64(markers_ref_fit)
    trajectory_prop = reset_view2float64(trajectory_prop)
    rotkinematics_kasa = reset_view2float64(rotkinematics_kasa)
    rotkinematics_fitz = reset_view2float64(rotkinematics_fitz)
    deformation = reset_view2float64(deformation)

    allclose_kasa_fitz = np.allclose(markers_fit["kasa"][:, :2], markers_fit["fitz"][:, :2], atol=1e-9, equal_nan=True)
    diff_kasa_fitz = markers_fit["kasa"][:, :2] - markers_fit["fitz"][:, :2]
    result_summary = {
        #### meta
        "num_frames": num_frames,
        "num_markers_cage": num_markers_cage,
        #### fitting information
        "allclose_kasa_fitz": allclose_kasa_fitz,
        "diff_kasa_fitz": np.nanmax(diff_kasa_fitz),
        #### markers_ref
        "cage_radius_ref": markers_ref_fit["kasa"][0, 2],
        "cage_radii_ref": markers_ref_fit["fitz"][0, 2:4],
        "markers_azimuth_kasa": np.degrees(np.atan2(markers_ref[0, :, 1] - markers_ref_fit["kasa"][0, np.newaxis, 1], markers_ref[0, :, 0] - markers_ref_fit["kasa"][0, np.newaxis, 0])),
        "markers_azimuth_fitz": np.degrees(np.atan2(markers_ref[0, :, 1] - markers_ref_fit["fitz"][0, np.newaxis, 1], markers_ref[0, :, 0] - markers_ref_fit["fitz"][0, np.newaxis, 0])),
        #### trajectory
        "kasa_trajectory": trajectory_prop["kasatrj_kasa"],
        "fitz_trajectory": trajectory_prop["fitztrj_kasa"],
        "cage_rpm_avg_kasa": rotkinematics_kasa["cage_Rvx_avg"]/2/np.pi*60,
        "cage_rpm_avg_fitz": rotkinematics_fitz["cage_Rvx_avg"]/2/np.pi*60,

        #### deformation
        "roundness_markers_avg": np.nanmean(deformation["roundness"]),
        "centrifugal_expansion_avg": np.nanmean(deformation["centrifugal_expansion_kasa"]),
        "roundness_fitz_avg": np.nanmean(deformation["roundness_fitz"]),
    }
    for _k, _v in result_summary.items():
        if isinstance(_v, np.ndarray):
            result_summary[_k] = _v.tolist()
    with open(outdir / f"{prefix}_summary.json", 'w', encoding="utf-8") as f:
        json.dump(result_summary, f, indent=4)
    loggaer_cage_fitting.measure_time("main", mode='e')



def summarize_information(tc, sc, datamappath=r"D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx", outdir=config.ROOT/"results"/"test"):
    datamapld = data_handler.DataMapLoader(datamappath)
    testinfo = datamapld.extract_testinfo(tc)
    shootinginfo = datamapld.extract_info_from_tcsc(tc, sc)

    summary = helper.merge_dicts(testinfo, shootinginfo)
    print(summary)

    with open(outdir/f"tc{tc}_{sc}sc_testinfo.json", 'w', encoding="utf-8") as f:
        json.dump(summary, f)


if __name__ == '__main__':
    print("---- test ----")

    outdir = config.ROOT / "results" / "test"
    outdir.mkdir(exist_ok=True, parents=True)

    #### sample data
    # datamappath = Path(r"D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    # datamapld = data_handler.DataMapLoader(datamappath)
    # print(datamapld.summary)

    datadir = config.ROOT / "sampledata" / "SIMPLE50"
    datapath_markers = list(datadir.rglob(r"*ROT_REV_*markers.csv"))[0]
    datapath_zero = list(datadir.rglob(r"*zero.csv"))[0]
    coorddl = data_handler.CoordDataLoader(datapath_markers, datapath_zero, data_format="csv")

    # datadir_par = Path(r"N:\1005_tyn\tema")
    # datadir = list(datadir_par.glob(r"tc11*"))[0]
    # print(f"datadir: {datadir.exists()}")
    # datapath_markers = list(datadir.rglob(r"*sc06*.txt"))[0]
    # datapath_zero = list(datadir.rglob(r"*zero.txt"))[0]
    # coorddl = data_handler.CoordDataLoader(datapath_markers, datapath_zero, data_format="tema")
    # print(f"target file:")
    # print(f"{datapath_markers}")
    # print(f"{datapath_zero}")

    analyze_cage(coorddl.cage_markers, coorddl.cage_markers_zero, fps=coorddl.fps, outdir=outdir, prefix=coorddl.code)
    # fps = 10000
    # num_frames = 10001
    # pixel2mm = 1
    # system_center = list(np.zeros(2))
    # num_cage_markers = 8
    # num_ring_markers = 1
    # num_markers = 9
    # info = {
    #     "fps": fps,
    #     "num_frames": num_frames,
    #     "duration": num_frames / fps,
    #     "pixel2mm": pixel2mm,
    #     "system_center": system_center,
    #     "num_cage_markers": num_cage_markers,
    #     "num_ring_markers": num_ring_markers,
    #     "num_markers": num_markers,
    # }
    # import json
    # with open(config.ROOT/"results"/"test"/"camera_info.json", "w") as f:
    #     json.dump(info, f)


    # summarize_information(1, 2)

    print("complete")

