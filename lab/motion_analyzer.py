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
            kasa_geom_error[f] = np.full(3, np.nan)
        else:
            fitz_geom_error[f] = _fitzinfo["geom_error_mean"], _fitzinfo["geom_error_max"], _fitzinfo["geom_error_std"]
    results = {
        "kasa": kasa,
        "kasainfo": kasainfo,
        "fitz": fitz,
        "fitzinfo": fitzinfo
    }
    return results

def analyze_trajectory(fit):
    kasatrj_kasa, kasatrj_kasainfo = myfitting.kasa_circle(fit["kasa"][:, :2])
    kasatrj_avg = np.nanmean(fit["kasa"][:, :2])
    fitztrj_kasa, fitztrj_kasainfo = myfitting.kasa_circle(fit["fitz"][:, :2])
    fitztrj_avg = np.nanmean(fit["fitz"][:, :2])
    results = {
        "kasatrj_kasa": kasatrj_kasa,
        "kasatrj_kasainfo": kasatrj_kasainfo,
        "kasarrj_avg": kasatrj_avg,
        "fitztrj_kasa": fitztrj_kasa,
        "fitztrj_kasinfo": fitztrj_kasainfo,
        "fitztrj_avg": fitztrj_avg,
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
        "markers_azimuth": markers_azimuth,
        "cage_Rx": cage_Rx,
        "cage_Rx_const": cage_Rx_const,
        "cage_Rvx": cage_Rvx,
        "cage_Rvx_avg": cage_Rvx_avg,
        "cage_revolution_speed": revolution_speed,
    }
    return results

def calc_deformation(markers, markers_ref, fitting, fitting_ref):
    deformation_markers = myfitting.calc_elliptical_deformation(markers, markers_ref)
    centrifugal_expansion_kasa = fitting["kasa"][:, :2] - fitting_ref["kasa"][:, :2]
    roundness_fitz = fitting["fitz"][:, 3] - fitting["fitz"][:, 4]
    results = {
        "deformation_markers": deformation_markers,
        "centrifugal_expansion_kasa": centrifugal_expansion_kasa,
        "roundness_fitz": roundness_fitz
    }
    return results

def analyze_cage(markers, markers_ref, fps, outdir=config.ROOT/"results"/"test", prefix=""):
    loggaer_cage_fitting = mylogger.MyLogger(f"{prefix}_analyze_cage", outdir=outdir)
    loggaer_cage_fitting.measure_time("main", mode='s')
    markers_fit = perform_cage_fitting(markers)
    markers_ref_fit = perform_cage_fitting(markers_ref)
    trajectory_prop = analyze_trajectory(markers_fit)
    rotkinematics_kasa = calc_rotational_kinematics(markers, markers_fit["kasa"], fps)
    rotkinematics_fitz = calc_rotational_kinematics(markers, markers_fit["fitz"], fps)
    deformation = calc_deformation(markers, markers_ref, markers_fit, markers_ref_fit)
    #### output
    tmpdir = outdir / "tmp"
    tmpdir.mkdir(exist_ok=True, parents=True)
    pkldict = {
        "markers": markers,
        "markers_ref": markers_ref,
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
    allclose_kasa_fitz = np.allclose(markers_fit["kasa"][:, :2], markers_fit["fitz"][:, :2], atol=1e-9, equal_nan=True)
    diff_kasa_fitz = markers_fit["kasa"][:, :2] - markers_fit["fitz"][:, :2]
    result_summary = {
        #### meta
        "num_frames": markers.shape[0],
        "num_markers_cage": markers.shape[1],
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
        "roundness_markers_avg": np.nanmean(deformation["deformation_markers"]["roundness"]),
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

