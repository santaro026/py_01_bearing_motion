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
    cage_kasa = np.zeros((num_frames, 3))
    cage_kasainfo = []
    kasa_geom_error = np.zeros((num_frames, 3)) # mean, max, std
    for f in range(num_frames):
        cage_kasa[f], _kasainfo = myfitting.kasa_circle(markers[f])
        cage_kasainfo.append(_kasainfo)
        if _kasainfo is None:
            kasa_geom_error[f] = np.full(3, np.nan)
        else:
            kasa_geom_error[f] = _kasainfo["geom_error_mean"], _kasainfo["geom_error_max"], _kasainfo["geom_error_std"]
    #### fitzgibbon fitting for ellipse
    cage_fitz = np.zeros((num_frames, 5))
    cage_fitzinfo = []
    fitz_geom_error = np.zeros((num_frames, 3)) # mean, max, std
    for f in range(num_frames):
        cage_fitz[f], _fitzinfo = myfitting.fitzgibbon_ellipse(markers[f], allow_nan=False)
        cage_fitzinfo.append(_fitzinfo)
        if _fitzinfo is None:
            kasa_geom_error[f] = np.full(3, np.nan)
        else:
            fitz_geom_error[f] = _fitzinfo["geom_error_mean"], _fitzinfo["geom_error_max"], _fitzinfo["geom_error_std"]
    results = {
        "cage_kasa": cage_kasa,
        "cage_kasainfo": cage_kasainfo,
        "cage_fitz": cage_fitz,
        "cage_fitzinfo": cage_fitzinfo
    }
    return results

def compute_rotational_kinematics(markers, cage, fps):
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]
    frame = np.arange(num_frames)
    t = frame / fps
    #### calculate rotation speed
    def calc_rotkinematics(markers, cage):
        markers_angles = np.zeros((num_frames, num_markers))
        for i in range(num_markers):
            markers_angles[:, i] = np.arctan2(markers[:, i, 1]-cage[:, 1], markers[:, i, 0]-cage[:, 0])
        markers_angles = np.unwrap(markers_angles, axis=0)
        markers_angular_displacement = markers_angles - markers_angles[0]
        cage_Rx = np.nanmean(markers_angular_displacement, axis=1)
        cage_Rvx = np.gradient(cage_Rx, t)
        cage_Rvx_avg = np.nanmean(cage_Rvx)
        cage_Rx_const = cage_Rvx * t
        cage_polar = mycoord.CoordTransformer2d.cartesian2polar(cage)
        revolution_speed = np.gradient(np.unwrap(cage_polar[:, 1]), t)
        results = {
            "markers_angles": markers_angles,
            "cage_Rx": cage_Rx,
            "cage_Rx_const": cage_Rx_const,
            "cage_Rvx": cage_Rvx,
            "cage_Rvx_avg": cage_Rvx_avg,
            "cage_revolution_speed": revolution_speed,
        }
        return results

def output(markers, markers_ref, fps, outdir=config.ROOT/"results"/"test", prefix=""):
    result_summary = {
        "num_frames": num_frames,
        "num_markers_cage": num_markers,

        #### kasa
        "cage_kasa_ref": cage_kasa_ref,
        "markers_angles_kasa": rotkinematics_kasa["markers_angles"][0],
        "kasa_trajectory_xyr": kasatrj_xyr,
        "cage_rpm_avg_kasa": rotkinematics_kasa["cage_Rvx_avg"]/2/np.pi*60,
        "centrifugal_expansion_avg": np.nanmean(centrifugal_expansion),

        #### fitz
        "cage_fitz_ref": cage_fitz_ref,
        "markers_angles_fitz": rotkinematics_fitz["markers_angles"][0],
        "fitz_trajectory_xyr": fitztrj_xyr,
        "cage_rpm_avg_fitz": rotkinematics_fitz["cage_Rvx_avg"]/2/np.pi*60,
        "roundness_fitz_avg": np.nanmean(roundness_fitz),

        #### markers
        "roundness_markers_avg": np.nanmean(deformation_markers["roundness"]),
    }
    for _k, _v in result_summary.items():
        if isinstance(_v, np.ndarray):
            result_summary[_k] = _v.tolist()
    with open(outdir / f"{prefix}_summary.json", 'w', encoding="utf-8") as f:
        json.dump(result_summary, f)

    tmpdir = outdir / "tmp"
    tmpdir.mkdir(exist_ok=True, parents=True)
    pkldict = {
        "markers": markers,
        "markers_ref": markers_ref,
        "cage_kasa_ref": cage_kasa_ref,
        "cage_kasa": cage_kasa,
        "cage_fitz_ref": cage_fitz_ref,
        "cage_fitz": cage_fitz,
    }
    with open(tmpdir/f"{prefix}_data.pkl", "wb") as f:
        pickle.dump(pkldict, f)

    loggaer_cage_fitting.measure_time("output", mode='e')
    loggaer_cage_fitting.measure_time("main", mode='e')

    def compare_2array(a, b, atol=1e-8, equal_nan=False, name=""):
        print(f"compare {name}")
        same = np.allclose(a, b, atol=atol, equal_nan=equal_nan)
        print(f"same: {same}")
        diff = a - b
        print(f"max difference: {np.max(diff, axis=0)}")
    # compare_2array(cage_kasa, cage_fitz, name="kasa and fitz")


def analyze_cage(markers, markers_ref, fps, outdir=config.ROOT/"results"/"test", prefix=""):
    loggaer_cage_fitting = mylogger.MyLogger(f"{prefix}_cage_fitting", outdir=outdir)
    loggaer_cage_fitting.measure_time("main", mode='s')
    markers_fit = perform_cage_fitting(markers)
    markers_ref_fit = perform_cage_fitting(markers_ref)
    rotkinematics_kasa = compute_rotational_kinematics(markers, markers_fit["cage_kasa"])
    rotkinematics_fitz = compute_rotational_kinematics(markers, markers_fit["cage_fitz"])


def perform_cage_fitting(markers, markers_ref, fps, outdir=config.ROOT/"results"/"test", prefix=""):
    loggaer_cage_fitting = mylogger.MyLogger(f"{prefix}_cage_fitting", outdir=outdir)
    loggaer_cage_fitting.measure_time("main", mode='s')
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]
    frame = np.arange(num_frames)
    t = frame / fps
    loggaer_cage_fitting.binfo(f"#### inputdata")
    loggaer_cage_fitting.binfo(f"markers: {markers.shape}")
    loggaer_cage_fitting.binfo(f"markers_ref: {markers_ref.shape}")
    loggaer_cage_fitting.binfo(f"fps: {fps}")

    #### kasa fitting for circle
    loggaer_cage_fitting.measure_time("kasa_fitting_for_circle", mode='s')
    cage_kasa_ref, markers_ref_kasainfo = myfitting.kasa_circle(markers_ref[0])
    loggaer_cage_fitting.binfo(f"#### kasa fitting for markers_ref")
    loggaer_cage_fitting.binfo(f"cage_kasa_ref: {cage_kasa_ref}")
    loggaer_cage_fitting.binfo(f"markers_ref_kasainfo: {markers_ref_kasainfo}")

    cage_kasa = np.zeros((num_frames, 3))
    cage_kasainfo = []
    kasa_geom_error = np.zeros((num_frames, 3)) # mean, max, std
    for f in range(num_frames):
        cage_kasa[f], _kasainfo = myfitting.kasa_circle(markers[f])
        cage_kasainfo.append(_kasainfo)
        if _kasainfo is None:
            kasa_geom_error[f] = np.full(3, np.nan)
        else:
            kasa_geom_error[f] = _kasainfo["geom_error_mean"], _kasainfo["geom_error_max"], _kasainfo["geom_error_std"]
    loggaer_cage_fitting.binfo(f"#### kasa fitting for markers")
    loggaer_cage_fitting.binfo(f"cage_kasa: {np.nanmean(cage_kasa, axis=0)}, {cage_kasa.shape}")
    loggaer_cage_fitting.binfo(f"max kasa_geom_error across all time step (mean, max, std): {np.nanmax(kasa_geom_error, axis=0)}")

    centrifugal_expansion = cage_kasa[:, 2] - cage_kasa_ref[2]
    loggaer_cage_fitting.binfo(f"average of centerifugal_expansion: {np.nanmean(centrifugal_expansion)}")

    kasatrj_xyr, kasatrj_lsminfo = myfitting.kasa_circle(cage_kasa[:, :2])
    loggaer_cage_fitting.binfo(f"#### kasa fitting for cage_trajectory:\n{kasatrj_lsminfo}")
    loggaer_cage_fitting.binfo(f"cage_trajectory xyr: {kasatrj_xyr}")
    loggaer_cage_fitting.binfo(f"cage_trajectory avg (reference):{np.nanmean(cage_kasa, axis=0)}")

    loggaer_cage_fitting.measure_time("kasa_fitting_for_circle", mode='e')


    #### fitzgibbon fitting for ellipse
    loggaer_cage_fitting.measure_time("fitzgibbon_fitting_for_ellipse", mode='s')
    cage_fitz_ref, markers_ref_fitzinfo = myfitting.kasa_circle(markers_ref[0])
    markers_ref_ellipse_deviation = cage_fitz_ref[2:4] - cage_kasa[:, 2:3]
    loggaer_cage_fitting.binfo(f"#### fitzgibbon fitting for markers_ref")
    loggaer_cage_fitting.binfo(f"cage_kasa_ref: {cage_fitz_ref}")
    loggaer_cage_fitting.binfo(f"deviation from the ideal circle: {markers_ref_ellipse_deviation}")
    loggaer_cage_fitting.binfo(f"markers_ref_fitzinfo: {markers_ref_fitzinfo}")

    cage_fitz = np.zeros((num_frames, 5))
    cage_fitz_info = []
    fitz_geom_error = np.zeros((num_frames, 3)) # mean, max, std
    for f in range(num_frames):
        cage_fitz[f], _fitz_info = myfitting.fitzgibbon_ellipse(markers[f], allow_nan=False)
        cage_fitz_info.append(_fitz_info)
        if _fitz_info is None:
            kasa_geom_error[f] = np.full(3, np.nan)
        else:
            fitz_geom_error[f] = _fitz_info["geom_error_mean"], _fitz_info["geom_error_max"], _fitz_info["geom_error_std"]
    loggaer_cage_fitting.binfo(f"#### fitzgibbon fitting for markers\nxyabtheta: {np.nanmean(cage_fitz, axis=0)}, {cage_fitz.shape}")
    loggaer_cage_fitting.binfo(f"max difference of cage center between circle and fitzgibbon fitting: {np.nanmax(cage_kasa[:, :2] - cage_fitz[:, :2], axis=0)}")
    loggaer_cage_fitting.binfo(f"max fitz_geom_error across all time step (mean, max, std): {np.nanmax(fitz_geom_error, axis=0)}")

    roundness_fitz = cage_fitz[:, 2] - cage_fitz[:, 3]
    loggaer_cage_fitting.binfo(f"roundness avg: {np.nanmean(roundness_fitz)}")

    fitztrj_xyr, fitztrj_kasainfo = myfitting.kasa_circle(cage_fitz[:, :2])
    loggaer_cage_fitting.binfo(f"#### lsm fitting for cage_trajectory:\n{fitztrj_kasainfo}")
    loggaer_cage_fitting.binfo(f"cage_trajectory xyr (kasa): {fitztrj_xyr}")
    loggaer_cage_fitting.binfo(f"cage_trajectory avg (reference):{np.nanmean(cage_fitz[:, :2], axis=0)}")

    loggaer_cage_fitting.measure_time("fitzgibbon_fitting_for_ellipse", mode='e')

    #### calculate rotation speed
    loggaer_cage_fitting.measure_time("calc_rotspeed", mode='s')

    def calc_rotkinematics(cage):
        markers_angles = np.zeros((num_frames, num_markers))
        for i in range(num_markers):
            markers_angles[:, i] = np.arctan2(markers[:, i, 1]-cage[:, 1], markers[:, i, 0]-cage[:, 0])
        markers_angles = np.unwrap(markers_angles, axis=0)
        markers_angular_displacement = markers_angles - markers_angles[0]
        cage_Rx = np.nanmean(markers_angular_displacement, axis=1)
        cage_Rvx = np.gradient(cage_Rx, t)
        cage_Rvx_avg = np.nanmean(cage_Rvx)
        cage_Rx_const = cage_Rvx * t
        cage_polar = mycoord.CoordTransformer2d.cartesian2polar(cage_kasa)
        revolution_speed = np.gradient(np.unwrap(cage_polar[:, 1]), t)
        res = {
            "markers_angles": markers_angles,
            "markers_angular_displacement": markers_angular_displacement,
            "cage_Rx": cage_Rx,
            "cage_Rx_const": cage_Rx_const,
            "cage_Rvx": cage_Rvx,
            "cage_Rvx_avg": cage_Rvx_avg,
            "cage_revolution_speed": revolution_speed,
        }
        return res

    rotkinematics_kasa = calc_rotkinematics(cage_kasa[:, :2])
    rotkinematics_fitz = calc_rotkinematics(cage_fitz[:, :2])
    loggaer_cage_fitting.binfo(f"average cage rotation speed (kasa): {rotkinematics_kasa["cage_Rvx_avg"]/2/np.pi*60:.1f} [rpm]\ninitial angle of marker0 (kasa): {np.degrees(rotkinematics_kasa["markers_angles"][0, 0])} [degree]")
    loggaer_cage_fitting.binfo(f"average cage rotation speed (fitz): {rotkinematics_fitz["cage_Rvx_avg"]/2/np.pi*60:.1f} [rpm]\ninitial angle of marker0 (fitz): {np.degrees(rotkinematics_fitz["markers_angles"][0, 0])} [degree]")
    loggaer_cage_fitting.measure_time("calc_rotspeed", mode='e')

    #### calculate deformation from markers
    loggaer_cage_fitting.measure_time("deformation_markers", mode='s')
    deformation_markers = myfitting.calc_elliptical_deformation(markers, markers_ref)
    loggaer_cage_fitting.binfo(f"average roundness: {np.nanmean(deformation_markers["roundness"])}")
    loggaer_cage_fitting.measure_time("deformation_markers", mode='e')

    #### output
    loggaer_cage_fitting.measure_time("output", mode='s')
    # headers = helper.get_headers(num_markers=num_markers)
    # header_cage_kasa = headers["cage_kasa"]
    # header_cage_fitz = headers["cage_fitz"]
    # header_markers = headers["markers"]
    # cage_kasa_arr = np.column_stack([
    #     cage_kasa,
    #     rotkinematics_kasa["cage_Rx"],
    # ])
    # cage_kasa_df = pl.DataFrame(cage_kasa_arr, schema=header_cage_kasa)
    # cage_fitz_arr = np.column_stack([
    #     cage_fitz,
    #     rotkinematics_kasa["cage_Rx"],
    # ])
    # cage_fitz_df = pl.DataFrame(cage_fitz_arr, schema=header_cage_fitz)

    # markers_arr = np.column_stack([
    #     markers.reshape(num_frames, -1),
    # ])
    # markers_ref_arr = markers_ref.reshape(1, -1)
    # markers_df = pl.DataFrame(markers_arr, schema=header_markers)
    # markers_ref_df = pl.DataFrame(markers_ref_arr, schema=header_markers)

    # cage_kasa_df.write_csv(outdir / f"{prefix}_cage_kasa.csv")
    # cage_fitz_df.write_csv(outdir / f"{prefix}_cage_fitz.csv")
    # markers_df.write_csv(outdir / f"{prefix}_markers.csv")
    # markers_ref_df.write_csv(outdir / f"{prefix}_markers_ref.csv")

    result_summary = {
        "num_frames": num_frames,
        "num_markers_cage": num_markers,

        #### kasa
        "cage_kasa_ref": cage_kasa_ref,
        "markers_angles_kasa": rotkinematics_kasa["markers_angles"][0],
        "kasa_trajectory_xyr": kasatrj_xyr,
        "cage_rpm_avg_kasa": rotkinematics_kasa["cage_Rvx_avg"]/2/np.pi*60,
        "centrifugal_expansion_avg": np.nanmean(centrifugal_expansion),

        #### fitz
        "cage_fitz_ref": cage_fitz_ref,
        "markers_angles_fitz": rotkinematics_fitz["markers_angles"][0],
        "fitz_trajectory_xyr": fitztrj_xyr,
        "cage_rpm_avg_fitz": rotkinematics_fitz["cage_Rvx_avg"]/2/np.pi*60,
        "roundness_fitz_avg": np.nanmean(roundness_fitz),

        #### markers
        "roundness_markers_avg": np.nanmean(deformation_markers["roundness"]),
    }
    for _k, _v in result_summary.items():
        if isinstance(_v, np.ndarray):
            result_summary[_k] = _v.tolist()
    with open(outdir / f"{prefix}_summary.json", 'w', encoding="utf-8") as f:
        json.dump(result_summary, f)

    tmpdir = outdir / "tmp"
    tmpdir.mkdir(exist_ok=True, parents=True)
    pkldict = {
        "markers": markers,
        "markers_ref": markers_ref,
        "cage_kasa_ref": cage_kasa_ref,
        "cage_kasa": cage_kasa,
        "cage_fitz_ref": cage_fitz_ref,
        "cage_fitz": cage_fitz,
    }
    with open(tmpdir/f"{prefix}_data.pkl", "wb") as f:
        pickle.dump(pkldict, f)

    loggaer_cage_fitting.measure_time("output", mode='e')
    loggaer_cage_fitting.measure_time("main", mode='e')

    def compare_2array(a, b, atol=1e-8, equal_nan=False, name=""):
        print(f"compare {name}")
        same = np.allclose(a, b, atol=atol, equal_nan=equal_nan)
        print(f"same: {same}")
        diff = a - b
        print(f"max difference: {np.max(diff, axis=0)}")
    # compare_2array(cage_kasa, cage_fitz, name="kasa and fitz")


def summarize_information(tc, sc, datamappath=r"D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx", outdir=config.ROOT/"results"/"test"):
    datamapld = data_handler.DataMapLoader(datamappath)
    testinfo = datamapld.extract_testinfo(tc)
    shootinginfo = datamapld.extract_info_from_tcsc(tc, sc)

    summary = helper.merge_dicts(testinfo, shootinginfo)
    print(summary)

    with open(outdir/f"tc{tc}_{sc}sc_testinfo.json", 'w', encoding="utf-8") as f:
        json.dump(summary, f)

def main(markers, markers_ref, fps, outdir):
    perform_cage_fitting(markers, markers_ref, fps, outdir)


if __name__ == '__main__':
    print("---- test ----")

    # outdir = config.ROOT / "results" / "test"
    # outdir.mkdir(exist_ok=True, parents=True)

    #### sample data
    # datamappath = Path(r"D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    # datamapld = data_handler.DataMapLoader(datamappath)
    # print(datamapld.summary)

    # datadir = config.ROOT / "sampledata" / "SIMPLE50"
    # datapath_markers = list(datadir.rglob(r"*ROT_REV_*markers.csv"))[0]
    # datapath_zero = list(datadir.rglob(r"*zero.csv"))[0]
    # coorddl = data_handler.CoordDataLoader(datapath_markers, datapath_zero, data_format="csv")

    datadir_par = Path(r"N:\1005_tyn\tema")
    datadir = list(datadir_par.glob(r"tc11*"))[0]
    print(f"datadir: {datadir.exists()}")
    datapath_markers = list(datadir.rglob(r"*sc06*.txt"))[0]
    datapath_zero = list(datadir.rglob(r"*zero.txt"))[0]
    print(f"target file:")
    print(f"{datapath_markers}")
    print(f"{datapath_zero}")

    coorddl = data_handler.CoordDataLoader(datapath_markers, datapath_zero, data_format="tema")
    perform_cage_fitting(coorddl.cage_markers, coorddl.cage_markers_zero, fps=coorddl.fps, prefix=coorddl.code)
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

