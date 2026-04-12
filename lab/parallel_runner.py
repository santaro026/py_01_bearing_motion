"""
Created on Fri Apr 10 16:32:15 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path
import multiprocessing
import argparse
import os
import datetime

from mymods import mylogger
import config
import helper
import data_handler

def make_input_list(datadir, outdir, filepattern=r"tc*.txt"):
    temafiles = datadir.rglob(filepattern)
    markerfiles = []
    zerofiles = []
    for f in temafiles:
        if "zero" in f.name:
            zerofiles.append(f)
        else:
            markerfiles.append(f)
    print(f"zero files: {zerofiles}")
    print(f"target files: {len(markerfiles)}\n{'\n'.join(map(str, markerfiles))}")
    # zero = pl.read_csv(zerofiles[0], has_header=True, infer_schema_length=True).cast(pl.Float64, strict=False).to_numpy()[:8, 1:3]
    # print(f"zero: {zero.shape}")
    inputlist = []
    for mf in markerfiles:
        # markers = pl.read_csv(mf, has_header=True, infer_schema_length=True).cast(pl.Float64).to_numpy()[:, 1:]
        dl = data_handler.CoordDataLoader(mf, zerofiles[0], data_format="csv")
        inputlist.append([dl.cage_markers, dl.cage_markers_zero, dl.fps, outdir, dl.code])
        # print(f"{dl.cage_markers.shape}, {dl.cage_markers_zero.shape}, {dl.fps}")
    return inputlist

def test_run(txt):
    print("test_run\n==========")
    print(str(txt))

if __name__ == "__main__":
    print(f"---- run {__file__}----")
    multiprocessing.set_start_method("spawn") # for linux

    parser = argparse.ArgumentParser(description="run main process")
    parser.add_argument("--mode", type=str, choices=["test", "prod", "production"], default="test", help="execution mode")
    parser.add_argument("--tc", type=str, default="tc*", help="matching pattern for target data")
    args = parser.parse_args()

    runmode = args.mode
    testcode = args.tc

    date_str = datetime.datetime.today().strftime('%y%m%d')
    if runmode == 'test':
        outdir = config.ROOT / 'results' / f'{date_str}_cage_visualization_{config.VERSION}_{runmode}'
    elif runmode == 'prod' or runmode == 'production':
        outdir = config.ROOT / 'results' / f'{date_str}_cage_visualization_{config.VERSION}'
    outdir.mkdir(parents=True, exist_ok=True)
    logger_parallel = mylogger.MyLogger("logger_parallel", outdir=outdir)
    logger_parallel.measure_time("main", mode='s')


    # datadir = Path(r"N:\1005_tyn\tema\tc01_240610_40BNR10TYN_PA46GF25")
    datadir = Path(config.ROOT/"sampledata"/"SIMPLE50")
    inputlist = make_input_list(datadir, outdir=outdir, filepattern=r"*.csv")


    # tgtdirlist = list(datadir.iterdir())
    # _tgtdirlistbr = '\n'.join(map(str, tgtdirlist))
    # _len = len(tgtdirlist)
    # logger_parallel.binfo(f'workdirectories list: ({_len} datasets)\n{_tgtdirlistbr}')


    import main_computation
    ncpu = 8
    with multiprocessing.Pool(ncpu) as p:
        results = p.starmap(main_computation.analyze_cage, inputlist)


    logger_parallel.measure_time("main", mode='e')



