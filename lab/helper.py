"""
Created on Fri Apr 10 19:31:39 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import re
from pathlib import Path


def merge_dicts(*args, check_consistency=True):
    d = {}
    for k, v in args[0].items():
        d[k] = v
    for _arg in args[1:]:
        for k, v in _arg.items():
            if k in d.keys() and d[k] != v:
                if check_consistency:
                    raise ValueError(
                        f"[WARNING]: duplicated key {k} was found with different values:\n"
                        f"{d[k]} and {v}"
                        )
            else:
                d[k] = v
    return d



def get_headers(num_markers=8):
    header_cage_kasa = [
        "cy", "cz", "radius", "Rx"
    ]
    header_cage_fitz = [
        "cy", "cz", "major", "minor", "theta", "Rx"
    ]
    header_markers = []
    for i in range(num_markers):
        header_markers.append(f"p{i}y")
        header_markers.append(f"p{i}z")
    headers = {
        "cage_kasa": header_cage_kasa,
        "cage_fitz": header_cage_fitz,
        "markers": header_markers,
    }
    return headers

def parse_filename(filename):
    tc_match = re.search(r"tc(\d+)", filename)
    sc_match = re.search(r"sc(\d+)", filename)
    fps_match = re.search(r"(\d+)fps", filename)
    rpm_match = re.search(r"(\d+)rpm", filename)
    rec_match = re.search(r"rec(\d+)", filename)
    tc = int(tc_match.group(1)) if tc_match else None
    sc = int(sc_match.group(1)) if sc_match else None
    fps = int(fps_match.group(1)) if fps_match else None
    rpm = int(rpm_match.group(1)) if rpm_match else None
    rec = int(rec_match.group(1)) if rec_match else None
    info = {
        "tc": tc,
        "sc": sc,
        "rec": rec,
        "fps": fps,
        "rpm": rpm,
    }
    return info

def get_unique_path(output_file: Path):
    output_file_new = output_file
    count = 1
    while output_file_new.exists():
        print(f'[DEBUG] There is already a file with the same name in the destination: {output_file}')
        output_file_new = output_file.parent / (output_file.stem + f'_{count}' + output_file.suffix)
        count += 1
    return output_file_new

def get_unique_path2(filepath, max_num=100):
    # curdir = Path.cwd()
    filepath = Path(filepath).resolve()
    targetdir = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix
    match = re.match(r'^(.*)_(\d+)$', stem) # basname_n.ext
    if match:
        base = match.group(1)
        n = int(match.group(2))
    else:
        base = stem
        n = 0
    same_suffix_files = [f.name for f  in targetdir.iterdir() if f.is_file() and f.suffix == suffix]
    new_name = filepath.name
    while new_name in same_suffix_files:
        n += 1
        new_name = f"{base}_{n}{suffix}"
        if (n >= max_num):
            raise ValueError('too many files with the same name exist. change output file name, or change max permitted number.')
    return targetdir / new_name


def trange2frange(trange, fps):
    if not isinstance(trange, (list, np.ndarray)):
        raise ValueError(f"trange must be list or ndarray: you passed {type(trange)}")
    if np.ndim(np.asarray(trange)) != 1:
        raise ValueError(f"trange must be 1-d list or ndarray: you passed {np.ndim(np.asarray(trange))}")
    st, et = trange
    sf, ef = int(st*fps), int(et * fps)
    return [sf, ef]

def tranges2franges(tranges, fps):
    franges = []
    for trange in tranges:
        frange = trange2frange(trange, fps)
        franges.append(frange)
    return franges

def frange2trange(frange, fps):
    if not isinstance(frange, (list, np.ndarray)):
        raise ValueError(f"frange must be list or ndarray: you passed {type(frange)}")
    if np.ndim(np.asarray(frange)) != 1:
        raise ValueError(f"frange must be 1-d list or ndarray: you passed {np.ndim(np.asarray(frange))}")
    sf, ef = frange
    st, et = sf/fps, ef/fps
    return [st, et]

def franges2tranges(franges, fps):
    tranges = []
    for frange in franges:
        trange = frange2trange(frange, fps)
        tranges.append(trange)
    return tranges

def extract_frames_forNrot(cage_rotspeed, fps, num_rot=3, start_time=0):
    period = 2 * np.pi / cage_rotspeed
    nperiod = period * num_rot
    num_frames = int(nperiod * fps)
    sf = int(start_time * fps)
    ef = sf + num_frames
    frange = [sf, ef]
    return frange





def load_summary(tgtdir):
    df = pl.read_csv(tgtdir/'summary.csv', has_header=True, infer_schema_length=50000)
    return df

def load_csv(tgtdir, tgtfname):
    df = pl.read_csv(tgtfname, has_header=True, infer_schema_length=5000)
    return df

def load_pklfig(tgtfile):
    with open(tgtfile, 'rb') as f:
        fig = dill.load(f)
    axs = [fig.axes]
    print(axs)
    return fig, axs

def load_pklcsv(tgtfile):
    with open(tgtfile, 'rb') as f:
        data = dill.load(f)
    print(f'{tgtfile.name}: {data.shape}')
    return data



if __name__ == "__main__":
    print("---- run ----")

    p = Path(r"D:/200_python/01_cage_visualization/data/test")
    print(p.exists())



