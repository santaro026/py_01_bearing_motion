"""
Created on Thu Mar 26 22:57:04 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path
import json
from dataclasses import dataclass, asdict
import config

@dataclass
class Test:
    tc: int
    size: str
    cage: str
    material: str
    detail: str
    PCD: float
    Dw: float
    Dp_measured: float
    Dl_measured: float
    dp_measured: float
    dl_measured: float
    Dp_drawing: float
    Dl_drawing: float
    dp_drawing: float
    dl_drawing: float
    noise_results: str

@dataclass
class Camera:
    fps: int
    num_frames: int
    duration: float
    shutter_speed: float
    pixel2mm: float
    Bring_area_pixel: float
    Bring_area_mm2: float
    Bring_center: list
    num_markers: int
    num_markers_cage: int
    num_markers_ring: int
    cage_circle_radius: float
    cage_ellipse_ab: list

@dataclass
class Audio:
    sample_rate: int
    num_samples: int
    duration: float

@dataclass
class Summary:
    test: Test
    camera: Camera
    audio: Audio

def set_summary(datamapfile):
    datamap_loader = DataMapLoader(datamapfile)
    summary = datamap_loader.summary.to_dicts()
    test = Test(
        tc=summary["test_code"],
        size=summary["size"],
        cage="TYN",
        material="PA46GF25",
        detail="this is sammple note",
        PCD=53.3,
        Dw=5.953,
        Dp_measured=None,
        Dl_measured=None,
        dp_measured=None,
        dl_measured=None,
        Dp_drawing=6.25,
        Dl_drawing=58.7,
        dp_drawing=6.25-5.953,
        dl_drawing=0.553,
        noise_results="x"
    )
    camera = Camera(
        fps=8000,
        num_frames=50000,
        duration=8000/50000,
        shutter_speed=1/500000,
        pixel2mm=(50/10000)**0.5,
        Bring_area_pixel=10000,
        Bring_area_mm2=50,
        Bring_center=[40, 40],
        num_markers=9,
        num_markers_cage=8,
        num_markers_ring=1,
        cage_circle_radius=25,
        cage_ellipse_ab=[26, 24]
    )
    audio = Audio(
        sample_rate=48000,
        num_samples=480000,
        duration=10
    )

if __name__ == "__main__":
    print("---- run ----")

    from data_handler import DataMapLoader

    datamapfile = Path("D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    datamap_loader = DataMapLoader(datamapfile)
    # print(datamap_loader.datamap.columns)
    test = datamap_loader.summary.to_dicts()

    outdir = config.ROOT / "results" / "test"
    # datamap_loader.summary.write_json(outdir/"test.json")
    # datamap_loader.summary.write_ndjson(outdir/"testnd.json")

    with open(outdir/"test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)


