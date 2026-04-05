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
import data_handler

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
    camera: Camera
    audio: Audio

def set_summary(datamapfile):
    datamap_loader = data_handler.DataMapLoader(datamapfile)
    summary = datamap_loader.summary.to_dicts()
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

    datamappath = Path("D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    datamaploader = data_handler.DataMapLoader(datamappath)
    # print(datamap_loader.datamap.columns)

    tc = 2
    sc = 7


    with open(config.ROOT/"results"/"test.json", 'w', encoding="utf-8") as f:
        json.dump(datamaploader.summary.to_dicts(), f, indent=2)

    with open(config.ROOT/"results"/"test2.json", 'w', encoding="utf-8") as f:
        json.dump(datamaploader.extract_info_from_tcsc(tc, sc), f, indent=2)





