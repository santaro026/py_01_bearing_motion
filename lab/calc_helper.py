
"""
Created on Wed Aug 27 08:42:28 2025
@author: honda-shin

this module provides some auxiliary calculations to improve readability and consistency

"""
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from scipy import signal

import re
from pathlib import Path
import json

from sintamods import mytools
from sintamods import myfitting
from sintamods import mydataclass
from sintamods import myplot

import config

class TimeSeriesDataProcessor():
    @staticmethod
    def extract_runs(mask):
        runs = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        if len(runs) == 0:
            runs = None
        return runs
    def __init__(self, num_frames=10000, fps=10000):
        self.num_frames = num_frames
        self.fps = fps
        self.duration = self.fps * (self.num_frames - 1)
        self.t = np.linspace(0, self.duration, self.num_frames)

    def convert_trange2frange(self, time_range=None):
        if time_range is None:
                frame_range = None
                frame_id = None
        else:
            st = time_range[0]
            et = time_range[1]
            sf = st * self.fps
            ef = et * self.fps
            sf = int(sf + 1) if sf - int(sf) != 0 else int(sf)
            ef = int(ef + 1) if ef - int(ef) != 0 else int(ef)
            frame_range = [sf, ef]
            frame_id = np.arange(sf, ef).astype(np.int64)
        return frame_range, frame_id

    def convert_tranges2franges(self, time_ranges=None):
        if time_ranges is None:
            frame_ranges = None
            frame_ids = None
        else:
            if isinstance(time_ranges, np.ndarray): time_ranges = time_ranges.tolist()
            if isinstance(time_ranges[0], (int, float)): time_ranges = [time_ranges]
            frame_ranges, frame_ids = [], []
            for time_range in time_ranges:
                st = time_range[0]
                et = time_range[1]
                sf = st * self.fps
                ef = et * self.fps
                sf = int(sf + 1) if sf - int(sf) != 0 else int(sf)
                ef = int(ef + 1) if ef - int(ef) != 0 else int(ef)
                frame_range, frame_id = self.convert_trange2frange(time_range=time_range)
                frame_ranges.append(frame_range)
                frame_ids.append(frame_id)
        return frame_ranges, frame_ids

    def merge_ranges(self, time_ranges1, time_ranges2, tmax=None):
        if isinstance(time_ranges1, np.ndarray): time_ranges1 = time_ranges1.tolist()
        if isinstance(time_ranges2, np.ndarray): time_ranges2 = time_ranges2.tolist()
        if time_ranges1 is None and time_ranges2 is None:
            merged = None
        elif time_ranges1 is None and time_ranges2 is not None:
            merged = time_ranges2
        elif time_ranges1 is not None and time_ranges2 is None:
            merged = time_ranges1
        else:
            # if not any(isinstance(r, list) for r in time_ranges1):
            #     time_ranges1 = [time_ranges1]
            # if not any(isinstance(r, list) for r in time_ranges2):
            #     time_ranges2 = [time_ranges2]
            if isinstance(time_ranges1[0], (int, float)): time_ranges1 = [time_ranges1]
            if isinstance(time_ranges2[1], (int, float)): time_ranges2 = [time_ranges2]
            all_ranges = sorted(time_ranges1 + time_ranges2, key=lambda x: x[0])
            merged = []
            for current in all_ranges:
                if not merged or merged[-1][1] < current[0]:
                    merged.append(current)
                else:
                    merged[-1][1] = max(merged[-1][1], current[1])
        if tmax is None:
            return merged
        elif tmax is not None and merged is not None:
            merged_filtered = []
            if not any(isinstance(_m, list) for _m in merged):
                merged = [merged]
            for _m in merged:
                st, et = _m
                if st < tmax and et < tmax:
                    merged_filtered.append([st, et])
                elif st < tmax and tmax < et:
                    merged_filtered.append([st, tmax])
                elif tmax < st:
                    pass
            if len(merged_filtered) == 0:
                merged_filtered = None
        return merged_filtered

    def get_mask(self, time_ranges):
        if isinstance(time_ranges, np.ndarray): time_ranges = time_ranges.tolist()
        if time_ranges == None:
            return [False] * self.num_frames
        mask = [False] * self.num_frames
        for _s, _e in time_ranges:
            _m = (_s < self.t) & (self.t < _e)
            mask = mask | _m
        return mask

    def calc_rms(self, sound, window_time=0.1):
        num_perwin = int(window_time * self.fps)
        _kernel = np.ones(num_perwin) / num_perwin
        sound_pw = signal.fftconvolve(sound**2, _kernel, mode='same')
        sound_pw[:num_perwin//2] = np.nan
        sound_pw[-num_perwin//2:] = np.nan
        sound_rms = np.sqrt(sound_pw)
        return sound_rms

    def detect_noise(self, sound, window_time=0.1, threshold_factor=0.1, threshold=None):
        sound_rms = self.calc_rms(self, sound=sound, window_time=window_time)
        sound_rms_range = (np.nanmin(sound_rms), np.nanmax(sound_rms))
        threshold = sound_rms_range[0] * threshold_factor if threshold is None else threshold
        mask = sound_rms > threshold
        okruns = mytools.extract_runs(mask) # get runs [start, end), with end not included.
        okruns_id = np.where(mask)[0]
        ngruns = mytools.extract_runs(~mask)
        ngruns_id = np.where(~mask)[0]
        return okruns, okruns_id, ngruns, ngruns_id, threshold


if __name__ == '__main__':
    print('---- test ----\n')

    processor = TimeSeriesDataProcessor(num_frames=20, fps=1)

    time_ranges = np.array([0, 3])
    time_ranges = [[0.2, 21.1], [32.2, 35]]

    frame_ranges, frame_ids = processor.convert_tranges2franges(time_ranges=time_ranges)

    print(frame_ranges)
    print(frame_ids)

    time_ranges1 = [[0, 1], [3.2, 3.8], [8, 9]]
    time_ranges2 = [[0.6, 2], [4.0, 6]]
    # time_ranges2 = None

    merged_filter = processor.merge_ranges(time_ranges1, time_ranges2)
    print(merged_filter)

    time_ranges = [[2, 6], [18, 20.1]]
    time_mask = processor.get_mask(time_ranges=time_ranges)
    print(f'mask: {time_mask}')

    runs = TimeSeriesDataProcessor.extract_runs(time_mask)
    print(f'runs: {runs}')



