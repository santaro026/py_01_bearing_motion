"""
Created on Fri Mar 06 18:39:00 2026
@author: santaro

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

import config

eps = 1e-12
p0 = 20e-6

class TimeSeriesProcessor():
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

    def __init__(self, sound=None, num_frames=None, fs=48000):
        if sound is None and num_frames is None:
            raise ValueError(f"sound or num_frames must be defined, both is None")
        self.fs = fs
        self.sound = sound
        self.num_frames = num_frames if sound is None else len(sound)
        self.duration = self.fs * (self.num_frames - 1)
        self.t = np.linspace(0, self.duration, self.num_frames)

    def _time2frame(self, time_range):
        st = time_range[0]
        et = time_range[1]
        sf = st * self.fs
        ef = et * self.fs
        sf = int(sf + 1) if sf - int(sf) != 0 else int(sf)
        ef = int(ef + 1) if ef - int(ef) != 0 else int(ef)
        frame_range = [sf, ef]
        frame_id = np.arange(sf, ef).astype(np.int64)
        return frame_range, frame_id
    def time2frame(self, time_ranges):
        if isinstance(time_ranges, np.ndarray): time_ranges = time_ranges.tolist()
        if isinstance(time_ranges[0], (int, float)): time_ranges = [time_ranges]
        frame_ranges, frame_ids = [], []
        for time_range in time_ranges:
            st = time_range[0]
            et = time_range[1]
            sf = st * self.fs
            ef = et * self.fs
            sf = int(sf + 1) if sf - int(sf) != 0 else int(sf)
            ef = int(ef + 1) if ef - int(ef) != 0 else int(ef)
            frame_range, frame_id = self._time2frame(time_range=time_range)
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
            if isinstance(time_ranges1[0], (int, float)): time_ranges1 = [time_ranges1]
            if isinstance(time_ranges2[0], (int, float)): time_ranges2 = [time_ranges2]
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
            _m = (_s <= self.t) & (self.t <= _e)
            mask = mask | _m
        return mask

    def calc_rms(self, sound=None, window_time=0.1, edge="nan"):
        if sound is None: sound = self.sound
        num_perwin = int(window_time * self.fs)
        _kernel = np.ones(num_perwin) / num_perwin
        sound_pw = signal.fftconvolve(sound**2, _kernel, mode='same')
        if edge == "cut":
            sound_pw[:num_perwin//2] = np.nan
            sound_pw[-num_perwin//2:] = np.nan
        elif edge == "pad":
            sound_pw[:num_perwin//2] = sound_pw[num_perwin//2]
            sound_pw[-num_perwin//2:] = sound_pw[-num_perwin//2]
        sound_rms = np.sqrt(sound_pw)
        return sound_rms

    @staticmethod
    def pw2db(signal, x0=1):
        db = 10 * np.log10((signal + eps) / x0**2)
        return db
    @staticmethod
    def mag2db(signal, x0=1):
        db = 20 * np.log10((signal + eps) / x0)
        return db
    @staticmethod
    def pa2db(rms, p0=20e-6):
        db = 20 * np.log10((rms + eps) / p0)
        return db

    def detect_noise_rms(self, sound=None, window_time=0.01, threshold_factor=0.2, threshold=None, mode="log"):
        if sound is None: sound = self.sound
        sound_rms = self.calc_rms(sound=sound, window_time=window_time, edge="pad")
        print(f"sound_rms: {sound_rms.shape}")
        if mode == "log":
            sound_rms = TimeSeriesProcessor.pa2db(sound_rms)
        sound_rms_range = (np.nanmin(sound_rms), np.nanmax(sound_rms))
        change = sound_rms_range[1] - sound_rms_range[0]
        threshold = sound_rms_range[0] + change * threshold_factor if threshold is None else threshold
        mask = sound_rms > threshold
        noisy_runs = TimeSeriesProcessor.extract_runs(mask) # get runs [start, end], with end not included.
        noisy_id = np.where(mask)[0]
        silent_rns = TimeSeriesProcessor.extract_runs(~mask) # get runs that is not OK
        silent_id = np.where(~mask)[0]
        result = {
            "noisy_runs": noisy_runs,
            "silent_runs": silent_rns,
            "noisy_id": noisy_id,
            "silent_id": silent_id,
            "rms": sound_rms,
            "threshold": threshold,
        }
        return result






if __name__ == '__main__':
    print('---- test ----\n')

    processor = TimeSeriesProcessor(num_frames=20, fs=1)

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

    runs = TimeSeriesProcessor.extract_runs(time_mask)
    print(f'runs: {runs}')



