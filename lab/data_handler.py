"""
Created on Sat Nov 29 13:14:50 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from scipy import signal
from pydub import AudioSegment

import re
from pathlib import Path
from dataclasses import dataclass, field

from mymods import myfitting

import config

def csvdata2npdata(csvdata):
    csvdata = np.asarray(csvdata)
    num_frames = len(csvdata)
    npdata = csvdata.reshape(num_frames, -1, 2)
    return npdata

def npdata2csvdata(npdata):
    num_frames, num_points, dimension = npdata.shape
    csvdata = npdata.reshape(num_frames, int(num_points * dimension))
    return csvdata

class DataMapLoader:
    @staticmethod
    def load_datamap(datamapfile):
        df = pl.read_excel(datamapfile, sheet_name="all", has_header=True, drop_empty_cols=True, drop_empty_rows=True, infer_schema_length=1000)
        datamap = df.select(
            pl.col("test_code").cast(pl.Int32, strict=False),
            pl.col("shooting_code").cast(pl.Int32, strict=False),
            pl.col("date").cast(pl.String, strict=False),
            pl.col("cage").cast(pl.String, strict=False),
            pl.col("material").cast(pl.String, strict=False),
            pl.col("commanded_rot_speed").cast(pl.Int32, strict=False),
            pl.col("fps").cast(pl.Int32, strict=False),
            pl.col("recording_number").cast(pl.Int32, strict=False),
            pl.col("sample_rate").cast(pl.Int32, strict=False),
            #### sound flag based on auditory input
            pl.col("CG").cast(pl.String, strict=False),
            pl.col("CGL").cast(pl.String, strict=False),
            pl.col("CGW").cast(pl.String, strict=False),
            pl.col("CGR").cast(pl.String, strict=False),
            pl.col("CGQ").cast(pl.String, strict=False),
            pl.col("CGQ2").cast(pl.String, strict=False),
            pl.col("CGT").cast(pl.String, strict=False),
            pl.col("Q").cast(pl.String, strict=False),
            pl.col("CK").cast(pl.String, strict=False),
            #### motihon flag based on high-speed camera data, plot-visual-based qualitative assessment
            pl.col("high_speed_whirl").cast(pl.String, strict=False),
            pl.col("cage_resonance").cast(pl.String, strict=False),
            pl.col("gravity_vibration").cast(pl.String, strict=False),
        ).drop_nulls(subset=["test_code"])
        df_summary = pl.read_excel(datamapfile, sheet_name="summary", has_header=True, drop_empty_cols=True, drop_empty_rows=True, infer_schema_length=1000)
        summary = df_summary.select(
            pl.col("test_code").cast(pl.Int32, strict=False),
            pl.col("date").cast(pl.String, strict=False),
            pl.col("bearing").cast(pl.String, strict=False),
            pl.col("cage").cast(pl.String, strict=False),
            pl.col("camera").cast(pl.String, strict=False),
            pl.col("laser_doppler").cast(pl.String, strict=False),
            pl.col("sound").cast(pl.String, strict=False),
            pl.col("material").cast(pl.String, strict=False),
            pl.col("detail").cast(pl.String, strict=False),
            pl.col("PCD").cast(pl.Float64, strict=False),
            pl.col("Dw").cast(pl.Float64, strict=False),
            pl.col("Dp_measured").cast(pl.Float64, strict=False),
            pl.col("Dl_measured").cast(pl.Float64, strict=False),
            pl.col("dp_measured").cast(pl.Float64, strict=False),
            pl.col("dl_measured").cast(pl.Float64, strict=False),
            pl.col("Dp_drawing").cast(pl.Float64, strict=False),
            pl.col("Dl_drawing").cast(pl.Float64, strict=False),
            pl.col("dp_drawing").cast(pl.Float64, strict=False),
            pl.col("dl_drawing").cast(pl.Float64, strict=False),
            pl.col("noise_result").cast(pl.String, strict=False),
        ).drop_nulls(subset=["test_code", "date", "camera"]).filter(pl.col("laser_doppler").is_null())
        return datamap, summary
    def __init__(self, datamappath):
        self._path = datamappath
        self._datamap, self._summary = DataMapLoader.load_datamap(self._path)
    @property
    def path(self):
        return self._path
    @property
    def datamap(self):
        return self._datamap
    @property
    def summary(self):
        return self._summary

    def extract_info_from_tcsc(self, tc, sc):
        info = self.datamap.filter((pl.col("test_code") == tc) & (pl.col("shooting_code") == sc)).to_dicts()
        if len(info) == 0:
            raise ValueError(f"no data was found in datamap by tc{tc}-sc{sc}.")
        elif len(info) > 1:
            raise RuntimeError(f"multiple data ({len(info)} was found by tc{tc}-sc{sc}.")
        return info[0]

    def extract_info_from_rec(self, rec):
        info = self.datamap.filter(pl.col("recording_number") == rec).to_dicts()
        if len(info) == 0:
            raise ValueError(f"no data was found in datamap by rec{rec}.")
        elif len(info) > 1:
            raise RuntimeError(f"multiple data ({len(info)} was found by rec{rec}.")
        return info[0]

    def extract_testinfo(self, tc):
        info = self.summary.filter((pl.col("test_code")) == tc).to_dicts()
        if len(info) == 0:
            raise ValueError(f"no data was found in datamap by tc{tc}.")
        elif len(info) > 1:
            raise RuntimeError(f"multiple data ({len(info)} was found by tc{tc}.")
        return info[0]

    def __repr__(self):
        return (
            f"DataMapLoader:\n"
            f"path: {self.path}\n"
            f"num_data: {self.datamap.shape[0]}\n"
            f"num_test: {self.summary.shape[0]}\n"
        )

class CoordDataLoader:
    def __init__(self, data_path, zero_data_path, data_format="tema", num_cage_markers=8, zero_data_format="tema", pixel2mm_reference_mode="area", reference_value=np.pi*(49.1/2)**2, dimension=2):
        self._data_path = data_path
        self._zero_data_path = zero_data_path
        filenameinfo = CoordDataLoader.parse_filename(self._data_path.name)
        self._tc = filenameinfo["tc"]
        self._sc = filenameinfo["sc"]
        self._rec = filenameinfo["rec"]
        self._fps = filenameinfo["fps"]
        self._rpm = filenameinfo["rpm"]
        self._dimension = dimension
        self._num_cage_markers = num_cage_markers
        self.t_data, self.cage_markers_pixel, self.ring_markers_pixel = CoordDataLoader.load_markers(data_path=self._data_path, data_format=data_format, num_cage_markers=self._num_cage_markers, dimension=self._dimension)
        self._num_frames = len(self.t_data)
        self.cage_markers_zero_pixel, self.ring_center_zero_pixel, self.ring_area_zero_pixel = CoordDataLoader.load_markers_zero(data_path=self._zero_data_path, data_format=zero_data_format, num_cage_markers=self._num_cage_markers, dimension=self._dimension)
        self._pixel2mm_reference_mode = pixel2mm_reference_mode
        self._reference_value = reference_value
        self.pixel2mm = CoordDataLoader.calc_scaling_factor_pixel2mm(measured_value=self.ring_area_zero_pixel, reference_value=self._reference_value, reference_mode=self._pixel2mm_reference_mode)
        self.t = np.arange(self._num_frames) / self._fps
        self.ring_center_zero = (self.pixel2mm * self.ring_center_zero_pixel)
        self.cage_markers = self.pixel2mm * self.cage_markers_pixel - self.ring_center_zero
        self.cage_markers_zero = (self.pixel2mm * self.cage_markers_zero_pixel - self.ring_center_zero)
        self.ring_markers = self.pixel2mm * self.ring_markers_pixel - self.ring_center_zero if self.ring_markers_pixel is not None else None
        self._duration = float(self.t[-1] - self.t[0])
    @property
    def data_path(self):
        return self._data_path
    @property
    def zero_data_path(self):
        return self._zero_data_path
    @property
    def tc(self):
        return self._tc
    @property
    def sc(self):
        return self._sc
    @property
    def rpm(self):
        return self._rpm
    @property
    def fps(self):
        return self._fps
    @property
    def num_frames(self):
        return self._num_frames
    @property
    def duration(self):
        return self._duration
    @property
    def num_cage_markers(self):
        return self._num_cage_markers
    @property
    def dimension(self):
        return self._dimension
    @property
    def rec(self):
        return self._rec
    @property
    def pixel2mm_reference_mode(self):
        return self._pixel2mm_reference_mode
    @property
    def reference_value(self):
        return self._reference_value

    @staticmethod
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

    @staticmethod
    def load_markers(data_path=None, data_format="tema", num_cage_markers=8, dimension=2):
        if data_format == "tema":
            skip_rows = 3
            skip_columns = 0
            separator = '\t'
        data = pl.read_csv(data_path, has_header=False, skip_rows=skip_rows, separator=separator, infer_schema_length=50000).cast(pl.Float64, strict=False).to_numpy()[:, skip_columns:]
        t = data[:, 0]
        if t[0] != 0:
            raise ValueError(f"loaded data does not start from 0 [sec], {data_path}, start form {t[0]}")
        points = data[:, 1:]
        if points.shape[1] % dimension == 0: # check data shape
            num_points = points.shape[1] // dimension
            num_ring_markers = num_points - num_cage_markers
            if not (num_points == num_cage_markers or num_points == num_cage_markers + 1):
                raise RuntimeError(f"loading data shape does not match: {data_path}, num_points is {num_points}")
        else:
            raise RuntimeError(f"the number of coordinate data points is odd, confirm the input data: {data_path}")
        num_frames = len(t)
        points = points.reshape((num_frames, num_points, dimension))
        cage_markers = points[:, :num_cage_markers]
        if num_ring_markers > 0:
            ring_markers = points[:, num_cage_markers:num_cage_markers+num_ring_markers]
        else:
            ring_markers = None
        return t, cage_markers, ring_markers

    @staticmethod
    def load_markers_zero(data_path, data_format="tema", num_cage_markers=8, dimension=2):
        if data_format == "tema":
            data = pl.read_csv(data_path, has_header=False, skip_rows=3, separator='\t', infer_schema_length=10).cast(pl.Float64, strict=False).to_numpy()[:, 1:]
            cage_markers = data[:-1, :dimension].astype(float)[np.newaxis, :, :]
            ring_center = data[-1, :dimension].astype(float)[np.newaxis, np.newaxis, :]
            ring_area = data[-1, dimension].astype(float) # float
            if cage_markers.shape[1] != num_cage_markers:
                raise RuntimeError(f"loading data shape does not match: {data_path}")
        return cage_markers, ring_center, ring_area

    @staticmethod
    def calc_scaling_factor_pixel2mm(measured_value=1, reference_value=1, reference_mode="area"):
        if reference_mode == "area":
            scaling_factor_pixel2mm = np.sqrt(reference_value/measured_value)
        return scaling_factor_pixel2mm

    def __repr__(self):
        ring_center_zero_preview = np.array2string(self.ring_center_zero.squeeze(), precision=9, separator=", ")
        ring_center_zero_pixel_preview = np.array2string(self.ring_center_zero_pixel, precision=9, separator=", ")
        return (
            f"data_path: {self.data_path}\n"
            f"zero_data_path: {self.zero_data_path}\n"
            f"tc: {self.tc}, sc: {self.sc}\n"
            f"rpm: {self.rpm}, rec: {self.rec}\n"
            f"fps: {self.fps} [frame/sec], duration: {self.duration} [sec], num_frames: {self.num_frames}, dimension: {self.dimension}\n"
            f"biring area: {self.ring_area_zero_pixel} [pixel] ({self.reference_value} [mm**2]), pexel2mm: {self.pixel2mm}\n"
            f"ring_center: {ring_center_zero_preview} [mm] ({ring_center_zero_pixel_preview} [pixel])\n"
            f"cage_markers: {self.cage_markers.shape}, ring_markers: {self.ring_markers.shape}"
        )

@dataclass
class CoordSeries:
    loader: CoordDataLoader
    datamap: DataMapLoader | None = None
    def __post_init__(self):
        if self.datamap is not None:
            info = self.datamap.extract_info_from_tcsc(self.loader.tc, self.loader.sc)
            if self.loader.rpm != info["commanded_rot_speed"]:
                raise ValueError(f"data condition of rpm does not match.\nfilename info: {self.loader.rpm}, datamap info: {info["commanded_rot_speed"]}")
            if self.loader.fps !=  info["fps"]:
                raise ValueError(f"data condition of fps does not match.\nfilename info: {self.loader.fps}, datamap info: {info["fps"]}")
            if self.loader.rec != info["recording_number"]:
                raise ValueError(f"data condition of rec does not match.\nfilename info: {self.loader.rec}, datamap info: {info["recording_number"]}")

    @property
    def meta(self) -> dict[str, int | float]:
        return {
            "tc": self.loader.tc,
            "sc": self.loader.sc,
            "fps": self.loader.fps,
            "duration": self.loader.duration,
            "num_frames": self.loader.num_frames,
            "num_cage_markers": self.loader.num_cage_markers,
            "pixel2mm": self.loader.pixel2mm
        }
    @property
    def t(self) -> np.ndarray:
        return self.loader.t
    @property
    def cage_markers_zero(self) -> np.ndarray:
        if self.loader.dimension == 2:
            x = np.zeros(1)[:, np.newaxis, np.newaxis]
            xs = np.broadcast_to(x, (1, self.loader.num_cage_markers, 1))
            cage_markers_zero = np.concatenate([xs, self.loader.cage_markers_zero], axis=-1)
        else:
            cage_markers_zero = self.loader.cage_markers_zero
        return cage_markers_zero
    @property
    def cage_markers(self) -> np.ndarray:
        if self.loader.dimension == 2:
            x = np.zeros(self.loader.num_frames)[:, np.newaxis, np.newaxis]
            xs = np.broadcast_to(x, (self.loader.num_frames, self.loader.num_cage_markers, 1))
            cage_markers = np.concatenate([xs, self.loader.cage_markers], axis=-1)
        else:
            cage_markers = self.loader.cage_markers
        return cage_markers
    @property
    def ring_markers(self) -> np.ndarray:
        if self.loader.dimension == 2:
            x = np.zeros(self.loader.num_frames)[:, np.newaxis, np.newaxis]
            ring_markers = np.concatenate([x, self.loader.ring_markers], axis=-1)
        else:
            ring_markers = self.loader.ring_markers
        return ring_markers
    @property
    def ring_center_zero(self) -> np.ndarray:
        if self.loader.dimension == 2:
            x = np.zeros(1)[:, np.newaxis, np.newaxis]
            ring_center_zero = np.concatenate([x, self.loader.ring_center_zero], axis=-1)
        else:
            ring_center_zero = self.loader.ring_center_zero
        return ring_center_zero

    def __repr__(self):
        return f"CoordLoader:\n{repr(self.loader)}"

class AudioDataLoader:
    @staticmethod
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

    @staticmethod
    def identify_data_channel(name, kind, unit):
        print(f"name, kind, unit: {name}, {kind}, {unit}")
        patterns = {
        "sound" : {
            "name": ["rec", "mic", "teds"],
            "kind": ["sound pressure"],
            "unit": ["pa"]
            },
        "trigger" : {
            "name": ["trigger", "rearright"],
            "kind": ["voltage"],
            "unit": ["v"]
            },
        "velocity" : {
            "name": ["velocity"],
            "kind": ["sound velocity"],
            "unit": ["m/s"]
            },
        "displacement" : {
            "name": ["displacement", "displaement"],
            "kind": ["displacement"],
            "unit": ["m"]
            }
        }
        for k, v in patterns.items():
            if name.lower() in v["name"]: _name = k
            if kind.lower() in v["kind"]: _kind = k
            if unit.lower() in v["unit"]: _unit = k
        if _name == _kind == _unit:
            data_type = _name
        else:
            raise ValueError(f"data type doesnt match: (name, kind, unit) = ({_name}, {_kind}, {_unit})")
        return data_type

    @staticmethod
    def load_sound(data_path=None):
        data_format = data_path.suffix
        if data_format == ".mat":
            matdata = scipy.io.loadmat(data_path)
            struct_array = matdata["shdf"]
            void_entry = struct_array[0][0]
            fields = void_entry.dtype.names
            # print(fields)
            #### id check
            chns = void_entry["Chn"][0]
            chn_names = []
            chn_kinds = []
            chn_units = []
            data_types = []
            for _meta in chns:
                _name = _meta[4][0]
                _kind = _meta[6][0]
                _unit = _meta[7][0]
                chn_names.append(str(_name))
                chn_kinds.append(str(_kind))
                chn_units.append(str(_unit))
                data_type = AudioDataLoader.identify_data_channel(_name, _kind, _unit)
                data_types.append(data_type)
            # print(f"names: {chn_names}")
            # print(f"kinds: {chn_kinds}")
            # print(f"units: {chn_units}")
            # print(f"data_types: {data_types}")
            id_sound = data_types.index("sound")
            t = void_entry["Absc1Data"][0]
            sound = void_entry["Data"][id_sound]
            if t[0] != 0:
                raise ValueError(f"input audio data has problem, time data does not start form 0, start from {t[0]}.")
        elif data_format == ".csv":
            data_types = []
            data_start_id = None
            data_info = {}
            data_line_pattern = re.compile(r"^\s*[-+]?\d+(\.\d+)?(e[-+]?\d+)?\s*,", re.IGNORECASE) # first element is numeric then comma
            with data_path.open('r', encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if data_line_pattern.match(stripped):
                    data_start_id = i
                    break
                if ':' in stripped:
                    key, value = stripped.split(':', 1)
                    data_info[key.strip()] = [v for v in value.strip().split(',')][:-1] # cut '' data due to format "name,kind,unit," (last comma)
            channels = [v for v in data_info.keys() if "Channel" in v]
            for ch in channels:
                _d = data_info[ch]
                data_type = AudioDataLoader.identify_data_channel(_d[0], _d[1], _d[2])
                data_types.append(data_type)
            id_sound = data_types.index("sound") + 2
            # print(f"data_types: {data_types}")
            # print(f"id_sound: {id_sound}")
            # df = pl.read_csv(data_path, has_header=False, skip_rows=data_start_id, separator=',', infer_schema_length=1000).cast(pl.Float64, strict=False)
            df = pl.scan_csv(data_path, has_header=False, skip_rows=data_start_id, separator=',', infer_schema_length=1000).select([pl.col("column_1"), pl.col(f"column_{id_sound}")]).cast(pl.Float64, strict=False).collect()
            data = df.to_numpy().astype(float)
            t = data[:, 0]
            sound = data[:, 1]
            if t[0] != 0:
                raise ValueError(f"input audio data has problem, time data does not start form 0, start from {t[0]}.")
        elif data_format.lower() == ".wav":
            audio = AudioSegment.from_file(data_path)
            channels = audio.channels
            sample_rate = audio.frame_rate
            duration = audio.duration_seconds
            sample_width = audio.sample_width
            bit_depth = sample_width * 8
            num_frames = int(audio.frame_count())
            t = np.arange(0, duration, 1/sample_rate)
            sound0 = AudioDataLoader.normalize_sound(np.array(audio.get_array_of_samples())[0::channels], bit_depth)
            sound1 = AudioDataLoader.normalize_sound(np.array(audio.get_array_of_samples())[1::channels], bit_depth)
            trigger_idx, sound_idx = AudioDataLoader.classify_sound_trigger_channel(sound0[:1000], sound1[:1000])
            # print(f"sound_idx: {sound_idx}")
            sound = [sound0, sound1][sound_idx]
            return t, sound
        return t, sound
    @staticmethod
    def normalize_sound(sound, bit_depth):
        return sound.astype(np.float64) / float(2**(bit_depth-1))
    @staticmethod
    def classify_sound_trigger_channel(d0, d1):
        # slope0 = np.max(np.abs(np.diff(d0)))
        # slope1 = np.max(np.abs(np.diff(d1)))
        d0 = d0 - np.nanmean(d0)
        d1 = d1 - np.nanmean(d1)
        rms0 = np.sqrt(np.mean(d0**2))
        rms1 = np.sqrt(np.mean(d1**2))
        # print(f"rms0, rms1: {rms0}, {rms1}")
        #### trriger should be large slope, low RMS
        # trigger_idx = 0 if (slope0 > slope1 and rms0 < rms1) else 1
        trigger_idx = 0 if rms0 < rms1 else 1
        sound_idx = 1 - trigger_idx
        return trigger_idx, sound_idx
    def __init__(self, data_path):
        self._data_path = data_path
        filenameinfo = CoordDataLoader.parse_filename(self._data_path.name)
        self._tc = filenameinfo["tc"]
        self._sc = filenameinfo["sc"]
        self._rec = filenameinfo["rec"]
        self._t, self._sound = AudioDataLoader.load_sound(self._data_path)
        self._num_samples = len(self._t)
        self._duration = float(self._t[-1] - self._t[0])
        self._sample_rate = float(1 / (self._t[1] - self._t[0]))
    @property
    def data_path(self):
        return self._data_path
    @property
    def tc(self):
        return self._tc
    @tc.setter
    def tc(self, value):
        if self._tc is None:
            self._tc = value
        else: raise AttributeError("tc has already certain value, you cannot rewrite tc.")
    @property
    def sc(self):
        return self._sc
    @sc.setter
    def sc(self, value):
        if self._sc is None:
            self._sc = value
        else: raise AttributeError("sc has already certain value, you cannot rewrite sc.")
    @property
    def rec(self):
        return self._rec
    @property
    def sample_rate(self):
        return self._sample_rate
    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value
    @property
    def num_samples(self):
        return self._num_samples
    @property
    def duration(self):
        return self._duration
    @property
    def t(self):
        return self._t
    @property
    def sound(self):
        return self._sound
    def __repr__(self):
        t_preview = np.array2string(self.t[:5], precision=10, separator=", ")
        sound_preview = np.array2string(self.sound[:5], precision=10, separator=", ")
        return (
            f"data_path: {self.data_path}\n"
            f"tc: {self.tc}, sc: {self.sc}, rec: {self.rec}\n"
            f"sample_rate: {self.sample_rate} [Hz], duration: {self.duration} [sec], num_samples: {self.num_samples}\n"
            f"t: {self.t.shape}, {t_preview}\n"
            f"sound: {self.sound.shape}, {sound_preview}"
        )

@dataclass
class AudioSeries:
    loader: AudioDataLoader
    datamap: DataMapLoader | None = None
    def __post_init__(self):
        if self.datamap is not None:
            info = self.datamap.extract_info_from_rec(self.loader.rec)
            if self.loader.tc is None:
                self.loader.tc = info["test_code"]
            if self.loader.sc is None:
                self.loader.sc = info["shooting_code"]
            sample_rate_accuracy = abs(1 - self.loader.sample_rate/info["sample_rate"])
            if sample_rate_accuracy >= 0.01:
                raise ValueError(f"data condition of sample_rate does not match.\ndatamap info: {info["sample_rate"]}, actual data: {self.loader.sample_rate}")

    @property
    def meta(self) -> dict[str, int | float]:
        return {
            "tc": self.loader.tc,
            "sc": self.loader.sc,
            "rec": self.loader.rec,
            "sample_rate": self.loader.sample_rate,
            "duration": self.loader.duration,
            "num_samples": self.loader.num_samples
        }
    @property
    def t(self) -> np.ndarray:
        return self.loader.t
    @property
    def sound(self) -> np.ndarray:
        return self.loader.sound

    def __repr__(self):
        return f"AudioLoader:\n{repr(self.loader)}"

@dataclass
class Series:
    _coord: CoordSeries | None = None
    _audio: AudioSeries | None = None
    # datamap_loader: DataMapLoader | None = None
    aligned_reference: str | None = None
    cache: dict[str, object] = field(default_factory=dict)
    _tc: int | None = None
    _sc: int | None = None
    def __post_init__(self):
        if self._coord: self.tc = self.coord.meta["tc"]
        if self._audio: self.sc = self.coord.meta["sc"]

    def __repr__(self):
        coord_repr = None
        audio_repr = None
        if self.coord:
            coord_repr = repr(self.coord)
        if self.audio:
            audio_repr = repr(self.audio)
        return (
            f"tc: {self.tc}, sc: {self.sc}\ncoord:\n{coord_repr}\naudio:\n{audio_repr}"
        )

    @property
    def tc(self):
        return self._tc
    @tc.setter
    def tc(self, value):
        self._tc = value
    @property
    def sc(self):
        return self._sc
    @sc.setter
    def sc(self, value):
        self._sc = value
    @property
    def coord(self):
        return self._coord
    @coord.setter
    def coord(self, value):
        if not isinstance(value, CoordSeries):
            raise TypeError(f"coord must be CoordSeries object, {type(value)} was passed.")
        if self._coord:
            raise TypeError(f"coord already has a CoordSeries object {self.coord}, you cannot overwrite.")
        elif self._coord is None:
            self._coord = value
            self.tc = self.coord.meta["tc"]
            self.sc = self.coord.meta["sc"]
    @property
    def audio(self):
        return self._audio
    @audio.setter
    def audio(self, value):
        if not isinstance(value, AudioSeries):
            raise TypeError(f"audio must be audioSeries object, {type(value)} was passed.")
        if self._audio:
            raise TypeError(f"audio already has a CoordSeries object {self.audio}, you cannot overwrite.")
        elif self._audio is None:
            self._audio = value
    @property
    def meta(self):
        coord_meta = None
        audio_meta = None
        if self.coord:
            coord_meta = self.coord.meta
        if self.audio:
            audio_meta = self.audio.meta
        return (
            f"tc: {self.tc}, sc: {self.sc}, coord: {coord_meta}, audio: {audio_meta}"
        )

    def has_both(self) -> bool:
        return (self.coord is not None) and (self.audio is not None)

@dataclass
class HandlerConfig:
    resample_mode: str = "linear"
    interpolation_fill: str = "edge"
    logging_level: str = "INFO"

class DataSeriesHandler:
    def __init__(self, config: HandlerConfig | None = None):
        self.config = config
        self.seriesmap: dict[tuple[int, int], Series] = {}
        self.datamaploader: DataMapLoader | None = None
        self.unloaded_coord: list[Path] = []
        self.unloaded_audio: list[Path] = []
        self._logs: list[tuple[str, str]] = []

    @staticmethod
    def search_zero_coord_file(tgtdir, tgtfile):
        tc_match = re.search(r"tc(\d+)", tgtfile.name)
        tc = tc_match.group(1) if tc_match else None
        suffix = tgtfile.suffix
        p = None
        if tc:
            p = tgtdir.glob(f"tc{tc}_sc00*{suffix}")
        if p is None:
            p = tgtdir.glob(f"zero*{suffix}")
        if p is None:
            raise FileNotFoundError(f"zero coord file was not found for {tgtfile}.")
        p = list(p)
        if len(p) != 1:
            raise FileNotFoundError(f"zero data file must be a single, but {len(p)} file was found.")
        return p[0]

    def add_coord_file(self, data_path, zero_data_path):
        try:
            loader = CoordDataLoader(data_path=data_path, zero_data_path=zero_data_path)
            tc = loader.tc
            sc = loader.sc
        except Exception as e:
            self._log("ERROR", f"Coord add failed: {data_path} ({e})")
            self.unloaded_coord.append(data_path)
            return None
        if (tc is None) or (sc is None):
            series = None
        else:
            series = self.seriesmap.get((tc, sc))
        if series is None:
            series = Series()
            self.seriesmap[(tc, sc)] = series
        series.coord = CoordSeries(loader)
        self._log("INFO", f"Coord added: tc={tc}, sc={sc}, file={data_path}")
        return series

    def add_audio_file(self, data_path):
        try:
            loader = AudioDataLoader(data_path)
            rec = loader.rec
            info = self.datamaploader.extract_info_from_rec(rec)
            tc = info["test_code"]
            sc = info["shooting_code"]
        except Exception as e:
            self._log("ERROR", f"Audio add failed: {data_path} ({e})")
            self.unloaded_audio.append(data_path)
            return None
        if tc is None or sc is None:
            series = None
        else:
            series = self.seriesmap.get((tc, sc))
        if series is None:
            series = Series()
            self.seriesmap[(tc, sc)] = series
        series.audio = AudioSeries(loader)
        self._log("INFO", f"Audio added: tc={tc}, sc={sc}, file={data_path}")
        return series

    def scan_directory(self, coord_dir, audio_dir, datamap_dir, coord_glob, audio_glob, datamap_glob):
        datamap_list = list(datamap_dir.glob(datamap_glob))
        if len(datamap_list) != 1:
            raise FileNotFoundError(f"multiple datamap file was found, it msut be a single file")
        self.datamaploader = DataMapLoader(datamap_list[0])
        for p in coord_dir.glob(coord_glob):
            if p.match(r"*sc00*"):
                continue
            zero_data_path = DataSeriesHandler.search_zero_coord_file(coord_dir, p)
            self.add_coord_file(p, zero_data_path)
        if audio_dir and audio_glob:
            for p in audio_dir.glob(audio_glob):
                self.add_audio_file(p)
        self.seriesmap = dict(sorted(self.seriesmap.items(), key=lambda x: (x[0][0], x[0][1])))

    def report_pairing(self):
        paired = sum(1 for s in self.seriesmap.values() if s.has_both())
        nocoord = [f"{s.tc}-{s.sc}" for s in self.seriesmap.values() if s.coord is None]
        noaudio = [f"{s.tc}-{s.sc}" for s in self.seriesmap.values() if s.audio is None]
        return {
            "paired_count": paired,
            "num_series": len(self.seriesmap),
            "missing_coord": nocoord,
            "missing_audio": noaudio,
            "unloaded_coord_files": [p.name for p in self.unloaded_coord],
            "unloaded_audio_files": [p.name for p in self.unloaded_audio],
        }

    def select_series(self, tc, sc):
        dataseries = self.seriesmap[(tc, sc)]
        datamapinfo = self.datamaploader.extract_info_from_tcsc(tc, sc)
        return dataseries, datamapinfo

    def filter(self, tc, sc):
        for s in self.seriesmap.values():
            if s.coord is None:
                continue
            m = s.coord.meta
            if tc is not None and m.get("tc") != tc:
                continue
            if sc is not None and m.get("sc") != sc:
                continue
            yield s

    #### editing
    def align_series(self, tc, sc, reference="coord", t0_offset=0.0):
        s = self.seriesmap.get((tc, sc))
        if s is None:
            self._log("WARN", f"align skipped because data was not found: (tc, sc) = ({tc}, {sc})")
            return None
        if not s.has_both():
            self._log("WARN", f"align skipped because coord or audio data was missing: (tc, sc) = ({tc}, {sc})")
            return None
        t_coord = s.coord.t
        t_audio = s.audio.t + t0_offset
        sound = s.audio.sound
        aligned_sound = np.interp(t_coord, t_audio, sound)
        self.aligned_reference = reference
        s.cache["aligned"] = {
            "t": t_coord,
            "coord_cage_markers": s.coord.cage_markers,
            "coord_ring_markers": s.coord.ring_markers,
            "audio":aligned_sound,
        }
        return s.cache["aligned"]

    #### editing
    def slice(self, tc, sc, t_start, t_end):
        s = self.seriesmap.get(tc, sc)
        if not s or "aligned" not in s.cache:
            self._log("WARN", f"slice skipped: tc, sc = ({tc}, {sc})")
            return None
        data = s.cache["aligned"]
        t = data["t"]
        idx = (t >= t_start) & (t <= t_end)
        out = {
            "t": t[idx],
            "coord": data["coord"][idx],
            "audio": data["audio"][idx],
        }
        return out

    def _log(self, level, message):
        if self.config:
            if self.config.logging_level in ("DEBUG", "INFO", "WARN", "ERROR"):
                self._logs.append((level, message))


def load_map_csv(csv_path, xrange=(0, 1, 0.05), yrange=(6, 7, 0.05), xlabel=r'$\delta l$ [mm]', ylabel=r'$D_p$ [mm]'):
    if isinstance(csv_path, str): csv_path = Path(csv_path)
    df = pl.read_csv(csv_path, has_header=False, infer_schema_length=1000)
    df_ndarray = df.to_numpy()
    delta_l_mesh = df_ndarray[0, 1:].astype(np.float16)
    Dp_mesh = df_ndarray[1:, 0].astype(np.float16)
    result = df_ndarray[1:, 1:].astype(np.int8)
    print(f'delta_l_mesh, Dp_mesh, result: {delta_l_mesh.shape}, {Dp_mesh.shape}, {result.shape}')
    # fig, ax = set_colormap(xmesh=delta_l_mesh, ymesh=Dp_mesh, z=result, xrange=xrange, yrange=yrange, xlabel=xlabel, ylabel=ylabel, note=f'{csv_path.stem}')
    fig, ax = set_colormap(xmesh=delta_l_mesh, ymesh=Dp_mesh, z=result, xrange=xrange, yrange=yrange, xlabel=xlabel, ylabel=ylabel, note=f'{csv_path.stem}')
    return fig, ax

def set_colormap(xmesh, ymesh, z, xrange=(0, 1, 0.05), yrange=(6, 7, 0.05), xlabel=r'$\delta l$ [mm]', ylabel=r'$D_p$ [mm]', note=''):
    from matplotlib.colors import ListedColormap
    import matplotlib.ticker as ticker
    # plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
    # fig, axs = plotter.myfig(slide=True)
    # ax = axs[0]
    # ax.set_aspect(1)
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.99, bottom=0.05)
    cmap = ListedColormap([(0, 0, 0, 0.2), 'white'])
    ax.pcolormesh(xmesh, ymesh, z, cmap=cmap, vmin=0, vmax=0.1, shading='nearest')
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.set_xticks(np.arange(xrange[0], xrange[1], xrange[2]))
    # ax.set_yticks(np.arange(yrange[0], yrange[1], yrange[2]))
    ax.set_yticks(np.arange(np.floor(yrange[0]*10**1)/10**1, yrange[1], yrange[2]))
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which='major', lw=0.7)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which='minor', lw=0.4)
    ax.set_aspect(1)
    fig.text(0.7, 0.01, note, ha='left', va='center', fontsize=8)
    return fig, ax

def set_colormap2(xmesh, ymesh, z, xrange=(0, 1, 0.1), yrange=(6, 7, 0.1), xlabel=r'$\delta l$ [mm]', ylabel=r'$D_p$ [mm]', note=''):
    from matplotlib.colors import ListedColormap
    import matplotlib.ticker as ticker
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
    fig, axs = plotter.myfig(slide=True)
    ax = axs[0]
    cmap = ListedColormap([(0, 0, 0, 0.2), 'white'])
    ax.pcolormesh(xmesh, ymesh, z, cmap=cmap, vmin=0, vmax=0.1, shading='nearest')
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.set_xticks(np.arange(xrange[0], xrange[1]*1.001, xrange[2]))
    ax.set_yticks(np.arange(np.floor(yrange[0]*10**1)/10**1, yrange[1]*1.001, yrange[2]))
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which='major', lw=1)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which='minor', lw=0.1)
    ax.set_aspect(1)
    # fig.text(0.7, 0.01, note, ha='left', va='center', fontsize=8)
    return fig, ax


if __name__ == "__main__":
    print("---- test ----")

    datadir = config.ROOT / "data" / "sampledata"
    datamapfile = Path("D:/1005_tyn/02_experiments_and_analyses/list_visualization_test.xlsx")
    datamap_loader = DataMapLoader(datamapfile)
    datamap = datamap_loader.datamap
    # print(datamap.head())
    # print(datamap.columns)
    summary = datamap_loader.summary.drop_nulls(subset=["noise_result"])
    print(summary.columns)
    print(summary)
    # info = datamap_loader.extract_info_from_tcsc(1, 2)
    # print(info)
    # datamap.write_csv(datadir/"datamap.csv")

    # all_list = dataseries.list_all()
    # for l in all_list:
        # print(l)

    # audio_ld = AudioDataLoader(datadir/"251127_002_5000rpm_silnet.WAV")
    # audio_ld = AudioDataLoader(datadir/"REC3002.csv")
    # audio_ld = AudioDataLoader(datadir/"REC3405.csv")
    # audio_ld0 = AudioDataLoader(datadir/"REC3405.mat")
    # audio_ld1 = AudioDataLoader(datadir/"REC3405.wav")

    # print(repr(audio_ld))

    # t_error = audio_ld0.t - audio_ld1.t
    # sound_error = audio_ld0.sound - audio_ld1.sound
    # print(f"max of error: {np.nanmax(np.abs(sound_error))}")

    # fig, ax = plt.subplots(figsize=(15, 8))
    # ax.plot(audio_ld.t, audio_ld.sound, lw=0.4, c='b')
    # ax.plot(np.arange(len(t_error)), t_error, lw=0.4, c='b')
    # ax.plot(np.arange(len(t_error)), sound_error, lw=0.4, c='b')
    # ax.plot(audio_ld0.t, audio_ld0.sound, lw=0.4, c='b')
    # ax.plot(audio_ld1.t, audio_ld1.sound, lw=0.4, c='r')
    # ax.set(ylim=(-10, 10))
    # ax.axhline(y=0, lw=0.1, c='k')
    # plt.show()


    csv_path = Path(r"D:/200_python/02_specification_for_v2cage/results/251017_Dpdlmap/Dp_vs_deltal_40BNRv2.csv")
    # csv_path = config.ROOT/"results"/"251017_Dpdlmap"/"Dp_vs_deltal_70BNRv2.csv"
    # fig, ax = load_map_csv(csv_path, xrange=(0, 1, 0.05), yrange=(6, 7, 0.05)) # argument xrange and yrange difine the plot range as  (start, end, step)
    fig, ax = load_map_csv(csv_path, xrange=(0.1, 0.7, 0.1), yrange=(6, 6.6, 0.1)) # argument xrange and yrange difine the plot range as  (start, end, step)
    # fig, ax = load_map_csv(csv_path, xrange=(0, 1, 0.1), yrange=(8.8, 9.8, 0.1)) # argument xrange and yrange difine the plot range as  (start, end, step)


    # Mapping from noise_result to annotation text
    marker_map = {
        "o": "o",
        "x": "x",
        "z": "+",
        "^": "^",
    }
    color_map = {
        "o": "g",
        "x": "r",
        "z": "k",
        "^": "b",
    }
    zorder_map = {
        "o": 10,
        "x": 100,
        "z": 10,
        "^": 20,
    }

    for row in summary.iter_rows(named=True):
        print("***")
        # Get the marker type from noise_result
        noise = row["noise_result"]
        # Decide annotation text
        mark = marker_map.get(noise)
        color = color_map.get(noise)
        zorder = zorder_map.get(noise)
        # Example positions (replace with your real columns)
        x = row["dl_measured"]
        y = row["Dp_measured"]
        if x is None: x = row["dl_drawing"]
        if y is None: y = row["Dp_drawing"]
        ax.scatter(x, y, marker=mark, color=color, s=100, zorder=zorder)
        ax.annotate(
            row["cage"],                 # Text to draw
            (x, y),               # Position (x, y)
            textcoords="offset points",
            xytext=(5, 5),         # Offset from point
            ha="left",
            va="bottom",
            fontsize=10,
            color="red" if mark == "x" else "black"
        )

    # annotate_point = (0.2, 6.2)
    # ax.scatter(annotate_point[0], annotate_point[1], marker='x', c='r', s=100)
    # ax.annotate('sample', xy=annotate_point, xytext=(0.4, 1), textcoords='offset points', fontsize=20)



    plt.show(block=True)
    # fig.savefig(outdir/f"{csv_path.stem}.png")





