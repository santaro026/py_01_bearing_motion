"""
Created on Thu Jan 22 00:16:17 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import librosa
from pydub import AudioSegment
from pydub.playback import play

HOME = Path("/home/sintaro")

def librosa_test(filename):
    print(f"target file: {filename.name}, {filename.exists()}")
    y, sr = librosa.load(filename, sr=48000)
    t = np.arange(len(y))/sr
    print(f"y: {y.size}, sr: {sr}")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Estimated temp: {tempo[0]:.2f} beat per minute")
    print(f"beat frames: {beat_frames.shape}")
    # print(f"beat frames: {beat_frames}")
    # beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # print(f"beat_times: {beat_times}")
    # fig, ax = plt.subplots(figsize=(20, 12))
    # ax.plot(t, y, lw=1, c='b')
    # for x in beat_times:
        # ax.axvline(x=x, ymin=-100, ymax=100, lw=0.4, c='k')
    # plt.show()

    hop_length = 512
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    print(f"y_percussive tempo: {tempo[0]:.2f}")
    # fig, ax = plt.subplots(figsize=(20, 12))
    # ax.plot(t, y_harmonic, lw=0.2, c='g')
    # ax.plot(t, y_percussive, lw=0.2, c='r')
    # plt.show()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    print(f"mfcc: {mfcc.shape}")
    mfcc_delta = librosa.feature.delta(mfcc)
    print(f"mfcc_delta: {mfcc_delta.shape}")
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
    print(f"beta_mfcc_delta: {beat_mfcc_delta.shape}")
    # librosa.display.specshow(mfcc, x_axis="time")
    # librosa.display.specshow(mfcc_delta, x_axis="time")
    # librosa.display.specshow(beat_mfcc_delta, x_axis="time")
    # plt.show()

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
    print(f"chromagram: {chromagram.shape}")
    print(f"beat_chromagram: {beat_chroma.shape}")
    # fig, ax = plt.subplots(figsize=(20, 12))
    # colors = plt.cm.hsv(np.linspace(0, 1, 13))[:12] # shpae is (12, 4)
    # colors = plt.cm.tab20(np.arange(12)) # shape is (12, 4)
    # fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    # base_colors = plt.cm.hsv(np.linspace(0, 1, 13))[:12]
    # colors = [base_colors[fifths_order.index(i)] for i in range(12)]
    # for i in range(12):
        # ax.plot(np.arange(chromagram.shape[1]), chromagram[i], lw=0.2, c=colors[i])
    # librosa.display.specshow(chromagram, x_axis="time")
    # librosa.display.specshow(beat_chroma, x_axis="time")
    # plt.show()

    # summarize features for deep learning
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    print(f"beat_features: {beat_features.shape}")

    # fft
    n_fft = 2**10
    hop_length = n_fft // 2
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(figsize=(20, 12))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz")
    # librosa.display.specshow(D, sr=sr, hop_length=512, x_axis="time", y_axis="hz")
    # plt.show()

    do_play = 0
    if do_play:
        audio_segment = AudioSegment(
            (y * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2, # 16bit = 2byte
            channels=1
        )
        play(audio_segment)

    return fig, ax

if __name__ == "__main__":
    print("---- run ----")
    print(f"home: {HOME.name}, {HOME.exists()}")

    # filename = HOME / "data" / "sampledata" / "mp3" / "canon.mp3"
    # filename = HOME / "data" / "sampledata" / "m4a" / "whitenoise_direct.m4a"
    # filename = HOME / "data" / "sampledata" / "m4a" / "whitenoise_smallbox.m4a"
    # filename = HOME / "data" / "sampledata" / "m4a" / "whitenoise_widebox.m4a"
    knock_widebox = HOME / "data" / "sampledata" / "m4a" / "knock_widebox.m4a"
    knock_smallbox = HOME / "data" / "sampledata" / "m4a" / "knock_smallbox.m4a"

    # filename = librosa.example("nutcracker")
    # print(f"sample: {type(filename)}, {filename}")

    fig1, ax1 = librosa_test(knock_widebox)
    fig2, ax2 = librosa_test(knock_smallbox)
    plt.show()


