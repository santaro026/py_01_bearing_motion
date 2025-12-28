"""
Created on Mon Sep 01 12:30:38 2025
@author: santa


"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
import time

import config

class Logger:
    def __init__(self):
        self.log_entries = []
        self.mgs_entries = []
        self.time_entries = {}
        self.prgbar_entries = []

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if type(message) != str:
            message = str(message)
        self.log_entries.append(f'[{timestamp}] {message}')

    def msg(self, message):
        if type(message) != str:
            message = str(message)
        self.mgs_entries.append(f'{message}')

    def rectime(self, mode, name='main'):
        if mode == 0:
            self.log(message=f'"{name}" start')
            self.time_entries[f'st_{name}'] = time.perf_counter()
        elif mode == 1:
            et = time.perf_counter()
            excution_time = et - self.time_entries[f'st_{name}']
            self.log(message=f'"{name}" complete')
            self.log(message=f'excution time of "{name}": {excution_time} [sec]')
            return excution_time

    def printlog(self):
        print('\n**** printlog\n')
        print('\n'.join(self.log_entries))

    def printmsg(self):
        print('\n**** printmsg\n')
        print('\n'.join(self.mgs_entries))

    def print_progressbar(self, current, total, bar_length=40, row=1):
        percent = round(current / total * 100)
        filled_length = int(current / total * bar_length)
        bar = "#" * filled_length + " " * (bar_length - filled_length)
        printmsg = f'{percent:3.0f} % |{bar}| {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        print(printmsg, flush=True)
        if row > 1:
            for i in range(row-1):
                print(f'      |{bar}|', flush=True)
        self.prgbar_entries.append(printmsg)

    def export(self, outdir=None, outfname=""):
        outdir = config.ROOT / 'results' / 'output' if outdir is None else outdir
        count = 1
        outfile_buf = outdir / (outfname + "_log.txt")
        outfile = outfile_buf
        while True:
            try:
                with open(outfile, "x") as f:
                    f.write("\n".join(self.log_entries))
                    break
            except FileExistsError:
                outfile = outfile_buf.parent / (str(outfile_buf.stem) + f'_o{count}' + str(outfile_buf.suffix))
                count += 1
        count = 1
        outfile_buf = outdir / (outfname + "_msg.txt")
        outfile = outfile_buf
        while True:
            try:
                with open(outfile, "x") as f:
                    f.write("\n".join(self.mgs_entries))
                    break
            except FileExistsError:
                outfile = outfile_buf.parent / (str(outfile_buf.stem) + f'_o{count}' + str(outfile_buf.suffix))
                count += 1
        count = 1
        outfile_buf = outdir / (outfname + "_progress.txt")
        outfile = outfile_buf
        while True:
            try:
                with open(outfile, "x") as f:
                    f.write("\n".join(self.prgbar_entries))
                    break
            except FileExistsError:
                outfile = outfile_buf.parent / (str(outfile_buf.stem) + f'_o{count}' + str(outfile_buf.suffix))
                count += 1