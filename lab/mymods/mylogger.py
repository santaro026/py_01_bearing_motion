"""
Created on Wed Oct 15 14:43:29 2025
@author: santaro



"""


import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path
import time

import logging
from datetime import datetime
import config

class MyLogger(logging.Logger):
    def __init__(self, name, outdir=Path.cwd(), mode="w"):
        super().__init__(name)
        outdir.mkdir(parents=True, exist_ok=True)
        self.setLevel(logging.DEBUG)
        self.full_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        self.blank_formatter = logging.Formatter('%(message)s')
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.DEBUG)
        log_fpath = outdir / f"{name}.log"
        self.file_handler = logging.FileHandler(log_fpath, mode=mode) # mode=a: add, mode=w: overwrite
        self.file_handler.setLevel(logging.INFO)
        if not self.handlers:
            self.addHandler(self.console_handler)
            self.addHandler(self.file_handler)
        self.set_all_formatters(self.full_formatter)
        if mode == 'a':
            if log_fpath.exists() and log_fpath.stat().st_size > 0:
                self.put_line(f"\n\n---------- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ----------\n")
            else:
                self.put_line(f"---------- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ----------\n")
        self.cache = {}
    def set_all_formatters(self, formatter):
        for handler in self.handlers:
            handler.setFormatter(formatter)
    def binfo(self, message):
        self.set_all_formatters(self.blank_formatter)
        self.log(logging.INFO, message)
        self.set_all_formatters(self.full_formatter)
    def measure_time(self, name, mode):
        if mode == 's':
            st = time.perf_counter()
            self.cache[f"{name}_st"] = st
            self.info(f"\"{name}\" start >>>>>>>>>>")
        elif mode == 'e':
            et = time.perf_counter()
            self.cache[f"{name}_et"] = et
            elapsed_time = et - self.cache[f"{name}_st"]
            self.cache[f"{name}_elapsed"] = elapsed_time
            self.info(f">>>>>>>>>> \"{name}\" completed, elapsed time: {elapsed_time:.4f} sec\n")


if __name__ == '__main__':
    print('---- test ----')

    # logger.debug('debug message')
    # logger.info('information')
    # logger.warning('warning')

    logger = MyLogger(name=__name__, mode='a')
    logger.debug('debug test')
    logger.info('test')
    logger.warning('****')
    logger.put_line("line test")

    # logger2 = MyLogger(name="logger2")
    # logger2.info("test2")
    # logger2.debug("debug test2")
    # logger2.warning("**** 2")

    # logger = MyLogger(name=__name__)
    # logger.warning('**** second time')
