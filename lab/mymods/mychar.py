"""
Created on Wed Apr 08 18:17:53 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path


def normalize_kanji2(x: str | None) -> str | None:
    if x is None:
        return None
    for src, dst in KANJI_MAP.items():
        x = x.replace(src, dst)
    return x

def normalize_kanji(expr) -> str | None:
    if expr is None:
        return None
    for src, dst in KANJI_MAP.items():
        expr = expr.str.replace_all(src, dst, literal=True)
    return expr


KANJI_MAP = {
    "﨑": "崎",
    "齋": "斎",
    "穗": "穂",
}

