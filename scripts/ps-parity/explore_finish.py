"""Search the edge finish's settings on the tuning half; report the best on the held-out half.

usage: explore_finish.py PHOTOS_A IN_A PS_A [PHOTOS_B IN_B PS_B ...]
finish(a) = curve(drop(a)): values under `cut` go to background, then
x^g / (k(1-x)^g + x^g). Photoshop's portrait finish is cut 0.2, g 1.3136, k 1.94.
"""
import hashlib
import itertools
import os
import sys

import numpy as np
from PIL import Image

sets = [sys.argv[i : i + 3] for i in range(1, len(sys.argv), 3)]
data = []
for photos, cut_dir, ps_dir in sets:
    for f in sorted(os.listdir(ps_dir)):
        stem = os.path.splitext(f)[0]
        path = os.path.join(cut_dir, stem + ".png")
        if not os.path.exists(path):
            continue
        a = np.asarray(Image.open(path).convert("RGBA"), np.float32)[..., 3] / 255
        ps = np.asarray(Image.open(os.path.join(ps_dir, f)).convert("RGB"), np.float32)[..., 0] / 255
        # Only the soft pixels change, so keep just those (and their Photoshop values) plus the
        # fixed error everywhere else.
        soft = (a > 0) & (a < 1)
        rest = np.abs(a[~soft] - ps[~soft]).sum()
        tune = hashlib.sha1(stem.encode()).digest()[0] % 2 == 0
        data.append((stem, tune, a[soft], ps[soft], rest))


def finish(a, cut, g, k):
    a = np.where(a < cut, 0.0, a)
    hi, lo = np.power(a, g), np.power(1 - a, g)
    return hi / (k * lo + hi + 1e-12)


def score(params, half):
    return np.mean([(np.abs(finish(a, *params) - p).sum() + r) / 1000 for _, t, a, p, r in data if t == half])


grid = list(itertools.product([0.0, 0.1, 0.2, 0.3], [1.0, 1.3136, 1.7, 2.2], [1.0, 1.94, 3.0, 4.5]))
base = (0.0, 1.0, 1.0)
ranked = sorted(grid, key=lambda p: score(p, True))
print(f"none           tune {score(base, True):7.2f}  held {score(base, False):7.2f}")
print(f"photoshop      tune {score((0.2, 1.3136, 1.94), True):7.2f}  held {score((0.2, 1.3136, 1.94), False):7.2f}")
for p in ranked[:6]:
    print(f"cut {p[0]:.1f} g {p[1]:.2f} k {p[2]:.2f}  tune {score(p, True):7.2f}  held {score(p, False):7.2f}")
