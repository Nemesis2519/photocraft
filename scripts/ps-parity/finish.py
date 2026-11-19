"""Does Photoshop's edge finish, applied to our cut-outs, bring them closer to Photoshop's masks?

usage: finish.py PHOTOS IN PS_MASKS
Photoshop's finishing steps, from the private reconstruction's constants (not Adobe code):
  drop:   in the edge band, alpha under 20% becomes background (merge_matte's weight rule)
  curve:  S-curve x^g / (k(1-x)^g + x^g), g = 1.3136, k = 49^(0.13 g)
  guided: guided filter on a 3x3-blurred grey image, radius max(short side / 320, 5), eps 0.01^2
Prints mean SAD (thousands) per variant on the tuning and held-out halves (split by name hash).
"""
import hashlib
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

PHOTOS, IN, PS = sys.argv[1], sys.argv[2], sys.argv[3]
G = 1.3136
K = math.exp(math.log(49.00005) * 0.13 * G)


def curve(a):
    hi, lo = np.power(a, G), np.power(1 - a, G)
    return hi / (K * lo + hi + 1e-12)


def drop(a):
    return np.where((a > 0) & (a < 0.2), 0.0, a)


def guided(rgb, a):
    grey = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, (3, 3), 0)
    r = int(max(min(a.shape) / 320, 5))
    k = (2 * r + 1, 2 * r + 1)
    box = lambda d: cv2.boxFilter(d, cv2.CV_32F, k)
    mg, ma = box(grey), box(a)
    slope = (box(grey * a) - mg * ma) / (box(grey * grey) - mg * mg + 1e-4)
    return np.clip(box(slope) * grey + box(ma - slope * mg), 0, 1)


VARIANTS = {
    "ours": lambda rgb, a: a,
    "drop": lambda rgb, a: drop(a),
    "curve": lambda rgb, a: curve(a),
    "guided": lambda rgb, a: guided(rgb, a),
    "drop+curve": lambda rgb, a: curve(drop(a)),
    "all three": lambda rgb, a: guided(rgb, curve(drop(a))),
}
sums = {v: {True: [], False: []} for v in VARIANTS}
for f in sorted(os.listdir(PS)):
    stem = os.path.splitext(f)[0]
    if not os.path.exists(os.path.join(IN, stem + ".png")):
        continue
    photo = [p for p in os.listdir(PHOTOS) if os.path.splitext(p)[0] == stem][0]
    rgb = np.asarray(Image.open(os.path.join(PHOTOS, photo)).convert("RGB"))
    a = np.asarray(Image.open(os.path.join(IN, stem + ".png")).convert("RGBA"), np.float32)[..., 3] / 255
    ps = np.asarray(Image.open(os.path.join(PS, f)).convert("RGB"), np.float32)[..., 0] / 255
    tune = hashlib.sha1(stem.encode()).digest()[0] % 2 == 0
    for name, fn in VARIANTS.items():
        sums[name][tune].append(np.abs(fn(rgb, a) - ps).sum() / 1000)
for name in VARIANTS:
    t, h = sums[name][True], sums[name][False]
    print(f"{name:11} tune SAD {np.mean(t):7.1f}   held-out SAD {np.mean(h):7.1f}   (n={len(t)}+{len(h)})")
