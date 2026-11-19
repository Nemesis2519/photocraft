"""Score our masks against Photoshop's Select Subject masks.

usage: score.py PHOTOS PS_MASKS OUT label=dir [label=dir ...]
Each dir holds our Remove Background PNGs (alpha = selection), named like the photos.
Photoshop masks are RGB PNGs, value = selection * 255.

Per photo and method:
  IoU       of the hard masks at 0.5 (1.0 = same shape)
  SAD       sum of |ours - PS| over the whole image, in thousands (matting papers' unit)
  edge MSE  mean squared difference on the edge band (x1000)
  grad      gradient error on the edge band (x1000), how differently the edge is shaped
  extra     pixels we select at least 25% more than PS does (smudges, halo, background kept)
  missing   pixels we select at least 25% less than PS does (lost hair, cut-off parts)
The edge band is PS's soft pixels plus 15 px either side of its 50% line.
A picture per photo: red = we keep more than PS, cyan = we keep less.
"""
import hashlib
import os
import sys

import cv2
import numpy as np
from PIL import Image

PHOTOS, PS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
METHODS = [a.split("=", 1) for a in sys.argv[4:]]
os.makedirs(OUT, exist_ok=True)


def grad(a):
    a = cv2.GaussianBlur(a, (0, 0), 1.4)
    return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))


rows, totals = [], {label: np.zeros(6) for label, _ in METHODS}
names = sorted(os.path.splitext(f)[0] for f in os.listdir(PS) if f.lower().endswith(".png"))
# SPLIT=tune or SPLIT=held scores a fixed half of the photos, chosen by a hash of the name, so
# settings are tuned on one half and reported on the other.
SPLIT = os.environ.get("SPLIT", "all")
if SPLIT != "all":
    names = [n for n in names if (hashlib.sha1(n.encode()).digest()[0] % 2 == 0) == (SPLIT == "tune")]
for name in names:
    ps = np.asarray(Image.open(os.path.join(PS, name + ".png")).convert("RGB"), np.float32)[..., 0] / 255
    h, w = ps.shape
    hard = (ps > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    band = ((ps > 0.02) & (ps < 0.98)) | (cv2.dilate(hard, k) != cv2.erode(hard, k))
    found = [f for f in os.listdir(PHOTOS) if os.path.splitext(f)[0] == name]
    photo = np.asarray(Image.open(os.path.join(PHOTOS, found[0])).convert("L"), np.float32) / 255
    for label, d in METHODS:
        path = os.path.join(d, name + ".png")
        if not os.path.exists(path):
            continue
        ours = np.asarray(Image.open(path).convert("RGBA"), np.float32)[..., 3] / 255
        if ours.shape != ps.shape:
            ours = cv2.resize(ours, (w, h), interpolation=cv2.INTER_LINEAR)
        diff = ours - ps
        a, b = ours > 0.5, ps > 0.5
        iou = (a & b).sum() / max((a | b).sum(), 1)
        sad = np.abs(diff).sum() / 1000
        mse = (diff[band] ** 2).mean() * 1000
        ge = ((grad(ours) - grad(ps))[band] ** 2).sum() / 1000
        extra, missing = int((diff > 0.25).sum()), int((diff < -0.25).sum())
        v = np.array([iou, sad, mse, ge, extra, missing])
        totals[label] += v
        rows.append((name, label, v))
        # Red where we keep more than Photoshop, cyan where we keep less, on the dimmed photo.
        base = (photo * 0.45)[..., None].repeat(3, 2)
        more, less = np.clip(diff, 0, 1)[..., None], np.clip(-diff, 0, 1)[..., None]
        pic = base * (1 - more - less) + more * np.array([1, 0.15, 0.15]) + less * np.array([0.1, 0.9, 1])
        Image.fromarray((np.clip(pic, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, f"{name}-{label}.png"))

print(f"{'photo':16} {'method':10} {'IoU':>7} {'SAD':>8} {'edgeMSE':>8} {'grad':>8} {'extra':>8} {'missing':>8}")
for name, label, v in rows:
    print(f"{name:16} {label:10} {v[0]:7.4f} {v[1]:8.1f} {v[2]:8.2f} {v[3]:8.1f} {int(v[4]):8d} {int(v[5]):8d}")
n = max(len(names), 1)
for label, t in totals.items():
    print(f"{'MEAN':16} {label:10} {t[0]/n:7.4f} {t[1]/n:8.1f} {t[2]/n:8.2f} {t[3]/n:8.1f} {int(t[4]/n):8d} {int(t[5]/n):8d}")
