"""The channel-levels hair mask, automated: does it beat ViTMatte where contrast is strong?

usage: explore_levels.py PHOTOS_A IN_A PS_A [PHOTOS_B IN_B PS_B ...]
By hand, the classic method picks the colour channel that separates subject from background best,
then crushes it with Levels so the subject goes white and the background black. Here that is done
per area of the edge band: F and B are the mean subject and background colours nearby (from our own
mask's sure pixels); the channel with the largest |F - B| is chosen, and Levels maps B to 0 and F to
1. Where |F - B| beats `contrast`, that alpha replaces ours (or is averaged with it).
"""
import hashlib
import os
import sys

import cv2
import numpy as np
from PIL import Image

BAND = 12
sets = [sys.argv[i : i + 3] for i in range(1, len(sys.argv), 3)]


def local_mean(img, weight, r):
    k = (2 * r + 1, 2 * r + 1)
    num = cv2.boxFilter(img * weight[..., None], -1, k, normalize=False)
    den = cv2.boxFilter(weight, -1, k, normalize=False)
    return num / np.maximum(den, 1e-6)[..., None], den


rows = []
for photos, cut_dir, ps_dir in sets:
    for f in sorted(os.listdir(ps_dir)):
        stem = os.path.splitext(f)[0]
        path = os.path.join(cut_dir, stem + ".png")
        if not os.path.exists(path):
            continue
        photo = [p for p in os.listdir(photos) if os.path.splitext(p)[0] == stem][0]
        rgb = np.asarray(Image.open(os.path.join(photos, photo)).convert("RGB"), np.float32) / 255
        a = np.asarray(Image.open(path).convert("RGBA"), np.float32)[..., 3] / 255
        ps = np.asarray(Image.open(os.path.join(ps_dir, f)).convert("RGB"), np.float32)[..., 0] / 255
        k = np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)
        fg = cv2.erode((a > 0.95).astype(np.uint8), k).astype(np.float32)
        bg = cv2.erode((a < 0.05).astype(np.uint8), k).astype(np.float32)
        band = (fg == 0) & (bg == 0)
        r = max(int(0.02 * max(a.shape)), 8)
        F, nf = local_mean(rgb, fg, r)
        B, nb = local_mean(rgb, bg, r)
        diff = F - B
        best = np.abs(diff).argmax(2)
        pick = lambda m: np.take_along_axis(m, best[..., None], 2)[..., 0]
        d, i0, b0 = pick(diff), pick(rgb), pick(B)
        levels = np.clip((i0 - b0) / np.where(np.abs(d) < 1e-3, 1e-3, d), 0, 1)
        ok = band & (nf > 0) & (nb > 0)
        tune = hashlib.sha1(stem.encode()).digest()[0] % 2 == 0
        rows.append((tune, a, ps, levels, np.abs(d), ok))


def sad(tune, contrast, how):
    out = []
    for t, a, ps, levels, d, ok in rows:
        if t != tune:
            continue
        use = ok & (d > contrast)
        new = {"replace": levels, "mean": (levels + a) / 2}[how] if how else a
        b = np.where(use, new, a) if how else a
        out.append(np.abs(b - ps).sum() / 1000)
    return np.mean(out)


print(f"ViTMatte as is          tune {sad(True, 0, None):7.2f}  held {sad(False, 0, None):7.2f}")
for how in ("replace", "mean"):
    for c in (0.15, 0.25, 0.35, 0.5):
        print(f"levels {how:7} > {c:.2f}  tune {sad(True, c, how):7.2f}  held {sad(False, c, how):7.2f}")
