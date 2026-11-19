"""A subject model, then ViTMatte on the edge band, as the app does, for scoring against Photoshop.

usage: coarse_test.py MODEL.onnx SIDE VITMATTE.onnx PHOTOS OUT
Writes OUT/<photo>.png with the selection as alpha. Run with Lite at 640 as a control: it should
score like the app's own output, which says this script stands for the app.
"""
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL, SIDE, VIT, PHOTOS, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
os.makedirs(OUT, exist_ok=True)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
TILE, OVERLAP = 384, 64
BAND = int(os.environ.get("BAND", "12"))  # px either side of the edge that ViTMatte decides


def subject(sess, photo):
    w, h = photo.size
    small = np.asarray(photo.resize((SIDE, SIDE), Image.BILINEAR), np.float32) / 255
    x = ((small - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[-1].squeeze().astype(np.float32)
    if out.min() < 0 or out.max() > 1:
        out = 1 / (1 + np.exp(-out))
    return np.asarray(Image.fromarray(out).resize((w, h), Image.BILINEAR), np.float32)


def vitmatte(vit, image, mask):
    H, W = mask.shape
    k = np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)
    fg, bg = cv2.erode((mask > 0.95).astype(np.uint8), k), cv2.erode((mask < 0.05).astype(np.uint8), k)
    tri = np.full(mask.shape, 0.5, np.float32)
    tri[fg == 1], tri[bg == 1] = 1.0, 0.0
    acc, wsum = np.zeros_like(mask), np.zeros_like(mask)
    ramp = lambda n: np.minimum(np.minimum(np.arange(n) + 1, n - np.arange(n)), OVERLAP) / OVERLAP
    for y0 in range(0, H, TILE - OVERLAP):
        for x0 in range(0, W, TILE - OVERLAP):
            y0c, x0c = min(y0, max(H - TILE, 0)), min(x0, max(W - TILE, 0))
            sl = (slice(y0c, y0c + TILE), slice(x0c, x0c + TILE))
            if not (tri[sl] == 0.5).any():
                continue
            h, w = tri[sl].shape
            x = np.zeros((1, 4, TILE, TILE), np.float32)
            x[0, :3, :h, :w] = ((image[sl] - 0.5) / 0.5).transpose(2, 0, 1)
            x[0, 3, :h, :w] = tri[sl]
            a = vit.run(None, {vit.get_inputs()[0].name: x})[0][0, 0, :h, :w]
            wt = np.outer(ramp(h), ramp(w)).astype(np.float32)
            acc[sl] += a * wt
            wsum[sl] += wt
    return np.where((tri == 0.5) & (wsum > 0), np.clip(acc / np.maximum(wsum, 1e-6), 0, 1), mask)


sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
vit = ort.InferenceSession(VIT, providers=["CPUExecutionProvider"])
for f in sorted(os.listdir(PHOTOS)):
    photo = Image.open(os.path.join(PHOTOS, f)).convert("RGB")
    # The app reads big photos reduced; ViTMatte works at up to 2048 px.
    if max(photo.size) > 2048:
        s = 2048 / max(photo.size)
        work = photo.resize((round(photo.width * s), round(photo.height * s)), Image.BILINEAR)
    else:
        work = photo
    start = time.perf_counter()
    mask = subject(sess, work)
    took = time.perf_counter() - start
    a = vitmatte(vit, np.asarray(work, np.float32) / 255, mask)
    a = np.asarray(Image.fromarray(a).resize(photo.size, Image.BILINEAR), np.float32)
    rgba = np.dstack([np.asarray(photo), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, os.path.splitext(f)[0] + ".png"))
    print(f"{f}: subject model {took:.2f}s on the CPU", flush=True)
