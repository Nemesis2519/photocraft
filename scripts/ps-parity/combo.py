"""Lite decides the subject, HR draws its edge: inside Lite's edge band (sure pixels eroded by B px)
the alpha comes from HR, elsewhere from Lite."""
import os, sys
import cv2, numpy as np
from PIL import Image
LITE, HR, OUT, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
HOW = sys.argv[5] if len(sys.argv) > 5 else "hr"  # hr: HR alone in the band; min; mean
os.makedirs(OUT, exist_ok=True)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * B + 1, 2 * B + 1))
for f in os.listdir(LITE):
    if not f.endswith(".png"):
        continue
    l = np.asarray(Image.open(os.path.join(LITE, f)).convert("RGBA"))
    h = np.asarray(Image.open(os.path.join(HR, f)).convert("RGBA"))[..., 3]
    a = l[..., 3]
    sure_in, sure_out = cv2.erode((a > 242).astype(np.uint8), k), cv2.erode((a < 13).astype(np.uint8), k)
    band = (sure_in == 0) & (sure_out == 0)
    out = l.copy()
    pick = {"hr": h, "min": np.minimum(h, a), "mean": ((h.astype(np.uint16) + a) // 2).astype(np.uint8)}[HOW]
    out[..., 3] = np.where(band, pick, a)
    Image.fromarray(out, "RGBA").save(os.path.join(OUT, f))
