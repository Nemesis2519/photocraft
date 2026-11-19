# Select Subject against Photoshop

Private-branch tooling (`feat/local-ml`): scores PhotoCraft's Select Subject / Remove Background
against Photoshop's own Select Subject on the same photos. Needs a licensed Photoshop on Windows
(tested with 2023, 24.5) and a Python with numpy, opencv-python, pillow and onnxruntime.

1. **Photoshop masks.** `ps_masks.jsx` runs Select Subject (no Select and Mask) on every JPEG or
   PNG in a folder and saves the selection as a PNG mask (value = selection × 255):

   ```powershell
   $ps = New-Object -ComObject Photoshop.Application
   $ps.DoJavaScriptFile("C:\...\scripts\ps-parity\ps_masks.jsx", @("C:/photos", "C:/masks"))
   $ps.Quit()
   ```

   Select Subject in Photoshop 24.x can run on the device or in the cloud
   (Preferences › Image Processing); the script uses whichever is set, so note which.

2. **Our cut-outs.** `run_models.sh PHOTOS OUT auto [birefnet-hr-matting ...]` runs Remove
   Background through the release CLI with the app's model store.

3. **Score.** `score.py PHOTOS MASKS OUT label=dir ...` prints IoU, SAD, edge MSE, gradient error
   and the pixels we keep more or less than Photoshop, and writes a picture per photo (red: we keep
   more, cyan: we keep less). `SPLIT=tune` or `SPLIT=held` scores a fixed half, so settings tuned
   on one half are reported on the other.

`coarse_test.py` runs a subject model plus ViTMatte in Python (for models the app doesn't ship),
`combo.py` mixes two cut-outs on the edge band. Results are in [`docs/local-ml.md`](../../docs/local-ml.md).

Test photos aren't committed. The 66-photo set is CC0 from Wikimedia Commons; its
`manifest.tsv` (file, title, page, licence) lives beside the photos.
