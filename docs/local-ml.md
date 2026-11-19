# Local models for Select Subject and Remove Background

> **Private branch.** `feat/local-ml` is a personal fork and must stay private. Unlike the rest of
> PhotoCraft it is **not clean-room**: `crates/ml/src/staged.rs`, `crates/algo/src/matting/hair.rs`,
> `crates/algo/src/matting/shared_sampling.rs`, the edge finish in `crates/algo/src/matting/band.rs`
> and the edge snap in
> `crates/engine/src/cmd/model_cmds.rs` use settings from a local reverse-engineered reconstruction
> of Photoshop's pipelines. This breaks rule 3 of [`AGENTS.md`](../AGENTS.md), so none of it may go
> upstream as it stands. Adobe's model weights are not used, decrypted or redistributed.

## Pipeline

Select Subject and Remove Background, on the `auto` preference with models installed:

1. BiRefNet Lite runs on the whole image at 640 px, with its size pinned so the graph folds at load.
   The image is read reduced only as far as keeps both sides at 640 or more, since the model
   stretches its input square: read by the long side, a phone portrait lost two thirds of its width.
2. U²-Net runs at 320 px as a check. If BiRefNet kept under 35% of what U²-Net found, it has lost a
   small subject: BiRefNet reruns on a padded crop around that subject and the two are blended by
   U²-Net's confidence.
3. With **ViTMatte Small** installed (Preferences › Integrations), the edge is solved by a matting
   model: a trimap from the mask (sure pixels eroded 12 px, the rest unknown) and ViTMatte on
   384 px tiles over the unknown band only, at up to 2048 px, overlapping by 64 px and blended.
   Only the band changes. It is the stage the reconstruction shows Photoshop's portrait path has
   (a trimap and a matting network) and the public models lacked. On the integrated Radeon GPU
   (DirectML) a repeat click went from about 2 s to 1.4 s on a 736 × 1308 photo and 2.5–5.0 s on
   1600–1920 px photos; hair, fur, spokes, nets and the double-delete outline all came out
   cleaner. Its weights are Apache-2.0 but were trained on Composition-1k, Adobe's dataset for
   research, so the licence of what they learned is a grey area. Run at half resolution it was
   faster but visibly worse, so it isn't.
4. Without ViTMatte, a guided filter snaps the edge to the picture, around the subject only, at up
   to 2048 px. It changes only pixels the model was unsure of (between 2% and 98%) and those 1 px
   from them. Unlimited, it smeared faint coverage up to 14 px into what the model was sure of, and
   deleting the subject and then the background left that as a halo.

`auto` picks BiRefNet Lite, then ISNet, then the classical method. Every GPU route falls back to the
CPU if the model fails there. Inference never falls back silently to another method, and never starts
a download. The desktop app loads the chosen model and U²-Net in the background at startup, so the
first click doesn't wait for them; a click during that waits for it rather than failing.

Any soft edge leaves a faint line when the subject and then the background are deleted, since an
edge pixel selected a fraction `a` keeps `a(1 − a)` of itself through both. The two changes above
took the user's test photo from 15,691 soft edge pixels to 8,947 (BiRefNet's own output has 6,950).

## Refine Hair

Refine Hair is a Select and Mask edit (`{"tool":"hair"}` in `select.refineEdge`'s `edits`), so it is
undoable, cached by the preview, and gives the same result on OK.

There is **no learned hair detector**. The selection stands in for the trimap network: its soft edge
and a band around it become the unknown region, and Shared Sampling (Gastal and Oliveira, 2010) solves
that region from the image, with no matting network. So it refines the whole edge, hair or not, and is
a button rather than part of Select Subject: it gains most on fur, hair against a contrasting
background and thin structures such as spokes, and it can change clean product edges for no gain.

Where the foreground and background colours nearly match the solve is close to arbitrary, so each
pixel carries a confidence and low-confidence pixels keep the selection they had. A small guided
filter then cleans the result, which costs a 1 px strand some of its opacity.

Refine Hair adds about 0.24 s on the six 1600–1920 px test photos.

## Measured

Release CLI, Ryzen 5 5500U with Radeon integrated graphics on DirectML, median of six photos:
6.4 s for the first click (process start and model loads included), 2.1 s for a repeat click.
CUDA, CoreML and Linux routes have not been run.

Against Photoshop 2023's own Select Subject (24.5, run by script, no Select and Mask) on the seven
test photos, Remove Background with ViTMatte matched at a mean IoU of 0.896 and SAD of 67.6
(thousands). Hair is close: the portrait scored 0.982 and the space photo 0.996. Without ViTMatte
it scored 0.895 and 66.7. ViTMatte was better on the hair portrait (SAD 30.5 vs 39.7) and thin
structures, and slightly worse on fur and the product shot. Almost all of the gap is which object
counts as the subject, not the edge. The product shot alone (0.694) is 60% of it: the counter,
grinder and machine parts. The rest comes from the dog's shadow under its belly, a leash and a bike's
hanging net, which we keep and Photoshop drops. Stretching our edge's alpha to make it firmer, as
Photoshop's is, lowered SAD by 1–3% at most.

Other subject models, scored the same way with ViTMatte on the edge (mean IoU, mean SAD):
BiRefNet Lite 0.896 / 67.6; full BiRefNet (onnx-community/BiRefNet-ONNX, Swin-L, 1024 px) 0.899 /
60.2, all of that from the product shot, at 23 s per photo on the CPU against 2 s; BiRefNet HR
0.836 / 101.6, best on hair and fur (portrait 0.985) but it lost part of the astronaut, the bike and
the product; ISNet 0.735 / 152.5. None is worth switching to, so Lite stays.

A wider set of 66 CC0 photos from Wikimedia Commons (animals, people with leashes, bags and bikes,
dark portraits, products) gave a mean IoU of 0.876 and a median of 0.951. On 45 of them the two
agree on the subject (IoU over 0.9). On the other 21, each disagreement is about which things the
subject includes, and Photoshop is neither consistently more nor less inclusive. It keeps a bird's
branches, a cat's stand and a bush. It drops a bust's pedestal, a teddy's umbrella, a saucer and
the trouser leg above a sneaker, and on a guitarist it took only the guitar. Some of its choices
look like mistakes: it drops half a bottle, a sneaker's sole and a cat's tail. No rule reproduces
those choices; they come from Adobe's model and its training data.

Tried against these scores and rejected:

- A 24 or 40 px ViTMatte band instead of 12: worse on every photo. The big dog's wet fringe, which
  Photoshop cuts between the strands, stays solid because ViTMatte keeps it too.
- BiRefNet HR's edge on Lite's subject (alone, the smaller of the two, or their mean): better
  on hair and fur by 11–27% in SAD, worse on the bike and product shot, and HR takes about 55 s a
  photo on this machine.
- Firming the edge (alpha stretched from [0.2, 1]): best on the tuning half of the 66 photos,
  0.8% better on the held-out half, and it costs faint hair.
- A cast-shadow rule: the dog's "shadow" was its wet fringe over shaded grass, not a shadow.

Photoshop's own edge finish, from the reconstruction's constants, applied to our cut-outs
(`scripts/ps-parity/finish.py`): faint values under 20% dropped to background, then its S-curve
`x^g / (k(1−x)^g + x^g)` (g = 1.3136, k ≈ 1.94). Together they lowered SAD by 0.8% on the
held-out half of the 66 photos and 2% on the 7. They improved 63 of 73 photos, most on dark
subjects against dark backgrounds (a black cat by 28%), and cost up to 8% on faint hair (a white
beard). Its guided-filter snap made things worse (1.1%), since ViTMatte has already fitted the edge.
It is now applied to ViTMatte's band (`band::finish`); the release app has not been rebuilt with it.

Against upstream v0.21.0 (2026-11-06), whose Remove Background is the classical selector with no
model (the same `segment::subject` code and edge settings as `"model": "classical"` here), on all
73 photos against Photoshop's masks: v0.21.0 mean IoU 0.522, median 0.539; this pipeline with the
finish 0.878 and 0.949. This pipeline was better on 62 photos, tied on 8 and worse on 3. 50 of 73
came out above 0.9 against 13, and v0.21.0 found no subject at all in one photo. The classical
method takes about 1.5 s a photo and runs everywhere with no download.

BiRefNet HR fails with "invalid image dimensions" on photos with a side over 2048 px (seen on
tall 1920-wide photos).

Which Select Subject mode made the Photoshop masks, device or cloud (Preferences › Image
Processing), wasn't recorded. Cloud is meant to be the more accurate, so if they came from device
mode, the "mistakes" above may be device mode's, and the masks should be made again in cloud mode.

## Not done

- Object Aware stays greyed out. It needs a model that refines inside a given mask; a salient-object
  model run on a crop can pick a different object.
- `select.refineEdge` runs a workspace Select Subject again at full size on OK, on the calling thread.
- Recorded actions don't keep which model a workspace Select Subject used.

## Credits

The model crate and download flow come from upstream PR #1199 (merged, then reverted) and the
contextual task bar from #2815. #8231, an open PR for optional local models, was studied alongside
them. Model licences are in
[`crates/ml/licenses/`](../crates/ml/licenses). The BiRefNet Lite export is
[senty-au/BiRefNet_lite-ONNX-dynamic](https://huggingface.co/senty-au/BiRefNet_lite-ONNX-dynamic)
(MIT), downloaded at a pinned revision and hash, not shipped in the repository.
