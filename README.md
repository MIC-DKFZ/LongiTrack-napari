# LongiTrack-napari

An interactive napari viewer for **point-prompted longitudinal lesion tracking** with
[LongiSeg](https://github.com/MIC-DKFZ/LongiSeg).

[![arXiv](https://img.shields.io/badge/arXiv-2605.23118-b31b1b.svg)](https://arxiv.org/abs/2605.23118)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Click a lesion in a baseline scan. The point is propagated to the follow-up scan by
[uniGradICON](https://github.com/uncbiag/uniGradICON) registration, you accept it or drag it
somewhere better, and the LongiSeg tracking network segments that lesion in the follow-up
scan -- and, from the same prompt, in the baseline scan too.

See [documentation/workflow.md](documentation/workflow.md) for what happens on a click.

## Quick start

```bash
git clone https://github.com/MIC-DKFZ/LongiTrack-napari.git
cd LongiTrack-napari
uv sync
uv run longitrack-napari
```

Then **Plugins -> LongiSeg Tracking**. On first open it asks whether the model should run on a
local or a remote backend; picking local starts it straight away and warms the registration
network in the background.

First run downloads the tracking weights into `~/.cache/huggingface` and the
uniGradICON weights into `~/.cache/longitrack`. Both are cached.

No data at hand? **Example** fetches one pair with five annotated lesions
from the [PanTrack](https://huggingface.co/datasets/mrokuss/PanTrack) dataset. To fetch a
different pair up front:

```bash
uv run longitrack-model sample --list
uv run longitrack-model sample -p PanTrack_009 -i 0
```

## The workflow

| Step | What you do | What happens |
| --- | --- | --- |
| 1 | *Initialize model* | Loads the weights and warms them. Once per session; clicking it while the backend is still starting just queues it. |
| 2 | Open a baseline and a follow-up scan, or press *Example* | Both scans are prepared for registration and segmentation in the background. |
| 3 | *Set baseline points*, then click each lesion | Each point is one tracked lesion and gets a row in the list. Click the button again to stop adding. |
| 4 | *Propagate* | uniGradICON registers the pair and maps every not-yet-registered point across. |
| 5 | Verify each proposal: **Accept**, **Edit** (click the right spot, then *Accept*) or **Skip** | The viewer steps through the lesions one at a time and locks each point you accept. |
| 6 | *Segment* | The network segments every accepted lesion, in both scans. |

Step 1 is optional -- *Propagate* and *Segment* load the model themselves if needed. *Track*
does steps 4-6 in one go, accepting every proposal without asking.

The bin on a scan row closes that scan and everything from it; *Clear everything* resets to
just after *Initialize model*. *Cancel* unlocks the panel immediately; a step already on the GPU finishes by itself and its
result is dropped.

### Annotating several scan pairs

*Next pair...* replaces both scans at once. *Pair list...* takes a JSON file and
steps through it, one pair per click, with no dialogs:

```json
[
  {"baseline_scan": "patient_001/bl.nii.gz", "followup_scan": "patient_001/fu.nii.gz"},
  {"baseline_scan": "patient_002/bl.nii.gz", "followup_scan": "patient_002/fu.nii.gz"}
]
```

Relative paths resolve against the list file's folder.

### Several lesions

Every baseline point is one tracked lesion with its own labels layer per timepoint. A point that
has not moved keeps what it already has -- its follow-up point (including a correction you
dragged in) and its segmentation -- so adding a fifth lesion costs one lesion's work, not five.
Moving a baseline point only makes that row need registering again.

### The lesion list

One row per lesion: its number in the colour it is drawn in, where it stands (*baseline prompt set*,
*proposed*, *verified*, or its volume in ml once segmented), an eye to show or hide it, and a bin.
Hover a row for the actual coordinates; clicking one lines both canvases up on that lesion.

Right-click a row for the rest: accept it, change its colour, propagate that one point again, undo
its verification, or remove it.

**Accepting locks a lesion.** Both its points are frozen afterwards -- dragging snaps back -- so an
accepted pair cannot drift out of sync with what was segmented from it. Undoing the verification
unlocks them and drops that lesion's segmentation. Anything not yet accepted stays fully movable,
baseline included.

### Options

**Speed/Quality slider** -- *Fastest* (no refinement, no TTA) through *Best* (50 refinement steps,
TTA). Refinement steps are uniGradICON instance optimisation; TTA is mirroring in the
segmentation. Fastest is interactive; Best spends considerably longer on the registration.

The device is picked automatically, every fold in the model folder is used, and the baseline
lesion is always segmented alongside the follow-up one -- the network works on a pair, so the
baseline mask comes from the same run with the prompt on both sides.

## Two scans, two canvases

The plugin owns two `ViewerModel`s, one per timepoint, side by side; napari's own canvas and
layer list are hidden. Each canvas has its own camera, so zooming one scan leaves the other
alone, and its own slider and contrast controls.

The two scans share no physical frame, so slice *n* on the left and slice *n* on the right are
unrelated anatomy. When a pair is propagated (or you click its row) the world-mm offset between
the two points is recorded and the follow-up slider follows the baseline one through it. The
offset is display only -- coordinates handed to the network are untouched.

### Exporting

*Export segmentations...* asks for a folder and writes

```
<scan-identifier>_bl_<lesion>.nii.gz   one binary mask per lesion per timepoint
<scan-identifier>_fu_<lesion>.nii.gz
inference_meta.json                    the prompts behind them
```

`<scan-identifier>` is the scan's filename without its nnU-Net channel suffix (`_0000`). Masks
keep the scan's spacing, origin and direction. `inference_meta.json` is a list, one entry per
pair, each `{"baseline_scan": ..., "followup_scan": ..., "lesions": {"<id>": {"bl_point": ...,
"fu_point_corrected": ..., "fu_point_propagated": ...}}}`.

Exporting several pairs into one folder is the expected way to work through a batch: each call
merges its pair into `inference_meta.json` instead of overwriting it. Export refuses to
overwrite an existing mask file and names every collision instead.

## Where the model comes from

A LongiSeg model folder (`dataset.json`, `plans.json`, `fold_*/checkpoint_final.pth`), either on
the Hugging Face Hub or on disk. The *Location* field decides, and is filled at start-up from
`$LONGITRACK_MODEL_DIR` if set, otherwise `$LONGITRACK_HF_REPO` or the default
`MIC-DKFZ/LongiSeg-Tracking`. A path that exists is read as a folder; anything else must be a
repo id and is downloaded.

```bash
uv run longitrack-model info                       # what is in the resolved model?
uv run longitrack-model download --folds 0         # pull one fold
uv run longitrack-model upload /path/to/folder -r owner/name
```

`upload` pushes only `dataset.json`, `plans.json` and each fold's final checkpoint.

## Remote GPU backend

Answer **Remote** when the plugin asks where the model should run, and it prompts for the
endpoint and your Ed25519 private key right away. The viewer stays local; model loading,
preprocessing, registration, segmentation and export run on the server, and your scans are
uploaded to it. A local backend reads your files in place and copies nothing.

## From code

```python
from longitrack_napari.app import launch

viewer, widget = launch("baseline.nii.gz", "followup.nii.gz", block=True)
```

`widget.open_scan("baseline", path)` swaps a scan later. Without the GUI,
`longitrack_napari.inference.TrackingEngine` runs the same pipeline and takes the arguments the
widget does not expose (`device`, `disable_tta`, `folds`).

## Coordinate conventions

`longitrack_napari.geometry` is the only place that converts between them:

- **index** -- `(z, y, x)` as `SimpleITKIO` reads it. What napari shows, and this package's interface.
- **xyz** -- ITK voxel order, used by LongiSeg's tracking json and `PairRegistration.propagate`.
- **preprocessed** -- after transpose, crop and resample to the plan's spacing. What the network sees.

Scans must come from files on disk: an array pasted into a layer has no spacing, origin or path
to register against. A scan opened through **File -> Open** is moved into whichever panel is free.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Citing

This viewer is the front end for **Exploiting Longitudinal Context in Clinician-Verified
Interactive Lesion Tracking** ([arXiv:2605.23118](https://arxiv.org/abs/2605.23118)). Please cite it
together with [LongiSeg](https://doi.org/10.1007/978-3-031-72069-7_7) for the segmentation
framework and [uniGradICON](https://arxiv.org/abs/2403.05780) for the registration.

```bibtex
@article{kirchhoff2026longitrack,
  title   = {Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking},
  author  = {Kirchhoff, Yannick and Rokuss, Maximilian and Mertens, Daniel Philipp and
             F{\"u}ller, David and Hamm, Benjamin and Schreyer, Andreas and Ritter, Oliver and
             Maier-Hein, Klaus},
  journal = {arXiv preprint arXiv:2605.23118},
  year    = {2026}
}

@inproceedings{rokuss2024longitudinal,
  title     = {Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting},
  author    = {Rokuss, Maximilian R. and Kirchhoff, Yannick and Roy, Saikat and Kovacs, Balint and
               Ulrich, Constantin and Wald, Tassilo and Zenk, Maximilian and Denner, Stefan and
               Isensee, Fabian and Vollmuth, Philipp and Kleesiek, Jens and Maier-Hein, Klaus},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages     = {64--74},
  year      = {2024},
  publisher = {Springer}
}

@inproceedings{tian2024unigradicon,
  title     = {uniGradICON: A Foundation Model for Medical Image Registration},
  author    = {Tian, Lin and Greer, Hastings and Kwitt, Roland and Vialard, Fran{\c{c}}ois-Xavier and
               San Jos{\'e} Est{\'e}par, Ra{\'u}l and Bouix, Sylvain and Rushmore, Richard and
               Niethammer, Marc},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages     = {749--760},
  year      = {2024},
  publisher = {Springer}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
