# LongiTrack-napari

An interactive [napari](https://napari.org) viewer for **point-prompted longitudinal lesion
tracking** with [LongiSeg](https://github.com/MIC-DKFZ/LongiSeg), backed by
[LongiTrack-backend](https://github.com/MIC-DKFZ/LongiTrack-backend).

[![arXiv](https://img.shields.io/badge/arXiv-2605.23118-b31b1b.svg)](https://arxiv.org/abs/2605.23118)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Click a lesion in a baseline scan. The point is propagated to the follow-up scan by
[uniGradICON](https://github.com/uncbiag/uniGradICON) registration, you accept it or drag it
somewhere better, and the LongiSeg tracking network segments that lesion in the follow-up
scan -- and, from the same prompt, in the baseline scan too.

See [documentation/workflow.md](documentation/workflow.md) for what happens on a click.

## Choose where the backend runs

The plugin has two modes:

- **Local backend — Linux only.** Runs the GPU model process on the same machine as napari. Use the full install.
- **Remote backend — macOS and Linux.** Keeps napari lightweight and runs model work on a separate Linux GPU server.
  Use the remote-only install; scans are uploaded to the server for processing.

A full install can do either and switches between them in the plugin; see
[Using a remote GPU backend](#using-a-remote-gpu-backend).

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the plugin once:

```bash
git clone https://github.com/MIC-DKFZ/LongiTrack-napari.git
cd LongiTrack-napari
```

### Local installation (Linux only)

```bash
uv sync
uv run longitrack-napari
```

### Remote-only installation (macOS and Linux)

```bash
uv sync --no-default-groups
uv run longitrack-napari
```

### Opening the plugin

`longitrack-napari` starts napari with the panel already docked, optionally on a pair of scans:

```bash
uv run longitrack-napari --baseline patient_001/bl.nii.gz --followup patient_001/fu.nii.gz
```

In a napari you started yourself, the panel is **Plugins -> LongiSeg Tracking**, and the pair
is also available as a sample under **File -> Open Sample**.


## The workflow

| Step | What you do | What happens |
| --- | --- | --- |
| 1 | *Initialize model* | Loads the weights and warms them. Once per session; clicking it while the backend is still starting just queues it. |
| 2 | Open a baseline and a follow-up scan, or press *Load PanTrack example* | Both scans are prepared for registration and segmentation in the background. |
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

Two buttons in the *Scans* box move you through a batch:

- ***Load next pair*** replaces both scans at once and resets the session, the same as
  replacing either scan on its own. On its own it asks for the two scans in two file
  dialogs, and leaves everything untouched if you cancel either one.
- ***Upload pair list*** picks a JSON file listing the pairs up front. *Load next pair* then
  stops asking and just opens the next entry, so a whole batch needs no file dialogs at all.
  The button is disabled until a list is loaded, and goes quiet again at the end of the list.

The list is a JSON array, one object per pair:

```json
[
  {"baseline_scan": "patient_001/bl.nii.gz", "followup_scan": "patient_001/fu.nii.gz"},
  {"baseline_scan": "patient_002/bl.nii.gz", "followup_scan": "patient_002/fu.nii.gz"}
]
```

Relative paths resolve against the list file's own folder, so a list can sit next to the data
it points at and the whole folder stays movable.

The usual batch loop is: *Upload pair list* once, then for each pair *Set baseline points* ->
*Propagate* -> verify -> *Segment* -> *Export segmentations...* into the same folder ->
*Load next pair*. Exporting every pair into one folder is what the export format is built for
(see [Exporting](#exporting)).

### Several lesions

Every baseline point is one tracked lesion with its own labels layer per timepoint. A point that
has not moved keeps what it already has -- its follow-up point (including a correction you
dragged in) and its segmentation -- so adding a fifth lesion costs one lesion's work, not five.
Moving a baseline point only makes that row need registering again.

### The lesion list

One row per lesion: its number in the colour it is drawn in, where it stands (*baseline prompt set*,
*proposed*, *verified*, or its volume in ml once segmented), an eye to show or hide it, and a bin.
Hover a row for the actual coordinates; clicking one lines both canvases up on that lesion and
reopens Accept/Edit/Skip for it, even if it was already accepted -- handy for correcting a point
without waiting for its turn in a fresh walk. A segmented lesion can still be reopened, but Edit
is greyed out; undo its verification first if the point itself needs to change.

Right-click a row for the rest: accept it, change its colour, propagate that one point again, undo
its verification, or remove it.

**Accepting locks a lesion.** Both its points are frozen afterwards -- dragging snaps back -- so an
accepted pair cannot drift out of sync with what was segmented from it. Undoing the verification
unlocks them and drops that lesion's segmentation. Anything not yet accepted stays fully movable,
baseline included.

### Options

**Speed/Quality slider** -- five stops, starting on *Balanced*:

| Stop | Registration refinement | Segmentation TTA |
| --- | --- | --- |
| Fastest | none | off |
| Fast | none | on |
| Balanced *(default)* | 10 steps | on |
| Detailed | 25 steps | on |
| Best | 50 steps | on |

Refinement steps are uniGradICON instance optimisation; TTA is mirroring in the segmentation.
*Fastest* is interactive; *Best* spends considerably longer on the registration. Moving the
slider drops cached results, so the next *Propagate* or *Segment* redoes the work.

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

Scans must come from files on disk: an array pasted into a layer has no spacing, origin or path
to register against. A scan opened through **File -> Open** is moved into whichever panel is free.

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
the Hugging Face Hub or on disk. The *Source* dropdown and the *Location* field next to it decide
which: a path that exists is read as a folder, anything else must be a repo id and is downloaded
on first use. *Initialize model* is what acts on them.

If several checkpoints are available (published as their own `LongiTrack_v<version>` folder), the
newest is used by default; append `@1.0` to the *Location* to pin an older one instead.

Fields the plugin fills for you at start-up, so a lab can preset them once:

| Environment variable | What it pre-fills |
| --- | --- |
| `$LONGITRACK_MODEL_DIR` | *Location*, switched to *Local folder* |
| `$LONGITRACK_HF_REPO` | *Location*, as a Hub repo id. Default `ykirchhoff/LongiTrack` |
| `$LONGITRACK_REMOTE_BACKEND` | the remote server address, and opens in *Remote* mode |

Managing model folders outside the plugin -- inspecting, downloading, publishing -- is the
backend's job; see the [LongiTrack-backend README](https://github.com/MIC-DKFZ/LongiTrack-backend).

## Using a remote GPU backend

The model work can run on a separate Linux GPU server instead of your own machine. The viewer
uploads the scans; model loading, registration, segmentation and export all happen there.

Someone has to be running a `LongiTrack-backend` server for you first, and has to have added your
SSH account to its `longitrack` group -- both are covered in the
[LongiTrack-backend README](https://github.com/MIC-DKFZ/LongiTrack-backend). Once they have, ask
them for the server's address. Then, on your own machine:

1. Create an identity and register it with the server, once per server:

   ```bash
   uv run create_and_push_remote_id --name gpu-cluster --ssh you@gpu-server --owner yourname
   ```

   The private key stays with you, under `~/.config/longitrack/remote/`; only the public
   half goes to the server. `--name` is what you will pick in the plugin, `--owner` is you, and it
   is what the server records next to the key.

2. In the *Model* box, set **Backend** to *Remote TCP server*. A dialog asks for the server
   address and which identity to use; *Connect* authenticates and switches over. The address
   field takes a plain host name, or the whole `CONNECT host:8765` line the server prints.

3. *Test connection* next to the address checks that a server is reachable there without
   connecting to it -- useful when a connection attempt failed and you want to know which half
   is at fault.

The **Backend** dropdown switches modes at any time, not just at start-up. Switching closes the
current backend and prepares the scans and model again on the new one, so anything already
segmented is dropped -- it does not travel between backends.

On a full install the plugin asks once, at start-up, whether to run local or remote. A remote-only
install skips the question and opens the connect dialog directly, and its *Local process* entry
will tell you it has no GPU backend to start.

Remote TCP is authenticated by public key but **not encrypted**. Use a trusted network, or an SSH
tunnel to a loopback-only server.


## Development

```bash
uv sync --extra dev            # add --no-default-groups off Linux: the GPU group is Linux only
uv run pytest
uv run ruff check .
```

The tests that need real weights skip themselves unless `$LONGITRACK_MODEL_DIR` points at a
LongiSeg tracking model folder; everything else runs without a GPU.

## Citing

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

@misc{napari2019,
  title  = {napari: a multi-dimensional image viewer for Python},
  author = {{napari contributors}},
  year   = {2019},
  doi    = {10.5281/zenodo.3555620}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
