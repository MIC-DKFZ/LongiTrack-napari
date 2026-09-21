# What happens on a click

```
 baseline.nii.gz                                     followup.nii.gz
        |                                                   |
        |  click (z, y, x)                                  |
        v                                                   |
  baseline point ------- uniGradICON registration --------> propagated point
        |            (Speed/Quality slider: refinement steps)      |
        |                                                   [ accept / drag ]
        |                                                          |
        |                                                          v
        |                                                  follow-up point
        |                                                          |
        v                                                          |
  +-----------------------------+                                  |
  | baseline duplicated as its  |                                  |
  | own follow-up               |                                  |
  |   bl_data = baseline        |                                  |
  |   fu_data = baseline        |                                  |
  |   both points = the click   |                                  |
  |   bl_seg   = empty          |                                  |
  +-----------------------------+                                  |
        |                                                          |
        v                                                          |
  baseline lesion mask                                             |
        |                                                          |
        |                                                          v
        |                                            +-----------------------------+
        |                                            | tracking forward pass       |
        |                                            |   bl_data = baseline        |
        |                                            |   fu_data = follow-up       |
        |                                            |   bl_seg  = empty           |
        |                                            |   points  = both prompts    |
        |                                            +-----------------------------+
        |                                                          |
        v                                                          v
   labels layer                                              labels layer
  "baseline lesion"                                       "follow-up lesion"
                                \                    /
                                 volume, and its change
```

The network input is a single patch, `patch_size` centred on the follow-up point --
shifted just enough to stay inside the volume when the point sits near an edge -- with
the baseline patch taken at the *same offset* from the baseline point, so the two
patches are aligned on the lesion. Five channels go in, in this order: follow-up
image, baseline image, baseline lesion mask, follow-up point as a Gaussian blob,
baseline point as a Gaussian blob. This is
`longitrack_backend.inference._predict_cached_patch` -- a small adapter around
LongiSeg's own `tracking_inference.predict_patch` geometry, fold loop, and predictor
calls, kept local (not a change to the upstream helper itself) because the upstream
version builds its prompt maps on CPU, which fails once the cached scan tensors it
concatenates against already live on CUDA (see below).

The baseline lesion mask channel is empty in both predictions -- filled with zeros,
the same way `LongiSeg_predict_tracking` runs without `--labels_path`. Neither
prediction is an input to the other, so the order of the two is free; both run one
after the other on the same backend process (a separate OS process from the GUI --
see below).

Every fold the model folder ships contributes: each fold's weights are loaded in turn
and the logits are summed before the softmax. Test-time augmentation (mirroring) is
controlled by the panel's Speed/Quality slider, not always on -- only "Fastest" disables it.
The four stops above it ("Fast", "Balanced", "Detailed", "Best") all enable it, and differ in
how many uniGradICON refinement steps the registration gets.

Only the small patch actually predicted is resampled back to the scan's native
resolution (`longitrack_backend.inference.TrackingEngine._restore_patch`), not the whole volume -- nnU-Net's
own export path resamples the *entire* probability volume even though everything outside
the patch is known background, which dominates wall-clock time on a full-body scan for no
benefit. The resampled patch is placed at its own bounding box and shown as its own
labels layer, one per lesion, in that scan's canvas.

# Backend/frontend split

None of the above runs in the napari/Qt process, and none of it lives in this package.
Everything that touches torch is the separate `longitrack-backend` distribution; the plugin
depends on it but the GUI never imports it. `longitrack_napari._widget` only knows
`longitrack_napari.backend.client.BackendClient`, which starts (or connects to)
`longitrack_backend.server` as a separate OS process and talks to it over a length-prefixed
JSON+binary-frame protocol (`longitrack_backend.protocol`) on a Unix domain socket locally, or a
TCP socket for a backend running elsewhere. This is deliberate: PyTorch/CUDA calls holding the GIL on a
background thread were starving the Qt event loop on the same interpreter, which is
what made the UI feel frozen during a forward pass before the split. Segmentation
masks cross that connection run-length encoded (falling back to raw only if RLE would
be larger), since a lesion mask is almost entirely one background value.

## One thread for all CUDA work

Every request in `longitrack_backend.server._GPU_REQUESTS` runs on `_GpuWorker`'s single thread. A thread's first
CUDA forward pass pays a per-thread initialization that every new thread pays again, and the
server answers each request on a fresh connection thread. One thread pays it once, during
`initialize`. Uploads, sessions and export stay off it.

## Two independent job channels

The GUI dispatches on two separate channels -- `_pool`/`_busy` for anything touching the scans,
points or lesions, and `_model_pool`/`_model_busy` for model loading -- so neither blocks the
other, and a second model click queues instead of being refused.

## Cancel

Cancel closes the client socket so the panel frees immediately; the call already on the GPU
runs itself out and its result is dropped.

## Switching backends

The *Backend* dropdown is the only thing that changes mode, and only when the user picks in it --
`activated`, not `currentIndexChanged`, so the code paths that merely *display* a mode
(`_set_backend_mode`, the `$LONGITRACK_REMOTE_BACKEND` pre-fill, reverting a cancelled dialog)
cannot re-enter the switch. `_on_backend_mode_selected` then closes the old client and opens the
new one through `_replace_backend`, which drops the cached scans, uploads and model state with it:
nothing is shared between a local and a remote backend.

# Verified tracking, one lesion at a time

The panel follows the paper's Verified Tracking loop: registration proposes a follow-up
point, the clinician accepts or corrects it, and only then does the network segment.

*Propagate* registers every baseline point that needs it and then walks the results, one
lesion at a time: the two canvases centre on it and the bar under the list offers
**Accept**, **Edit** (click the right spot, then *Accept*) and **Skip**. Accepting locks
both of that lesion's points (`_accept_row` snapshots them; `_enforce_locked_points`
reverts any later drag), so what gets segmented is exactly what was verified. *Segment*
then runs every accepted lesion, and *Track* does the whole thing without asking.

Clicking a row (`_on_table_clicked`) reopens this same bar for that one lesion --
`_jump_to_verification` moves it to the front of `_verify_rows` without dropping whatever
else was still pending, so a detour to fix an already-accepted point resumes the original
walk afterwards. Edit stays available post-accept (`_apply_click_edit` bypasses the lock
directly), but is disabled once the lesion is segmented; the segmentation itself is the
source of truth at that point, and "Undo verification" is the sanctioned way back.

# Moving between scan pairs

"Load next pair" (`_widget._on_open_pair`) replaces both scans at once and resets
the session, same as replacing either scan on its own does -- the normal way to move on
to the next case. With no pair list loaded it just asks for the two scans via two file
dialogs, aborting untouched if either is cancelled. "Upload pair list" parses a JSON
file of `{"baseline_scan": path, "followup_scan": path}` entries (`_parse_pair_list`,
relative paths resolved against the list file's own folder) into `self._pair_list`;
while `self._pair_index` has not reached the end of it, "Load next pair" opens
that entry instead of prompting, so annotating a whole batch needs no dialogs after the
list is loaded.
