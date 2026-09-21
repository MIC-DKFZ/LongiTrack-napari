import json
import os
import threading

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")

from longitrack_napari._widget import (  # noqa: E402
    _COL_INDEX,
    _COL_REMOVE,
)


@pytest.fixture(scope="module")
def app():
    from napari.plugins import _initialize_plugins
    from qtpy.QtWidgets import QApplication

    instance = QApplication.instance() or QApplication([])
    _initialize_plugins()  # napari does this at start-up; the reader is needed to open scans
    return instance


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    # a real QMessageBox blocks forever offscreen, with nothing to click it
    from qtpy.QtWidgets import QMessageBox

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *_a, **_k: None))


def build_widget():
    # a ViewerModel has no window: the widget is driven through its ViewerModels
    from napari.components import ViewerModel

    from longitrack_napari._widget import LongiTrackWidget

    return LongiTrackWidget(ViewerModel())


def load_scans(widget):
    # two scans of the same patient: different slice counts, unrelated origins
    scale = (0.4, 0.6, 0.6)
    baseline = widget.vm["baseline"].add_image(np.zeros((200, 64, 64), np.int16), name="baseline", scale=scale)
    followup = widget.vm["followup"].add_image(np.zeros((80, 64, 64), np.int16), name="follow-up", scale=scale)
    widget._ct["baseline"] = baseline
    widget._ct["followup"] = followup
    widget._refresh_scan_labels()
    return baseline, followup


@pytest.fixture
def bare(app):
    return build_widget()


@pytest.fixture
def widget(bare):
    load_scans(bare)
    return bare


def prompt_layers(widget, baseline_points, followup_points):
    baseline = widget._prepare_baseline_points()
    baseline.data = np.asarray(baseline_points, dtype=float)
    followup = widget._points_layer("followup")
    followup.data = np.asarray(followup_points, dtype=float)
    return baseline, followup


def _write_scan(path):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.zeros((6, 8, 8), np.int16))
    image.SetSpacing((0.7, 0.7, 3.0))
    sitk.WriteImage(image, str(path))
    return path


# ------------------------------------------------------------- next scan pair -
def test_open_pair_consumes_a_loaded_pair_list_in_order(widget, tmp_path):
    bl1, fu1 = _write_scan(tmp_path / "bl1_0000.nii.gz"), _write_scan(tmp_path / "fu1_0000.nii.gz")
    bl2, fu2 = _write_scan(tmp_path / "bl2_0000.nii.gz"), _write_scan(tmp_path / "fu2_0000.nii.gz")
    widget._pair_list = [
        {"baseline_scan": str(bl1), "followup_scan": str(fu1)},
        {"baseline_scan": str(bl2), "followup_scan": str(fu2)},
    ]
    widget._pair_index = 0

    widget._on_open_pair()
    assert widget._ct["baseline"].name == "bl1_0000"
    assert widget._ct["followup"].name == "fu1_0000"
    assert widget._pair_index == 1

    widget._on_open_pair()
    assert widget._ct["baseline"].name == "bl2_0000"
    assert widget._ct["followup"].name == "fu2_0000"
    assert widget._pair_index == 2


def test_load_pair_list_resolves_relative_paths_against_the_json_files_folder(bare, tmp_path):
    _write_scan(tmp_path / "bl_0000.nii.gz")
    list_path = tmp_path / "pairs.json"
    list_path.write_text(
        json.dumps([{"baseline_scan": "bl_0000.nii.gz", "followup_scan": "/abs/fu_0000.nii.gz"}])
    )

    pairs = bare._parse_pair_list(list_path)
    assert pairs == [
        {"baseline_scan": str(tmp_path / "bl_0000.nii.gz"), "followup_scan": "/abs/fu_0000.nii.gz"}
    ]


# ------------------------------------------------------------- propagation ---
def test_an_unmoved_prompt_keeps_its_follow_up_point(widget):
    # a baseline point that has not moved since its last registration is not re-registered
    prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._anchors = [[150, 32, 32]]

    # the user appends a second baseline prompt
    widget._layer("baseline", "baseline prompt").data = np.array([[150.0, 32, 32], [160, 20, 20]])

    baseline_points = widget._baseline_points()
    assert not widget._row_needs_registration(0, baseline_points), "the corrected point should be reused"
    assert widget._row_needs_registration(1, baseline_points), "the new prompt has never been registered"


def test_focus_follows_a_newly_added_baseline_point(widget):
    baseline = widget._prepare_baseline_points()
    baseline.data = np.array([[150.0, 32, 32]])
    baseline.mode = "add"

    baseline.data = np.array([[150.0, 32, 32], [60.0, 12, 50]])

    assert widget._focus == 1


def test_a_moved_prompt_is_propagated_again(widget):
    prompt_layers(widget, [[151, 32, 32]], [[40, 30, 35]])
    widget._anchors = [[150, 32, 32]]
    assert widget._row_needs_registration(0, widget._baseline_points())


def test_a_failed_propagation_keeps_the_prompts_paired(widget):
    # dropping the failure would leave 2 baseline and 1 follow-up point
    from longitrack_backend.registration import PointPropagation

    prompt_layers(widget, [[150, 32, 32], [160, 20, 20]], [])
    points = widget._baseline_points()
    widget._on_propagated((
        [0, 1],
        [
            PointPropagation(points[0], [40, 30, 35]),
            PointPropagation(points[1], None, error="outside the image"),
        ],
    ))

    followup_points = widget._followup_points()
    assert len(followup_points) == 2
    assert followup_points[0] == [40, 30, 35]
    # the failed one falls back into the follow-up volume so it can be dragged
    assert followup_points[1] == [79, 20, 20]
    assert "could not be registered" in widget.log_view.toPlainText()


def propagate_two_prompts(widget):
    # one prompt propagated and kept, then a second one added on a far away slice
    from longitrack_backend.registration import PointPropagation

    prompt_layers(widget, [[150, 32, 32]], [])
    widget._on_propagated(([0], [PointPropagation([150, 32, 32], [40, 30, 35])]))

    widget._layer("baseline", "baseline prompt").data = np.array([[150.0, 32, 32], [60, 12, 50]])
    widget._on_propagated(([1], [PointPropagation([60, 12, 50], [15, 14, 48])]))


def test_placing_and_running_never_moves_the_baseline_point(widget):
    baseline = widget._ct["baseline"]
    prompt_layers(widget, [[150, 32, 32]], [])
    before = widget._layer("baseline", "baseline prompt").data.copy()
    world_before = baseline.data_to_world([150, 32, 32])

    widget._lock_views([150, 32, 32], [40, 30, 35])
    widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16))
    widget._focus_on("baseline", [150, 32, 32])

    assert np.array_equal(widget._layer("baseline", "baseline prompt").data, before)
    assert np.allclose(baseline.data_to_world([150, 32, 32]), world_before)
    assert np.allclose(widget._layer("baseline", "baseline prompt").translate, 0)
    assert np.allclose(baseline.translate, 0)


# ------------------------------------------------------------ point table ---
def _propagation(baseline_point, followup_point):
    from longitrack_backend.registration import PointPropagation

    return PointPropagation(baseline_point, followup_point)


def test_removing_a_row_keeps_the_rest_paired(widget):
    prompt_layers(widget, [[150, 32, 32], [60, 12, 50]], [[40, 30, 35], [15, 14, 48]])
    widget._anchors = [[150, 32, 32], [60, 12, 50]]

    widget.point_table.cellWidget(0, _COL_REMOVE).click()

    assert widget._baseline_points() == [[60, 12, 50]]
    assert widget._followup_points() == [[15, 14, 48]]
    assert widget._anchors == [[60, 12, 50]], "the anchors have to shift with the prompts"
    assert widget.point_table.rowCount() == 1
    assert widget.point_table.item(0, _COL_INDEX).text() == "1"


def test_clear_everything_closes_both_scans_and_clears_the_gpu_scan_cache(widget):
    # only the per-scan caches go; the model itself is never touched
    calls = []

    class FakeBackend:
        def is_running(self):
            return True

        def invalidate_registration(self, **kwargs):
            calls.append(kwargs)

        def release_scan(self, path):
            pass

    widget._backend = FakeBackend()
    widget._on_clear_everything()

    assert widget._ct["baseline"] is None
    assert widget._ct["followup"] is None
    assert len(widget.vm["baseline"].layers) == 0
    assert len(widget.vm["followup"].layers) == 0
    assert any(call.get("clear_scans") for call in calls), "must ask the backend to drop its scan cache"


# --------------------------------------------------------------- model switch -
def test_reinitializing_the_model_clears_the_lesion_cache(widget, tmp_path):
    widget._lesions = ["1", "2"]
    widget.export_button.setEnabled(True)

    info = {
        "folder": str(tmp_path),
        "labels": ["background", "lesion"],
        "dataset": "case",
        "device": "cpu",
        "folds": (0,),
        "patch_size": (32, 32, 32),
        "spacing": (1.0, 1.0, 1.0),
    }
    widget._on_model_loaded(info)
    assert widget._lesions == []
    assert not widget.export_button.isEnabled()


def test_initialize_can_be_queued_while_another_model_job_runs(widget):
    # the backend queues model work anyway, so the click is remembered, not refused
    widget._start_model(lambda: None, lambda result: None, "Failed", "Starting local backend")
    assert widget.load_model_button.isEnabled()

    started = []
    widget._start_model(lambda: started.append(True), lambda result: None, "Failed", "Initializing model")
    assert widget._model_queued is not None, "the second click must be queued, not dropped"
    assert "queued" in widget.log_view.toPlainText().lower()

    widget._on_model_job_done((widget._model_job_token, None))
    assert widget._model_queued is None
    assert widget._model_busy, "the queued job started on its own"


def test_a_plain_data_load_leaves_the_quality_slider_usable(widget):
    widget._start(lambda: None, lambda result: None, "Failed", "Downloading", lock_quality=False)
    assert widget.quality_slider.isEnabled()

    widget._on_job_done((widget._job_token, None))


def test_the_model_and_a_session_job_can_both_be_in_flight_at_once(widget):
    widget._start(lambda: "segmented", lambda result: None, "Failed", "Working")
    widget._start_model(lambda: {"folder": "/x", "device": "cpu"}, lambda info: None, "Failed", "Initializing model")

    assert widget._busy
    assert widget._model_busy
    assert widget.cancel_button.isEnabled()

    widget._on_job_done((widget._job_token, "segmented"))
    assert not widget._busy
    assert widget._model_busy, "the model job must not be affected by the session job finishing"
    assert widget.cancel_button.isEnabled(), "still busy overall"

    widget._on_model_job_done((widget._model_job_token, {"folder": "/x", "device": "cpu"}))
    assert not widget._model_busy
    assert not widget.cancel_button.isEnabled()


def test_on_model_loaded_does_not_wipe_lesions_while_a_session_job_is_running(widget, tmp_path):
    widget._lesions = ["1"]
    widget.export_button.setEnabled(True)
    widget._busy = True  # a session job (e.g. a segment) is in flight

    widget._on_model_loaded({"folder": str(tmp_path), "device": "cpu"})

    assert widget._lesions == ["1"], "a concurrently-running session job still owns this state"
    assert widget.export_button.isEnabled()
    widget._busy = False


# ------------------------------------------------ propagation vs. correction provenance -
def test_a_correction_survives_a_later_unrelated_propagation(widget):
    from longitrack_backend.registration import PointPropagation

    prompt_layers(widget, [[150, 32, 32]], [])
    widget._on_propagated(([0], [PointPropagation([150, 32, 32], [40, 30, 35])]))

    # the user drags the follow-up point into a corrected position
    widget._layer("followup", "follow-up prompt").data = np.array([[45.0, 31, 34]])

    # a second, unrelated lesion is added and propagated
    widget._layer("baseline", "baseline prompt").data = np.array([[150.0, 32, 32], [60, 12, 50]])
    widget._on_propagated(([1], [PointPropagation([60, 12, 50], [15, 14, 48])]))

    assert widget._followup_points()[0] == [45, 31, 34], "the correction must stick"
    assert widget._registration_proposals[0] == [40, 30, 35], "the original registration output must survive"


# --------------------------------------------------------------- accept controls -
class _FakePropagateBackend:
    def __init__(self, propagations):
        self.propagations = propagations
        self.calls: list[list] = []

    def ensure_started(self, progress=None):
        return False

    def propagate(self, baseline_path, followup_path, points, shape, **kwargs):
        self.calls.append(list(points))
        return self.propagations


def test_segment_all_is_disabled_until_something_is_accepted(widget):
    prompt_layers(widget, [[150, 32, 32]], [])
    assert not widget.segment_button.isEnabled()

    widget._set_followup_point(0, [40, 30, 35])
    widget._refresh_point_table()
    assert not widget.segment_button.isEnabled(), "registered but not yet accepted"

    widget._on_toggle_accept(0, True)
    widget._refresh_point_table()
    assert widget.segment_button.isEnabled()


def test_load_next_pair_is_disabled_until_a_pair_list_is_loaded(bare):
    assert not bare.open_pair_button.isEnabled()

    bare._pair_list = [{"baseline_scan": "a", "followup_scan": "b"}]
    bare._refresh_action_buttons()
    assert bare.open_pair_button.isEnabled()


def test_track_all_propagates_accepts_and_segments_everything(bare, tmp_path, qtbot):
    import SimpleITK as sitk
    from longitrack_backend.registration import PointPropagation

    baseline_path, followup_path = tmp_path / "bl_0000.nii.gz", tmp_path / "fu_0000.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((200, 64, 64), np.int16)), str(baseline_path))
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((80, 64, 64), np.int16)), str(followup_path))
    bare.open_scan("baseline", baseline_path)
    bare.open_scan("followup", followup_path)

    prompt_layers(bare, [[150, 32, 32]], [])

    track_calls = []

    class FakeBackend:
        def ensure_started(self, progress=None):
            return False

        def propagate(self, baseline_path, followup_path, points, shape, **kwargs):
            return [PointPropagation(points[0], [40, 30, 35])]

        def track(self, baseline_path, baseline_point, followup_path, followup_point, **kwargs):
            track_calls.append(list(followup_point))
            return {"followup": {"mask": np.zeros((80, 64, 64), np.uint8), "volume_ml": 0.0, "voxels": 0},
                    "baseline": None}

    bare._backend = FakeBackend()
    bare._backend_initialized = True
    for path in (baseline_path, followup_path):
        disk_path = str(path.absolute())
        bare._scan_upload_futures.pop(disk_path, None)
        bare._backend_scan_paths[disk_path] = disk_path

    bare._on_track_all()
    # Track chains segment off propagate's completion, so poll for the outcome
    # rather than for a fixed number of signals
    qtbot.waitUntil(lambda: any(lesion_id is not None for lesion_id in bare._lesions), timeout=5000)

    assert track_calls == [[40, 30, 35]]
    assert bare._accepted == [True]
    assert bare._lesions[0] is not None


def test_propagate_all_only_touches_rows_needing_registration(bare, tmp_path, qtbot):
    import SimpleITK as sitk
    from longitrack_backend.registration import PointPropagation

    baseline_path, followup_path = tmp_path / "bl_0000.nii.gz", tmp_path / "fu_0000.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((200, 64, 64), np.int16)), str(baseline_path))
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((80, 64, 64), np.int16)), str(followup_path))
    bare.open_scan("baseline", baseline_path)
    bare.open_scan("followup", followup_path)

    prompt_layers(bare, [[150, 32, 32], [60, 12, 50]], [])
    # row 0 is already registered and unmoved; only row 1 should go out
    bare._on_propagated(([0], [PointPropagation([150, 32, 32], [40, 30, 35])]))

    backend = _FakePropagateBackend([PointPropagation([60, 12, 50], [15, 14, 48])])
    bare._backend = backend
    bare._backend_initialized = True
    for path in (baseline_path, followup_path):
        disk_path = str(path.absolute())
        bare._scan_upload_futures.pop(disk_path, None)
        bare._backend_scan_paths[disk_path] = disk_path

    with qtbot.waitSignal(bare._bridge.done, timeout=5000):
        bare._on_propagate_all()

    assert backend.calls == [[[60, 12, 50]]]


def test_segment_all_only_touches_accepted_rows(bare, tmp_path, qtbot):
    import SimpleITK as sitk
    from longitrack_backend.registration import PointPropagation

    baseline_path, followup_path = tmp_path / "bl_0000.nii.gz", tmp_path / "fu_0000.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((200, 64, 64), np.int16)), str(baseline_path))
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((80, 64, 64), np.int16)), str(followup_path))
    bare.open_scan("baseline", baseline_path)
    bare.open_scan("followup", followup_path)

    prompt_layers(bare, [[150, 32, 32], [60, 12, 50]], [])
    bare._on_propagated((
        [0, 1],
        [PointPropagation([150, 32, 32], [40, 30, 35]), PointPropagation([60, 12, 50], [15, 14, 48])],
    ))
    bare._on_toggle_accept(1, True)  # only row 1 is accepted; row 0 is left pending

    calls = []

    class FakeBackend:
        def ensure_started(self, progress=None):
            return False

        def track(self, baseline_path, baseline_point, followup_path, followup_point, **kwargs):
            calls.append(list(baseline_point))
            return {"followup": {"mask": np.zeros((80, 64, 64), np.uint8), "volume_ml": 0.0, "voxels": 0},
                    "baseline": None}

    bare._backend = FakeBackend()
    bare._backend_initialized = True
    for path in (baseline_path, followup_path):
        disk_path = str(path.absolute())
        bare._scan_upload_futures.pop(disk_path, None)
        bare._backend_scan_paths[disk_path] = disk_path

    with qtbot.waitSignal(bare._bridge.done, timeout=5000):
        bare._on_segment_all()

    assert calls == [[60, 12, 50]]
    assert bare._lesions[0] is None
    assert bare._lesions[1] is not None


def test_segmenting_sends_the_hand_corrected_point_but_the_original_proposal(bare, tmp_path, qtbot):
    # the corrected point and the original proposal travel separately to the backend
    import SimpleITK as sitk
    from longitrack_backend.registration import PointPropagation

    baseline_path, followup_path = tmp_path / "bl_0000.nii.gz", tmp_path / "fu_0000.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((200, 64, 64), np.int16)), str(baseline_path))
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((80, 64, 64), np.int16)), str(followup_path))
    bare.open_scan("baseline", baseline_path)
    bare.open_scan("followup", followup_path)

    prompt_layers(bare, [[150, 32, 32]], [])
    bare._on_propagated(([0], [PointPropagation([150, 32, 32], [40, 30, 35])]))

    # correct the follow-up point by hand, then propagate a second, unrelated lesion
    bare._layer("followup", "follow-up prompt").data = np.array([[45.0, 31, 34]])
    bare._layer("baseline", "baseline prompt").data = np.array([[150.0, 32, 32], [60, 12, 50]])
    bare._on_propagated(([1], [PointPropagation([60, 12, 50], [15, 14, 48])]))

    calls = []

    class FakeBackend:
        def ensure_started(self, progress=None):
            pass

        def track(self, baseline_path, baseline_point, followup_path, followup_point, **kwargs):
            calls.append({"followup_point": list(followup_point), "propagated_point": kwargs["propagated_point"]})
            return {"followup": {"mask": np.zeros((80, 64, 64), np.uint8), "volume_ml": 0.0, "voxels": 0},
                    "baseline": None}

    bare._backend = FakeBackend()
    bare._backend_initialized = True
    for path in (baseline_path, followup_path):
        disk_path = str(path.absolute())
        bare._scan_upload_futures.pop(disk_path, None)
        bare._backend_scan_paths[disk_path] = disk_path

    with qtbot.waitSignal(bare._bridge.done, timeout=5000):
        bare._segment_rows([0])

    assert calls[0]["followup_point"] == [45, 31, 34]
    assert calls[0]["propagated_point"] == [40, 30, 35], "must send the original proposal, not the correction"


# ------------------------------------------------------------ locking and undo -
def test_accepting_a_row_locks_both_its_points_against_dragging(widget):
    baseline, followup = prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._on_toggle_accept(0, True)

    baseline.data = np.array([[10.0, 11, 12]])
    followup.data = np.array([[20.0, 21, 22]])

    assert widget._baseline_points() == [[150, 32, 32]]
    assert widget._followup_points() == [[40, 30, 35]]


def test_a_registered_but_unaccepted_baseline_point_stays_movable(widget):
    baseline, _followup = prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    baseline.data = np.array([[10.0, 11, 12]])
    assert widget._baseline_points() == [[10, 11, 12]]


def test_unaccepting_a_segmented_row_removes_its_segmentation(widget):
    prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._on_toggle_accept(0, True)
    name = widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16))
    widget._lesions = ["1"]
    widget._lesion_layers = {"1": {"followup": name}}
    widget.export_button.setEnabled(True)

    widget._on_toggle_accept(0, False)

    assert widget._layer("followup", name) is None
    assert widget._lesions == [None]
    assert widget._lesion_layers == {}
    assert not widget.export_button.isEnabled()
    # unaccepting unlocks the points again too
    widget._layer("baseline", "baseline prompt").data = np.array([[1.0, 2, 3]])
    assert widget._baseline_points() == [[1, 2, 3]]


def test_points_get_a_black_border_and_the_focused_one_is_highlighted(widget):
    from longitrack_napari._widget import POINTS_LAYER

    prompt_layers(widget, [[150, 32, 32], [60, 12, 50]], [[40, 30, 35], [15, 14, 48]])
    widget._set_focus(1)

    borders = np.asarray(widget._layer("followup", POINTS_LAYER["followup"]).border_color)
    assert np.allclose(borders[0], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(borders[1], [1.0, 1.0, 1.0, 1.0])


def test_segmenting_a_lesion_keeps_its_point_visible(widget):
    from longitrack_napari._widget import POINTS_LAYER

    prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._on_toggle_accept(0, True)
    name = widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16))
    widget._lesions = ["1"]
    widget._lesion_layers = {"1": {"followup": name}}
    widget._refresh_point_table()

    assert bool(widget._layer("followup", POINTS_LAYER["followup"]).shown[0])


def test_the_focused_lesions_mask_is_brighter(widget):
    prompt_layers(widget, [[150, 32, 32], [60, 12, 50]], [[40, 30, 35], [15, 14, 48]])
    name0 = widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16), suffix="1")
    name1 = widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16), suffix="2")
    widget._lesions = ["1", "2"]
    widget._lesion_layers = {"1": {"followup": name0}, "2": {"followup": name1}}

    widget._set_focus(1)

    assert widget._layer("followup", name1).opacity > widget._layer("followup", name0).opacity


def test_a_segmented_lesions_point_is_drawn_above_its_mask(widget):
    from longitrack_napari._widget import POINTS_LAYER

    prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    name = widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16))

    layers = widget.vm["followup"].layers
    points = widget._layer("followup", POINTS_LAYER["followup"])
    assert layers.index(points) > layers.index(widget._layer("followup", name)), (
        "the mask must not be drawn on top of the point, or its border becomes invisible"
    )


def test_a_segmented_points_fill_is_transparent(widget):
    from longitrack_napari._widget import POINTS_LAYER

    prompt_layers(widget, [[150, 32, 32], [60, 12, 50]], [[40, 30, 35], [15, 14, 48]])
    name = widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16))
    widget._lesions = ["1", None]
    widget._lesion_layers = {"1": {"followup": name}}
    widget._refresh_point_table()

    layer = widget._layer("followup", POINTS_LAYER["followup"])
    alphas = np.asarray(layer.face_color)[:, 3]
    assert alphas[0] == 0.0
    assert alphas[1] == 1.0


def test_mask_color_matches_the_row_regardless_of_segmentation_order(widget):
    from longitrack_napari._widget import LESION_COLORS

    prompt_layers(
        widget,
        [[150, 32, 32], [60, 12, 50], [10, 10, 10]],
        [[40, 30, 35], [15, 14, 48], [5, 5, 5]],
    )
    name_row2 = widget._add_labels("followup", np.ones((80, 64, 64), np.uint16), suffix="1", row=2)
    widget._lesions = [None, None, "1"]
    name_row0 = widget._add_labels("followup", np.ones((80, 64, 64), np.uint16), suffix="2", row=0)
    widget._lesions = ["2", None, "1"]

    color_row2 = widget._layer("followup", name_row2).colormap.color_dict[1]
    color_row0 = widget._layer("followup", name_row0).colormap.color_dict[1]
    assert np.allclose(color_row2, LESION_COLORS[2])
    assert np.allclose(color_row0, LESION_COLORS[0])


def test_delete_registration_resets_the_row_to_unregistered(widget):
    baseline, followup = prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._anchors = [[150, 32, 32]]
    widget._registration_proposals = [[40, 30, 35]]
    widget._on_toggle_accept(0, True)

    widget._on_delete_registration(0)

    assert not widget._row_has_followup(0)
    assert widget._anchors == [None]
    assert widget._registration_proposals == [None]
    assert widget._accepted == [False]
    # the baseline point (and the lesion's row) survives -- only the registration is gone
    assert widget._baseline_points() == [[150, 32, 32]]
    # and it is movable again, and re-registerable
    followup.data = np.array([[7.0, 8, 9]])
    assert widget._followup_points() == [[7, 8, 9]]


def test_removing_a_prompt_keeps_a_later_locked_rows_lock_aligned(widget):
    # row 1's lock must shift down with its data when row 0 goes
    baseline, followup = prompt_layers(widget, [[150, 32, 32], [60, 12, 50]], [[40, 30, 35], [15, 14, 48]])
    widget._on_toggle_accept(1, True)

    widget._on_remove_prompt(0)

    assert widget._baseline_points() == [[60, 12, 50]]
    assert widget._followup_points() == [[15, 14, 48]]
    # the row that is now index 0 is still locked (it used to be row 1)
    baseline.data = np.array([[1.0, 2, 3]])
    followup.data = np.array([[4.0, 5, 6]])
    assert widget._baseline_points() == [[60, 12, 50]]
    assert widget._followup_points() == [[15, 14, 48]]


def test_a_cancelled_jobs_late_result_is_dropped(widget):
    calls = []
    widget._start(lambda: None, lambda result: calls.append(result), "Failed", "Working")
    token = widget._job_token
    widget._on_cancel()

    # the job's own thread has no idea it was cancelled and reports success anyway
    widget._on_job_done((token, "late result"))
    assert calls == [], "a cancelled job's result must never reach on_return"
    assert not widget._busy


def test_cancelling_track_all_does_not_contaminate_a_later_propagate(widget):
    # cancelling before Track's propagate callback runs must not leave its flag stuck,
    # or a later Propagate would continue into accept-and-segment on its own
    from longitrack_backend.registration import PointPropagation

    prompt_layers(widget, [[150, 32, 32]], [])
    widget._track_all_pending = True
    widget._busy = True

    widget._on_cancel()
    assert not widget._track_all_pending

    segment_all_calls = []
    widget._on_segment_all = lambda: segment_all_calls.append(True)
    widget._on_propagated(([0], [PointPropagation([150, 32, 32], [40, 30, 35])]))

    assert widget._accepted == [False], "a plain propagate must not auto-accept"
    assert segment_all_calls == []


# --------------------------------------------- cancel restarts a wedged backend -
class _FakeBackendWithActiveRequest:
    # a started GPU call is not interruptible in place -- see BackendClient.kill()
    def __init__(self):
        self.killed = threading.Event()
        self.cancel_active_called = False

    def has_active_request(self):
        return True

    def kill(self):
        self.killed.set()

    def cancel_active(self):
        self.cancel_active_called = True


class _FakeBackendIdle:
    # nothing is actually in flight against it right now
    def __init__(self):
        self.killed = False
        self.cancel_active_called = False

    def has_active_request(self):
        return False

    def kill(self):
        self.killed = True

    def cancel_active(self):
        self.cancel_active_called = True



def test_editing_a_proposal_moves_the_point_to_the_clicked_position(widget):
    from longitrack_napari._widget import POINTS_LAYER

    prompt_layers(widget, [[150, 32, 32], [60, 12, 50]], [[40, 30, 35], [15, 14, 48]])
    widget._begin_verification([0, 1])
    widget._on_verify_edit()

    layer = widget._layer("followup", POINTS_LAYER["followup"])
    assert layer.mode == "add"
    assert not widget.verify_edit_button.isEnabled()

    layer.data = np.vstack([np.asarray(layer.data), [[20.0, 25, 25]]])  # what a click does

    assert len(layer.data) == 2
    assert widget._followup_points()[0] == [20, 25, 25]
    widget._on_verify_accept()
    assert widget._accepted[0]
    assert widget._followup_points()[0] == [20, 25, 25]


def test_editing_an_already_accepted_point_actually_sticks(widget):
    from longitrack_napari._widget import POINTS_LAYER

    prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._on_toggle_accept(0, True)
    widget._on_table_clicked(0)  # reopens Accept/Edit/Skip for the accepted row
    widget._on_verify_edit()

    layer = widget._layer("followup", POINTS_LAYER["followup"])
    layer.data = np.vstack([np.asarray(layer.data), [[20.0, 25, 25]]])  # what a click does

    assert widget._followup_points()[0] == [20, 25, 25], "must not snap back to the locked point"


def test_clicking_a_row_reopens_verification_for_an_accepted_point(widget):
    prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._on_toggle_accept(0, True)
    assert widget._verify_rows == []

    widget._on_table_clicked(0)

    assert widget._verify_rows == [0]
    assert widget.verify_edit_button.isEnabled()
    assert "accepted" in widget.verify_label.text()


def test_clicking_a_segmented_row_does_not_reopen_verification(widget):
    prompt_layers(widget, [[150, 32, 32]], [[40, 30, 35]])
    widget._on_toggle_accept(0, True)
    name = widget._add_labels("followup", np.zeros((80, 64, 64), np.uint16))
    widget._lesions = ["1"]
    widget._lesion_layers = {"1": {"followup": name}}

    widget._on_table_clicked(0)

    assert widget._verify_rows == [], "nothing left to accept, edit or skip once it's segmented"


def test_clicking_a_different_row_keeps_the_pending_ones_queued(widget):
    prompt_layers(widget, [[150, 32, 32], [60, 12, 50], [10, 10, 10]], [[40, 30, 35], [15, 14, 48], [5, 5, 5]])
    widget._begin_verification([0, 1])
    widget._on_toggle_accept(2, True)  # row 2 is done and not part of the pending walk

    widget._on_table_clicked(2)
    assert widget._verify_rows == [2, 0, 1]

    widget._on_verify_skip()
    assert widget._verify_rows == [0, 1], "the original walk resumes once the detour is dismissed"



# ------------------------------------------------- switching the backend mode -
class _FakeRunningBackend:
    """A backend that is up, in one mode or the other."""

    def __init__(self, remote: bool):
        self.is_remote = remote

    def is_running(self) -> bool:
        return True


def test_selecting_remote_while_local_runs_opens_the_connect_dialog_once(bare, monkeypatch):
    # regression: _set_backend_mode used to re-enter the handler, so confirming the
    # dialog re-opened it, and cancelling out of the loop dropped back to Local
    bare._backend = _FakeRunningBackend(remote=False)
    calls = []
    monkeypatch.setattr(type(bare), "_on_connect_remote_server", lambda _self, **kw: calls.append(kw))

    # drive it the way the user does, through the combo box
    bare.backend_mode.setCurrentIndex(1)
    bare.backend_mode.activated.emit(1)

    assert calls == [{"switching": True}]


def test_showing_a_decided_mode_does_not_start_a_switch(bare, monkeypatch):
    bare._backend = _FakeRunningBackend(remote=False)
    monkeypatch.setattr(
        type(bare), "_on_connect_remote_server", lambda _self, **_kw: pytest.fail("must not reconnect")
    )

    bare._set_backend_mode(True)

    assert bare._remote_mode()
    assert bare.remote_endpoint_widget.isVisibleTo(bare)


def test_selecting_remote_with_no_backend_running_still_offers_the_connect_dialog(bare, monkeypatch):
    calls = []
    monkeypatch.setattr(type(bare), "_on_connect_remote_server", lambda _self, **kw: calls.append(kw))

    bare._on_backend_mode_selected(1)

    assert calls == [{"switching": False}]


def test_selecting_local_without_a_local_backend_reverts_to_remote(bare, monkeypatch):
    bare._backend = _FakeRunningBackend(remote=True)
    monkeypatch.setattr(type(bare), "_local_backend_available", staticmethod(lambda: False))
    monkeypatch.setattr(
        type(bare), "_start_local_backend_warmup", lambda _self, **_kw: pytest.fail("no local backend to start")
    )
    errors = []
    monkeypatch.setattr(type(bare), "_error", lambda _self, title, error: errors.append(title))

    bare._on_backend_mode_selected(0)

    assert errors == ["No local backend"]
    assert bare._remote_mode()
