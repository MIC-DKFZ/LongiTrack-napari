from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image, Points
from qtpy.QtCore import QObject, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ._reader import SUPPORTED_SUFFIXES
from .geometry import snap_to_volume
from .model import DEFAULT_REPO_ID
from .pantrack import DEFAULT_PAIR_INDEX as PANTRACK_PAIR
from .pantrack import DEFAULT_PATIENT as PANTRACK_PATIENT

BASELINE_POINTS = "baseline prompt"
FOLLOWUP_POINTS = "follow-up prompt"
BASELINE_SEG = "baseline lesion"
FOLLOWUP_SEG = "follow-up lesion"
# temporary overlay shown in the follow-up panel while a registration runs
BLEND_LAYER = "registering"

ROLES = ("baseline", "followup")
REMOTE_BACKEND_PORT = 8765
ROLE_WORDS = {"baseline": "baseline", "followup": "follow-up"}
POINTS_LAYER = {"baseline": BASELINE_POINTS, "followup": FOLLOWUP_POINTS}
SEG_LAYER = {"baseline": BASELINE_SEG, "followup": FOLLOWUP_SEG}
# A bright, color-blind-friendly categorical sequence (Okabe–Ito).
LESION_COLORS = np.asarray(
    [
        (0.0000, 0.4471, 0.6980, 1.0),  # blue
        (0.8353, 0.3686, 0.0000, 1.0),  # vermillion
        (0.0000, 0.6196, 0.4510, 1.0),  # bluish green
        (0.9412, 0.8941, 0.2588, 1.0),  # yellow
        (0.3373, 0.7059, 0.9137, 1.0),  # sky blue
        (0.8000, 0.4745, 0.6549, 1.0),  # reddish purple
        (0.9020, 0.6235, 0.0000, 1.0),  # orange
        (0.3000, 0.3000, 0.3000, 1.0),  # charcoal
    ],
    dtype=float,
)

# Slider positions: only registration refinement varies, TTA is always on.
QUALITY_MODES = (
    ("Fastest", None, True, "no refinement, no TTA"),
    ("Fast", None, False, "no refinement, TTA"),
    ("Balanced", 10, False, "10 refinement steps, TTA"),
    ("Detailed", 25, False, "25 refinement steps, TTA"),
    ("Best", 50, False, "50 refinement steps, TTA"),
)

# how many slider steps above and below the current slice still draw a point.
SLICE_MARGIN = 4

# point_table columns: (header label, header tooltip).
_COL_INDEX = 0
_COL_STATUS, _COL_VISIBILITY, _COL_REMOVE = 1, 2, 3
_TABLE_COLUMNS = (
    ("#", "Lesion number, in the colour it is drawn in."),
    ("", "Where this lesion stands: proposed, verified, or its segmented volume."),
    ("\N{EYE}", "Show or hide this lesion."),
    ("", ""),
)

_FILE_FILTER = "Medical volumes (" + " ".join(f"*{suffix}" for suffix in SUPPORTED_SUFFIXES) + ");;All files (*)"


class _Bridge(QObject):
    # everything crossing back from the worker thread to the GUI thread.
    message = Signal(str)
    done = Signal(object)
    failed = Signal(object)
    model_done = Signal(object)
    model_failed = Signal(object)


class _DaemonWorker:
    # a daemon thread: shutdown never has to wait for it
    def __init__(self, name: str) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            fn, args = self._queue.get()
            fn(*args)

    def submit(self, fn, *args) -> None:
        self._queue.put((fn, args))

    def shutdown(self, wait: bool = False) -> None:
        pass  # nothing to release: the thread is a daemon and dies with the process


def _resolve_pair_path(base_dir: Path, raw: str) -> str:
    candidate = Path(raw).expanduser()
    return str(candidate if candidate.is_absolute() else base_dir / candidate)


def layer_path(layer) -> Path | None:
    stored = (layer.metadata or {}).get("longitrack_path")
    if stored:
        return Path(stored)
    source = getattr(layer, "source", None)
    if source is not None and getattr(source, "path", None):
        return Path(source.path)
    return None


class LongiTrackWidget(QWidget):
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        # one ViewerModel per timepoint: each canvas keeps its own camera and sliders
        self.vm: dict[str, ViewerModel] = {role: ViewerModel(ndisplay=2) for role in ROLES}
        self._ct: dict[str, Image | None] = {role: None for role in ROLES}
        self._qtv: dict = {}
        self._scan_dock = None

        # the only thing here that knows about torch/CUDA: a client for the backend process
        from .backend.client import BackendClient

        self._backend = BackendClient()
        self._backend_initialized = False
        self._preload_paths: set[str] = set()
        # Client file paths map to content-addressed scan copies owned by the backend.
        self._backend_scan_paths: dict[str, str] = {}
        self._scan_upload_futures: dict[str, object] = {}
        self._scan_upload_generation = 0
        self._segmentation_preload_paths: set[str] = set()
        self._model_folder: Path | None = None
        self._export_folder: str | None = None
        # per row: the lesion_id its segmentation is stored under on the backend
        self._lesions: list[str | None] = []
        # per-row colour overrides; they survive a model change, like the points themselves
        self._custom_colors: list[np.ndarray | None] = []
        # lesion_id -> {role: labels-layer name}
        self._lesion_layers: dict[str, dict[str, str]] = {}
        # monotonic, never reused: a re-segment reuses the same backend cache entry
        self._lesion_counter = 0
        # baseline point as of each row's last successful propagation
        self._anchors: list[list[int] | None] = []
        # raw registration output, kept apart from the (possibly corrected) follow-up layer
        self._registration_proposals: list[list[int] | None] = []
        # True while a row's follow-up entry is only a placeholder
        self._placeholder: list[bool] = []
        # per row: has the user accepted the current follow-up point
        self._accepted: list[bool] = []
        # per row and role: the point as it stood when the row was accepted, else None
        self._locked_points: dict[str, list[list[int] | None]] = {"baseline": [], "followup": []}
        # guards _enforce_locked_points against reacting to the very revert it just made
        self._enforcing_lock = False
        # set by Track while its propagate step is in flight, so _on_propagated carries on
        self._track_all_pending = False
        # queued {"baseline_scan", "followup_scan"} entries from a pair-list JSON
        self._pair_list: list[dict[str, str]] = []
        self._pair_index = 0
        # the verification walk after a propagation: which rows still need a verdict
        self._row_volumes: dict[str, tuple] = {}
        self._verify_rows: list[int] = []
        self._verify_editing = False
        self._verify_edit_count = 0
        self._applying_edit = False
        # the session channel: everything that touches the scans, prompts or results
        self._busy = False
        self._pending: tuple | None = None
        # bumped on every dispatch and by Cancel: a result carrying an old token is dropped
        self._job_token = 0
        # the "model" channel: Initialize model only.
        self._model_busy = False
        self._model_pending: tuple | None = None
        # one model job may wait behind the one in flight, so Initialize stays clickable
        self._model_queued: tuple | None = None
        self._model_job_token = 0
        self._activity = QTimer(self)
        self._activity.setInterval(120)
        self._activity.timeout.connect(self._tick_activity)
        self._activity_phase = 0.0
        self._activity_started = 0.0
        # what each currently-active channel is doing, e.g.
        self._activity_labels: dict[str, str] = {}
        # only the registration gets the canvas animation
        self._blend_channels: set[str] = set()
        # z offset in world mm between the two sliders, so one lesion shows on both at once
        self._slice_offset = 0.0
        self._syncing = False
        # guards a table write driven by layer data from being taken for a user edit
        self._syncing_table = False
        # which lesion the sliders are lined up on: only one pair fits at a time
        self._focus = 0
        # the device is chosen when the model actually loads, in the backend
        self._device: str | None = None
        self._width_claimed = False
        self._width_claim_attempts = 0
        # one long-lived worker thread, so the GUI never blocks on a job
        self._pool = _DaemonWorker("longitrack")
        # preload work must never make a user action wait in the GUI queue
        self._preload_pool = _DaemonWorker("longitrack-preload")
        # its own pool: Initialize model must never sit behind a session job
        self._model_pool = _DaemonWorker("longitrack-model")

        self._bridge = _Bridge()
        self._bridge.message.connect(self._log)
        self._bridge.done.connect(self._on_job_done)
        self._bridge.failed.connect(self._on_job_failed)
        self._bridge.model_done.connect(self._on_model_job_done)
        self._bridge.model_failed.connect(self._on_model_job_failed)

        self._build_ui()
        self._install_canvases()
        self.vm["baseline"].dims.events.point.connect(self._link_dims)
        for role in ROLES:
            # every layer, whenever and however it got there, needs its projection mode set
            self.vm[role].layers.events.inserted.connect(lambda _=None, r=role: self._apply_projection(r))
            # binds a role's self._ct to whatever Image lands in its ViewerModel
            self.vm[role].layers.events.inserted.connect(lambda event, r=role: self._on_role_layer_inserted(r, event))
        # File > Open and drops outside the two canvases still land in the host viewer
        self.viewer.layers.events.inserted.connect(self._on_stray_layer)
        self._refresh_scan_labels()

        self._log("Load a baseline and a follow-up scan, then click the lesion in the baseline.")
        if hasattr(self.viewer, "window"):
            QTimer.singleShot(0, self._ask_backend_startup_mode)

    # ------------------------------------------------------------------ ui ---
    def _build_ui(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)
        layout.addWidget(self._build_model_box())
        layout.addWidget(self._build_image_box())
        layout.addWidget(self._build_prompt_box())
        layout.addWidget(self._build_action_box())
        layout.addWidget(self._build_log_box(), stretch=1)
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._content = content
        for button in content.findChildren(QPushButton):
            if button is not self.add_point_button:
                button.clicked.connect(self._stop_adding_points)

        self.model_source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_source.setMinimumContentsLength(6)

        # buttons keep their normal size policy, so nothing shrinks below its label
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        # a horizontal scrollbar rather than squeezed columns if the dock gets narrow
        scroll.horizontalScrollBar().rangeChanged.connect(lambda _lo, _hi: scroll.horizontalScrollBar().setValue(0))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _install_canvases(self) -> None:
        # a ViewerModel has no window: headless tests stop here, the rest is display only
        window = getattr(self.viewer, "window", None)
        if window is None:
            return
        try:
            from napari.qt import QtViewer

            holder = QSplitter(Qt.Orientation.Horizontal)
            for role in ROLES:
                # both viewers must be built while their ViewerModel is still empty
                qtv = QtViewer(self.vm[role])
                controls = qtv.controls
                _ = qtv.layers

                pane = QWidget()
                column = QVBoxLayout(pane)
                column.setContentsMargins(0, 0, 0, 0)
                column.setSpacing(2)
                title = QLabel(ROLE_WORDS[role])
                title.setStyleSheet("font-weight: bold; color: gray;")
                column.addWidget(title)
                column.addWidget(qtv, stretch=1)

                # window/level etc.
                toggle = QToolButton()
                toggle.setText("Image controls")
                toggle.setCheckable(True)
                toggle.setChecked(False)
                toggle.setArrowType(Qt.ArrowType.RightArrow)
                toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
                toggle.setStyleSheet("QToolButton { border: none; }")
                controls.setVisible(False)
                toggle.toggled.connect(
                    lambda checked, c=controls, b=toggle: (
                        c.setVisible(checked),
                        b.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow),
                    )
                )
                column.addWidget(toggle)
                column.addWidget(controls)
                self._qtv[role] = qtv
                holder.addWidget(pane)

            # "left", not "right": two docks in the same area stack vertically
            window.add_dock_widget(holder, name="scans", area="left", add_vertical_stretch=False)
            self._scan_dock = holder
            # napari's own canvas and layer docks are dead weight now
            window._qt_window.centralWidget().hide()
            window._qt_viewer.dockLayerList.setVisible(False)
            window._qt_viewer.dockLayerControls.setVisible(False)
        except Exception as error:  # noqa: BLE001 - a broken layout must not kill the plugin
            self._log(f"Could not embed the two canvases: {error}")
            traceback.print_exc()

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Model")
        form = QFormLayout(box)

        self.backend_mode = QComboBox()  # internal state: 0 local, 1 remote
        self.backend_mode.addItems(["Local process", "Remote TCP server"])
        self.backend_mode.currentIndexChanged.connect(self._on_backend_mode_changed)
        self.backend_status = QLabel("Local backend")
        self.backend_status.setStyleSheet("color: gray;")
        form.addRow("Backend", self.backend_status)

        self.remote_host = QLineEdit("127.0.0.1")
        self.remote_host.setPlaceholderText("GPU server host name")
        self.remote_port = QLineEdit(str(REMOTE_BACKEND_PORT))
        self.test_remote_button = QPushButton("Test connection")
        self.test_remote_button.clicked.connect(self._on_test_remote_backend)
        remote_row = QHBoxLayout()
        remote_row.setContentsMargins(0, 0, 0, 0)
        remote_row.addWidget(self.remote_host, stretch=1)
        remote_row.addWidget(self.test_remote_button)
        self.remote_endpoint_widget = QWidget()
        self.remote_endpoint_widget.setLayout(remote_row)
        self.remote_endpoint_widget.setVisible(False)
        self.remote_notice = QLabel(
            "Remote mode reads scans, models, and export folders on the server. Use shared paths, or an SSH tunnel "
            "to a loopback-only server. TCP uses public-key authentication but is not encrypted."
        )
        self.remote_notice.setWordWrap(True)
        self.remote_notice.setStyleSheet("color: #8a6d3b;")
        self.remote_notice.setVisible(False)
        self._remote_private_key = str(Path.home() / ".config" / "longitrack-napari" / "remote" / "id_ed25519")

        self.model_source = QComboBox()
        self.model_source.addItems(["Hugging Face Hub", "Local folder"])
        self.model_source.currentIndexChanged.connect(self._on_model_source_changed)
        form.addRow("Source", self.model_source)

        self.model_location = QLineEdit(DEFAULT_REPO_ID)
        self.model_location.setToolTip("A Hugging Face repo id like 'owner/name', or a LongiSeg model folder.")
        self.browse_model = QPushButton("...")
        self.browse_model.setMaximumWidth(32)
        self.browse_model.setEnabled(False)
        self.browse_model.clicked.connect(self._on_browse_model)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.model_location)
        row.addWidget(self.browse_model)
        container = QWidget()
        container.setLayout(row)
        form.addRow("Location", container)

        # what the panel shows is what is used (see resolve_model_folder)
        env_folder = os.environ.get("LONGITRACK_MODEL_DIR", "").strip()
        if env_folder:
            self.model_source.setCurrentIndex(1)
            self.model_location.setText(env_folder)

        endpoint = os.environ.get("LONGITRACK_REMOTE_BACKEND", "").strip()
        if endpoint:
            self.remote_host.setText(endpoint)
            self.backend_mode.setCurrentIndex(1)
        self._on_backend_mode_changed(self.backend_mode.currentIndex())

        self.load_model_button = QPushButton("Initialize model")
        self.load_model_button.setToolTip("Downloads the model if needed and gets it ready to segment.")
        self.load_model_button.clicked.connect(self._on_load_model)
        form.addRow(self.load_model_button)

        # the device is not decided yet -- it is chosen fresh when Initialize model runs
        self.model_status = QLabel("not initialized")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet("color: gray;")
        form.addRow(self.model_status)
        return box

    def _build_image_box(self) -> QGroupBox:
        box = QGroupBox("Scans")
        grid = QGridLayout(box)

        self.scan_name: dict[str, QLabel] = {}
        self.scan_buttons: dict[str, tuple[QPushButton, QPushButton]] = {}
        for row, role in enumerate(ROLES):
            word = ROLE_WORDS[role]
            name = QLabel("none")
            name.setStyleSheet("color: gray;")
            name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.scan_name[role] = name
            open_button = self._icon_button(QStyle.StandardPixmap.SP_DialogOpenButton, f"Open a {word} scan.")
            open_button.clicked.connect(lambda _=False, r=role: self._on_open_image(r))
            remove_button = self._icon_button(
                QStyle.StandardPixmap.SP_TrashIcon, "Close this scan and everything derived from it."
            )
            remove_button.clicked.connect(lambda _=False, r=role: self._on_remove_scan(r))
            self.scan_buttons[role] = (open_button, remove_button)
            grid.addWidget(QLabel(word.capitalize()), row, 0)
            grid.addWidget(name, row, 1)
            grid.addWidget(open_button, row, 2)
            grid.addWidget(remove_button, row, 3)

        self.image_status = QLabel("")
        self.image_status.setWordWrap(True)
        self.image_status.setStyleSheet("color: gray;")
        grid.addWidget(self.image_status, 2, 0, 1, 4)

        # optional JSON list of pairs, stepped through one per click
        scan_actions = QHBoxLayout()
        scan_actions.setContentsMargins(0, 0, 0, 0)
        self.load_pair_list_button = QPushButton("Pair list...")
        self.load_pair_list_button.setToolTip(
            "Loads a JSON list of baseline/follow-up scan pairs; 'Load next scan pair' then "
            "opens them in order without asking each time."
        )
        self.load_pair_list_button.clicked.connect(self._on_load_pair_list)
        scan_actions.addWidget(self.load_pair_list_button)

        # replaces both scans at once, so a new baseline never pairs with an old follow-up
        self.open_pair_button = QPushButton("Next pair...")
        self.open_pair_button.setToolTip(
            "Opens a new baseline and follow-up scan together, resetting the session. Takes "
            "the next entry from a loaded pair list if there is one, otherwise asks for both scans."
        )
        self.open_pair_button.clicked.connect(self._on_open_pair)
        scan_actions.addWidget(self.open_pair_button)

        self.pantrack_button = QPushButton("Example")
        self.pantrack_button.setToolTip(f"Downloads and opens a sample case from {PANTRACK_PATIENT}.")
        self.pantrack_button.clicked.connect(self._on_load_pantrack)
        scan_actions.addWidget(self.pantrack_button)
        scan_action_row = QWidget()
        scan_action_row.setLayout(scan_actions)
        grid.addWidget(scan_action_row, 3, 0, 1, 4)
        return box

    def _icon_button(self, pixmap, tooltip: str) -> QPushButton:
        button = QPushButton("")
        button.setIcon(self.style().standardIcon(pixmap))
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(28, 28)
        button.setToolTip(tooltip)
        return button

    def _build_prompt_box(self) -> QGroupBox:
        box = QGroupBox("Prompt")
        layout = QVBoxLayout(box)

        self.add_point_button = QPushButton("Set baseline points")
        self.add_point_button.setCheckable(True)
        self.add_point_button.setToolTip("Click lesions in the baseline scan. Click again to stop adding.")
        self.add_point_button.toggled.connect(self._on_toggle_add_points)
        layout.addWidget(self.add_point_button)

        self.point_table = QTableWidget(0, len(_TABLE_COLUMNS))
        self.point_table.setHorizontalHeaderLabels([label for label, _tip in _TABLE_COLUMNS])
        for column, (_label, tip) in enumerate(_TABLE_COLUMNS):
            if tip:
                self.point_table.horizontalHeaderItem(column).setToolTip(tip)
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.point_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.point_table.setMinimumHeight(110)
        self.point_table.setMaximumHeight(190)
        self.point_table.verticalHeader().setDefaultSectionSize(26)
        self.point_table.setToolTip("One row per tracked lesion, paired by order.")
        header = self.point_table.horizontalHeader()
        header.setSectionResizeMode(_COL_INDEX, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeMode.Stretch)
        for column in (_COL_VISIBILITY, _COL_REMOVE):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.point_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.point_table.customContextMenuRequested.connect(self._on_table_context_menu)
        self.point_table.cellClicked.connect(self._on_table_clicked)
        layout.addWidget(self.point_table)

        self.verify_label = QLabel("")
        self.verify_label.setWordWrap(True)
        self.verify_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.verify_accept_button = QPushButton("Accept")
        self.verify_accept_button.setToolTip("The proposed point is on the lesion: keep it and go to the next one.")
        self.verify_accept_button.clicked.connect(self._on_verify_accept)
        self.verify_edit_button = QPushButton("Edit")
        self.verify_edit_button.setToolTip("Click the lesion in the follow-up scan, then Accept.")
        self.verify_edit_button.clicked.connect(self._on_verify_edit)
        self.verify_skip_button = QPushButton("Skip")
        self.verify_skip_button.setToolTip("Leave this one undecided and move on.")
        self.verify_skip_button.clicked.connect(self._on_verify_skip)
        verify_buttons = QHBoxLayout()
        verify_buttons.setContentsMargins(0, 0, 0, 0)
        for button in (self.verify_accept_button, self.verify_edit_button, self.verify_skip_button):
            verify_buttons.addWidget(button)
        # question above the buttons: a one-line message never has to widen the panel
        verify_column = QVBoxLayout()
        verify_column.setContentsMargins(0, 0, 0, 0)
        verify_column.setSpacing(4)
        verify_column.addWidget(self.verify_label)
        verify_column.addLayout(verify_buttons)
        self.verify_bar = QWidget()
        self.verify_bar.setLayout(verify_column)
        # keeps its space when hidden, so the panel does not jump
        policy = self.verify_bar.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.verify_bar.setSizePolicy(policy)
        self.verify_bar.setVisible(False)
        layout.addWidget(self.verify_bar)

        bulk_row = QHBoxLayout()
        bulk_row.setContentsMargins(0, 0, 0, 0)
        self.propagate_button = QPushButton("Propagate")
        self.propagate_button.setToolTip("Registers every unregistered baseline point, then verifies each.")
        self.propagate_button.clicked.connect(self._on_propagate_all)
        self.segment_button = QPushButton("Segment")
        self.segment_button.setToolTip("Segments every lesion whose follow-up point you accepted.")
        self.segment_button.clicked.connect(self._on_segment_all)
        self.track_button = QPushButton("Track")
        self.track_button.setToolTip("Propagate, accept every proposal and segment, without verifying each point.")
        self.track_button.clicked.connect(self._on_track_all)
        for button in (self.propagate_button, self.segment_button, self.track_button):
            bulk_row.addWidget(button)
        layout.addLayout(bulk_row)



        scale_labels = QHBoxLayout()
        scale_labels.setContentsMargins(0, 0, 0, 0)
        scale_labels.addWidget(QLabel("Speed"))
        scale_labels.addStretch()
        self.quality_label = QLabel()
        scale_labels.addWidget(self.quality_label)
        scale_labels.addStretch()
        scale_labels.addWidget(QLabel("Quality"))
        layout.addLayout(scale_labels)
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(0, len(QUALITY_MODES) - 1)
        self.quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_slider.setTickInterval(1)
        self.quality_slider.setValue(2)
        self._quality_index = self.quality_slider.value()
        self.quality_slider.setToolTip("Registration refinement and segmentation TTA. Resets cached results.")
        self.quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_quality_label()
        self.quality_slider.valueChanged.connect(self._on_quality_changed)
        layout.addWidget(self.quality_slider)

        self.prompt_status = QLabel("no baseline point yet")
        self.prompt_status.setWordWrap(True)
        self.prompt_status.setStyleSheet("color: gray;")
        layout.addWidget(self.prompt_status)
        return box

    def _build_action_box(self) -> QGroupBox:
        box = QGroupBox("Results")
        layout = QVBoxLayout(box)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.export_button = QPushButton("Export segmentations...")
        self.export_button.setToolTip("Writes each lesion's masks and metadata to a folder.")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        buttons.addWidget(self.export_button, stretch=1)
        self.clear_points_button = QPushButton("Clear everything")
        self.clear_points_button.setToolTip("Resets to right after model initialization: no prompts, no results.")
        self.clear_points_button.clicked.connect(self._on_clear_everything)
        buttons.addWidget(self.clear_points_button)
        layout.addLayout(buttons)

        self.activity = QLabel("")
        self.activity.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setVisible(False)  # a trough sitting at zero just looks broken
        layout.addWidget(self.progress)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Unlocks the panel; a step already on the GPU finishes and is dropped.")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        status = QHBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.addWidget(self.activity, stretch=1)
        status.addWidget(self.cancel_button)
        layout.addLayout(status)
        return box

    def _build_log_box(self) -> QGroupBox:
        box = QGroupBox("Log")
        layout = QVBoxLayout(box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.WidgetWidth)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_view.setMaximumHeight(150)
        layout.addWidget(self.log_view)
        return box

    # device selection lives in the backend (see backend/server.py)

    # --------------------------------------------------------------- logging -
    def _log(self, message: str) -> None:
        self.log_view.append(message)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _emit(self, message: str) -> None:
        # A daemon preload job can finish after napari has destroyed this widget.
        try:
            self._bridge.message.emit(message)
        except RuntimeError as error:
            if "has been deleted" not in str(error):
                raise

    def _error(self, title: str, error: BaseException) -> None:
        self._log(f"<span style='color:#d9534f;'>{title}: {error}</span>")
        traceback.print_exception(type(error), error, error.__traceback__)
        QMessageBox.critical(self, title, str(error))

    # ----------------------------------------------------------------- state -
    def _set_busy(self, busy: bool, what: str = "", blend: bool = False, lock_quality: bool = True) -> None:
        # session channel only -- Initialize model has its own, see _set_model_busy
        if busy:
            self._start_activity("session", what, blend)
        else:
            self._stop_activity("session")
        for widget in (
            self.add_point_button,
            self.clear_points_button,
            self.load_pair_list_button,
            self.open_pair_button,
            self.pantrack_button,
            self.point_table,
            *(button for pair in self.scan_buttons.values() for button in pair),
        ):
            widget.setEnabled(not busy)
        # only a job that can leave a registration/segmentation half-done needs this locked;
        # a plain data load has nothing in flight for a changed setting to race with
        self.quality_slider.setEnabled(not busy or not lock_quality)
        self.export_button.setEnabled(not busy and any(lesion_id is not None for lesion_id in self._lesions))
        self._refresh_action_buttons()
        # the caller-given `busy`, which may be set before self._busy itself is
        self._sync_progress_and_cancel(session_busy=busy, model_busy=self._model_busy)

    def _set_model_busy(self, busy: bool, what: str = "") -> None:
        # model channel only -- see _set_busy
        if busy:
            self._start_activity("model", what)
        else:
            self._stop_activity("model")
        # nothing is disabled here: clicking Initialize during the warm-up queues it
        self.backend_mode.setEnabled(not busy)
        self.browse_model.setEnabled(self.model_source.currentIndex() == 1)
        self._sync_progress_and_cancel(session_busy=self._busy, model_busy=busy)

    def _sync_progress_and_cancel(self, session_busy: bool, model_busy: bool) -> None:
        # progress bar and Cancel are shared: either channel running shows activity
        any_busy = session_busy or model_busy
        self.progress.setVisible(any_busy)
        self.cancel_button.setEnabled(any_busy)

    def _refuse_if_busy(self, action: str) -> bool:
        # the single place that says no while a job is reading or writing self._ct
        if not self._busy:
            return False
        # a routine "please wait" is not worth a modal; the log line already says it
        self._log(f"Something is already running; wait for it to finish before {action}.")
        return True

    def _on_model_source_changed(self, index: int) -> None:
        local = index == 1
        self.browse_model.setEnabled(local)
        if local and self.model_location.text().strip() == DEFAULT_REPO_ID:
            self.model_location.setText("")
        if not local and not self.model_location.text().strip():
            self.model_location.setText(DEFAULT_REPO_ID)

    def _remote_endpoint(self) -> tuple[str, int]:
        host = self._normalise_remote_host(self.remote_host.text())
        if not host:
            raise ValueError("Remote backend host cannot be empty.")
        return host, REMOTE_BACKEND_PORT

    @staticmethod
    def _normalise_remote_host(value: str) -> str:
        """Accept either a hostname or the server's printed ``CONNECT host:8765`` line."""
        host = value.strip()
        if host.upper().startswith("CONNECT "):
            host = host[8:].strip()
        if host.endswith(f":{REMOTE_BACKEND_PORT}"):
            host = host[: -(len(str(REMOTE_BACKEND_PORT)) + 1)]
        if not host or any(character.isspace() for character in host):
            raise ValueError("Enter the server address, or paste its CONNECT address line.")
        return host

    def _remote_mode(self) -> bool:
        return self.backend_mode.currentIndex() == 1

    def _on_backend_mode_changed(self, index: int) -> None:
        remote = index == 1
        if self._backend.is_running() and remote != self._backend.is_remote:
            self.backend_mode.blockSignals(True)
            self.backend_mode.setCurrentIndex(1 if self._backend.is_remote else 0)
            self.backend_mode.blockSignals(False)
            self._log("Restart the plugin to switch backend.")

    def _ask_backend_startup_mode(self) -> None:
        """Ask once whether this napari instance owns compute or is a remote client."""
        if self._backend.is_running():
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("LongiTrack backend")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText("Should the model run on a local or remote backend?")
        local = dialog.addButton("Local", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Remote", QMessageBox.ButtonRole.ActionRole)
        dialog.exec()
        if dialog.clickedButton() is local:
            self._start_local_backend_warmup()
        else:
            self.backend_mode.setCurrentIndex(1)
            self.backend_status.setText("Remote: not connected")
            self._on_connect_remote_server()

    def _start_local_backend_warmup(self) -> None:
        def work():
            self._configure_backend(False, None)
            self._backend.ensure_started(progress=self._emit)
            # registration does not depend on the segmentation model, so warm it first
            try:
                self._backend.warm_up_registration(progress=self._emit)
            except Exception as error:  # noqa: BLE001 - a warm-up must never fail start-up
                self._emit(f"Could not warm the registration network ({error}).")

        def ready(_result=None) -> None:
            self.backend_status.setText("Local backend ready")
            self._log("Local backend ready: registration warmed, model not loaded yet.")

        self._start_model(work, ready, "Local backend startup failed", "Starting local backend")

    def _backend_configuration(self) -> tuple[bool, tuple[str, int] | None]:
        remote = self._remote_mode()
        return remote, self._remote_endpoint() if remote else None

    def _configure_backend(self, remote: bool, endpoint: tuple[str, int] | None) -> None:
        if not remote:
            if self._backend.is_remote:
                raise RuntimeError(
                    "This widget is already connected to a remote backend; restart it to use local mode."
                )
            return
        if self._backend.is_running():
            if not self._backend.is_remote:
                raise RuntimeError("This widget already owns a local backend; restart it to use a remote server.")
            return
        if endpoint is None:
            raise RuntimeError("Remote backend endpoint was not configured.")
        self._backend.connect_tcp(
            *endpoint,
            private_key=self._remote_private_key,
        )

    def _on_connect_remote_server(self) -> None:
        if self._backend.is_running() and self._backend_initialized:
            self._log("Restart the plugin to switch backend.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Connect remote LongiTrack server")
        form = QFormLayout(dialog)
        host = QLineEdit(self.remote_host.text())
        key_dir = Path.home() / ".config" / "longitrack-napari" / "remote"
        private_keys = []
        if key_dir.is_dir():
            private_keys = sorted(path.name for path in key_dir.iterdir() if path.is_file() and path.suffix != ".pub")
        if not private_keys:
            self._error(
                "No remote identity",
                RuntimeError("Create one first: uv run create_remote_id --name <server-name>"),
            )
            return
        private_key = QComboBox()
        private_key.addItems(private_keys)
        form.addRow("Host", host)
        form.addRow("Private key", private_key)
        warning = QLabel(
            "Remote TCP uses public-key authentication but is not encrypted. Only connect over a trusted network."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #8a6d3b;")
        form.addRow(warning)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Connect")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            endpoint = (self._normalise_remote_host(host.text()), REMOTE_BACKEND_PORT)
            key_path = key_dir / private_key.currentText()
        except ValueError as error:
            self._error("Invalid remote backend", error)
            return
        self.remote_host.setText(endpoint[0])
        if self._backend.is_running():
            self._discard_prefetch_backend()
        self._remote_private_key = str(key_path)
        self.backend_mode.setCurrentIndex(1)
        self._start_model(
            lambda: self._configure_backend(True, endpoint),
            lambda _: self._on_remote_connected(*endpoint),
            "Remote backend connection failed",
            "Connecting to remote backend",
        )

    def _on_remote_connected(self, host: str, port: int) -> None:
        self.backend_status.setText(f"Remote: {host}:{port}")
        self.backend_status.setStyleSheet("color: #3c763d;")
        self._log(f"Connected to authenticated remote backend {host}:{port}.")
        self._queue_scan_preloads(
            [path for layer in self._ct.values() if layer is not None if (path := layer_path(layer)) is not None]
        )
        self._queue_segmentation_preloads(
            [path for layer in self._ct.values() if layer is not None if (path := layer_path(layer)) is not None]
        )

    def _on_test_remote_backend(self) -> None:
        try:
            host, port = self._remote_endpoint()
        except ValueError as error:
            self._error("Invalid remote backend", error)
            return
        from .backend.client import BackendClient

        self._start_model(
            lambda: BackendClient.probe_tcp(host, port),
            lambda info: self._on_remote_backend_tested(host, port, info),
            "Remote backend connection failed",
            "Testing remote backend",
        )

    def _on_remote_backend_tested(self, host: str, port: int, info: dict) -> None:
        self.model_status.setText(f"Remote backend {host}:{port} is reachable ({info['service']}).")
        self.model_status.setStyleSheet("color: #3c763d;")
        self._log(f"Remote backend {host}:{port} is reachable. Initialize model to use it.")

    def _on_browse_model(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a LongiSeg model folder")
        if folder:
            self.model_location.setText(folder)

    # ---------------------------------------------------------------- layers -
    def _layer(self, role: str, name: str):
        layers = self.vm[role].layers
        return layers[name] if name in layers else None

    def _apply_projection(self, role: str) -> None:
        # a prompt has to stay visible while scrolling, which needs thick slices
        vm = self.vm[role]
        margin = tuple(SLICE_MARGIN if axis == 0 else 0 for axis in range(vm.dims.ndim))
        vm.dims.margin_left_step = margin
        vm.dims.margin_right_step = margin
        for layer in vm.layers:
            # reading out_of_slice_display at all raises a FutureWarning in napari 0.9
            layer.projection_mode = "rescale_linear" if isinstance(layer, Points) else "none"

    def _refresh_scan_labels(self) -> None:
        messages = []
        for role in ROLES:
            layer = self._ct[role]
            self.scan_name[role].setText(layer.name if layer is not None else "none")
            self.scan_name[role].setToolTip(str(layer_path(layer)) if layer is not None else "")
            if layer is not None and layer_path(layer) is None:
                messages.append(
                    f"The {ROLE_WORDS[role]} scan has no file on disk; open it through this widget or drop the file "
                    f"onto its panel."
                )
        self.image_status.setText(" ".join(messages))
        self._update_prompt_status()
        self._refresh_point_table()

    def _on_stray_layer(self, event) -> None:
        # only fires for File > Open and for drops that missed both panels
        layer = getattr(event, "value", None)
        if layer is None or layer.name == BLEND_LAYER:
            return
        QTimer.singleShot(0, lambda: self._adopt_stray(layer))

    def _adopt_stray(self, layer) -> None:
        if layer not in self.viewer.layers or layer.name == BLEND_LAYER:
            return
        self.viewer.layers.remove(layer)
        free = next((role for role in ROLES if self._ct[role] is None), None)
        if free is None or not isinstance(layer, Image):
            self._log(f"Ignored '{layer.name}': drop the scan onto one of the two panels.")
            return
        # _on_role_layer_inserted binds self._ct[free] and refreshes the labels once this lands
        self.vm[free].add_layer(layer)
        self.vm[free].reset_view()
        self._log(f"Put '{layer.name}' in the {ROLE_WORDS[free]} panel. Next time drop the scan onto that panel.")

    def _on_role_layer_inserted(self, role: str, event) -> None:
        # any Image landing in a role's ViewerModel becomes that role's scan
        layer = getattr(event, "value", None)
        if not isinstance(layer, Image) or layer.name == BLEND_LAYER or layer is self._ct[role]:
            return
        if self._busy:
            # a job in flight captured the old path; let it finish rather than rebind it
            self._log(f"Busy: ignoring the new {ROLE_WORDS[role]} scan '{layer.name}' until the current job finishes.")
            return
        previous = self._ct[role]
        self._ct[role] = layer
        self._reset_session()
        self._refresh_scan_labels()
        if previous is not None and (previous_path := layer_path(previous)) is not None:
            self._queue_scan_release(previous_path)
        paths = [layer_path(scan) for scan in self._ct.values() if scan is not None]
        self._queue_scan_preloads([path for path in paths if path is not None])

    def _require_paths(self) -> tuple[Image, Path, Image, Path]:
        baseline, followup = self._ct["baseline"], self._ct["followup"]
        if baseline is None or followup is None:
            raise RuntimeError("Open a baseline and a follow-up scan first.")
        baseline_path, followup_path = layer_path(baseline), layer_path(followup)
        missing = [role for role, path in (("baseline", baseline_path), ("follow-up", followup_path)) if path is None]
        if missing:
            raise RuntimeError(
                f"The {' and '.join(missing)} layer is not backed by a file. Both the registration and the "
                f"preprocessing read from disk, so open the scans with 'Open...' or by dragging the files into napari."
            )
        return baseline, baseline_path, followup, followup_path

    def _on_load_pantrack(self) -> None:
        emit = self._emit

        def work():
            from .pantrack import download_pair

            return download_pair(patient=PANTRACK_PATIENT, pair_index=PANTRACK_PAIR, progress=emit)

        self._log(f"Loading the {PANTRACK_PATIENT} example...")
        self._start(
            work, self._on_pantrack_loaded, "Loading the PanTrack example failed", "Downloading PanTrack",
            lock_quality=False,
        )

    def _on_pantrack_loaded(self, pair) -> None:
        from ._sample_data import pair_to_layers

        for role, (data, kwargs, _) in zip(ROLES, pair_to_layers(pair), strict=True):
            self.vm[role].layers.clear()
            # _on_role_layer_inserted binds self._ct[role] and resets the session for us
            self.vm[role].add_image(data, **kwargs)
            self.vm[role].reset_view()

        self._log(
            f"{pair.patient}: {pair.baseline} -> {pair.followup}. "
            f"Press 'Set baseline point' and click the lesion."
        )
        self._refresh_scan_labels()

    def open_scan(self, role: str, path: str | Path):
        # public so a script can set a session up without clicking through the dialogs
        if role not in ROLES:
            raise ValueError(f"role must be 'baseline' or 'followup', not {role!r}.")
        if self._busy:
            raise RuntimeError("A job is still running; wait for it to finish before opening a new scan.")
        path = str(Path(path).expanduser())
        vm = self.vm[role]
        # a failed read must leave the previous scan, and everything from it, untouched
        try:
            added = vm.open(path, plugin="longitrack-napari")
        except Exception:
            added = vm.open(path)
        if not added:
            raise RuntimeError(f"Could not open {path}.")

        layer = added[0]
        # _on_role_layer_inserted has already bound self._ct[role] and reset the session
        if self._ct[role] is not layer:
            self._ct[role] = layer
            self._reset_session()
        # only now drop what was in the panel before
        for stale in [existing for existing in list(vm.layers) if existing is not layer]:
            vm.layers.remove(stale)
        vm.reset_view()
        self._log(f"Opened {role}: {path} {tuple(layer.data.shape)}")
        self._refresh_scan_labels()
        self._queue_scan_preload(path)
        return layer

    def _queue_scan_preload(self, path: str | Path) -> None:
        # everything the backend needs for this scan, in the background
        self._queue_scan_preloads([path])
        self._queue_segmentation_preloads([path])

    def _queue_scan_preloads(self, paths: Sequence[str | Path]) -> None:
        """Prepare scans on the backend; a remote one first needs its own copy of them."""
        disk_paths = []
        for path in paths:
            disk_path = str(Path(path).expanduser().absolute())
            if disk_path not in self._preload_paths:
                self._preload_paths.add(disk_path)
                disk_paths.append(disk_path)
        if not disk_paths:
            return
        backend = self._backend
        generation = self._scan_upload_generation
        emit = self._emit

        def work() -> dict[str, str]:
            uploaded: dict[str, str] = {}
            try:
                # A remote client is connected before scans can be queued.
                backend.ensure_started(progress=emit)
                for disk_path in disk_paths:
                    # a local backend reads the same file; only a remote one needs the bytes sent
                    uploaded[disk_path] = (
                        backend.upload_scan(disk_path, progress=emit)
                        if getattr(backend, "is_remote", False)
                        else disk_path
                    )
                # Registration has no dependency on LongiSeg.
                preload_registration = getattr(backend, "preload_registration_scans", None)
                if preload_registration is not None:
                    try:
                        preload_registration(list(uploaded.values()), progress=emit)
                    except Exception as error:  # noqa: BLE001 - preload is opportunistic
                        emit(f"Could not prepare registration inputs yet ({error}).")
                if self._scan_upload_generation == generation and self._backend is backend:
                    self._backend_scan_paths.update(uploaded)
                return uploaded
            except Exception as error:  # noqa: BLE001 - preparation is opportunistic
                names = ", ".join(Path(path).name for path in disk_paths)
                emit(f"Could not upload {names} to the backend ({error}).")
                raise
            finally:
                for disk_path in disk_paths:
                    self._preload_paths.discard(disk_path)

        future = self._preload_pool.submit(work)
        for disk_path in disk_paths:
            self._scan_upload_futures[disk_path] = future

    def _queue_segmentation_preloads(self, paths: Sequence[str | Path]) -> None:
        """Populate LongiSeg cache after weights load, before the first Segment action."""
        if not self._backend_initialized:
            return
        disk_paths = [str(Path(path).expanduser().absolute()) for path in paths]
        disk_paths = [path for path in dict.fromkeys(disk_paths) if path not in self._segmentation_preload_paths]
        if not disk_paths:
            return
        self._segmentation_preload_paths.update(disk_paths)
        backend = self._backend
        emit = self._emit

        def work() -> None:
            try:
                backend_paths = [self._backend_scan_path(path) for path in disk_paths]
                preload = getattr(backend, "load_scans", None)
                if preload is not None:
                    preload(backend_paths, progress=emit)
            except Exception as error:
                for path in disk_paths:
                    self._segmentation_preload_paths.discard(path)
                emit(f"Could not prepare LongiSeg scans yet ({error}).")

        self._preload_pool.submit(work)

    def _backend_scan_path(self, path: str | Path) -> str:
        """The path the backend should read this scan from."""
        disk_path = str(Path(path).expanduser().absolute())
        if not getattr(self._backend, "is_remote", False):
            return disk_path  # same filesystem, nothing was ever copied
        future = self._scan_upload_futures.get(disk_path)
        if future is not None:
            future.result()
        backend_path = self._backend_scan_paths.get(disk_path)
        if backend_path is None:
            raise RuntimeError(f"The backend copy of {Path(disk_path).name} is not available.")
        return backend_path

    def _queue_scan_release(self, path: str | Path) -> None:
        if not self._backend.is_running():
            return
        disk_path = str(Path(path).expanduser().absolute())
        backend_path = self._backend_scan_paths.get(disk_path)
        if backend_path is not None:
            self._preload_pool.submit(lambda: self._backend.release_scan(backend_path))

    def _discard_prefetch_backend(self) -> None:
        """Replace a local upload-only backend before connecting a remote one."""
        from .backend.client import BackendClient

        self._scan_upload_generation += 1
        self._segmentation_preload_paths.clear()
        self._preload_paths.clear()
        self._scan_upload_futures.clear()
        self._backend_scan_paths.clear()
        self._backend.cancel_active()
        self._backend = BackendClient()
        self._backend_initialized = False

    def _on_open_image(self, role: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Open the {ROLE_WORDS[role]} scan", "", _FILE_FILTER)
        if not path:
            return
        try:
            self.open_scan(role, path)
        except Exception as error:
            self._error("Could not open the scan", error)

    def _on_load_pair_list(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load a scan pair list", "", "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            pairs = self._parse_pair_list(Path(path))
        except Exception as error:
            self._error("Could not load the pair list", error)
            return
        self._pair_list = pairs
        self._pair_index = 0
        self._queue_scan_preloads(
            [scan for pair in pairs for scan in (pair["baseline_scan"], pair["followup_scan"])]
        )
        self._log(f"Loaded {len(pairs)} scan pair(s) from {path}. Click 'Load next scan pair' to open the first one.")

    @staticmethod
    def _parse_pair_list(path: Path) -> list[dict[str, str]]:
        # the keys inference_meta.json uses, but with openable paths
        entries = json.loads(path.read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path} must contain a non-empty JSON list of scan pairs.")
        base_dir = path.parent
        pairs = []
        for entry in entries:
            if not isinstance(entry, dict) or "baseline_scan" not in entry or "followup_scan" not in entry:
                raise ValueError(f"Every entry needs 'baseline_scan' and 'followup_scan', got: {entry!r}.")
            pairs.append(
                {
                    "baseline_scan": _resolve_pair_path(base_dir, entry["baseline_scan"]),
                    "followup_scan": _resolve_pair_path(base_dir, entry["followup_scan"]),
                }
            )
        return pairs

    def _on_open_pair(self) -> None:
        if self._refuse_if_busy("loading the next scan pair"):
            return
        if self._pair_index < len(self._pair_list):
            entry = self._pair_list[self._pair_index]
            try:
                self.open_scan("baseline", entry["baseline_scan"])
                self.open_scan("followup", entry["followup_scan"])
            except Exception as error:
                self._error("Could not open the next scan pair", error)
                return
            self._pair_index += 1
            self._log(f"Opened pair {self._pair_index}/{len(self._pair_list)} from the pair list.")
            if self._pair_index >= len(self._pair_list):
                self._log("That was the last pair in the list; the next click will ask for scans manually.")
            return

        baseline_path, _ = QFileDialog.getOpenFileName(self, "Open the baseline scan", "", _FILE_FILTER)
        if not baseline_path:
            return
        followup_path, _ = QFileDialog.getOpenFileName(self, "Open the follow-up scan", "", _FILE_FILTER)
        if not followup_path:
            return
        try:
            self.open_scan("baseline", baseline_path)
            self.open_scan("followup", followup_path)
        except Exception as error:
            self._error("Could not open the scan pair", error)

    # ---------------------------------------------------------------- points -
    def _points_layer(self, role: str) -> Points | None:
        reference = self._ct[role]
        if reference is None:
            return None
        name = POINTS_LAYER[role]
        existing = self._layer(role, name)
        if isinstance(existing, Points):
            existing.scale = reference.scale
            return existing

        kwargs = dict(
            name=name,
            ndim=reference.data.ndim,
            scale=reference.scale,
            size=6,
            face_color=LESION_COLORS[0],
        )
        empty = np.empty((0, reference.data.ndim))
        try:
            layer = self.vm[role].add_points(empty, border_color="white", **kwargs)
        except TypeError:  # napari < 0.5 called it edge_color
            layer = self.vm[role].add_points(empty, edge_color="white", **kwargs)
        self._connect_points(role, layer)
        self._refresh_point_table()
        return layer

    def _connect_points(self, role: str, layer: Points) -> None:
        if layer.metadata.get("longitrack_connected"):
            return
        layer.events.data.connect(lambda _=None, r=role: self._on_points_changed(r))
        layer.metadata["longitrack_connected"] = True

    def _on_points_changed(self, role: str) -> None:
        if role == "baseline":
            self._on_baseline_points_changed()
        else:
            self._on_followup_points_changed()
        self._refresh_point_table()

    def _row_color(self, row: int) -> np.ndarray:
        custom = self._get_at(self._custom_colors, row, None)
        if custom is not None:
            return np.asarray(custom, dtype=float)
        lesion_id = self._get_at(self._lesions, row, None)
        try:
            color_index = int(lesion_id) - 1 if lesion_id is not None else row
        except ValueError:
            color_index = row
        return LESION_COLORS[color_index % len(LESION_COLORS)]

    def _apply_prompt_colors(self) -> None:
        """Keep every row's two prompt markers in the same categorical color."""
        for role in ROLES:
            layer = self._layer(role, POINTS_LAYER[role])
            if not isinstance(layer, Points) or not len(layer.data):
                continue
            layer.face_color = np.asarray([self._row_color(row) for row in range(len(layer.data))], dtype=float)

    @staticmethod
    def _mask_colormap(data: np.ndarray, color: np.ndarray):
        from napari.utils.colormaps import DirectLabelColormap

        # a layer is one lesion: map every foreground value, so the colour never depends
        flat = data.reshape(-1)
        present = np.flatnonzero(np.bincount(flat)) if flat.dtype.kind in "ui" else np.unique(flat)
        color_dict = {None: np.zeros(4, dtype=float)}
        color_dict.update({int(value): color for value in present if int(value) != 0})
        return DirectLabelColormap(color_dict=color_dict)

    def _set_row_color(self, row: int, color: np.ndarray) -> None:
        color = np.asarray(color, dtype=float)
        if color.shape != (4,):
            raise ValueError("A lesion color must be RGBA.")
        self._set_at(self._custom_colors, row, color, None)
        self._apply_prompt_colors()
        for layer in self._row_lesion_layers(row):
            layer.colormap = self._mask_colormap(np.asarray(layer.data), color)
        self._refresh_point_table()

    def _on_choose_row_color(self, row: int) -> None:
        color = self._row_color(row)
        initial = QColor.fromRgbF(*[float(value) for value in color])
        selected = QColorDialog.getColor(initial, self, f"Lesion {row + 1} color")
        if selected.isValid():
            self._set_row_color(
                row, np.asarray([selected.redF(), selected.greenF(), selected.blueF(), selected.alphaF()], dtype=float)
            )

    def _prepare_baseline_points(self) -> Points | None:
        layer = self._points_layer("baseline")
        if layer is None:
            return None
        self._connect_points("baseline", layer)
        # click-to-add only reaches a Points layer that is active in its own viewer
        self.vm["baseline"].layers.selection.active = layer
        return layer

    def _on_toggle_add_points(self, adding: bool) -> None:
        layer = self._prepare_baseline_points() if adding else self._layer("baseline", POINTS_LAYER["baseline"])
        if adding and layer is None:
            QMessageBox.information(self, "No baseline scan", "Open a baseline scan first.")
            self.add_point_button.setChecked(False)
            return
        if isinstance(layer, Points):
            # "pan_zoom": points stay draggable, but a click no longer adds one
            layer.mode = "add" if adding else "pan_zoom"
        self._log("Click each lesion in the baseline scan." if adding else "Stopped adding baseline points.")

    def _on_baseline_points_changed(self, event=None) -> None:
        # nothing is invalidated here: a row's status is derived when the table refreshes
        self._enforce_locked_points("baseline")
        self._update_prompt_status()

    def _on_followup_points_changed(self, event=None) -> None:
        if self._apply_click_edit():
            return
        # editing a follow-up point is a correction, not a reason to drop the row
        self._enforce_locked_points("followup")
        self._update_prompt_status()

    def _enforce_locked_points(self, role: str) -> None:
        # an accepted row's points are frozen (see _accept_row): revert any move
        if self._enforcing_lock:
            return
        layer = self._layer(role, POINTS_LAYER[role])
        locked = self._locked_points[role]
        if not isinstance(layer, Points) or not locked:
            return
        data = np.asarray(layer.data, dtype=float)
        fixed = data.copy()
        changed = False
        for row, point in enumerate(locked):
            if point is None or row >= len(data) or np.allclose(data[row], point, atol=1e-6):
                continue
            fixed[row] = point
            changed = True
        if not changed:
            return
        self._enforcing_lock = True
        try:
            layer.data = fixed
        finally:
            self._enforcing_lock = False
        self._log(f"That {ROLE_WORDS[role]} point is locked -- unaccept its row to move it again.")

    def _role_points(self, role: str) -> list[list[int]]:
        layer = self._layer(role, POINTS_LAYER[role])
        reference = self._ct[role]
        if not isinstance(layer, Points) or reference is None or len(layer.data) == 0:
            return []
        shape = reference.data.shape
        return [
            [int(np.clip(round(float(c)), 0, s - 1)) for c, s in zip(point, shape, strict=True)]
            for point in layer.data
        ]

    def _baseline_points(self) -> list[list[int]]:
        return self._role_points("baseline")

    def _followup_points(self) -> list[list[int]]:
        return self._role_points("followup")

    def _set_followup_point(self, row: int, point: Sequence[int]) -> None:
        # row N can be set before row N-1: skipped rows are padded with a placeholder
        layer = self._points_layer("followup")
        baseline_points = self._baseline_points()
        ndim = len(point)
        current = np.asarray(layer.data, dtype=float) if len(layer.data) else np.empty((0, ndim))
        gap_start = len(current)
        if row < len(current):
            current = current.copy()
            current[row] = point
        else:
            gap = [baseline_points[i] if i < len(baseline_points) else point for i in range(gap_start, row)]
            current = np.vstack([current, np.asarray([*gap, point], dtype=float)])
        layer.data = current
        layer.mode = "select"
        self.vm["followup"].layers.selection.active = layer
        for gap_row in range(gap_start, row):
            self._set_at(self._placeholder, gap_row, True, False)
        self._set_at(self._placeholder, row, False, False)

    @staticmethod
    def _get_at(values: list, index: int, default):
        return values[index] if 0 <= index < len(values) else default

    @staticmethod
    def _set_at(values: list, index: int, value, default) -> None:
        while len(values) <= index:
            values.append(default)
        values[index] = value

    def _row_has_followup(self, row: int) -> bool:
        # a real follow-up point: registered, placed or corrected, never a placeholder
        return row < len(self._followup_points()) and not self._get_at(self._placeholder, row, False)

    def _row_needs_registration(self, row: int, baseline_points: Sequence[Sequence[int]]) -> bool:
        if not self._row_has_followup(row):
            return True
        anchor = self._get_at(self._anchors, row, None)
        return anchor is None or row >= len(baseline_points) or list(anchor) != list(baseline_points[row])

    def _update_prompt_status(self) -> None:
        baseline_points = self._baseline_points()
        if not baseline_points:
            self.prompt_status.setText("no baseline point yet")
            return
        lesions = f"{len(baseline_points)} lesion{'s' if len(baseline_points) != 1 else ''}"
        registered = sum(1 for row in range(len(baseline_points)) if self._row_has_followup(row))
        segmented = sum(1 for lesion_id in self._lesions if lesion_id is not None)
        if registered:
            text = f"{lesions}, {registered} registered, {segmented} segmented."
        else:
            text = f"{lesions} prompted at {baseline_points[0]}"
            if len(baseline_points) > 1:
                text += f" (+{len(baseline_points) - 1} more)"
        self.prompt_status.setText(text)

    @staticmethod
    def _centered(widget: QWidget) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.addWidget(widget)
        row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.setContentsMargins(0, 0, 0, 0)
        return container

    def _row_lesion_layers(self, row: int) -> list:
        lesion_id = self._get_at(self._lesions, row, None)
        if lesion_id is None:
            return []
        names = self._lesion_layers.get(lesion_id, {})
        return [layer for role, name in names.items() if (layer := self._layer(role, name)) is not None]

    @staticmethod
    def _row_prompt_shown(row: int, point_layers: dict) -> bool:
        return all(
            bool(layer.shown[row])
            for layer in point_layers.values()
            if isinstance(layer, Points) and row < len(layer.data) and row < len(layer.shown)
        )

    @staticmethod
    def _small_button(text: str, tooltip: str) -> QPushButton:
        button = QPushButton(text)
        button.setFixedSize(34, 24)
        button.setToolTip(tooltip)
        return button

    def _refresh_point_table(self) -> None:
        baseline_points, followup_points = self._baseline_points(), self._followup_points()
        self._apply_prompt_colors()
        point_layers = {role: self._layer(role, POINTS_LAYER[role]) for role in ROLES}
        table = self.point_table
        self._syncing_table = True
        table.blockSignals(True)
        try:
            table.setRowCount(max(len(baseline_points), len(followup_points)))
            for row in range(table.rowCount()):
                registered = self._row_has_followup(row)
                accepted = registered and self._get_at(self._accepted, row, False)
                lesion = self._row_lesion_layers(row)

                number = QTableWidgetItem(str(row + 1))
                number.setFlags(number.flags() & ~Qt.ItemFlag.ItemIsEditable)
                color = self._row_color(row)
                number.setForeground(QColor.fromRgbF(*color[:3]))
                table.setItem(row, _COL_INDEX, number)

                item = QTableWidgetItem(self._row_status(row, bool(lesion), accepted, registered))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                # the coordinates stay in the tooltip instead of taking up half the panel
                where = []
                if row < len(baseline_points):
                    where.append("baseline " + ", ".join(str(c) for c in baseline_points[row]))
                if registered:
                    where.append("follow-up " + ", ".join(str(c) for c in followup_points[row]))
                item.setToolTip("\n".join(where) or "no point yet")
                table.setItem(row, _COL_STATUS, item)

                shown = all(layer.visible for layer in lesion) if lesion else self._row_prompt_shown(row, point_layers)
                visibility = self._row_button(row, _COL_VISIBILITY, self._on_toggle_row_visibility)
                visibility.setText("\N{EYE}" if shown else "\N{CIRCLED DIVISION SLASH}")
                visibility.setToolTip("Show or hide this lesion.")

                remove = self._row_button(row, _COL_REMOVE, self._on_remove_prompt)
                remove.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
                remove.setToolTip("Remove this lesion.")
        finally:
            table.blockSignals(False)
            self._syncing_table = False
        self._refresh_action_buttons()

    def _refresh_action_buttons(self) -> None:
        points = self._baseline_points()
        rows = range(len(points))
        needs_registration = any(self._row_needs_registration(row, points) for row in rows)
        unsegmented = [row for row in rows if not self._row_lesion_layers(row)]
        ready = bool(points) and all(self._ct[role] is not None for role in ROLES)
        self.propagate_button.setEnabled(not self._busy and ready and needs_registration)
        self.track_button.setEnabled(not self._busy and ready and (needs_registration or bool(unsegmented)))
        can_segment = any(self._get_at(self._accepted, row, False) for row in unsegmented)
        self.segment_button.setEnabled(not self._busy and can_segment)

    def _stop_adding_points(self) -> None:
        self.add_point_button.setChecked(False)

    def _row_button(self, row: int, column: int, handler) -> QPushButton:
        """Reuse the row's button: replacing it on every refresh makes it flash elsewhere."""
        button = self.point_table.cellWidget(row, column)
        if button is None:
            button = QPushButton()
            button.setFlat(True)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.point_table.setCellWidget(row, column, button)
        try:
            button.clicked.disconnect()
        except TypeError:
            pass
        button.clicked.connect(lambda _=False, r=row: handler(r))
        return button

    def _row_status(self, row: int, segmented: bool, accepted: bool, registered: bool) -> str:
        """What this lesion is worth saying: its volume change once there is one."""
        volumes = self._row_volumes.get(self._get_at(self._lesions, row, None)) if segmented else None
        if volumes and volumes[1] is not None:
            followup = float(volumes[1])
            if volumes[0] is None:
                return f"{followup:.1f} ml"
            baseline = float(volumes[0])
            change = f"{100 * (followup - baseline) / baseline:+.0f}%" if baseline else "new"
            return f"{baseline:.1f} \N{RIGHTWARDS ARROW} {followup:.1f} ml  ({change})"
        if segmented:
            return "segmented"
        return "verified" if accepted else ("proposed" if registered else "baseline prompt set")

    def _on_table_context_menu(self, position) -> None:
        """Everything that is not worth a permanent column of its own."""
        row = self.point_table.rowAt(position.y())
        if row < 0:
            return
        menu = QMenu(self)
        accept = menu.addAction("Accept this point")
        accept.setEnabled(self._row_has_followup(row) and not self._get_at(self._accepted, row, False))
        recolor = menu.addAction("Change colour...")
        repropagate = menu.addAction("Propagate this point again")
        repropagate.setEnabled(self._row_has_followup(row))
        unaccept = menu.addAction("Undo verification")
        unaccept.setEnabled(self._get_at(self._accepted, row, False))
        menu.addSeparator()
        remove = menu.addAction("Remove lesion")
        chosen = menu.exec(self.point_table.viewport().mapToGlobal(position))
        if chosen is accept:
            self._accept_row(row)
            self._refresh_point_table()
        elif chosen is recolor:
            self._on_choose_row_color(row)
        elif chosen is repropagate:
            self._on_delete_registration(row)
        elif chosen is unaccept:
            self._unaccept_row(row)
            self._refresh_point_table()
        elif chosen is remove:
            self._on_remove_prompt(row)

    def _on_table_clicked(self, row: int, _column: int = 0) -> None:
        baseline_points, followup_points = self._baseline_points(), self._followup_points()
        if row < len(baseline_points) and row < len(followup_points):
            self._focus = row
            self._lock_views(baseline_points[row], followup_points[row])
        for role, points in (("baseline", baseline_points), ("followup", followup_points)):
            if row < len(points):
                self._focus_on(role, points[row])

    def _on_toggle_prompt_shown(self, row: int, value: bool) -> None:
        for role in ROLES:
            layer = self._layer(role, POINTS_LAYER[role])
            if not isinstance(layer, Points) or row >= len(layer.data):
                continue
            shown = np.asarray(layer.shown, dtype=bool).copy()
            if row < len(shown):
                shown[row] = value
                layer.shown = shown

    def _on_toggle_row_visibility(self, row: int) -> None:
        # one control for whatever the row shows: its masks, else its prompt point
        seg_layers = self._row_lesion_layers(row)
        if seg_layers:
            value = not all(layer.visible for layer in seg_layers)
            for layer in seg_layers:
                layer.visible = value
        else:
            point_layers = {role: self._layer(role, POINTS_LAYER[role]) for role in ROLES}
            self._on_toggle_prompt_shown(row, not self._row_prompt_shown(row, point_layers))
        self._refresh_point_table()

    def _on_toggle_prompt_visibility(self, row: int) -> None:
        """Toggle points independently of any segmentation layers in this row."""
        point_layers = {role: self._layer(role, POINTS_LAYER[role]) for role in ROLES}
        self._on_toggle_prompt_shown(row, not self._row_prompt_shown(row, point_layers))
        self._refresh_point_table()

    def _accept_row(self, row: int) -> None:
        # snapshot both points: _enforce_locked_points holds them here while accepted
        self._set_at(self._accepted, row, True, False)
        baseline_points, followup_points = self._baseline_points(), self._followup_points()
        self._set_at(
            self._locked_points["baseline"], row,
            list(baseline_points[row]) if row < len(baseline_points) else None, None,
        )
        self._set_at(
            self._locked_points["followup"], row,
            list(followup_points[row]) if row < len(followup_points) else None, None,
        )

    def _unaccept_row(self, row: int) -> None:
        # a segmentation no longer backed by an accepted pair is stale, so it goes too
        self._set_at(self._accepted, row, False, False)
        self._set_at(self._locked_points["baseline"], row, None, None)
        self._set_at(self._locked_points["followup"], row, None, None)
        lesion_id = self._get_at(self._lesions, row, None)
        if lesion_id is not None:
            self._set_at(self._lesions, row, None, None)
            self._remove_lesion_segmentation(lesion_id)
            self.export_button.setEnabled(any(lid is not None for lid in self._lesions))

    def _on_toggle_accept(self, row: int, value: bool) -> None:
        if value:
            self._accept_row(row)
        else:
            self._unaccept_row(row)
        self._refresh_point_table()

    def _remove_lesion_segmentation(self, lesion_id: str) -> None:
        """Drop one lesion's own segmentation layers, given its id (see _on_remove_prompt)."""
        for role, name in self._lesion_layers.pop(lesion_id, {}).items():
            layer = self._layer(role, name)
            if layer is not None:
                self.vm[role].layers.remove(layer)

    def _on_delete_registration(self, row: int) -> None:
        # back to never registered: drops the follow-up point and anything from it
        if not self._row_has_followup(row):
            return
        self._unaccept_row(row)
        layer = self._points_layer("followup")
        baseline_points = self._baseline_points()
        if isinstance(layer, Points) and row < len(layer.data):
            # the neutral placeholder _set_followup_point uses for a gap row
            fallback = baseline_points[row] if row < len(baseline_points) else layer.data[row]
            data = np.asarray(layer.data, dtype=float).copy()
            data[row] = fallback
            layer.data = data
        self._set_at(self._placeholder, row, True, False)
        self._set_at(self._anchors, row, None, None)
        self._set_at(self._registration_proposals, row, None, None)
        self._update_prompt_status()
        self._refresh_point_table()
        self._log(f"Deleted prompt {row + 1}'s registered follow-up point; Propagate all will run it again.")

    def _on_remove_prompt(self, row: int) -> None:
        # prompts are paired by index, so the row has to go from both layers at once
        removed = False
        for role in ROLES:
            layer = self._layer(role, POINTS_LAYER[role])
            if isinstance(layer, Points) and row < len(layer.data):
                # layer.data alone does not shift .shown, so every per-point array is trimmed too
                shown = np.asarray(layer.shown, dtype=bool).copy()
                if row < len(self._locked_points[role]):
                    del self._locked_points[role][row]
                layer.data = np.delete(np.asarray(layer.data, dtype=float), row, axis=0)
                if row < len(shown):
                    layer.shown = np.delete(shown, row)
                removed = True
        if not removed:
            return
        for values in (self._anchors, self._registration_proposals, self._accepted, self._placeholder):
            if row < len(values):
                del values[row]
        if row < len(self._custom_colors):
            del self._custom_colors[row]
        if row < len(self._lesions):
            lesion_id = self._lesions.pop(row)
            if lesion_id is not None:
                self._remove_lesion_segmentation(lesion_id)
        if self._focus >= row:
            self._focus = max(0, self._focus - 1)
        self.export_button.setEnabled(any(lesion_id is not None for lesion_id in self._lesions))
        self._update_prompt_status()
        self._refresh_point_table()
        self._log(f"Removed prompt {row + 1}.")

    def _focus_on(self, role: str, point: Sequence[float]) -> None:
        # move that panel's dims sliders to the world position of one of its voxels
        layer = self._ct[role]
        if layer is None:
            return
        try:
            world = layer.data_to_world([float(c) for c in point])
            self.vm[role].dims.set_point(range(len(world)), world)
        except Exception as error:
            self._log(f"Could not move the view to {list(point)}: {error}")

    def _lock_views(self, baseline_point: Sequence[float], followup_point: Sequence[float]) -> None:
        # the two scans share neither slice count nor physical frame
        baseline, followup = self._ct["baseline"], self._ct["followup"]
        if baseline is None or followup is None:
            return
        try:
            baseline_world = float(np.asarray(baseline.data_to_world(list(baseline_point)), dtype=float)[0])
            followup_depth = float(followup_point[0]) * float(np.asarray(followup.scale)[0])
            self._slice_offset = baseline_world - followup_depth
            self._link_dims()
        except Exception as error:
            self._log(f"Could not line the two scans up: {error}")

    def _unlock_views(self) -> None:
        self._slice_offset = 0.0
        self._link_dims()

    def _link_dims(self, *_) -> None:
        # baseline drives follow-up; set_point clamps, so an edge lesion parks on the nearest slice
        if self._syncing or self._ct["baseline"] is None or self._ct["followup"] is None:
            return
        self._syncing = True
        try:
            self.vm["followup"].dims.set_point(0, self.vm["baseline"].dims.point[0] - self._slice_offset)
        except Exception as error:
            self._log(f"Could not line the two sliders up: {error}")
        finally:
            self._syncing = False

    # ----------------------------------------------------------------- model -
    def _on_load_model(self) -> None:
        # its own channel (see _start_model): loading the model touches no scan state
        try:
            backend_config = self._backend_configuration()
        except ValueError as error:
            self._error("Invalid remote backend", error)
            return
        self._start_model(
            self._load_model_job(backend_config),
            self._on_model_loaded,
            "Loading the model failed",
            "Initializing model",
        )

    def _load_model_job(self, backend_config: tuple[bool, tuple[str, int] | None]):
        location = self.model_location.text().strip() or None
        local_model = self.model_source.currentIndex() == 1
        emit = self._emit

        def work():
            self._configure_backend(*backend_config)
            if self._backend.ensure_started(progress=emit):
                self._backend_initialized = False
            # a local backend opens the folder itself; only a remote one needs a copy
            backend_location = (
                self._backend.upload_model_folder(location, emit)
                if local_model and getattr(self._backend, "is_remote", False)
                else location
            )
            # device=None: the backend decides and reports back what it picked
            return self._backend.initialize(backend_location, device=None, progress=emit)

        return work

    def _on_model_loaded(self, info: dict) -> None:
        self._backend_initialized = True
        # a new segmentation model invalidates masks, never the registration
        if not self._busy:
            self._lesions = []
            self._remove_all_segmentation_layers()
            self.export_button.setEnabled(False)
        self._model_folder = Path(info["folder"])
        self._device = info["device"]
        if self.model_source.currentIndex() == 0:
            # the repo the user asked for, not the local path it was downloaded to
            location = self.model_location.text().strip() or DEFAULT_REPO_ID
            source = f'HuggingFace repo "{location}"'
        else:
            # just the folder's own name: the user may have pointed at a parent directory
            source = f'local folder "{self._model_folder.name}"'
        self.model_status.setText(f"Model initialized from {source}, running on {self._device}.")
        self.model_status.setStyleSheet("color: #3c763d;")
        open_scans = [
            path for layer in self._ct.values() if layer is not None if (path := layer_path(layer)) is not None
        ]
        self._queue_scan_preloads(open_scans)
        # scans opened before the model was ready still need preprocessing
        self._queue_segmentation_preloads(open_scans)

    def _quality_mode(self) -> tuple[str, int | None, bool, str]:
        return QUALITY_MODES[self.quality_slider.value()]

    def _update_quality_label(self) -> None:
        name, _, _, _ = self._quality_mode()
        self.quality_label.setText(name)

    def _on_quality_changed(self, _value: int) -> None:
        if _value == self._quality_index:
            return
        if self._followup_points() or any(lesion_id is not None for lesion_id in self._lesions):
            answer = QMessageBox.question(
                self,
                "Discard derived results?",
                "Changing the speed–quality setting invalidates propagated follow-up points "
                "and segmentation results. Baseline prompts will be kept, but you will need "
                "to propagate and segment again. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.quality_slider.blockSignals(True)
                self.quality_slider.setValue(self._quality_index)
                self.quality_slider.blockSignals(False)
                return
            self._discard_derived_results()
        self._quality_index = _value
        self._update_quality_label()
        # waited for, so a just-started propagation cannot overtake the reset
        self._invalidate_registration(clear_tracking=True, wait=True)

    def _discard_derived_results(self) -> None:
        """Keep baseline prompts but remove data dependent on the previous mode."""
        self._lesions = []
        self._anchors = []
        self._registration_proposals = []
        self._accepted = []
        self._locked_points = {"baseline": [], "followup": []}
        self._placeholder = []
        self._lesion_counter = 0
        followup_points = self._layer("followup", FOLLOWUP_POINTS)
        if isinstance(followup_points, Points):
            dimensions = followup_points.data.shape[1] if followup_points.data.ndim == 2 else 3
            followup_points.data = np.empty((0, dimensions))
        self._remove_all_segmentation_layers()
        self.export_button.setEnabled(False)
        self._update_prompt_status()
        self._refresh_point_table()
        self._log("Discarded propagated points and segmentation results after changing quality mode.")

    def _invalidate_registration(
        self,
        clear_tracking: bool = False,
        clear_scans: bool = False,
        wait: bool = False,
    ) -> None:
        # the backend holds the cached registration; nothing to do if it never started
        if self._backend.is_running():
            kwargs = {"clear_tracking": clear_tracking, "clear_scans": clear_scans}
            if wait:
                self._backend.invalidate_registration(**kwargs)
                return

            def invalidate_async() -> None:
                try:
                    self._backend.invalidate_registration(**kwargs)
                except Exception:  # noqa: BLE001 - best-effort cache cleanup during teardown
                    pass

            threading.Thread(target=invalidate_async, daemon=True).start()

    # ---------------------------------------------------------- verification -
    def _begin_verification(self, rows: Sequence[int]) -> None:
        """Walk the freshly proposed points, one lesion at a time (Accept / Edit)."""
        self._verify_rows = [row for row in rows if self._row_has_followup(row)]
        self._verify_editing = False
        self._show_current_verification()

    def _clear_point_selection(self) -> None:
        """Leave napari's select mode, so its selection handle stops sitting on the point."""
        layer = self._layer("followup", POINTS_LAYER["followup"])
        if isinstance(layer, Points):
            layer.selected_data = set()
            layer.mode = "pan_zoom"

    def _show_current_verification(self) -> None:
        if not self._verify_rows:
            self._clear_point_selection()
            self.verify_bar.setVisible(False)
            self._verify_editing = False
            self._refresh_point_table()
            accepted = sum(1 for row in range(len(self._baseline_points())) if self._get_at(self._accepted, row, False))
            if accepted:
                self._log(f"{accepted} point(s) verified. Press Segment.")
            return
        if not self._verify_editing:
            self._clear_point_selection()
        row = self._verify_rows[0]
        total = len(self._baseline_points())
        self.verify_bar.setVisible(True)
        self.verify_edit_button.setEnabled(not self._verify_editing)
        if self._verify_editing:
            self.verify_label.setText(f"Lesion {row + 1}/{total}: click, then Accept")
        else:
            self.verify_label.setText(f"Lesion {row + 1}/{total}: propagated correctly?")
        self._focus_row(row)
        self.point_table.selectRow(row)

    def _focus_row(self, row: int) -> None:
        baseline_points, followup_points = self._baseline_points(), self._followup_points()
        if row < len(baseline_points) and row < len(followup_points):
            self._focus = row
            self._lock_views(baseline_points[row], followup_points[row])
            self._focus_on("baseline", baseline_points[row])
            self._focus_on("followup", followup_points[row])

    def _advance_verification(self) -> None:
        if self._verify_rows:
            self._verify_rows.pop(0)
        self._verify_editing = False
        self._show_current_verification()

    def _on_verify_accept(self) -> None:
        if not self._verify_rows:
            return
        row = self._verify_rows[0]
        self._accept_row(row)
        self._refresh_point_table()
        self._advance_verification()
        self._prepare_lesion_layers(row)

    def _on_verify_edit(self) -> None:
        if not self._verify_rows or self._verify_editing:
            return
        layer = self._layer("followup", POINTS_LAYER["followup"])
        if not isinstance(layer, Points):
            return
        self._verify_editing = True
        self._verify_edit_count = len(layer.data)
        layer.selected_data = set()
        layer.mode = "add"
        self.vm["followup"].layers.selection.active = layer
        self._show_current_verification()

    def _apply_click_edit(self) -> bool:
        """While editing, a click in the follow-up canvas moves that lesion's point."""
        if not (self._verify_editing and self._verify_rows) or self._applying_edit:
            return False
        layer = self._layer("followup", POINTS_LAYER["followup"])
        if not isinstance(layer, Points):
            return False
        data = np.asarray(layer.data, dtype=float)
        if len(data) <= self._verify_edit_count:
            return False
        row = self._verify_rows[0]
        moved = data[: self._verify_edit_count].copy()
        if row >= len(moved):
            return False
        moved[row] = data[-1]
        self._applying_edit = True
        try:
            layer.data = moved
        finally:
            self._applying_edit = False
        self._set_at(self._placeholder, row, False, False)
        return True

    def _on_verify_skip(self) -> None:
        self._advance_verification()

    # ------------------------------------------------------------- propagate -
    def _on_propagate_all(self) -> None:
        try:
            _, _, _, _ = self._require_paths()
            baseline_points = self._baseline_points()
            if not baseline_points:
                raise RuntimeError("Click a lesion in the baseline scan first.")
        except Exception as error:
            QMessageBox.information(self, "Not ready", str(error))
            return
        todo = [row for row in range(len(baseline_points)) if self._row_needs_registration(row, baseline_points)]
        if not todo:
            self._log("Every prompt is already registered.")
            return
        self._propagate_rows(todo)

    def _on_track_all(self) -> None:
        # propagate, accept, and segment everything in one click: the "just do it" shortcut
        try:
            _, _, _, _ = self._require_paths()
            baseline_points = self._baseline_points()
            if not baseline_points:
                raise RuntimeError("Click a lesion in the baseline scan first.")
        except Exception as error:
            QMessageBox.information(self, "Not ready", str(error))
            return
        todo = [row for row in range(len(baseline_points)) if self._row_needs_registration(row, baseline_points)]
        if not todo:
            self._accept_and_segment_all(exclude=())
            return
        self._track_all_pending = True
        self._propagate_rows(todo)

    def _accept_and_segment_all(self, exclude: Sequence[int]) -> None:
        baseline_points = self._baseline_points()
        for row in range(len(baseline_points)):
            if row not in exclude and self._row_has_followup(row):
                self._accept_row(row)
        self._refresh_point_table()
        self._on_segment_all()

    def _propagate_rows(self, rows: list[int]) -> None:
        try:
            _, baseline_path, followup, followup_path = self._require_paths()
            baseline_points = self._baseline_points()
        except Exception as error:
            self._track_all_pending = False
            QMessageBox.information(self, "Not ready", str(error))
            return
        rows = [row for row in rows if row < len(baseline_points)]
        if not rows:
            self._track_all_pending = False
            return

        mode_name, refinement_steps, _, _ = self._quality_mode()
        shape = tuple(followup.data.shape)
        emit = self._emit
        pending = [baseline_points[row] for row in rows]

        self._log(
            f"Registering {len(pending)} point{'s' if len(pending) != 1 else ''} ({mode_name.lower()})..."
        )

        def work():
            if self._backend.ensure_started(progress=emit):
                self._backend_initialized = False
            backend_baseline_path = self._backend_scan_path(baseline_path)
            backend_followup_path = self._backend_scan_path(followup_path)
            propagations = self._backend.propagate(
                backend_baseline_path, backend_followup_path, pending, shape,
                refinement_steps=refinement_steps, progress=emit,
            )
            return rows, propagations

        self._start(
            work,
            self._on_propagated,
            "Registration failed",
            f"Registering {Path(baseline_path).name} -> {Path(followup_path).name}",
        )

    def _on_propagated(self, payload) -> None:
        rows, propagations = payload
        baseline_points = self._baseline_points()
        followup = self._ct["followup"]
        shape = tuple(followup.data.shape) if followup is not None else None

        failures, clipped = [], False
        for row, propagation in zip(rows, propagations, strict=True):
            if row >= len(baseline_points):
                continue
            if propagation.ok:
                self._set_followup_point(row, propagation.followup_index)
                self._set_at(self._registration_proposals, row, propagation.followup_index, None)
                self._set_at(self._anchors, row, list(baseline_points[row]), None)
                self._unaccept_row(row)  # every proposal is verified before it counts
                clipped = clipped or propagation.out_of_bounds
            else:
                # falling back to the baseline index keeps the prompts paired one to one, so the user can drag this
                failures.append(row)
                fallback = snap_to_volume(baseline_points[row], shape) if shape else list(baseline_points[row])
                self._set_followup_point(row, fallback)
                self._set_at(self._registration_proposals, row, None, None)

        if failures and len(failures) == len(rows):
            self._track_all_pending = False
            QMessageBox.warning(self, "Registration failed", "No point could be registered to the follow-up scan.")
            return
        if failures:
            numbers = ", ".join(str(row + 1) for row in failures)
            self._log(f"Point(s) {numbers} could not be registered; they sit at the baseline location, drag them.")
        if clipped:
            self._log("At least one registered point landed outside the follow-up volume and was clipped to its edge.")

        followup_points = self._followup_points()
        if rows and rows[0] < len(followup_points) and rows[0] < len(baseline_points):
            # otherwise the new follow-up prompt lands on a slice nobody is looking at
            self._focus = rows[0]
            self._lock_views(baseline_points[rows[0]], followup_points[rows[0]])
            self._focus_on("baseline", baseline_points[rows[0]])
        self._update_prompt_status()
        self._refresh_point_table()

        if not self._track_all_pending:
            self._begin_verification([row for row in rows if row not in failures])
        if self._track_all_pending:
            self._track_all_pending = False
            # rows that failed to register are not real follow-up points, so Track skips them
            self._accept_and_segment_all(exclude=failures)

    # --------------------------------------------------------------- segment -
    def _on_segment_all(self) -> None:
        baseline_points = self._baseline_points()
        rows = [
            row for row in range(len(baseline_points))
            if self._get_at(self._accepted, row, False) and self._row_has_followup(row)
        ]
        if not rows:
            self._log("No accepted prompt is ready to segment.")
            return
        self._segment_rows(rows)

    def _segment_rows(self, rows: list[int]) -> None:
        try:
            _, baseline_path, _, followup_path = self._require_paths()
            baseline_points = self._baseline_points()
            followup_points = self._followup_points()
        except Exception as error:
            QMessageBox.information(self, "Not ready", str(error))
            return
        rows = [row for row in rows if self._row_has_followup(row)]
        if not rows:
            QMessageBox.information(self, "Nothing to segment", "Register a follow-up point first.")
            return

        self._lock_views(baseline_points[rows[0]], followup_points[rows[0]])

        location = self.model_location.text().strip() or None
        local_model = self.model_source.currentIndex() == 1
        emit = self._emit
        disable_tta = self._quality_mode()[2]

        # a row gets a fresh, permanent id the first time it is segmented; re-segmenting (e.g.
        lesion_ids = []
        for row in rows:
            existing = self._get_at(self._lesions, row, None)
            if existing is None:
                self._lesion_counter += 1
                existing = str(self._lesion_counter)
            lesion_ids.append(existing)
        proposals = [self._get_at(self._registration_proposals, row, None) for row in rows]
        pairs = [(baseline_points[row], followup_points[row]) for row in rows]

        self._log(f"Segmenting {len(rows)} lesion{'s' if len(rows) != 1 else ''}...")

        def work():
            if self._backend.ensure_started(progress=emit):
                self._backend_initialized = False
            backend_baseline_path = self._backend_scan_path(baseline_path)
            backend_followup_path = self._backend_scan_path(followup_path)
            if not self._backend_initialized:
                # a local backend opens the folder itself; only a remote one needs a copy
                backend_location = (
                    self._backend.upload_model_folder(location, emit)
                    if local_model and getattr(self._backend, "is_remote", False)
                    else location
                )
                self._backend.initialize(backend_location, device=None, progress=emit)
                self._backend_initialized = True

            results = []
            for number, ((baseline_point, followup_point), lesion_id, proposal) in enumerate(
                zip(pairs, lesion_ids, proposals, strict=True), start=1
            ):
                emit(f"--- lesion {number}/{len(pairs)} ---")
                result = self._backend.track(
                    backend_baseline_path,
                    baseline_point,
                    backend_followup_path,
                    followup_point,
                    segment_baseline=True,
                    lesion_id=lesion_id,
                    lesion_number=int(lesion_id),
                    propagated_point=proposal,
                    disable_tta=disable_tta,
                    progress=emit,
                )
                results.append(result)
            return rows, lesion_ids, results

        self._start(work, self._on_segmented, "Segmentation failed",
                    f"Segmenting {len(rows)} lesion" + ("s" if len(rows) != 1 else ""))

    @staticmethod
    def _summarize(result: dict) -> str:
        # the same fields as TrackingResult.summary(), as a plain dict
        parts = [f"follow-up: {result['followup']['volume_ml']:.2f} ml ({result['followup']['voxels']} voxels)"]
        baseline = result.get("baseline")
        if baseline is not None:
            parts.insert(0, f"baseline: {baseline['volume_ml']:.2f} ml ({baseline['voxels']} voxels)")
            change = result["followup"]["volume_ml"] - baseline["volume_ml"]
            relative = f", {100.0 * change / baseline['volume_ml']:+.1f}%" if baseline["volume_ml"] > 0 else ""
            parts.append(f"change: {change:+.2f} ml{relative}")
        return " | ".join(parts)

    def _on_segmented(self, payload) -> None:
        rows, lesion_ids, results = payload
        try:
            self._require_paths()
        except Exception as error:
            self._log(f"Segmentation finished but the scan selection changed in the meantime: {error}")
            return

        # only this row's segmentation is replaced; every other row is left alone
        for row, lesion_id, result in zip(rows, lesion_ids, results, strict=True):
            volumes = tuple(
                (result.get(role) or {}).get("volume_ml") for role in ("baseline", "followup")
            )
            if volumes[1] is not None:
                self._row_volumes[lesion_id] = volumes
            old_layer_names = self._lesion_layers.get(lesion_id, {})
            new_layer_names: dict[str, str] = {}
            for role, mask_payload in (("followup", result["followup"]), ("baseline", result["baseline"])):
                if mask_payload is None:
                    continue
                bounds = mask_payload.get("bounds")
                if bounds is None:
                    self._log(f"The {ROLE_WORDS[role]} result has no local bounds; refusing to render it.")
                    continue
                name = self._add_labels(
                    role,
                    mask_payload["mask"].astype(np.uint16),
                    bounds=tuple(tuple(int(value) for value in pair) for pair in bounds),
                    suffix=lesion_id,
                )
                if name is not None:
                    new_layer_names[role] = name
            # a role that lost its layer is cleaned up rather than left stale
            for role, old_name in old_layer_names.items():
                if role not in new_layer_names:
                    layer = self._layer(role, old_name)
                    if layer is not None:
                        self.vm[role].layers.remove(layer)
            self._lesion_layers[lesion_id] = new_layer_names
            self._set_at(self._lesions, row, lesion_id, None)
            # the mask stands in for its prompt; the row's toggle brings the point back
            self._on_toggle_prompt_shown(row, False)
            self._log(f"lesion (row {row + 1}): {self._summarize(result)}")

        self.export_button.setEnabled(any(lesion_id is not None for lesion_id in self._lesions))
        self._refresh_point_table()

    def _on_export(self) -> None:
        lesion_ids = [lesion_id for lesion_id in self._lesions if lesion_id is not None]
        if not lesion_ids:
            QMessageBox.information(self, "Nothing to export", "Run a segmentation first.")
            return
        folder = self._export_folder
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Choose export location")
            if not folder:
                return
            self._export_folder = folder
            self._log(f"Export location: {folder}")

        choice, accepted = QInputDialog.getItem(
            self,
            "Export masks",
            "Materialize masks for:",
            ["Baseline and follow-up", "Baseline only", "Follow-up only"],
            0,
            False,
        )
        if not accepted:
            return
        timepoints = {
            "Baseline and follow-up": ["baseline", "followup"],
            "Baseline only": ["baseline"],
            "Follow-up only": ["followup"],
        }[choice]

        emit = self._emit

        def work():
            # the backend already holds these lesions: nothing to send along
            return self._backend.export(folder, lesion_ids, timepoints=timepoints, progress=emit)

        self._start(work, self._on_exported, "Export failed", "Writing files")

    def _on_exported(self, written) -> None:
        self._log(f"Exported {len(written)} file(s).")

    def _add_labels(
        self,
        role: str,
        data: np.ndarray,
        bounds: tuple[tuple[int, int], ...] | None = None,
        suffix: str = "",
    ) -> str | None:
        reference = self._ct[role]
        if reference is None:
            return None
        name = SEG_LAYER[role] if not suffix else f"{SEG_LAYER[role]} {suffix}"
        try:
            color = self._row_color(int(suffix) - 1) if suffix else LESION_COLORS[0]
        except ValueError:
            color = LESION_COLORS[0]
        colormap = self._mask_colormap(data, color)
        translate = reference.translate
        if bounds is not None:
            origin = np.asarray(reference.translate, dtype=float)
            scale = np.asarray(reference.scale, dtype=float)
            offset = np.asarray([start for start, _ in bounds], dtype=float)
            translate = origin + scale * offset
        existing = self._layer(role, name)
        if existing is not None:
            existing.data = data
            existing.scale = reference.scale
            existing.translate = translate
            existing.colormap = colormap
            existing.visible = True
            return name
        self.vm[role].add_labels(
            data, name=name, scale=reference.scale, translate=translate, opacity=0.55, colormap=colormap
        )
        return name

    def _prepare_lesion_layers(self, row: int) -> None:
        """Build this lesion's (empty) label layers now, so segmenting only fills them."""
        lesion_id = self._get_at(self._lesions, row, None)
        if lesion_id is None:
            lesion_id = str(self._lesion_counter + 1)
        for role in ROLES:
            if self._ct[role] is None:
                continue
            name = f"{SEG_LAYER[role]} {lesion_id}"
            if self._layer(role, name) is None:
                layer = self.vm[role].add_labels(
                    np.zeros((1, 1, 1), dtype=np.uint16), name=name, scale=self._ct[role].scale, opacity=0.55
                )
                layer.visible = False

    # -------------------------------------------------------------- activity -
    def _start_activity(self, channel: str, what: str, blend: bool = False) -> None:
        import time

        self._activity_labels[channel] = what or "Working"
        if len(self._activity_labels) == 1:
            # the first channel to start owns the elapsed-time origin
            self._activity_started = time.monotonic()
            self._activity_phase = 0.0
        if blend:
            self._blend_channels.add(channel)
            self._add_blend_overlay()
        self._tick_activity()
        self._activity.start()

    def _stop_activity(self, channel: str) -> None:
        self._activity_labels.pop(channel, None)
        self._blend_channels.discard(channel)
        if self._activity_labels:
            # another channel is still running -- keep the spinner and label going
            self._tick_activity()
            return
        self._activity.stop()
        self.activity.setText("")
        self._remove_blend_overlay()

    def _tick_activity(self) -> None:
        import math
        import time

        elapsed = time.monotonic() - self._activity_started
        dots = "." * (1 + int(elapsed * 2) % 3)
        label = " / ".join(self._activity_labels.values()) or "Working"
        self.activity.setText(f"{label}{dots}  {elapsed:.0f}s")
        overlay = self._layer("followup", BLEND_LAYER)
        if overlay is None:
            return
        # the baseline swells in and out over the follow-up while they are registered
        self._activity_phase += 0.12
        overlay.opacity = 0.5 - 0.5 * math.cos(self._activity_phase)

    def _add_blend_overlay(self) -> None:
        baseline, followup = self._ct["baseline"], self._ct["followup"]
        if baseline is None or followup is None:
            return
        self._remove_blend_overlay()
        vm = self.vm["followup"]
        point, active = self._panel_state(vm)
        # the baseline's own array, never a copy; additive blending keeps the follow-up bright
        overlay = vm.add_image(
            baseline.data,
            name=BLEND_LAYER,
            scale=baseline.scale,
            translate=baseline.translate,
            contrast_limits=list(baseline.contrast_limits),
            colormap="bop orange",
            blending="additive",
            opacity=0.0,
        )
        # an Image left on 'mean' would turn the overlay into a mean projection
        overlay.projection_mode = "none"
        self._restore_panel_state(vm, point, active)

    def _remove_blend_overlay(self) -> None:
        vm = self.vm["followup"]
        overlay = self._layer("followup", BLEND_LAYER)
        if overlay is None:
            return
        point, active = self._panel_state(vm)
        vm.layers.remove(overlay)
        self._restore_panel_state(vm, point, active)

    def _panel_state(self, vm) -> tuple:
        # otherwise the overlay moves the panel's slider and steals the active layer
        active = vm.layers.selection.active
        if active is not None and active.name == BLEND_LAYER:
            active = None
        return (vm.dims.point[0] if vm.dims.ndim else None), active

    def _restore_panel_state(self, vm, point, active) -> None:
        if point is not None:
            vm.dims.set_point(0, point)
        if active is not None and active in vm.layers:
            vm.layers.selection.active = active

    # ----------------------------------------------------------------- reset -
    def _remove_all_segmentation_layers(self) -> None:
        for role in ROLES:
            name = SEG_LAYER[role]
            for layer in list(self.vm[role].layers):
                if layer.name == name or layer.name.startswith(f"{name} "):
                    self.vm[role].layers.remove(layer)
        self._lesion_layers = {}

    def _reset_session(self) -> None:
        self._lesions = []
        self._custom_colors = []
        self._anchors = []
        self._registration_proposals = []
        self._accepted = []
        self._locked_points = {"baseline": [], "followup": []}
        self._placeholder = []
        self._lesion_counter = 0
        self._remove_all_segmentation_layers()
        # the backend's result cache is keyed on scan paths, so nothing else to clear
        self._slice_offset = 0.0
        self._focus = 0
        self.export_button.setEnabled(False)
        self._invalidate_registration()

    def _on_clear_everything(self) -> None:
        # back to right after model initialization: no scans, prompts or results
        removed = sum(len(layer.data) for role in ROLES if (layer := self._layer(role, POINTS_LAYER[role])))
        for role in ROLES:
            if self._ct[role] is None:
                continue
            path = layer_path(self._ct[role])
            self.vm[role].layers.clear()
            self._ct[role] = None
            if path is not None:
                self._queue_scan_release(path)
        self._reset_session()
        # waited for: nothing should look cleared while the backend still drops tensors
        self._invalidate_registration(clear_scans=True, wait=True)
        self._refresh_scan_labels()
        self._update_prompt_status()
        self._refresh_point_table()
        self._log(f"Cleared everything ({removed} point(s), both scans, cached GPU data).")

    def _on_remove_scan(self, role: str) -> None:
        if self._ct[role] is None:
            return
        if self._refuse_if_busy(f"removing the {ROLE_WORDS[role]} scan"):
            return
        path = layer_path(self._ct[role])
        self.vm[role].layers.clear()
        self._ct[role] = None
        self._reset_session()
        if path is not None:
            self._queue_scan_release(path)
        self._refresh_scan_labels()
        self._log(f"Removed the {ROLE_WORDS[role]} scan.")

    # --------------------------------------------------------------- workers -
    def _start(
        self, job, on_return, error_title: str, what: str = "Working", blend: bool = False, lock_quality: bool = True
    ) -> None:
        if self._busy:
            # the log line and the activity indicator already say something is running
            self._log("Something is already running; wait for it to finish.")
            return
        self._job_token += 1
        token = self._job_token
        self._busy = True
        self._pending = (on_return, error_title)
        self._set_busy(True, what, blend, lock_quality)
        self._pool.submit(self._run, job, token)

    def _run(self, job, token: int) -> None:
        # runs on the pool thread; results cross back as queued Qt signals
        try:
            result = job()
        except BaseException as error:  # noqa: BLE001 - reported in the GUI, not swallowed
            self._bridge.failed.emit((token, error))
        else:
            self._bridge.done.emit((token, result))

    def _finish(self) -> tuple:
        pending = self._pending or (None, "Failed")
        self._pending = None
        self._busy = False
        self._set_busy(False)
        return pending

    def _on_cancel(self) -> None:
        if not self._busy and not self._model_busy:
            return
        # bumping the token is enough: a late result carries the old one and is dropped
        message = self._cancel_backend_work()
        if self._busy:
            self._job_token += 1
            # otherwise this would contaminate a later, unrelated propagate
            self._track_all_pending = False
            self._finish()
        if self._model_busy:
            self._model_job_token += 1
            self._finish_model()
        self._log(message)

    def _cancel_backend_work(self) -> str:
        """Stop waiting for the backend; never restart it.

        Closing the socket frees the panel at once; the call already on the GPU runs itself
        out and its result is dropped.
        """
        cancel = getattr(self._backend, "cancel_active", None)
        if cancel is not None:
            cancel()
        return "Cancelled. The step already on the GPU finishes by itself; its result is dropped."

    def _on_job_done(self, payload) -> None:
        token, result = payload
        if token != self._job_token:
            return  # cancelled (or superseded) -- _finish() already ran when that happened
        on_return, _ = self._finish()
        if on_return is not None:
            on_return(result)

    def _on_job_failed(self, payload) -> None:
        token, error = payload
        if token != self._job_token:
            return
        _, error_title = self._finish()
        self._error(error_title, error)

    # ----------------------------------------------------------- model workers -
    def _start_model(self, job, on_return, error_title: str, what: str = "Working") -> None:
        # mirrors the session channel above, on its own pool
        if self._model_busy:
            self._model_queued = (job, on_return, error_title, what)
            self._log(f"{what} queued; it starts as soon as the current step finishes.")
            return
        self._model_job_token += 1
        token = self._model_job_token
        self._model_busy = True
        self._model_pending = (on_return, error_title)
        self._set_model_busy(True, what)
        self._model_pool.submit(self._run_model, job, token)

    def _run_model(self, job, token: int) -> None:
        try:
            result = job()
        except BaseException as error:  # noqa: BLE001 - reported in the GUI, not swallowed
            self._bridge.model_failed.emit((token, error))
        else:
            self._bridge.model_done.emit((token, result))

    def _finish_model(self) -> tuple:
        pending = self._model_pending or (None, "Failed")
        self._model_pending = None
        self._model_busy = False
        self._set_model_busy(False)
        return pending

    def _on_model_job_done(self, payload) -> None:
        token, result = payload
        if token != self._model_job_token:
            return
        on_return, _ = self._finish_model()
        if on_return is not None:
            on_return(result)
        self._start_queued_model_job()

    def _on_model_job_failed(self, payload) -> None:
        token, error = payload
        if token != self._model_job_token:
            return
        _, error_title = self._finish_model()
        self._error(error_title, error)
        self._start_queued_model_job()

    def _start_queued_model_job(self) -> None:
        queued, self._model_queued = self._model_queued, None
        if queued is not None:
            self._start_model(*queued)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # napari's own docks are already sized; a sizeHint alone does not claim width
        if self._width_claimed:
            return
        # deferred: showEvent can fire before napari restores its layout
        QTimer.singleShot(0, self._claim_panel_width)

    def _claim_panel_width(self) -> None:
        dock = self.parent()
        while dock is not None and not isinstance(dock, QDockWidget):
            dock = dock.parent()
        if dock is None:
            return
        window = dock.window()
        if not isinstance(window, QMainWindow):
            return
        # just wider than the widest row needs; set once, then the width is the user's
        target = self._content.sizeHint().width() + 28
        self._width_claim_attempts += 1
        if dock.width() < target:
            window.resizeDocks([dock], [target], Qt.Orientation.Horizontal)
            if self._width_claim_attempts < 8:
                QTimer.singleShot(150, self._claim_panel_width)
                return
        self._width_claimed = True

    def closeEvent(self, event):
        self._pool.shutdown(wait=False)
        self._preload_pool.shutdown(wait=False)
        # best-effort: closing the window must not hang on the backend process
        if self._backend.is_running():
            threading.Thread(target=self._backend.shutdown, daemon=True).start()
        for qtv in self._qtv.values():
            try:
                qtv.close()
            except Exception:  # noqa: BLE001 - shutting down, nothing to report to
                pass
        super().closeEvent(event)
