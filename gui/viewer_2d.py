"""
2D Medical Image Viewer
========================
Three-plane slice viewer (Axial/Sagittal/Coronal) with matplotlib
embedded in PyQt5. Supports mask overlay, window/level, brightness,
contrast, invert, and scrolling.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel,
    QComboBox, QFrame, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from medical.preprocessing import WINDOW_PRESETS, apply_window_level


class SliceCanvas(FigureCanvas):
    """Matplotlib canvas for displaying a single slice."""

    def __init__(self, title: str = "Axial", parent=None):
        self.fig = Figure(figsize=(4, 4), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.title = title
        self.ax.set_facecolor('#0d1b2a')
        self.ax.set_title(title, color='#74b9ff', fontsize=11, fontweight='bold', pad=8)
        self.ax.tick_params(colors='#555', labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color('#0f3460')

        self._image_handle = None
        self._mask_handle = None
        self.fig.tight_layout(pad=1.5)

    def display_slice(self, slice_data: np.ndarray, mask_data: np.ndarray = None):
        """Display a 2D slice with optional mask overlay."""
        if self._image_handle is None:
            self._image_handle = self.ax.imshow(
                slice_data, cmap='gray', aspect='auto',
                interpolation='bilinear'
            )
        else:
            self._image_handle.set_data(slice_data)
            self._image_handle.set_clim(vmin=slice_data.min(), vmax=slice_data.max())

        # Mask overlay
        if mask_data is not None:
            mask_rgba = np.zeros((*mask_data.shape, 4), dtype=np.float32)
            mask_rgba[mask_data > 0] = [1.0, 0.27, 0.38, 0.45]

            if self._mask_handle is None:
                self._mask_handle = self.ax.imshow(
                    mask_rgba, aspect='auto', interpolation='nearest'
                )
            else:
                self._mask_handle.set_data(mask_rgba)
        elif self._mask_handle is not None:
            self._mask_handle.set_data(
                np.zeros((*slice_data.shape, 4), dtype=np.float32)
            )

        self.draw_idle()

    def clear_mask(self):
        """Remove the mask overlay."""
        if self._mask_handle is not None:
            self._mask_handle.set_data(
                np.zeros((2, 2, 4), dtype=np.float32)
            )
            self._mask_handle = None
            self.draw_idle()


class Viewer2DWidget(QWidget):
    """Three-plane medical image viewer with advanced controls."""

    slice_changed = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume = None
        self.mask = None
        self.brightness = 0.0
        self.contrast = 1.0
        self.invert = False

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ── Top Control Bar ──
        control_bar = QHBoxLayout()

        # Window/Level preset
        wl_label = QLabel("Window:")
        wl_label.setStyleSheet("font-weight: bold; color: #74b9ff;")
        self.wl_combo = QComboBox()
        self.wl_combo.addItem("Auto (Min-Max)")
        for name in WINDOW_PRESETS.keys():
            self.wl_combo.addItem(name)
        self.wl_combo.currentTextChanged.connect(self._on_controls_changed)
        control_bar.addWidget(wl_label)
        control_bar.addWidget(self.wl_combo)

        control_bar.addSpacing(10)

        # Brightness
        br_label = QLabel("Brightness:")
        br_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedWidth(100)
        self.brightness_slider.valueChanged.connect(self._on_controls_changed)
        self.br_value_label = QLabel("0")
        self.br_value_label.setStyleSheet("color: #a0a0c0; font-size: 11px; min-width: 25px;")
        control_bar.addWidget(br_label)
        control_bar.addWidget(self.brightness_slider)
        control_bar.addWidget(self.br_value_label)

        control_bar.addSpacing(10)

        # Contrast
        ct_label = QLabel("Contrast:")
        ct_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(100)
        self.contrast_slider.valueChanged.connect(self._on_controls_changed)
        self.ct_value_label = QLabel("1.0")
        self.ct_value_label.setStyleSheet("color: #a0a0c0; font-size: 11px; min-width: 30px;")
        control_bar.addWidget(ct_label)
        control_bar.addWidget(self.contrast_slider)
        control_bar.addWidget(self.ct_value_label)

        control_bar.addSpacing(10)

        # Invert checkbox
        self.invert_check = QCheckBox("Invert")
        self.invert_check.setStyleSheet("color: #a0a0c0;")
        self.invert_check.stateChanged.connect(self._on_controls_changed)
        control_bar.addWidget(self.invert_check)

        control_bar.addStretch()

        # Slice info label
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("color: #00cec9; font-weight: bold;")
        control_bar.addWidget(self.info_label)

        main_layout.addLayout(control_bar)

        # ── Three canvases ──
        canvas_layout = QHBoxLayout()

        self.axial_canvas = SliceCanvas("Axial")
        self.sagittal_canvas = SliceCanvas("Sagittal")
        self.coronal_canvas = SliceCanvas("Coronal")

        for canvas in [self.axial_canvas, self.sagittal_canvas, self.coronal_canvas]:
            frame = QFrame()
            frame.setStyleSheet("border: 1px solid #0f3460; border-radius: 6px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(2, 2, 2, 2)
            fl.addWidget(canvas)
            canvas_layout.addWidget(frame)

        main_layout.addLayout(canvas_layout, stretch=1)

        # ── Sliders ──
        slider_layout = QHBoxLayout()

        self.axial_slider = self._make_slider("Axial")
        self.sagittal_slider = self._make_slider("Sagittal")
        self.coronal_slider = self._make_slider("Coronal")

        slider_layout.addLayout(self.axial_slider[0])
        slider_layout.addLayout(self.sagittal_slider[0])
        slider_layout.addLayout(self.coronal_slider[0])

        main_layout.addLayout(slider_layout)

    def _make_slider(self, name: str):
        layout = QVBoxLayout()
        label = QLabel(f"{name}: 0")
        label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        label.setAlignment(Qt.AlignCenter)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(0)
        slider.setValue(0)
        slider.valueChanged.connect(lambda v, n=name, l=label: self._on_slice_changed(n, v, l))

        layout.addWidget(label)
        layout.addWidget(slider)
        return layout, slider, label

    def _on_slice_changed(self, plane: str, value: int, label: QLabel):
        label.setText(f"{plane}: {value}")
        self._update_slices()
        self.slice_changed.emit(plane, value)

    def _on_controls_changed(self, *args):
        self.brightness = self.brightness_slider.value() / 100.0
        self.contrast = self.contrast_slider.value() / 100.0
        self.invert = self.invert_check.isChecked()
        self.br_value_label.setText(str(self.brightness_slider.value()))
        self.ct_value_label.setText(f"{self.contrast:.1f}")
        self._update_slices()

    def set_volume(self, volume: np.ndarray):
        """Set the 3D volume to display."""
        self.volume = volume.astype(np.float32)
        self.mask = None

        z, y, x = volume.shape

        # Block signals while updating sliders to prevent IndexError
        for _, slider, _ in [self.axial_slider, self.sagittal_slider, self.coronal_slider]:
            slider.blockSignals(True)

        self.axial_slider[1].setMaximum(z - 1)
        self.axial_slider[1].setValue(z // 2)

        self.sagittal_slider[1].setMaximum(x - 1)
        self.sagittal_slider[1].setValue(x // 2)

        self.coronal_slider[1].setMaximum(y - 1)
        self.coronal_slider[1].setValue(y // 2)

        # Unblock signals
        for _, slider, _ in [self.axial_slider, self.sagittal_slider, self.coronal_slider]:
            slider.blockSignals(False)

        self.info_label.setText(f"Volume: {x}x{y}x{z}")
        self._update_slices()

    def set_mask(self, mask: np.ndarray):
        self.mask = mask
        self._update_slices()

    def clear_mask(self):
        self.mask = None
        self.axial_canvas.clear_mask()
        self.sagittal_canvas.clear_mask()
        self.coronal_canvas.clear_mask()

    def _get_windowed_volume(self) -> np.ndarray:
        """Apply window/level + brightness/contrast to the volume."""
        if self.volume is None:
            return None

        preset = self.wl_combo.currentText()
        if preset == "Auto (Min-Max)":
            vmin, vmax = self.volume.min(), self.volume.max()
            if vmax - vmin == 0:
                result = np.zeros_like(self.volume)
            else:
                result = (self.volume - vmin) / (vmax - vmin)
        else:
            p = WINDOW_PRESETS[preset]
            result = apply_window_level(self.volume, p["center"], p["width"])

        # Apply brightness and contrast
        result = self.contrast * (result - 0.5) + 0.5 + self.brightness
        result = np.clip(result, 0, 1)

        # Invert
        if self.invert:
            result = 1.0 - result

        return result.astype(np.float32)

    def _update_slices(self):
        """Redraw all three slice views."""
        if self.volume is None:
            return

        windowed = self._get_windowed_volume()
        z, y, x = self.volume.shape

        # Clamp values to valid range
        az = min(self.axial_slider[1].value(), z - 1)
        sx = min(self.sagittal_slider[1].value(), x - 1)
        cy = min(self.coronal_slider[1].value(), y - 1)

        ax_mask = self.mask[az, :, :] if self.mask is not None else None
        sg_mask = self.mask[:, :, sx] if self.mask is not None else None
        cr_mask = self.mask[:, cy, :] if self.mask is not None else None

        self.axial_canvas.display_slice(windowed[az, :, :], ax_mask)
        self.sagittal_canvas.display_slice(windowed[:, :, sx], sg_mask)
        self.coronal_canvas.display_slice(windowed[:, cy, :], cr_mask)
