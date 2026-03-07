"""
3D Medical Image Viewer
========================
VTK render window embedded in PyQt5 for interactive 3D visualization.
Supports volume rendering and surface extraction.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QLabel, QFrame
)
from PyQt5.QtCore import Qt

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from visualization.vtk_volume import (
    numpy_to_vtk_image, create_volume_rendering, create_mask_rendering
)
from visualization.vtk_surface import extract_surface
from utils.config import VTK_PRESETS


class Viewer3DWidget(QWidget):
    """Interactive 3D viewer with VTK embedded in PyQt5."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume_data = None
        self.mask_data = None
        self.spacing = (1.0, 1.0, 1.0)
        self.current_actors = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Controls
        controls = QHBoxLayout()

        preset_label = QLabel("Render Preset:")
        preset_label.setStyleSheet("font-weight: bold; color: #74b9ff;")
        self.preset_combo = QComboBox()
        for name in VTK_PRESETS.keys():
            self.preset_combo.addItem(name)

        self.btn_volume = QPushButton("🔲 Volume Render")
        self.btn_surface = QPushButton("🔺 Surface View")
        self.btn_reset = QPushButton("🔄 Reset Camera")
        self.btn_clear = QPushButton("❌ Clear")

        self.btn_volume.clicked.connect(self._show_volume_rendering)
        self.btn_surface.clicked.connect(self._show_surface_view)
        self.btn_reset.clicked.connect(self._reset_camera)
        self.btn_clear.clicked.connect(self._clear_scene)

        controls.addWidget(preset_label)
        controls.addWidget(self.preset_combo)
        controls.addWidget(self.btn_volume)
        controls.addWidget(self.btn_surface)
        controls.addWidget(self.btn_reset)
        controls.addWidget(self.btn_clear)
        controls.addStretch()

        layout.addLayout(controls)

        # VTK Widget
        frame = QFrame()
        frame.setStyleSheet(
            "border: 2px solid #0f3460; border-radius: 8px; background-color: #0d1b2a;"
        )
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(2, 2, 2, 2)

        self.vtk_widget = QVTKRenderWindowInteractor(frame)
        frame_layout.addWidget(self.vtk_widget)
        layout.addWidget(frame, stretch=1)

        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.07, 0.12)  # Dark background
        self.renderer.SetBackground2(0.02, 0.03, 0.08)
        self.renderer.GradientBackgroundOn()

        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Interactor style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        # Orientation marker (axes widget)
        self._add_orientation_marker()

    def _add_orientation_marker(self):
        """Add XYZ orientation axes in the bottom-left corner."""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(20, 20, 20)
        axes.SetShaftTypeToCylinder()

        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(
            self.vtk_widget.GetRenderWindow().GetInteractor()
        )
        self.orientation_widget.SetViewport(0.0, 0.0, 0.15, 0.15)
        self.orientation_widget.EnabledOn()
        self.orientation_widget.InteractiveOff()

    def initialize(self):
        """Initialize the VTK interactor. Call after the widget is shown."""
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def set_volume(self, volume: np.ndarray, spacing=(1.0, 1.0, 1.0)):
        """Set volume data for 3D rendering."""
        self.volume_data = volume
        self.spacing = spacing

    def set_mask(self, mask: np.ndarray):
        """Set segmentation mask for rendering."""
        self.mask_data = mask

    def _clear_scene(self):
        """Remove all actors from the scene."""
        for actor in self.current_actors:
            self.renderer.RemoveActor(actor)
            self.renderer.RemoveVolume(actor)
        self.current_actors.clear()
        self.vtk_widget.GetRenderWindow().Render()

    def _show_volume_rendering(self):
        """Display volume rendering of the main image + mask overlay."""
        if self.volume_data is None:
            return

        self._clear_scene()
        preset = self.preset_combo.currentText()

        # Main volume
        vtk_image = numpy_to_vtk_image(self.volume_data, self.spacing)
        vol_actor = create_volume_rendering(vtk_image, preset)
        self.renderer.AddVolume(vol_actor)
        self.current_actors.append(vol_actor)

        # Mask overlay
        if self.mask_data is not None:
            mask_actor = create_mask_rendering(
                self.mask_data, self.spacing,
                color=(1.0, 0.27, 0.38), opacity=0.6
            )
            self.renderer.AddVolume(mask_actor)
            self.current_actors.append(mask_actor)

        self._reset_camera()

    def _show_surface_view(self):
        """Display surface extraction of the segmentation mask."""
        if self.mask_data is None:
            return

        self._clear_scene()

        # Surface from mask
        surface_actor = extract_surface(
            self.mask_data, self.spacing,
            color=(0.9, 0.27, 0.35)
        )
        self.renderer.AddActor(surface_actor)
        self.current_actors.append(surface_actor)

        self._reset_camera()

    def _reset_camera(self):
        """Reset the camera to show the full scene."""
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Azimuth(30)
        self.renderer.GetActiveCamera().Elevation(20)
        self.vtk_widget.GetRenderWindow().Render()
