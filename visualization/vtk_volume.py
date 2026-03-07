"""
VTK Volume Rendering
====================
Volume rendering of 3D medical data using VTK.
Supports CT and MRI presets with customizable transfer functions.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from utils.config import VTK_PRESETS


def numpy_to_vtk_image(volume: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    """Convert a numpy volume to vtkImageData.

    Args:
        volume: 3D numpy array (Z, Y, X)
        spacing: Voxel spacing (X, Y, Z) in mm
    """
    depth, height, width = volume.shape

    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(width, height, depth)
    vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_data.SetOrigin(0, 0, 0)

    # Flatten in Fortran order for VTK (X varies fastest)
    flat = volume.astype(np.float32).flatten(order='C')
    vtk_array = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("ImageScalars")
    vtk_data.GetPointData().SetScalars(vtk_array)

    return vtk_data


def create_volume_property(preset_name: str = "CT Bone") -> vtk.vtkVolumeProperty:
    """Create a VTK volume property with a medical rendering preset.

    Args:
        preset_name: One of the VTK_PRESETS keys
    """
    preset = VTK_PRESETS.get(preset_name, VTK_PRESETS["CT Bone"])

    # Opacity transfer function
    opacity_tf = vtk.vtkPiecewiseFunction()
    for val, op in preset["opacity"]:
        opacity_tf.AddPoint(val, op)

    # Color transfer function
    color_tf = vtk.vtkColorTransferFunction()
    for val, (r, g, b) in preset["color"]:
        color_tf.AddRGBPoint(val, r, g, b)

    # Volume property
    vol_prop = vtk.vtkVolumeProperty()
    vol_prop.SetScalarOpacity(opacity_tf)
    vol_prop.SetColor(color_tf)
    vol_prop.SetInterpolationTypeToLinear()
    vol_prop.ShadeOn()
    vol_prop.SetAmbient(0.3)
    vol_prop.SetDiffuse(0.6)
    vol_prop.SetSpecular(0.2)
    vol_prop.SetSpecularPower(10)

    return vol_prop


def create_volume_rendering(vtk_image_data: vtk.vtkImageData,
                            preset_name: str = "CT Bone") -> vtk.vtkVolume:
    """Create a VTK volume actor for rendering.

    Args:
        vtk_image_data: The vtkImageData of the volume
        preset_name: Rendering preset name

    Returns:
        vtkVolume actor ready to add to renderer
    """
    # Smart volume mapper (GPU if available, CPU fallback)
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_image_data)
    mapper.SetBlendModeToComposite()

    # Volume property
    vol_prop = create_volume_property(preset_name)

    # Volume actor
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(vol_prop)

    return volume


def create_mask_rendering(mask: np.ndarray, spacing=(1.0, 1.0, 1.0),
                          color=(1.0, 0.3, 0.3), opacity=0.5) -> vtk.vtkVolume:
    """Create a semi-transparent volume rendering of a segmentation mask.

    Args:
        mask: Binary mask (Z, Y, X) as uint8
        spacing: Voxel spacing
        color: RGB tuple for mask color
        opacity: Maximum opacity

    Returns:
        vtkVolume actor
    """
    vtk_data = numpy_to_vtk_image(mask.astype(np.float32) * 255, spacing)

    # Opacity
    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(0, 0.0)
    opacity_tf.AddPoint(127, opacity * 0.5)
    opacity_tf.AddPoint(255, opacity)

    # Color
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(0, 0, 0, 0)
    color_tf.AddRGBPoint(255, color[0], color[1], color[2])

    vol_prop = vtk.vtkVolumeProperty()
    vol_prop.SetScalarOpacity(opacity_tf)
    vol_prop.SetColor(color_tf)
    vol_prop.SetInterpolationTypeToLinear()
    vol_prop.ShadeOff()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_data)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(vol_prop)

    return volume
