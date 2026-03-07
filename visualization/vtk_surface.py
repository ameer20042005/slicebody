"""
VTK Surface Extraction
======================
Marching cubes surface extraction from segmentation masks.
Generates meshes for 3D visualization and STL export.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def extract_surface(mask: np.ndarray, spacing=(1.0, 1.0, 1.0),
                    threshold: float = 0.5,
                    smooth_iterations: int = 25,
                    color=(0.9, 0.3, 0.3)) -> vtk.vtkActor:
    """Extract an isosurface from a binary mask using marching cubes.

    Args:
        mask: Binary mask (Z, Y, X)
        spacing: Voxel spacing
        threshold: Isosurface threshold
        smooth_iterations: Number of smoothing iterations
        color: RGB surface color

    Returns:
        vtkActor with the extracted surface
    """
    depth, height, width = mask.shape

    # Create vtkImageData
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(width, height, depth)
    vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_data.SetOrigin(0, 0, 0)

    flat = mask.astype(np.float32).flatten(order='C')
    vtk_array = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data.GetPointData().SetScalars(vtk_array)

    # Marching Cubes
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(vtk_data)
    contour.SetValue(0, threshold)
    contour.ComputeNormalsOn()
    contour.Update()

    # Smooth the surface
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(smooth_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(0.001)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(smoother.GetOutputPort())
    mapper.ScalarVisibilityOff()

    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(0.85)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)

    return actor


def export_stl(mask: np.ndarray, filepath: str, spacing=(1.0, 1.0, 1.0),
               threshold: float = 0.5):
    """Export a segmentation mask as an STL mesh file.

    Args:
        mask: Binary mask (Z, Y, X)
        filepath: Output STL path
        spacing: Voxel spacing
        threshold: Isosurface threshold
    """
    depth, height, width = mask.shape

    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(width, height, depth)
    vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])

    flat = mask.astype(np.float32).flatten(order='C')
    vtk_array = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data.GetPointData().SetScalars(vtk_array)

    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(vtk_data)
    contour.SetValue(0, threshold)
    contour.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(contour.GetOutputPort())
    writer.Write()
