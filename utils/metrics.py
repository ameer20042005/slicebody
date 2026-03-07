"""
Metrics & Measurement Utilities
================================
Volume calculations, bounding box, surface area estimation.
"""

import numpy as np
from scipy import ndimage


def calculate_volume_cm3(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Calculate the volume of a binary mask in cm³.

    Args:
        mask: Binary mask (Z, Y, X)
        spacing: Voxel spacing (X, Y, Z) in mm

    Returns:
        Volume in cm³
    """
    voxel_vol_mm3 = float(np.prod(spacing))
    return mask.sum() * voxel_vol_mm3 / 1000.0


def calculate_bounding_box(mask: np.ndarray, spacing: np.ndarray) -> dict:
    """Calculate the bounding box of a binary mask.

    Returns:
        dict with keys: min_zyx, max_zyx, size_mm
    """
    if mask.sum() == 0:
        return {"min_zyx": (0, 0, 0), "max_zyx": (0, 0, 0), "size_mm": (0, 0, 0)}

    coords = np.argwhere(mask > 0)
    min_zyx = tuple(coords.min(axis=0).tolist())
    max_zyx = tuple(coords.max(axis=0).tolist())

    size_voxels = np.array(max_zyx) - np.array(min_zyx) + 1
    size_mm = tuple((size_voxels * spacing[::-1]).tolist())

    return {"min_zyx": min_zyx, "max_zyx": max_zyx, "size_mm": size_mm}


def estimate_surface_area_cm2(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Estimate surface area of segmented region in cm².

    Uses the simple gradient-based approach.
    """
    if mask.sum() == 0:
        return 0.0

    # Calculate surface voxels (boundary of the mask)
    eroded = ndimage.binary_erosion(mask).astype(np.uint8)
    surface = mask.astype(np.uint8) - eroded
    surface_count = surface.sum()

    # Approximate each surface voxel as having area = average of face areas
    sx, sy, sz = spacing
    avg_face_area = (sx * sy + sy * sz + sx * sz) / 3.0  # mm²

    return surface_count * avg_face_area / 100.0  # cm²


def count_components(mask: np.ndarray) -> int:
    """Count the number of connected components in a binary mask."""
    _, num = ndimage.label(mask)
    return num


def get_component_sizes_cm3(mask: np.ndarray, spacing: np.ndarray) -> list:
    """Get sizes of all connected components in cm³, sorted descending."""
    labeled, num = ndimage.label(mask)
    if num == 0:
        return []

    voxel_vol_mm3 = float(np.prod(spacing))
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    sizes_cm3 = sorted([s * voxel_vol_mm3 / 1000.0 for s in sizes], reverse=True)
    return sizes_cm3
