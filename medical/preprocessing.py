"""
Medical Image Preprocessing
============================
Resampling, normalization, windowing, and synthetic volume generation.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from medical.loader import MedicalImage


# ──────────────────────────────────────
# Window/Level Presets (center, width)
# ──────────────────────────────────────
WINDOW_PRESETS = {
    "CT Bone":        {"center": 300,  "width": 1500},
    "CT Lung":        {"center": -600, "width": 1500},
    "CT Soft Tissue": {"center": 40,   "width": 400},
    "CT Brain":       {"center": 40,   "width": 80},
    "CT Abdomen":     {"center": 40,   "width": 350},
    "CT Liver":       {"center": 60,   "width": 150},
    "MRI Default":    {"center": 500,  "width": 1000},
}


def apply_window_level(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply window/level (contrast) adjustment to a volume.

    Returns values in [0, 1] range.
    """
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed.astype(np.float32)


def normalize_min_max(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] using min-max scaling."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin == 0:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def normalize_zscore(volume: np.ndarray) -> np.ndarray:
    """Normalize volume using z-score (zero mean, unit variance)."""
    mean = volume.mean()
    std = volume.std()
    if std == 0:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - mean) / std).astype(np.float32)


def resample_isotropic(medical_image: MedicalImage, new_spacing=(1.0, 1.0, 1.0)) -> MedicalImage:
    """Resample a medical image to isotropic spacing.

    Args:
        medical_image: Input MedicalImage
        new_spacing: Target spacing in mm (x, y, z)

    Returns:
        Resampled MedicalImage
    """
    original_spacing = medical_image.sitk_image.GetSpacing()
    original_size = medical_image.sitk_image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(medical_image.sitk_image.GetDirection())
    resampler.SetOutputOrigin(medical_image.sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(
        float(medical_image.sitk_image.GetPixelIDValue())
        if medical_image.sitk_image.GetPixelID() == sitk.sitkFloat32
        else -1024
    )
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled = resampler.Execute(medical_image.sitk_image)
    return MedicalImage(resampled)


def generate_synthetic_volume(size=(128, 128, 64), spacing=(1.0, 1.0, 2.0)):
    """Generate a synthetic CT-like volume with embedded 'tumors' for testing.

    Creates a volume with:
    - Background at -1000 HU (air)
    - Body outline at 0-40 HU (soft tissue)
    - Embedded spherical 'tumors' at 60-100 HU
    - Bone-like structures at 300-700 HU

    Returns:
        MedicalImage with realistic-looking synthetic data
    """
    sx, sy, sz = size
    volume = np.full((sz, sy, sx), -1000, dtype=np.float32)  # Air

    # Create body ellipsoid (soft tissue)
    zz, yy, xx = np.mgrid[0:sz, 0:sy, 0:sx]
    cx, cy, cz = sx // 2, sy // 2, sz // 2
    body = (
        ((xx - cx) / (sx * 0.38)) ** 2 +
        ((yy - cy) / (sy * 0.42)) ** 2 +
        ((zz - cz) / (sz * 0.45)) ** 2
    ) < 1.0
    volume[body] = np.random.normal(30, 10, body.sum()).astype(np.float32)

    # Add spine (bone)
    spine = (
        ((xx - cx) / 8) ** 2 +
        ((yy - cy * 1.4) / 10) ** 2
    ) < 1.0
    spine_mask = body & spine
    volume[spine_mask] = np.random.normal(500, 100, spine_mask.sum()).astype(np.float32)

    # Add tumors (3 spheres at different locations)
    tumors = [
        (cx - 20, cy - 10, cz, 8),   # x, y, z, radius
        (cx + 15, cy + 5, cz + 5, 6),
        (cx - 5, cy + 15, cz - 8, 10),
    ]

    tumor_mask = np.zeros_like(volume, dtype=bool)
    for tx, ty, tz, tr in tumors:
        dist = np.sqrt(
            (xx - tx) ** 2 + (yy - ty) ** 2 + (zz - tz) ** 2
        )
        sphere = dist < tr
        valid = body & sphere
        volume[valid] = np.random.normal(80, 15, valid.sum()).astype(np.float32)
        tumor_mask[valid] = True

    # Convert to SimpleITK
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin((0.0, 0.0, 0.0))

    return MedicalImage(sitk_image), tumor_mask
