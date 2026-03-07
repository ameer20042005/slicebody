"""
AI Segmentation Models
======================
Baseline segmentation algorithms: Threshold, Otsu, Region Growing.
Designed with a common interface for easy plug-in of MONAI/nnU-Net models later.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from abc import ABC, abstractmethod


class BaseSegmenter(ABC):
    """Base class for all segmentation models."""

    name: str = "Base"

    @abstractmethod
    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        """Run segmentation on a 3D volume.

        Args:
            volume: 3D numpy array (Z, Y, X) in original intensity units

        Returns:
            Binary mask (Z, Y, X) as uint8 (0 or 1)
        """
        pass


class ThresholdSegmenter(BaseSegmenter):
    """Simple intensity thresholding."""

    name = "Threshold"

    def __init__(self, lower: float = 50, upper: float = 200):
        self.lower = lower
        self.upper = upper

    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        lower = kwargs.get("lower", self.lower)
        upper = kwargs.get("upper", self.upper)
        mask = ((volume >= lower) & (volume <= upper)).astype(np.uint8)
        return mask


class OtsuSegmenter(BaseSegmenter):
    """Automatic Otsu thresholding using SimpleITK."""

    name = "Otsu"

    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        sitk_image = sitk.GetImageFromArray(volume.astype(np.float32))
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(1)
        otsu_filter.SetOutsideValue(0)
        result = otsu_filter.Execute(sitk_image)
        mask = sitk.GetArrayFromImage(result).astype(np.uint8)
        return mask


class RegionGrowingSegmenter(BaseSegmenter):
    """Connected threshold region growing from a seed point."""

    name = "Region Growing"

    def __init__(self, lower: float = 30, upper: float = 120):
        self.lower = lower
        self.upper = upper

    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            volume: 3D numpy array
            seed: tuple (z, y, x) — seed point for region growing
            lower: lower intensity threshold
            upper: upper intensity threshold
        """
        seed = kwargs.get("seed", None)
        if seed is None:
            # Default: center of volume
            seed = (volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2)

        lower = kwargs.get("lower", self.lower)
        upper = kwargs.get("upper", self.upper)

        sitk_image = sitk.GetImageFromArray(volume.astype(np.float32))

        # Convert (z, y, x) to SimpleITK's (x, y, z) order
        seed_xyz = (int(seed[2]), int(seed[1]), int(seed[0]))

        seg = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[seed_xyz],
            lower=float(lower),
            upper=float(upper),
        )
        mask = sitk.GetArrayFromImage(seg).astype(np.uint8)
        return mask


class MorphologicalSegmenter(BaseSegmenter):
    """Threshold + morphological operations for cleaner results."""

    name = "Morphological"

    def __init__(self, lower: float = 50, upper: float = 200):
        self.lower = lower
        self.upper = upper

    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        lower = kwargs.get("lower", self.lower)
        upper = kwargs.get("upper", self.upper)

        # Threshold
        mask = ((volume >= lower) & (volume <= upper)).astype(np.uint8)

        # Morphological closing (fill small holes)
        struct = ndimage.generate_binary_structure(3, 2)
        mask = ndimage.binary_closing(mask, structure=struct, iterations=2).astype(np.uint8)

        # Remove small components
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            min_size = max(100, np.max(sizes) * 0.05) if len(sizes) > 0 else 100
            for i, s in enumerate(sizes, 1):
                if s < min_size:
                    mask[labeled == i] = 0

        return mask


class TotalSegmentatorSegmenter(BaseSegmenter):
    """TotalSegmentator — AI-based 117-organ segmentation.

    Downloads the model on first use (~1.5 GB).
    Requires CT data for best results.
    """

    name = "TotalSegmentator"

    def __init__(self, **kwargs):
        self.task = kwargs.get("task", "total")

    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        import tempfile
        import os
        import nibabel as nib

        task = kwargs.get("task", self.task)

        # Save volume as NIfTI temp file for TotalSegmentator
        temp_dir = tempfile.mkdtemp(prefix="ts_")
        input_path = os.path.join(temp_dir, "input.nii.gz")
        output_dir = os.path.join(temp_dir, "output")

        # Get spacing from kwargs or default
        spacing = kwargs.get("spacing", (1.0, 1.0, 1.0))

        sitk_img = sitk.GetImageFromArray(volume.astype(np.float32))
        sitk_img.SetSpacing(spacing)
        sitk.WriteImage(sitk_img, input_path)

        try:
            from totalsegmentator.python_api import totalsegmentator
            totalsegmentator(
                input=input_path,
                output=output_dir,
                task=task,
                fast=True,  # Use fast mode for speed
                ml=True,    # Multilabel output
            )

            # Load the combined segmentation
            output_file = os.path.join(output_dir, "combined.nii.gz")
            if not os.path.exists(output_file):
                # Try to combine individual masks
                combined = np.zeros_like(volume, dtype=np.uint8)
                for f in os.listdir(output_dir):
                    if f.endswith('.nii.gz'):
                        seg = nib.load(os.path.join(output_dir, f))
                        seg_data = seg.get_fdata()
                        combined[seg_data > 0] = 1
                return combined
            else:
                seg = nib.load(output_file)
                mask = (seg.get_fdata() > 0).astype(np.uint8)
                return mask

        except ImportError:
            raise RuntimeError(
                "TotalSegmentator not installed.\n"
                "Install with: pip install totalsegmentator"
            )
        except Exception as e:
            raise RuntimeError(f"TotalSegmentator failed: {str(e)}")


class BrainTumorSegmenter(BaseSegmenter):
    """Brain tumor detection using enhanced morphological analysis.

    Looks for high-intensity anomalies within the brain region
    using adaptive thresholding and shape analysis.
    Works best with MRI brain scans (T1 contrast-enhanced or FLAIR).
    """

    name = "Brain Tumor"

    def __init__(self, **kwargs):
        self.sensitivity = kwargs.get("sensitivity", 1.5)

    def segment(self, volume: np.ndarray, **kwargs) -> np.ndarray:
        sensitivity = kwargs.get("sensitivity", self.sensitivity)

        # Step 1: Detect brain region (remove skull/background)
        # Use Otsu to find tissue
        sitk_img = sitk.GetImageFromArray(volume.astype(np.float32))
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(1)
        otsu.SetOutsideValue(0)
        brain_mask = sitk.GetArrayFromImage(otsu.Execute(sitk_img)).astype(bool)

        # Fill holes in brain mask
        struct = ndimage.generate_binary_structure(3, 2)
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        brain_mask = ndimage.binary_closing(brain_mask, structure=struct, iterations=3)

        # Step 2: Within brain region, find abnormally bright areas
        brain_values = volume[brain_mask]
        if len(brain_values) == 0:
            return np.zeros_like(volume, dtype=np.uint8)

        mean_intensity = brain_values.mean()
        std_intensity = brain_values.std()

        # Tumor threshold: values significantly above brain mean
        tumor_threshold = mean_intensity + sensitivity * std_intensity

        # Step 3: Create tumor mask
        tumor_mask = np.zeros_like(volume, dtype=np.uint8)
        tumor_mask[(volume > tumor_threshold) & brain_mask] = 1

        # Step 4: Morphological cleanup
        # Opening to remove small noise
        tumor_mask = ndimage.binary_opening(
            tumor_mask, structure=struct, iterations=1
        ).astype(np.uint8)

        # Closing to fill small gaps
        tumor_mask = ndimage.binary_closing(
            tumor_mask, structure=struct, iterations=2
        ).astype(np.uint8)

        # Step 5: Remove very small components (likely noise)
        labeled, num_features = ndimage.label(tumor_mask)
        if num_features > 0:
            sizes = ndimage.sum(tumor_mask, labeled, range(1, num_features + 1))
            # Keep only components larger than 50 voxels
            min_size = max(50, np.max(sizes) * 0.02) if len(sizes) > 0 else 50
            for i, s in enumerate(sizes, 1):
                if s < min_size:
                    tumor_mask[labeled == i] = 0

        return tumor_mask


# Registry of available segmenters
SEGMENTERS = {
    "threshold": ThresholdSegmenter,
    "otsu": OtsuSegmenter,
    "region_growing": RegionGrowingSegmenter,
    "morphological": MorphologicalSegmenter,
    "totalsegmentator": TotalSegmentatorSegmenter,
    "brain_tumor": BrainTumorSegmenter,
}


def get_segmenter(name: str, **kwargs) -> BaseSegmenter:
    """Get a segmenter by name."""
    if name not in SEGMENTERS:
        raise ValueError(f"Unknown segmenter: {name}. Available: {list(SEGMENTERS.keys())}")
    return SEGMENTERS[name](**kwargs)

