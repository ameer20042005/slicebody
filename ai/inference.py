"""
AI Inference Pipeline
=====================
Orchestrates the full segmentation workflow:
Load → Preprocess → Segment → Postprocess → Return results.
"""

import numpy as np
from scipy import ndimage

from ai.model import BaseSegmenter, get_segmenter


class SegmentationResult:
    """Container for segmentation results with measurements."""

    def __init__(self, mask: np.ndarray, spacing: np.ndarray):
        self.mask = mask
        self.spacing = spacing
        self._compute_measurements()

    def _compute_measurements(self):
        """Compute measurements from the segmentation mask."""
        voxel_volume_mm3 = float(np.prod(self.spacing))
        self.num_voxels = int(self.mask.sum())
        self.volume_mm3 = self.num_voxels * voxel_volume_mm3
        self.volume_cm3 = self.volume_mm3 / 1000.0

        # Bounding box
        if self.num_voxels > 0:
            coords = np.argwhere(self.mask > 0)
            self.bbox_min = coords.min(axis=0).tolist()  # (z, y, x)
            self.bbox_max = coords.max(axis=0).tolist()
            bbox_size = (np.array(self.bbox_max) - np.array(self.bbox_min) + 1)
            self.bbox_size_mm = (bbox_size * self.spacing[::-1]).tolist()  # Convert to mm
        else:
            self.bbox_min = [0, 0, 0]
            self.bbox_max = [0, 0, 0]
            self.bbox_size_mm = [0, 0, 0]

        # Connected components
        labeled, self.num_components = ndimage.label(self.mask)
        if self.num_components > 0:
            component_sizes = ndimage.sum(self.mask, labeled, range(1, self.num_components + 1))
            self.largest_component_voxels = int(max(component_sizes))
            self.largest_component_cm3 = self.largest_component_voxels * voxel_volume_mm3 / 1000.0
        else:
            self.largest_component_voxels = 0
            self.largest_component_cm3 = 0.0

    def summary(self) -> str:
        """Return a text summary of the results."""
        lines = [
            "=== Segmentation Results ===",
            f"  Segmented Voxels:  {self.num_voxels:,}",
            f"  Total Volume:      {self.volume_cm3:.2f} cm3",
            f"  Components Found:  {self.num_components}",
            f"  Largest Component: {self.largest_component_cm3:.2f} cm3",
            f"  Bounding Box (mm): {self.bbox_size_mm[0]:.1f} x {self.bbox_size_mm[1]:.1f} x {self.bbox_size_mm[2]:.1f}",
        ]
        return "\n".join(lines)


class InferencePipeline:
    """Runs the full segmentation pipeline."""

    def __init__(self, segmenter_name: str = "morphological", **segmenter_kwargs):
        self.segmenter = get_segmenter(segmenter_name, **segmenter_kwargs)

    def run(self, volume: np.ndarray, spacing: np.ndarray, **kwargs) -> SegmentationResult:
        """Execute the segmentation pipeline.

        Args:
            volume: 3D numpy array (Z, Y, X)
            spacing: Voxel spacing (X, Y, Z) in mm
            **kwargs: Additional arguments passed to the segmenter

        Returns:
            SegmentationResult with mask and measurements
        """
        # Step 1: Segment (pass spacing for AI models that need it)
        kwargs["spacing"] = tuple(float(s) for s in spacing)
        mask = self.segmenter.segment(volume, **kwargs)

        # Step 2: Post-process — remove tiny components
        mask = self._postprocess(mask)

        # Step 3: Compute results
        return SegmentationResult(mask, spacing)

    def _postprocess(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove connected components smaller than min_size voxels."""
        labeled, num = ndimage.label(mask)
        if num == 0:
            return mask

        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        cleaned = np.zeros_like(mask)
        for i, s in enumerate(sizes, 1):
            if s >= min_size:
                cleaned[labeled == i] = 1

        return cleaned.astype(np.uint8)
