"""
Medical Image Loader
====================
Supports: DICOM series, NIfTI (.nii/.nii.gz), NRRD, MHD/RAW
Returns standardized numpy arrays with metadata.
"""

import os
import numpy as np
import SimpleITK as sitk


class MedicalImage:
    """Container for a loaded medical image with metadata."""

    def __init__(self, sitk_image: sitk.Image):
        self.sitk_image = sitk_image
        self.volume = sitk.GetArrayFromImage(sitk_image)  # (Z, Y, X)
        self.spacing = np.array(sitk_image.GetSpacing())  # (X, Y, Z)
        self.origin = np.array(sitk_image.GetOrigin())
        self.direction = np.array(sitk_image.GetDirection())
        self.size = np.array(sitk_image.GetSize())  # (X, Y, Z)

    @property
    def shape(self):
        return self.volume.shape

    @property
    def voxel_volume_mm3(self):
        """Volume of a single voxel in mm³."""
        return float(np.prod(self.spacing))

    def get_axial_slice(self, index: int) -> np.ndarray:
        index = np.clip(index, 0, self.volume.shape[0] - 1)
        return self.volume[index, :, :]

    def get_sagittal_slice(self, index: int) -> np.ndarray:
        index = np.clip(index, 0, self.volume.shape[2] - 1)
        return self.volume[:, :, index]

    def get_coronal_slice(self, index: int) -> np.ndarray:
        index = np.clip(index, 0, self.volume.shape[1] - 1)
        return self.volume[:, index, :]

    def __repr__(self):
        return (
            f"MedicalImage(shape={self.shape}, "
            f"spacing={self.spacing.tolist()}, "
            f"dtype={self.volume.dtype})"
        )


class ImageLoader:
    """Load medical images from various formats."""

    @staticmethod
    def load_dicom_series(directory: str, series_id: str = None) -> MedicalImage:
        """Load a DICOM series from a directory.

        If the directory contains multiple series (e.g. different scan phases),
        loads the series with the most slices (main acquisition).

        Args:
            directory: Path to folder containing DICOM files
            series_id: Optional specific Series Instance UID to load
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        reader = sitk.ImageSeriesReader()

        # Get all series IDs in the directory
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)

        if not series_ids:
            raise ValueError(f"No DICOM series found in: {directory}")

        if series_id and series_id in series_ids:
            # Use the specified series
            chosen_id = series_id
        else:
            # Pick the series with the most files (main acquisition)
            best_id = series_ids[0]
            best_count = 0
            for sid in series_ids:
                files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, sid)
                if len(files) > best_count:
                    best_count = len(files)
                    best_id = sid
            chosen_id = best_id

        dicom_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, chosen_id)

        if not dicom_files:
            raise ValueError(f"No DICOM files found for series: {chosen_id}")

        reader.SetFileNames(dicom_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()
        return MedicalImage(image)

    @staticmethod
    def load_nifti(filepath: str) -> MedicalImage:
        """Load a NIfTI file (.nii or .nii.gz)."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        image = sitk.ReadImage(filepath)
        return MedicalImage(image)

    @staticmethod
    def load_nrrd(filepath: str) -> MedicalImage:
        """Load an NRRD file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        image = sitk.ReadImage(filepath)
        return MedicalImage(image)

    @staticmethod
    def load_mhd(filepath: str) -> MedicalImage:
        """Load an MHD/RAW file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        image = sitk.ReadImage(filepath)
        return MedicalImage(image)

    @staticmethod
    def load_auto(path: str) -> MedicalImage:
        """Auto-detect format and load.
        If path is a directory, tries DICOM series.
        If path is a file, determines format by extension.
        """
        if os.path.isdir(path):
            return ImageLoader.load_dicom_series(path)

        ext = os.path.splitext(path)[1].lower()
        if path.lower().endswith('.nii.gz'):
            return ImageLoader.load_nifti(path)
        elif ext == '.nii':
            return ImageLoader.load_nifti(path)
        elif ext == '.nrrd':
            return ImageLoader.load_nrrd(path)
        elif ext == '.mhd':
            return ImageLoader.load_mhd(path)
        else:
            # Try generic SimpleITK reader
            try:
                image = sitk.ReadImage(path)
                return MedicalImage(image)
            except Exception as e:
                raise ValueError(f"Unsupported format: {ext}") from e

    @staticmethod
    def save_nifti(medical_image: MedicalImage, filepath: str):
        """Save a MedicalImage as NIfTI."""
        sitk.WriteImage(medical_image.sitk_image, filepath)

    @staticmethod
    def save_mask_nifti(mask: np.ndarray, reference: MedicalImage, filepath: str):
        """Save a segmentation mask as NIfTI, using reference image geometry."""
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_sitk.SetSpacing(reference.sitk_image.GetSpacing())
        mask_sitk.SetOrigin(reference.sitk_image.GetOrigin())
        mask_sitk.SetDirection(reference.sitk_image.GetDirection())
        sitk.WriteImage(mask_sitk, filepath)
