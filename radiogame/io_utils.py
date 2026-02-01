from __future__ import annotations
import numpy as np
import SimpleITK as sitk

def load_image(path: str) -> sitk.Image:
    """
    Load an image using SimpleITK.
    Supports common formats (jpg/png) and medical formats (nii, mhd, dicom series with extra code later).
    """
    return sitk.ReadImage(path)

def image_to_array_2d(img: sitk.Image) -> np.ndarray:
    """
    Convert a SimpleITK image to a 2D numpy array suitable for matplotlib imshow.
    If the image is 3D, take the middle slice along z.
    """
    arr = sitk.GetArrayFromImage(img)  # for 2D: (H,W); for 3D: (Z,H,W)
    if arr.ndim == 3:
        arr = arr[:,:,0]
    return arr

def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Simple normalization for display (not clinical windowing).
    """
    a = arr.astype(float)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return a
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo)
    return a
