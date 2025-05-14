import math
import numpy as np
from scipy.optimize import least_squares
import pydicom
import os

def normalize_image(image):
    """ Normalize the pixel values of an image to the range [0, 1]. """
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (image_max - image_min)
    return normalized_image
def maximum_intensity_projection(image, axis=0):
    return np.max(image, axis=axis)

def load_dcm_data(image_path):
    img_dcmset = []
    for root, _, filenames in os.walk(image_path):
        for filename in filenames:
            dcm_path = os.path.join(root,filename)
            dicom = pydicom.dcmread(dcm_path, force=True)
            img_dcmset.append(dicom)
    return img_dcmset
def sigmoid_contrast(img, gain=10, cutoff=0.5):
    """
    Apply a sigmoid-like contrast curve to the image.

    Parameters:
    - img: Input image (normalized to [0, 1] if float)
    - gain: Controls the steepness of the curve
    - cutoff: The center of the sigmoid (usually 0.5)

    Returns:
    - Adjusted image
    """
    img = img.astype(np.float32)
    img_norm = img / 255.0 if img.max() > 1 else img
    transformed = 1 / (1 + np.exp(-gain * (img_norm - cutoff)))
    return (transformed * 255).astype(np.uint8)
