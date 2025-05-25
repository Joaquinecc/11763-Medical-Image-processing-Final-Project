import numpy as np
import matplotlib.pyplot as plt
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


def plot_difference_map(fixed_volume: np.ndarray, moving_volume: np.ndarray, threshold: float = 0.2):
    """
    Plot a 3x3 grid showing voxel-wise differences between a fixed and moving volume.
    
    Parameters:
        fixed_volume (np.ndarray): The reference volume (3D array).
        moving_volume (np.ndarray): The aligned or predicted volume (3D array).
        threshold (float): Threshold to suppress small differences in the heatmap.
    """
    assert fixed_volume.shape == moving_volume.shape, "Volumes must have the same shape"

    depth, height, width = fixed_volume.shape

    # Choose 3 slice indices (¼, ½, ¾ positions)
    slice_indices = [
        (depth // 4, height // 4, width // 4),
        (depth // 2, height // 2, width // 2),
        (3 * depth // 4, 3 * height // 4, 3 * width // 4),
    ]

    titles = ['Axial', 'Coronal', 'Sagittal']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for row_idx, (idx_axial, idx_coronal, idx_sagittal) in enumerate(slice_indices):
        # Extract slices
        fixed_slices = [
            fixed_volume[idx_axial, :, :],
            fixed_volume[:, idx_coronal, :],
            fixed_volume[:, :, idx_sagittal]
        ]
        moving_slices = [
            moving_volume[idx_axial, :, :],
            moving_volume[:, idx_coronal, :],
            moving_volume[:, :, idx_sagittal]
        ]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]

            # Compute absolute normalized difference
            diff = np.abs(fixed_slices[col_idx] - moving_slices[col_idx])
            diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            highlight = np.where(diff_norm >= threshold, diff_norm, 0)

            ax.imshow(highlight, cmap='hot')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=12)
            if col_idx == 0:
                label = f"Slice Set {row_idx + 1}"
                ax.text(-0.2, 0.5, label, fontsize=14, color='black',
                        ha='right', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

def plot_coregistration_views(pixelarray_target: np.ndarray, pixelarray_moving: np.ndarray):
    """
    Plot 6 rows × 3 columns showing the target and moving volume slices
    (axial, coronal, sagittal) at 3 positions (1/4, 1/2, 3/4) of depth.

    Parameters:
        pixelarray_target (np.ndarray): 3D target (fixed) volume.
        pixelarray_moving (np.ndarray): 3D moving volume (registered or unregistered).
    """
    assert pixelarray_target.shape == pixelarray_moving.shape, "Volume shapes must match."

    depth, height, width = pixelarray_moving.shape

    # Choose 3 slice index sets (¼, ½, ¾ positions)
    slice_indices = [
        (depth // 4, height // 4, width // 4),
        (depth // 2, height // 2, width // 2),
        (3 * depth // 4, 3 * height // 4, 3 * width // 4),
    ]

    titles = ['Axial', 'Coronal', 'Sagittal']
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for view_set_idx, (idx_axial, idx_coronal, idx_sagittal) in enumerate(slice_indices):
        target_slices = [
            pixelarray_target[idx_axial, :, :],
            pixelarray_target[:, idx_coronal, :],
            pixelarray_target[:, :, idx_sagittal]
        ]
        moving_slices = [
            pixelarray_moving[idx_axial, :, :],
            pixelarray_moving[:, idx_coronal, :],
            pixelarray_moving[:, :, idx_sagittal]
        ]

        for row_offset, (slices, label) in enumerate([(target_slices, 'Reference'), (moving_slices, 'Input')]):
            row = view_set_idx * 2 + row_offset
            for col in range(3):
                ax = axes[row, col]
                ax.imshow(slices[col], cmap='bone')
                ax.axis('off')

                if row == 0:
                    ax.set_title(titles[col], fontsize=12)

            # Label each row (Target / Moving)
            axes[row, 0].text(-0.2, 0.5, label, fontsize=14, color='black',
                              ha='right', va='center', transform=axes[row, 0].transAxes)

    plt.tight_layout()
    plt.show()
