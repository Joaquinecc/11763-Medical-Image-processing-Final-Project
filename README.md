# Medical Imaging Final Project

This project implements two core tasks in medical imaging:
1. **DICOM Loading and 3D Visualization** of CT images and associated liver/tumor segmentations.
2. **3D Rigid Coregistration** between two CT volumes without using third-party registration libraries.

---

## 📁 Project Structure

| File/Folder                  | Description |
|-----------------------------|-------------|
| `task1_animation.py`        | Generates a rotating Maximum Intensity Projection (MIP) animation with tumor and liver overlays. |
| `task2_coregistration.ipynb`| Jupyter notebook that implements rigid registration between two CT volumes. |
| `utils.py`                  | Helper functions: image loading, projection, normalization, plotting, etc. |
| `results/`                  | Folder for storing outputs like generated `.gif` animations and projection images. |

---

## ✅ Tasks Completed

- ✅ **Task 1: DICOM Visualization**
  - Loads CT volume and segmentation masks using `pydicom`.
  - Aligns segmentation frames using DICOM metadata.
  - Applies sigmoid contrast adjustment and colormap for better tissue visualization.
  - Generates rotating MIP projections with colored overlays (green = liver, red = tumor).

- ✅ **Task 3: Rigid Coregistration**
  - Implements rigid transformation using 6 parameters (3 translation, 3 rotation).
  - Resamples input volume using `scipy.ndimage.affine_transform`.
  - Defines a custom MSE loss function and finite-difference gradient.
  - Optimizes using `scipy.optimize.minimize` with the 'Newton-CG' method.
  - Visual and numerical validation included.

---

## ▶️ How to Run

### 1. Rotating MIP Animation

```bash
python task1_animation.py \
  --ct_path path/to/CT_FOLDER \
  --tumor_path path/to/TUMOR_SEGMENTATION.dcm \
  --liver_path path/to/LIVER_SEGMENTATION.dcm \
  --n 36
