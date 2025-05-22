# Medical Imaging Final Project

This project implements two core tasks in medical image analysis:

1. **3D Visualization** of CT images and associated liver/tumor segmentations.
2. **Rigid Coregistration** between CT volumes using custom transformation and optimization logic.

---

## 📁 Project Structure

| File/Folder                  | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| `task1_animation.py`         | Generates rotating MIP projections with tumor and liver overlays.    |
| `task2_coregistration.ipynb` | Jupyter notebook for rigid registration between two CT volumes.      |
| `utils.py`                   | Helper functions for DICOM handling, projections, and visualization. |
| `results/`                   | Directory to save output images, animations, and evaluation plots.   |

---

## ✅ Tasks Completed

### Task 1: DICOM Loading and Tumor/Liver Visualization

* Loads CT volume and structured segmentation masks using `pydicom`.
* Aligns segmentation frames using `PerFrameFunctionalGroupsSequence` and `ImagePositionPatient`.
* Applies MIP projection across rotation angles and generates color-coded overlays.
* Exports `.gif` animations and static views with overlays.

### Task 2: 3D Rigid Coregistration

* Rigid transformation modeled with 6 parameters (3 translation, 3 Euler rotation).
* Interpolation performed with `scipy.ndimage.affine_transform`.
* MSE used as the loss function (RMSE tested but yielded lower performance).
* Finite differences used to approximate gradients.
* Optimized using `scipy.optimize.minimize` with the 'Newton-CG' method.
* Visual and numerical evaluation confirms successful alignment.

---

## ▶️ How to Run

### 1. Rotating MIP Animation (Task 1)

```bash
python task1_animation.py \
  --ct_path path/to/CT_FOLDER \
  --tumor_path path/to/TUMOR_SEGMENTATION.dcm \
  --liver_path path/to/LIVER_SEGMENTATION.dcm \
  --n 36
```

* Output: `results/MIP/Animation.gif` and individual projection frames.
* Make sure DICOM files are in correct folders.

### 2. Rigid Coregistration (Task 2)

Open the notebook:

```bash
jupyter notebook task2_coregistration.ipynb
```

* Follow the notebook cells to load, transform, optimize, and visualize.
* Outputs include axial/coronal/sagittal comparisons and error plots.

---

## 📦 Requirements

To install all necessary libraries:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
pydicom
matplotlib
scipy
tqdm
```


