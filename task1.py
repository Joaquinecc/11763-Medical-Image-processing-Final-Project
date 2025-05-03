import os
import matplotlib
import pydicom
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
from pathlib import Path
import os
from skimage.transform import resize  
from tqdm import tqdm 

def load_dcm_data(image_path):
    img_dcmset = []
    for root, _, filenames in os.walk(image_path):
        for filename in filenames:
            dcm_path = os.path.join(root,filename)
            dicom = pydicom.dcmread(dcm_path, force=True)
            img_dcmset.append(dicom)
    return img_dcmset

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)
def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)
def maximum_intensity_projection(image, axis=0):
    return np.max(image, axis=axis)


def create_seg_mask(dcm_mask,dcms,shape):
    slice_index_dcm=[float(dcm.SliceLocation) for dcm in dcms]
    slice_index_dcm.sort()
    shape= combined_pixelarray.shape
    slice_index_mask= [int(-a.PlanePositionSequence[0].ImagePositionPatient[-1]) for a in list(dcm_mask.PerFrameFunctionalGroupsSequence)]
    min_index= min(slice_index_mask)
    n_frames= int(dcm_mask.NumberOfFrames)
    #incemental= float(dcm_mask.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices)
    start_index = slice_index_dcm.index(min_index)
    end_index= start_index + n_frames
    seg_mask = np.zeros(shape)
    seg_mask[start_index:end_index] =  dcm_mask.pixel_array
    return seg_mask

if __name__ == '__main__':
    os.makedirs('results/MIP/', exist_ok=True)

    dcm_path='RadCTTACEomics_1193-20250418T131346Z-001/RadCTTACEomics_1193/10_AP_Ax2.50mm'
    dcms=load_dcm_data(dcm_path)
    #Sort by slice inde
    dcms.sort(key = (lambda x: float(x.SliceLocation)))
    # Stack DICOM images
    combined_pixelarray = np.stack([x.pixel_array for x in dcms], axis=0)


    dmc_tumor_path='RadCTTACEomics_1193-20250418T131346Z-001/RadCTTACEomics_1193/10_AP_Ax2.50mm_ManualROI_Tumor.dcm'
    dcm_tumor=pydicom.dcmread(dmc_tumor_path)
    #Load liver
    dmc_liver_path='RadCTTACEomics_1193-20250418T131346Z-001/RadCTTACEomics_1193/10_AP_Ax2.50mm_ManualROI_Liver.dcm'
    dcm_liver=pydicom.dcmread(dmc_liver_path)
    maks_liver=create_seg_mask(dcm_liver,dcms,combined_pixelarray.shape)
    maks_tumor = create_seg_mask(dcm_tumor, dcms, combined_pixelarray.shape)


    #Aspect for plotting
    img_min = np.amin(combined_pixelarray)
    img_max = np.amax(combined_pixelarray)
    slice_thickness= float(dcms[0].SliceThickness)
    pixel_spacing=float(dcms[0].PixelSpacing[0])
    aspect=slice_thickness / pixel_spacing
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    n=6
    cmap_bone = plt.get_cmap('bone')
    projections=[]
    # Loop through the images and process
    for idx, degree in tqdm(enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)), total=n, desc="Processing frames"):
        rotated_img = rotate_on_axial_plane(combined_pixelarray, degree)
        projection = maximum_intensity_projection(rotated_img,axis=1)
        projection = (projection - img_min) / (img_max - img_min) #Normalize.
        projection= cmap_bone(projection)[..., :3] #Ignore aplha
       
       
        # Rotate and project liver mask
        rotated_liver_mask = rotate_on_axial_plane(maks_liver, degree)
        rotated_liver_mask = maximum_intensity_projection(rotated_liver_mask,axis=1)
        rotated_liver_mask = [0.0, 1.0, 0.0] * rotated_liver_mask[..., np.newaxis] #Green
        rotated_liver_mask=rotated_liver_mask[...,:3]#Ignore alpha

        # Rotate and project tumor mask
        rotated_tumor_mask = rotate_on_axial_plane(maks_tumor, degree)
        rotated_tumor_mask = maximum_intensity_projection(rotated_tumor_mask, axis=1)
        rotated_tumor_mask = [1.0, 0.0, 0.0] * rotated_tumor_mask[..., np.newaxis]  # Red mask
        rotated_tumor_mask = rotated_tumor_mask[..., :3]  # Ignore alpha

        alpha=0.4
        overlay_img= projection * (1 - alpha * rotated_liver_mask) + rotated_liver_mask * alpha
        alpha=0.5
        overlay_img = overlay_img * (1 - alpha * rotated_tumor_mask) + rotated_tumor_mask * alpha

   
        # Display the image with overlay
        plt.imshow(overlay_img, cmap=cm, vmin=0, vmax=1, aspect=aspect)
        plt.axis('off')  # Turn off the axis
        plt.savefig(f'results/MIP/Projection_{idx}.png')  # Save image frame
        projections.append(overlay_img)  # Append for animation



    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=aspect)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=250, blit=True)
    anim.save('results/MIP/Animation.gif')  # Save animation
    plt.show()                              # Show animation
