import os
import matplotlib
import pydicom
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
import os
from tqdm import tqdm 
from utils import load_dcm_data,maximum_intensity_projection,sigmoid_contrast



def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """
    Rotate a 3D medical image around the axial plane (Z-axis).

    Parameters:
        img_dcm (np.ndarray): 3D volume (Z, Y, X).
        angle_in_degrees (float): Rotation angle in degrees.

    Returns:
        np.ndarray: Rotated 3D volume.
    """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)



def create_seg_mask(dcm_mask,dcms_full_patient):
    """
    Create a segmentation mask aligned to a full patient volume.

    This function aligns a multiframe segmentation DICOM to the full 3D volume by matching
    their respective slice positions using DICOM metadata (SliceLocation and ImagePositionPatient).

    Parameters:
        dcm_mask: DICOM segmentation object with multiple frames.
        dcms_full_patient: List of DICOM slices from the full patient scan.

    Returns:
        np.ndarray: 3D segmentation mask aligned to the patient volume.
    """
    #Sort all slice index
    slice_index_dcm=[float(dcm.SliceLocation) for dcm in dcms_full_patient]
    slice_index_dcm.sort()
    #Slice index Mask
    slice_index_mask= [int(-a.PlanePositionSequence[0].ImagePositionPatient[-1]) for a in list(dcm_mask.PerFrameFunctionalGroupsSequence)]
    initial_index_mask= min(slice_index_mask) #initial position
    
    #Find idex match
    start_index=None
    for i,slice_index_patiend in enumerate(slice_index_dcm):
        if  abs(slice_index_patiend-initial_index_mask)<=1.25:
            start_index = i

    end_index= start_index + int(dcm_mask.NumberOfFrames)
    shape_image=(len(dcms_full_patient),dcms_full_patient[0].Rows,dcms_full_patient[0].Columns)
    #Create empty mask
    seg_mask = np.zeros(shape_image)
    #Fill with value
    seg_mask[start_index:end_index] =  np.flip(dcm_mask.pixel_array,0) #A flip is necessary
    return seg_mask

if __name__ == '__main__':
    os.makedirs('results/MIP/', exist_ok=True)
    # Load and sort full patient volume
    dcm_path='RadCTTACEomics_1193-20250418T131346Z-001/RadCTTACEomics_1193/10_AP_Ax2.50mm'
    dcms_full_patient=load_dcm_data(dcm_path)
    dcms_full_patient.sort(key = (lambda x: float(x.SliceLocation))) #Sort by skuce index
    combined_pixelarray = np.stack([x.pixel_array for x in dcms_full_patient], axis=0)

    #Load tumor
    dmc_tumor_path='RadCTTACEomics_1193-20250418T131346Z-001/RadCTTACEomics_1193/10_AP_Ax2.50mm_ManualROI_Tumor.dcm'
    dcm_tumor=pydicom.dcmread(dmc_tumor_path)
    maks_tumor = create_seg_mask(dcm_tumor, dcms_full_patient)

    #Load liver
    dmc_liver_path='RadCTTACEomics_1193-20250418T131346Z-001/RadCTTACEomics_1193/10_AP_Ax2.50mm_ManualROI_Liver.dcm'
    dcm_liver=pydicom.dcmread(dmc_liver_path)
    maks_liver=create_seg_mask(dcm_liver,dcms_full_patient)

    #Clip Hue Value to focus on essential organs
    img_min, img_max = 0, 800
    # Set normalization and visualization parameters
    slice_thickness= float(dcms_full_patient[0].SliceThickness)
    pixel_spacing=float(dcms_full_patient[0].PixelSpacing[0])
    aspect=slice_thickness / pixel_spacing
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    n=12 # number of projections/frames
    cmap_bone = plt.get_cmap('bone')
    projections=[]
    # Loop through the images and process
    for idx, degree in tqdm(enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)), total=n, desc="Processing frames"):
        rotated_img = rotate_on_axial_plane(combined_pixelarray, degree)
        projection = maximum_intensity_projection(rotated_img,axis=1)

        #CLip values to focus on essential tissues
        projection[projection>=800]= img_max
        projection[projection <200]= img_min

        projection = (projection - img_min) / (img_max - img_min) #Normalize.
        projection=sigmoid_contrast(projection)  #Increse contrast for better visualization
        projection= cmap_bone(projection)[..., :3] #Add Cbone color range, ignore alpha value

        # Rotate and project liver mask
        rotated_liver_mask = rotate_on_axial_plane(maks_liver, degree)
        rotated_liver_mask = maximum_intensity_projection(rotated_liver_mask,axis=1)

        # # Rotate and project tumor mask
        rotated_tumor_mask = rotate_on_axial_plane(maks_tumor, degree)
        rotated_tumor_mask = maximum_intensity_projection(rotated_tumor_mask, axis=1)


        overlay_img=projection
        #add each overalay
        alpha=0.7
        green=np.array([0.0, 1.0, 0.0])
        mask=rotated_liver_mask>1
        overlay_img [mask]=  overlay_img [mask]*(1-alpha ) + green * rotated_liver_mask[..., np.newaxis][mask] * alpha


        alpha=0.7
        red=np.array([1.0, 0.0, 0.0] )
        mask=rotated_tumor_mask>1
        overlay_img [mask]=  overlay_img [mask]*(1-alpha ) + red * rotated_tumor_mask[..., np.newaxis][mask] * alpha

        # Clip to valid range for imshow
        overlay_img = np.clip(overlay_img, 0, 1)
        # Display the image with overlay
        plt.imshow(overlay_img, aspect=aspect)
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
