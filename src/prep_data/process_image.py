
import os
import numpy as np
import nibabel as nb
import scipy
from nilearn.image import threshold_img
from skimage.transform import resize
import matplotlib.pyplot as plt

def process_image(path:str, norm:str='max_dataset', resize_type:str='pad_PET', maximum:float=None, mask:str='50%') -> np.array:
    """
    Preprocess individual image. To use before training to normalize, resize, etc.

    Parameters
    ----------
    path: str
        Filepath of image
    norm: str
        Type of intensity normalization
    resize_type: str
        Resize/padding type to make images (96, 112, 96)
    maximum: float
        Maximum value to use in max normalization
    mask: str
        Threshold to apply to mask images (ex: '50%').
        Voxels with intensities less than the requested threshold will be set to zero.


    Returns
    -------
    numpy.array
        Processed array

    """

    # Apply mask or read filedir into array
    array_norm =None
    if '%' in mask:
        img = nb.load(path)
        masked_img = threshold_img(img, threshold=mask)
        array = masked_img.get_fdata()

    elif os.path.isfile(mask):
        img = nb.load(path)
        array = img.get_fdata()
        #print(array.shape)
        if resize_type == 'pad_PET':
            array = np.pad(array, ((6, 6), (7, 7), (11, 11)))

        array = mask_img(array, mask_file=mask)
    else:
        array = img_to_array(path)

    # Intensity normalization
    if norm == 'max_dataset':
        array_norm = array / maximum
    elif norm == 'max_img':
        array_norm = array / np.max(array)
    elif norm == 'zscore':
        array_norm = normalize_individual_zscore(array)

    #TODO: raise error

    # Resize arrays
    if resize_type == 'pad_SPECT':
        array_res = np.pad(array_norm, ((2, 3), (1, 2), (2, 3)))  ##adapt according to .nii or .img
    elif resize_type == 'pad_PET':
        if os.path.isfile(mask):
            array_res = np.pad(array_norm, ((2, 3), (1, 2), (2, 3)))
        else:
            array_res = np.pad(array_norm, ((8, 9), (8, 9), (13, 14)))
    elif resize_type == 'resize':
        array_res = resize(array_norm, output_shape=(96, 112, 96))


    return array_res

def img_to_array(filedir):
    """Read image in filedir into array"""
    img = nb.load(filedir)
    array = img.get_fdata()

    ## if we want to read another image type uncomment
    # img = sitk.ReadImage(filedir)
    # array = sitk.GetArrayFromImage(img)

    return array

def img_to_nii(img_filepath):
    img = nb.load(img_filepath)
    new_filepath = img_filepath.replace('.img', '.nii')
    nb.save(img, new_filepath)
    return new_filepath

def get_max_min_dataset(filelist):
    """Get the max and min value of arrays of images in filelist"""
    max_dataset = np.max(np.array([img_to_array(path) for path in filelist]))
    min_dataset = np.min(np.array([img_to_array(path) for path in filelist]))

    print('Max value: ' + str(max_dataset))
    print('Min value: ' + str(min_dataset))

    return max_dataset, min_dataset


def normalize_individual_max(array):
    """max normalization"""
    array = np.float64(array)
    ma = np.max(array)
    array = array / ma
    return array

def normalize_individual_min_max(array):
    """min-max normalization"""
    array = np.float64(array)
    ma = np.max(array)  # mean for prep_data centering
    mi = np.min(array)  # std for prep_data normalization
    array -= mi
    array /= ma-mi
    return array


def normalize_individual_zscore(array):
    """z-score normalization"""
    array = np.float64(array)
    mean = np.mean(array)  # mean for prep_data centering
    std = np.std(array)  # std for prep_data normalization
    array -= mean
    array /= std
    return array


def crop(array):
    """
    Inverse of the padding of SPECT images that we did in process_image, before inputing SPECT images to network.
    array_res = np.pad(array_norm, ((2, 3), (1, 2), (2, 3))).
    To get image in MNI space dimensions
    """

    array = array[2:-3, 1:-2, 2:-3]
    return array


def mask_img(array, mask_file='/home/leonor/Code/brain_masks/brainmask.nii'):
    """ Apply mask to image"""
    mask_img = nb.load(mask_file)
    mask = np.array(mask_img.dataobj)

    array_masked = array * mask
    return array_masked


def create_slice_figure(img_array, title, save_name=None):
    """ Print several slices of 3D array"""
    rotation = 90
    fig = plt.figure(figsize=(6,3.6), dpi=500)
    plt.title(title)
    plt.axis('off')
    print(img_array.shape)
    shape = int(img_array.shape[0] / 6)

    for i in range(1,6):
        fig.add_subplot(3, 6, i)
        #plt.tight_layout()
        img_slice = img_array[:,:,i*shape]
        img_slice = scipy.ndimage.rotate(img_slice, rotation)
        #matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')

        plt.axis('off')
        #plt.subplots_adjust(hspace=0.0001)

    #for i in range(6):
        fig.add_subplot(3, 6, i+6)
        #plt.tight_layout()
        img_slice = img_array[:,i*shape,:]
        img_slice = scipy.ndimage.rotate(img_slice, rotation)
        #matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')
        plt.axis('off')
        #plt.subplots_adjust(hspace=0.0001)

    #for i in range(6):
        fig.add_subplot(3, 6, i+12)
        #plt.tight_layout()
        img_slice = img_array[i*shape,:,:]
        img_slice= np.flip(img_slice, axis=0)
        img_slice = scipy.ndimage.rotate(img_slice, rotation)

        #matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')
        #plt.subplots_adjust(hspace=0.0001)

        plt.axis('off')

    plt.colorbar(cax=plt.axes([0.79, 0.16, 0.02, 0.7]), anchor=(np.min(img_array), np.max(img_array)))

    plt.subplots_adjust(wspace=0.03, hspace=0.0001)
    if save_name:
        plt.savefig(save_name)
    else:
        fig.show()



