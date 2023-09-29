
import numpy as np
import pandas as pd

from src.prep_data.process_image import process_image

def prep_data(df: pd.DataFrame, label_dict:dict, norm:str, resize_type:str, maximum:float = None, mask:str = None) -> \
        (np.array, np.array) :
    """
    Get filepaths and correspondent labels from dataframe (can be an excel/csv file). Get arrays of the images
    (and labels) after processing them according to the process_image function.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe with wanted filepaths and labels to preprocess and pass to numpy array
    label_dict: dict
        Dict with correspondence from label names to numbers. Ex: {'NC':0, 'PD':1}
    norm: str
        Type of intensity normalization to apply to images. Either max_dataset. max_img or z_score.
    resize_type: str
        Resize or padding type. PET image needs to be padded differently from SPECT due to different dimensions.
    maximum: float
        If norm is max_dataset, we can pass here the max value of dataset to normalize

    mask: str
        Threshold to apply to mask images (ex: '50%').
        Voxels with intensities less than the requested threshold will be set to zero.

    Returns
    -------
    numpy.array
        numpy arrays of images
    numpy.array
        numpy arrays of labels
    """

    filelist = list(df['img_paths'])
    label_list = list(df['labels'])

    img_arrays = []
    for path in filelist:
        array = process_image(path, norm=norm, resize_type=resize_type, maximum=maximum, mask=mask)
        img_arrays.append(array)

    labels = []
    for label in label_list:
        label_id = np.array(label_dict.get(label))
        labels.append(label_id)

    return np.array(img_arrays), np.array(labels)