
import numpy as np
from src.prep_data.process_image import process_image, get_max_min_dataset


def data_generator(pet_data, spect_data, paired_by_label, batch_size, length, intensity_norm, pet_maximum, spect_maximum, mask):

    while True:
        pet_batch, spect_batch = [], []
        count = 0

        # Shuffle prep_data
        pet_data_s = pet_data.sample(frac=1).reset_index()
        spect_data_s = spect_data.sample(frac=1).reset_index()

        # iterate until the desired length to get batches of pet + spect
        for idx in range(length):

            # get the idx PET and label
            pet_img_path = pet_data_s.at[idx, 'img_paths']
            pet_label = pet_data_s.at[idx, 'labels']

            pet_array = process_image(pet_img_path, norm=intensity_norm, resize_type='pad_PET', maximum=pet_maximum, mask=mask)
            pet_array = np.expand_dims(pet_array, axis=-1)

            pet_batch.append(pet_array)  # append to pet batch

            if paired_by_label:
                # get all spect prep_data of the same label of the pet label of the correspondent image
                spect_data_s_label = spect_data_s[spect_data_s['labels'] == pet_label].reset_index()

                # get the idx SPECT of the same PET label
                spect_img_path = spect_data_s_label.at[idx, 'img_paths']
            else:
                spect_img_path = spect_data_s.at[idx, 'img_paths']

            spect_array = process_image(spect_img_path, norm=intensity_norm, resize_type='pad_SPECT', maximum=spect_maximum, mask=mask)
            spect_array = np.expand_dims(spect_array, axis=-1)

            spect_batch.append(spect_array)  # append to spect batch

            count += 1
            if count == batch_size:
                yield np.array(pet_batch), np.array(spect_batch)
                count = 0
                pet_batch, spect_batch = [], []