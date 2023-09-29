# DAT-cycle-gan

Code to generate DaTSCAN SPECT from CFT PET scans and vice-versa.

## Usage

(1) Edit settings.py file to add your parameters:

    TRAIN_DIR - directory to save the results and weights of the model
    TEST_DIR - directory to save the generated images
    TRAIN_PET_CSV - csv file with the paths to the PET images to train and the labels (columns' names should be 'img_paths' and 'labels')
    TRAIN_SPECT_CSV - csv file with the paths to the SPECT images to train and the labels (columns' names should be 'img_paths' and 'labels')

(2) Run main.py:

    python3 main.py
