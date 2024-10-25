# DAT-cycle-gan

Code for the paper: "Dopaminergic PET to SPECT Domain Adaptation: A Cycle GAN translation approach".

Dopamine transporter imaging is routinely used in Parkinson's disease (PD) and atypical parkinsonian syndromes (APS) diagnosis. 
While [11C]CFT PET is prevalent in Asia with a large APS database, Europe relies on [123I]FP-CIT SPECT with limited APS data. 
Our aim was to develop a deep learning-based method to convert [11C]CFT PET images to [123I]FP-CIT SPECT images, 
facilitating multicenter studies and overcoming data scarcity to promote Artificial Intelligence (AI) advancements.

<img src="https://github.com/lopes-leonor/DAT-cycle-gan/blob/main/images/figure1.png" width="800" alt="Cycle GAN model">
Scheme of our Cycle GAN model

<img src="https://github.com/lopes-leonor/DAT-cycle-gan/blob/main/images/figure6.png" width="800" alt="Synthetic SPECT images">
Example of images of (a) normal controls and (b) Parkinson’s disease from real PET (upper row), real SPECT (middle row) and synthetic SPECT (bottom row). 

## Usage

(1) Edit settings.py file to add your parameters:

    TRAIN_DIR - directory to save the results and weights of the model
    TEST_DIR - directory to save the generated images
    TRAIN_PET_CSV - csv file with the paths to the PET images to train and the labels (columns' names should be 'img_paths' and 'labels')
    TRAIN_SPECT_CSV - csv file with the paths to the SPECT images to train and the labels (columns' names should be 'img_paths' and 'labels')
    ''' 
    # Directory
    PROJ_DIR - main directory to save results
    TRAIN_DIR - directory to save the results and weights of the model
    TEST_DIR - directory to save the generated images
    
    # Data
    TRAIN_PET_CSV - csv file with the paths to the PET images to train and the labels (columns' names should be 'img_paths' and 'labels')
    TRAIN_SPECT_CSV - csv file with the paths to the SPECT images to train and the labels (columns' names should be 'img_paths' and 'labels')
    TEST_CSV = csv file with paths to test images and labels (columns' names should be 'img_paths' and 'labels')
    
    # Training
    GPUS - GPUs to use
    EPOCHS - Mumber of epochs to train the model
    BATCH_SIZE - Batch size to use for training
    IMG_SHAPE - Image shape with channels in last dimension. Example: (96, 112, 96, 1)
    LOSS_LIST - List with loss functions for the adversarial model: discriminator loss, cycle consistency loss and identity loss. Example: ['binary_crossentropy', 'binary_crossentropy', 'mae', 'mae', 'mae', 'mae']
    LOSS_WEIGHTS - Weights for each loss. Example: [1.0, 1.0, 10.0, 10.0, 1.0, 1.0]
    MASK_FILE - brain mask nifti file to mask the images before training
    PAIRED_BY_LABEL - Whether to train with paired NC-NC and PD-PD images
    GEN_DISC_UPDATE - Defines discriminator training frequency relative to generator
    WEIGHTS_DIR - directory where model was saved to further test the model
    '''

(2) Run main.py - to train the model in the images from TRAIN_PET_CSV file and test in images from TEST_CSV:

    python3 main.py
