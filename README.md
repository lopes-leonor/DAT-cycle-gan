# DAT-cycle-gan

Code for the paper: "Dopaminergic PET to SPECT Domain Adaptation: A Cycle GAN translation approach".


Dopamine transporter imaging is routinely used in Parkinson's disease (PD) and atypical parkinsonian syndromes (APS) diagnosis. 
While [11C]CFT PET is prevalent in Asia with a large APS database, Europe relies on [123I]FP-CIT SPECT with limited APS data. 
Our aim was to develop a deep learning-based method to convert [11C]CFT PET images to [123I]FP-CIT SPECT images, 
facilitating multicenter studies and overcoming data scarcity to promote Artificial Intelligence (AI) advancements.

![Our Cycle GAN model](https://github.com/lopes-leonor/DAT-cycle-gan/blob/main/images/figure1.png)

<img src="https://github.com/lopes-leonor/DAT-cycle-gan/blob/main/images/figure6.png" width="300" alt="Synthetic SPECT images">

![Synthetic SPECT images](https://github.com/lopes-leonor/DAT-cycle-gan/blob/main/images/figure6.png)


## Usage

(1) Edit settings.py file to add your parameters:

    TRAIN_DIR - directory to save the results and weights of the model
    TEST_DIR - directory to save the generated images
    TRAIN_PET_CSV - csv file with the paths to the PET images to train and the labels (columns' names should be 'img_paths' and 'labels')
    TRAIN_SPECT_CSV - csv file with the paths to the SPECT images to train and the labels (columns' names should be 'img_paths' and 'labels')

(2) Run main.py:

    python3 main.py
