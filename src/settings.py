
### Configurations ###

# Directory
PROJ_DIR = '/home/leonor/Code/DAT-cycle-gan/results/test_run/'
TRAIN_DIR = PROJ_DIR + 'train/'
TEST_DIR = PROJ_DIR + 'genSPECT/'

# Data
TRAIN_PET_CSV = '/home/leonor/Code/DAT-cycle-gan/results/test_run/train_pet.csv'
TRAIN_SPECT_CSV = '/home/leonor/Code/DAT-cycle-gan/results/test_run/train_spect.csv'
TEST_CSV = '/home/leonor/Code/DAT-cycle-gan/results/test_run/test_pet.csv'

# Training
GPUS = '1'
EPOCHS = 3 # 200
BATCH_SIZE = 1
IMG_SHAPE = (96, 112, 96, 1)
LOSS_LIST = ['binary_crossentropy', 'binary_crossentropy', 'mae', 'mae', 'mae', 'mae']
LOSS_WEIGHTS = [1.0, 1.0, 10.0, 10.0, 1.0, 1.0]
MASK_FILE = '/home/leonor/Code/brain_masks/brainmask.nii'
PAIRED_BY_LABEL = True
GEN_DISC_UPDATE = 1
WEIGHTS_DIR = TRAIN_DIR + f'/model/generatorAToB_epoch_{EPOCHS}.hdf5'
