
import os
import pandas as pd

from src import settings
from src.prep_data.process_image import get_max_min_dataset
from src.prep_data.prep_data_df import prep_data
from src.training.train import train
from src.testing.test import test


class Run:
    def __init__(self, gpus, proj_dir):
        self.gpus = gpus
        self.proj_dir = proj_dir

        self.pet_df_dir = settings.PET_TRAIN_DF
        self.spect_df_dir = settings.SPECT_TRAIN_DF

        self.train_dir = self.proj_dir + '/train'
        self.fine_tune_dir = self.proj_dir + '/finetune'
        self.test_dir = self.proj_dir + '/test'


        self.label_dict = settings.LABELS

        if not os.path.exists(self.proj_dir):
            os.mkdir(self.proj_dir)

        # if not os.path.exists(self.pretrain_dir):
        #     os.mkdir(self.pretrain_dir)

        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)

        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)


    def prep_data(self):

        ### PET dataset ###
        self.pet_data_df = pd.read_csv(self.pet_df_dir)

            # Exclude prep_data not needed
        self.pet_data_df = self.pet_data_df[self.pet_data_df['exclude'] == 0]

            # Exclude classes/labels not needed for training
        self.pet_data_df = self.pet_data_df[self.pet_data_df['labels'] != 'MSA']
        self.pet_data_df = self.pet_data_df[self.pet_data_df['labels'] != 'PSP']

            # Select PET training prep_data
        self.pet_data_df = self.pet_data_df[self.pet_data_df['splits'] == 0]

        ### SPECT dataset ###
        self.spect_data_df = pd.read_csv(self.spect_df_dir)

            # Exclude prep_data not needed
        self.spect_data_df = self.spect_data_df[self.spect_data_df['exclude'] == 0]

            # Select SPECT training prep_data
        self.spect_data_df = self.spect_data_df[self.spect_data_df['splits'] == 0]

            # Exclude classes/labels not needed for training


        # lista = [10,11,12,14,15, 400,401,402,403, 600]
        # self.pet_data_df = self.pet_data_df.loc[lista, :]
        #
        # lista = [1, 2, 3, 4, 100, 300, 301, 302, 303, 500]
        # self.spect_data_df = self.spect_data_df.loc[lista, :]

        max_pet_dataset, min_pet_dataset = get_max_min_dataset(list(self.pet_data_df['img_paths']))
        max_spect_dataset, min_spect_dataset = get_max_min_dataset(list(self.spect_data_df['img_paths']))

        max_all = max(max_pet_dataset, max_spect_dataset)


        print('-' * 30)
        print('Preparing NC PET train prep_data...')
        print('-' * 30)

        self.pet_data_df_NC = self.pet_data_df[self.pet_data_df['labels'] == 'NC']

        self.pet_x_train_NC, self.pet_y_train_NC = prep_data(df=self.pet_data_df_NC,
                                                             label_dict= self.label_dict,
                                                             norm='max',
                                                             resize_type='pad_PET',
                                                             maximum=max_pet_dataset,
                                                             mask='50%')

        print(f'Training NC PET dataset: {len(self.pet_x_train_NC)}')


        print('-' * 30)
        print('Preparing NC SPECT train prep_data...')
        print('-' * 30)

        self.spect_data_df_NC = self.spect_data_df[self.spect_data_df['labels'] == 'NC']

        self.spect_x_train_NC, self.spect_y_train_NC= prep_data(df=self.spect_data_df_NC,
                                                             label_dict= self.label_dict,
                                                             norm='max',
                                                             resize_type='pad_SPECT',
                                                             maximum=max_spect_dataset,
                                                             mask='50%')


        print(f'Training NC SPECT dataset: {len(self.spect_x_train_NC)}')


        print('-' * 30)
        print('Preparing PD PET train prep_data...')
        print('-' * 30)

        self.pet_data_df_PD = self.pet_data_df[self.pet_data_df['labels'] == 'PD']

        self.pet_x_train_PD, self.pet_y_train_PD = prep_data(df=self.pet_data_df_PD,
                                                                   label_dict=self.label_dict,
                                                                   norm='max',
                                                                   resize_type='pad_PET',
                                                                   maximum=max_pet_dataset,
                                                                   mask='50%')

        print(f'Training PD PET dataset: {len(self.pet_x_train_PD)}')

        print('-' * 30)
        print('Preparing PD SPECT train prep_data...')
        print('-' * 30)

        self.spect_data_df_PD = self.spect_data_df[self.spect_data_df['labels'] == 'PD']

        self.spect_x_train_PD, self.spect_y_train_PD = prep_data(df=self.spect_data_df_PD,
                                                                       label_dict=self.label_dict,
                                                                       norm='max',
                                                                       resize_type='pad_SPECT',
                                                                       maximum=max_spect_dataset,
                                                                       mask='50%')

        print(f'Training PD SPECT dataset: {len(self.spect_x_train_PD)}')

    def training(self):
        # Training
        print('-' * 30)
        print('Training...')
        print('-' * 30)

        train(self.pet_x_train_NC, self.spect_x_train_NC, self.gpus, self.train_dir)
        train(self.pet_x_train_PD, self.spect_x_train_PD, self.gpus, self.fine_tune_dir, use_pretrained_model=True, pretrained_model_dir=self.train_dir)

    def testing(self, gpus):
        # Testing
        print('-' * 30)
        print('Testing...')
        print('-' * 30)

        self.test_data_df = pd.read_csv(self.pet_df_dir)

        # Exclude prep_data not needed
        self.pet_data_df = self.pet_data_df[self.pet_data_df['exclude'] == 0]

        # Exclude classes/labels not needed for training
        self.pet_data_df = self.pet_data_df[self.pet_data_df['labels'] != 'MSA']
        self.pet_data_df = self.pet_data_df[self.pet_data_df['labels'] != 'PSP']

        max_pet_dataset, min_pet_dataset = get_max_min_dataset(list(self.pet_data_df['img_paths']))

        self.test_data_df = self.pet_data_df[self.pet_data_df['splits'] == 1]

        print(f'Testing PET dataset: {len(self.test_data_df)}')


        self.x_test, self.y_test = prep_data(df=self.test_data_df,
                                                    label_dict=self.label_dict,
                                                    norm='max',
                                                    resize_type='pad_PET',
                                                    maximum=max_pet_dataset,
                                                    mask='50%')



        test(self.x_test, self.y_test, self.train_dir, gpus, self.test_dir)

if __name__ == '__main__':
    run = Run(gpus='2,3', proj_dir='/home/leonor/Code/cycle_gan_tf/results/run2/')

    run.prep_data()
    run.training()
    run.testing()

