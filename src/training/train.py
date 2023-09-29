
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import keras.backend as K
from keras.callbacks import TensorBoard

from src.models.cycle_gan import CycleGAN3D
from src.prep_data.process_image import get_max_min_dataset
from src.prep_data.data_generator import data_generator
K.set_image_data_format("channels_last")


def train(args):

    # Create train directory
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    # Read data
    pet_data = pd.read_csv(args.train_pet_csv)
    spect_data = pd.read_csv(args.train_spect_csv)

    # Get the smaller dataset to use as number of training pairs
    nc_pet = pet_data[pet_data['labels'] == 'NC']
    pd_pet = pet_data[pet_data['labels'] == 'PD']

    nc_spect = spect_data[spect_data['labels'] == 'NC']
    pd_spect = spect_data[spect_data['labels'] == 'PD']

    n_training_pairs = min(len(nc_pet), len(pd_pet), len(nc_spect), len(pd_spect))

    print('-' * 30)
    print('Training data:')
    print(f'PET NC: {len(nc_pet)}')
    print(f'PET PD: {len(pd_pet)}')
    print(' ')
    print(f'SPECT NC: {len(nc_spect)}')
    print(f'SPECT PD: {len(pd_spect)}')
    print(' ')
    print('Number of training pairs: ' + str(n_training_pairs))
    print('-' * 30)

    # Get maximum of datasets to normalize images
    pet_maximum, _ = get_max_min_dataset(list(pet_data['img_paths']))
    spect_maximum, _ = get_max_min_dataset(list(spect_data['img_paths']))

    # get data generator to yield batches of PET and SPECT images
    train_data_generator = data_generator(pet_data, spect_data, args.paired_by_label, args.batch_size,
                                          length=n_training_pairs,
                                          intensity_norm='max_dataset', pet_maximum=pet_maximum,
                                          spect_maximum=spect_maximum, mask=args.mask_file)


    # Get the Cycle GAN model
    model = CycleGAN3D(img_shape=args.img_shape, GPU=args.gpus,
                       loss_list=args.loss_list, loss_weights=args.loss_weights)

    # Create dir to save results
    log_dir = os.path.join(args.train_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.time()))
    tensorboard.set_model(model.generatorAToB)
    tensorboard.set_model(model.generatorBToA)
    tensorboard.set_model(model.discriminatorA)
    tensorboard.set_model(model.discriminatorB)

    # Create real and fake labels to train discriminator
    real_labels = np.ones((args.batch_size, 5, 6, 5, 1))
    fake_labels = np.zeros((args.batch_size, 5, 6, 5, 1))

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.keras.backend.set_session(sess)

        # writer = tf.summary.create_file_writer("/losses")

        # Add a loop, which will run for a specified number of epochs:
        for epoch in range(1, args.epochs + 1):

            # Create two lists to store losses
            gen_losses, dis_losses = [], []

            number_of_batches = int(n_training_pairs / args.batch_size)

            # iterate through the batches
            for index in tqdm(range(number_of_batches), desc=f'Epoch: {epoch}/{args.epochs}'):

                # Get batches of data from data generator
                batchA, batchB = next(train_data_generator)

                # Generate synthetic images from the real ones
                generatedB = model.generatorAToB.predict(batchA)
                generatedA = model.generatorBToA.predict(batchB)

                # Get the discriminator loss and train discriminators
                if index % args.gen_disc_update == 0:
                    dALoss1 = model.discriminatorA.train_on_batch(batchA, real_labels)
                    dALoss2 = model.discriminatorA.train_on_batch(generatedA, fake_labels)
                    dBLoss1 = model.discriminatorB.train_on_batch(batchB, real_labels)
                    dBLoss2 = model.discriminatorB.train_on_batch(generatedB, fake_labels)
                    d_loss = 0.5 * np.add(0.5 * np.add(dALoss1, dALoss2), 0.5 * np.add(dBLoss1, dBLoss2))

                    dis_losses.append(d_loss)

                adv_model = model.adversarial_model
                # Train the generators with the model loss
                g_loss = adv_model.train_on_batch([batchA, batchB],
                                                  [real_labels, real_labels, batchA, batchB, batchA, batchB])

                gen_losses.append(g_loss)

            print(f'G_loss: {np.mean(gen_losses)}\n    D_loss: {np.mean(dis_losses)}')

            # Save losses to Tensorboard
            # self.write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)
            # self.write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
            logs = {'generator_loss': np.mean(gen_losses), 'discriminator_loss': np.mean(dis_losses)}
            tensorboard.on_epoch_end(epoch, logs=logs)

            writer = tf.summary.create_file_writer("/losses")
            with writer.as_default():
                tf.summary.scalar('generator_loss', np.mean(gen_losses), step=epoch)
                writer.flush()

            with writer.as_default():
                tf.summary.scalar('discriminator_loss', np.mean(dis_losses), step=epoch)
                writer.flush()

            model_dir = os.path.join(args.train_dir, 'model')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            if epoch % 20 == 0:
                model.generatorAToB.save_weights(os.path.join(model_dir, 'generatorAToB_epoch_{}.hdf5'.format(epoch)))
                model.generatorBToA.save_weights(os.path.join(model_dir, 'generatorBToA_epoch_{}.hdf5'.format(epoch)))
                model.discriminatorA.save_weights(os.path.join(model_dir, 'discriminatorA_epoch_{}.hdf5'.format(epoch)))
                model.discriminatorB.save_weights(os.path.join(model_dir, 'discriminatorB_epoch_{}.hdf5'.format(epoch)))

    return pet_maximum, spect_maximum
