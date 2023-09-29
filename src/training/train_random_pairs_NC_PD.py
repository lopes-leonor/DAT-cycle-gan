
import time
import os
import numpy as np
from tqdm import tqdm

from src import settings
from src.models.cycle_gan import CycleGAN3D
from sklearn.utils import shuffle

import tensorflow as tf

import keras.backend as K
from keras.callbacks import TensorBoard

K.set_image_data_format("channels_last")


def data_generator(arrays_pet, arrays_spect, labels_pet, labels_spect, batch_size, length):
    while True:
        pet_batch, spect_batch = [], []
        count = 0

        # Shuffle arrays and labels
        arrays_pet_s, labels_pet_s = shuffle(arrays_pet, labels_pet)
        arrays_spect_s, labels_spect_s = shuffle(arrays_spect, labels_spect)

        # iterate until the desired length to get batches of pet + spect
        for j in range(length):

            # get the j'th PET
            pet = arrays_pet_s[j]
            pet = np.expand_dims(pet, axis=-1)
            pet_batch.append(pet) # append to pet batch

            # get the j'th label
            l_pet = labels_pet_s[j]


            # get all spect images of the same label of the pet label of the correspondent image
            indices = np.where(labels_spect_s == l_pet)[0]
            arrays_spect_new = arrays_spect_s[indices]

            # get the j'th SPECT of the same PET label
            spect = arrays_spect_new[j]
            spect = np.expand_dims(spect, axis=-1)
            spect_batch.append(spect) # append to spect batch

            count += 1
            if count == batch_size:
                yield np.array(pet_batch), np.array(spect_batch)
                count = 0
                pet_batch, spect_batch = [], []

def train(x_train_pet, x_train_spect, y_train_pet, y_train_spect, gpus, train_dir,
          gen_disc_update=1, n_training_pairs=None, use_pretrained_model=False, pretrained_model_dir=None):

    print('Number of training pairs: ' + str(n_training_pairs))

    # get prep_data generator to yield batches of PET and SPECT images
    train_data_generator = data_generator(x_train_pet, x_train_spect, y_train_pet, y_train_spect, batch_size=settings.BATCH_SIZE, length=n_training_pairs)

    real_labels = np.ones((settings.BATCH_SIZE, 5, 6, 5, 1))  ##was 7,7,1
    fake_labels = np.zeros((settings.BATCH_SIZE, 5, 6, 5, 1))

    log_dir = os.path.join(train_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = CycleGAN3D(img_shape=settings.IMG_SHAPE, GPU=gpus,
                        loss_list=settings.LOSS_LIST, loss_weights=settings.LOSS_WEIGHTS)

    if use_pretrained_model:
        model.generatorAToB.load_weights(pretrained_model_dir + '/model/generatorAToB_epoch_100.hdf5')
        model.generatorBToA.load_weights(pretrained_model_dir + '/model/generatorBToA_epoch_100.hdf5')
        model.discriminatorA.load_weights(pretrained_model_dir + '/model/discriminatorA_epoch_100.hdf5')
        model.discriminatorB.load_weights(pretrained_model_dir + '/model/discriminatorB_epoch_100.hdf5')

    tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.time()))
    tensorboard.set_model(model.generatorAToB)
    tensorboard.set_model(model.generatorBToA)
    tensorboard.set_model(model.discriminatorA)
    tensorboard.set_model(model.discriminatorB)


    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.keras.backend.set_session(sess)

        # writer = tf.summary.create_file_writer("/losses")

        # Add a loop, which will run for a specified number of epochs:
        for epoch in range(1, settings.EPOCHS +1):
            # Create two lists to store losses
            gen_losses, dis_losses = [] ,[]

            number_of_batches = int(n_training_pairs / settings.BATCH_SIZE)

            for index in tqdm(range(number_of_batches), desc= f'Epoch: {epoch}/{settings.EPOCHS}'):
                #print(f'Epoch: {epoch}/{settings.EPOCHS}\n  Batch: {index+1}/{number_of_batches}')
                batchA, batchB = next(train_data_generator)
                print(batchA[0].shape)

                generatedB = model.generatorAToB.predict(batchA)
                generatedA = model.generatorBToA.predict(batchB)

                if index % gen_disc_update == 0:

                    dALoss1 = model.discriminatorA.train_on_batch(batchA, real_labels)
                    dALoss2 = model.discriminatorA.train_on_batch(generatedA, fake_labels)
                    dBLoss1 = model.discriminatorB.train_on_batch(batchB, real_labels)
                    dBLoss2 = model.discriminatorB.train_on_batch(generatedB, fake_labels)
                    d_loss = 0.5 * np.add(0.5 * np.add(dALoss1, dALoss2), 0.5 * np.add(dBLoss1, dBLoss2))

                    dis_losses.append(d_loss)


                adv_model = model.adversarial_model
                g_loss = adv_model.train_on_batch([batchA, batchB],
                                                  [real_labels, real_labels, batchA, batchB, batchA, batchB])

                gen_losses.append(g_loss)


                #print(f'G_loss: {g_loss}\n    D_loss: {d_loss}')
                #print('metric names: {}'.format(names))

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

            model_dir = os.path.join(train_dir, 'model')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            if epoch % 20 == 0:
                model.generatorAToB.save_weights(os.path.join(model_dir, 'generatorAToB_epoch_{}.hdf5'.format(epoch)))
                model.generatorBToA.save_weights(os.path.join(model_dir, 'generatorBToA_epoch_{}.hdf5'.format(epoch)))
                model.discriminatorA.save_weights(os.path.join(model_dir, 'discriminatorA_epoch_{}.hdf5'.format(epoch)))
                model.discriminatorB.save_weights(os.path.join(model_dir, 'discriminatorB_epoch_{}.hdf5'.format(epoch)))
