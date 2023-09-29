
import os
import numpy as np
import pandas as pd
import nibabel as nb
import tensorflow as tf

from src.prep_data.process_image import process_image, create_slice_figure
from src.models.cycle_gan import CycleGAN3D



def test(test_csv, dataset_maximum, weights_dir, genPETorSPECT='SPECT', save_orig=True, args=None):

    # Create test directory
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    # Get test data
    test_data = pd.read_csv(test_csv)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.keras.backend.set_session(sess)

        # Get Cycle GAN model
        model = CycleGAN3D(img_shape=args.img_shape, GPU=args.gpus,
                           loss_list=args.loss_list, loss_weights=args.loss_weights)


        # Load weights of needed generator

        if genPETorSPECT == 'SPECT':
            model.generatorAToB.load_weights(weights_dir)
            resize_type = 'pad_PET'
        elif genPETorSPECT == 'PET':
            model.generatorBToA.load_weights(weights_dir)
            resize_type = 'pad_SPECT'
        else:
            print('Choose to generate either PET or SPECT.')


        test_imgs = []
        labels = []
        predictions = []

        # iterate through the test data
        for i in range(len(test_data)):
            # get image path and processed image
            img_path = test_data.at[i, 'img_paths']
            img = process_image(img_path, norm='max_dataset', resize_type=resize_type, maximum=dataset_maximum, mask=args.mask_file)
            test_imgs.append(img)

            # Get correspondent label
            label = test_data.at[i, 'labels']
            labels.append(label)

            # Sve original image
            if save_orig:
                nifti_img = nb.Nifti1Image(img[: ,: ,:], affine=np.eye(4))
                nb.save(nifti_img, args.test_dir + f'/orig_{label}_{str(i).zfill(3)}.nii')

            # Get generated images
            batch = []
            img = np.expand_dims(img, axis=-1)
            batch.append(img)
            prediction = model.generatorAToB.predict(np.array(batch))
            predictions.append(prediction)
            batch.clear()
            img = prediction[0, :, :, :, 0]

            nifti_img = nb.Nifti1Image(img, affine=np.eye(4))
            nb.save(nifti_img, args.test_dir + f'/pred_{label}_{str(i).zfill(3)}.nii')


        ## visualize 4 random images
        iter_list = list(np.random.randint(0 ,len(test_imgs), size=4))

        for i in iter_list:
            real = test_imgs[i][: ,: ,:]
            print(real.shape)
            create_slice_figure(real, str(i))

            pred = predictions[i][0 ,: ,: ,: ,0]
            print(pred.shape)
            create_slice_figure(pred, str(i))
