
"""
An adaptation from a 3D Cycle GAN implementation by Song Xue.
"""

import os

import tensorflow as tf
#from tensorflow_addons.layers import InstanceNormalization

import keras.backend as K
from keras.layers import Input, BatchNormalization, GroupNormalization, LeakyReLU, Activation, ZeroPadding3D, Conv3D, Conv3DTranspose, Add
#from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

K.set_image_data_format("channels_last")


class CycleGAN3D:
    """
    Class to build the cycle GAN model.

    Attributes
    ----------
    img_shape: tuple
        Dimensions of the input image to input the network. Already with channel in the last dimension of the array.
        Example: (96, 112, 96, 1)
    common_optimizer: keras.optimizers
        Define the common for the discriminator and generator
    loss_list: list [str]
        List with loss functions for the adversarial model: discriminator loss, cycle consistency loss and identity loss.
        Example ['binary_crossentropy', 'binary_crossentropy', 'mae', 'mae', 'mae', 'mae']
    loss_weights: list [float]
        List with the weights to apply to each loss in the previous list
    discriminatorA: keras.models.Model
        Discriminator network to discriminate real and generated images from batch A (PET images in our case)
    discriminatorB: keras.models.Model
        Discriminator network to discriminate real and generated images from batch B (SPECT images in our case)
    generatorAToB: keras.models.Model
        Generator to generate images B (SPECT) from images A (PET)
    generatorBToA: keras.models.Model
        Generator to generate images A (PET) from images B (SPECT)
    adversarial_model: keras.models.Model
        Adversarial model that is the base of the cycle GAN:
        1- Generates images using both of the generator networks
        2 - Reconstructs images back to original images
        3 - Makes both of the discriminator networks non-trainable
        4 - Gets discriminators outputs on generated images A and B
        5 - Applies the corresponding losses on loss_list

    Methods
    -------
    build_generator()
        Creates the generator network
    build_discriminator()
        Creates the discriminator network
    build_adversarial_model()
        Creates the adversarial model

    """

    def __init__(self, img_shape, GPU, loss_list, loss_weights):
        """

        Parameters
        ----------
        img_shape: tuple
            Dimensions of the input image to input the network. Already with channel in the last dimension of the array.
            Example: (96, 112, 96, 1)
        common_optimizer: keras.optimizers
            Define the common for the discriminator and generator
        loss_list: list [str]
            List with loss functions for the adversarial model: discriminator loss, cycle consistency loss and identity loss.
            Example ['binary_crossentropy', 'binary_crossentropy', 'mae', 'mae', 'mae', 'mae']
        loss_weights: list [float]
            List with the weights to apply to each loss in the previous list
        discriminatorA: keras.models.Model
            Discriminator network to discriminate real and generated images from batch A (PET images in our case)
        discriminatorB: keras.models.Model
            Discriminator network to discriminate real and generated images from batch B (SPECT images in our case)
        generatorAToB: keras.models.Model
            Generator to generate images B (SPECT) from images A (PET)
        generatorBToA: keras.models.Model
            Generator to generate images A (PET) from images B (SPECT)
        adversarial_model: keras.models.Model
            Adversarial model that is the base of the cycle GAN:
            1- Generates images using both of the generator networks
            2 - Reconstructs images back to original images
            3 - Makes both of the discriminator networks non-trainable
            4 - Gets discriminators outputs on generated images A and B
            5 - Applies the corresponding losses on loss_list

        """

        # GPU configs
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        tf.compat.v1.disable_eager_execution()

        self.img_shape = img_shape
        self.common_optimizer = Adam(0.0002, 0.5)

        self.loss_list = loss_list
        self.loss_weights = loss_weights

        self.discriminatorA = self.build_discriminator()
        self.discriminatorB = self.build_discriminator()

        self.discriminatorA.compile(loss=loss_list[0], optimizer=self.common_optimizer)
        self.discriminatorB.compile(loss=loss_list[0], optimizer=self.common_optimizer)

        self.generatorAToB = self.build_generator()
        self.generatorBToA = self.build_generator()

        # Then, create and compile the adversarial model
        self.adversarial_model = self.build_adversarial_model()
        self.adversarial_model.compile(loss=self.loss_list,
                                       loss_weights=self.loss_weights, #original: [1, 1, 10.0, 10.0, 1.0, 1.0]
                                       optimizer=self.common_optimizer)

    def build_generator(self):
        """
        Create the generator network using the hyperparameter values defined below
        """

        def residual_block(x):
            """
            Residual block
            """
            res = Conv3D(filters=128, kernel_size=3, strides=1, padding="same")(x)
            #res = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)(res)
            res = GroupNormalization(groups=1, axis=-1)(x)
            res = Activation('relu')(res)

            res = Conv3D(filters=128, kernel_size=3, strides=1, padding="same")(res)
            #res = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)(res)
            res = GroupNormalization(groups=1, axis=-1)(x)
            out = Add()([res, x])

            return out ##can I use this Add from keras????

        residual_blocks = 6
        input_layer = Input(self.img_shape)

        # First Convolution block
        x = Conv3D(filters=32, kernel_size=7, strides=1, padding="same")(input_layer)
        x = GroupNormalization(groups=1, axis=-1)(x) ##it was 1 in every instance norm
        x = Activation("relu")(x)

        # 2nd Convolution block
        x = Conv3D(filters=64, kernel_size=3, strides=2, padding="same")(x)
        x = GroupNormalization(groups=1, axis=-1)(x)
        x = Activation("relu")(x)

        # 3rd Convolution block
        x = Conv3D(filters=128, kernel_size=3, strides=2, padding="same")(x)
        x = GroupNormalization(groups=1, axis=-1)(x)
        x = Activation("relu")(x)

        # Residual blocks
        for _ in range(residual_blocks):
            x = residual_block(x)

        # Upsampling blocks

        # 1st Upsampling block
        x = Conv3DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = GroupNormalization(groups=1, axis=-1)(x)
        x = Activation("relu")(x)

        # 2nd Upsampling block
        x = Conv3DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = GroupNormalization(groups=1, axis=-1)(x)
        x = Activation("relu")(x)

        # Last Convolution layer
        x = Conv3D(filters=1, kernel_size=7, strides=1, padding="same")(x) ##changed filter from 3 to 1
        output = Activation('tanh')(x)

        model = Model(inputs=[input_layer], outputs=[output])
        return model

    def build_discriminator(self):
        """
        Creates the discriminator network using the hyperparameter values defined below
        """
        hidden_layers = 3

        input_layer = Input(self.img_shape)

        x = ZeroPadding3D(padding=(1, 1, 1))(input_layer)

        # 1st Convolutional block
        x = Conv3D(filters=64, kernel_size=4, strides=2, padding="valid")(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = ZeroPadding3D(padding=(1, 1, 1))(x)

        # 3 Hidden Convolution blocks
        for i in range(1, hidden_layers + 1):
            x = Conv3D(filters=2 ** i * 64, kernel_size=4, strides=2, padding="valid")(x)
            x = GroupNormalization(groups=1, axis=-1)(x)
            x = LeakyReLU(alpha=0.2)(x)

            x = ZeroPadding3D(padding=(1, 1, 1))(x)

        # Last Convolution layer
        output = Conv3D(filters=1, kernel_size=4, strides=1, activation="sigmoid")(x)

        model = Model(inputs=[input_layer], outputs=[output])
        return model

    def build_adversarial_model(self):
        """
        Creates the adversarial network
        """
        inputA = Input(self.img_shape)
        inputB = Input(self.img_shape)

        # Generates images using both of the generator networks
        generatedB = self.generatorAToB(inputA)
        generatedA = self.generatorBToA(inputB)

        # Reconstructs images back to original images
        reconstructedA = self.generatorBToA(generatedB)
        reconstructedB = self.generatorAToB(generatedA)

        # Generate Identity images
        generatedAId = self.generatorBToA(inputA)
        generatedBId = self.generatorAToB(inputB)

        # Makes both of the discriminator networks non-trainable
        self.discriminatorA.trainable = False
        self.discriminatorB.trainable = False

        # Discriminator outputs on generated images A and B
        probsA = self.discriminatorA(generatedA)
        probsB = self.discriminatorB(generatedB)

        adversarial_model = Model(inputs=[inputA, inputB],
                                  outputs=[probsA, probsB, reconstructedA, reconstructedB,
                                           generatedAId, generatedBId])
        return adversarial_model


