import numpy as np
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

def build_ae(n_adrs, latent_dim=512, bias_factor=-3.5):

    model = models.Sequential([
        layers.Input(shape=(n_adrs,)),
        layers.Dense(
            2048,
            activation=layers.LeakyReLU(0.1),
            kernel_initializer='he_normal',
            name='encoder_layer1'
        ),
        layers.BatchNormalization(name='encoder_bn1'),
        #layers.LeakyReLU(0.1),
        layers.Dropout(0.2, name='encoder_dropout1'),

        layers.Dense(
            1024,
            activation=layers.LeakyReLU(0.1),
            kernel_initializer='he_normal',
            name='encoder_layer2'
        ),
        layers.BatchNormalization(name='encoder_bn2'),
        layers.Dropout(0.2, name='encoder_dropout2'),

        layers.Dense(
            latent_dim,
            activation=layers.LeakyReLU(0.1),
            kernel_initializer='he_normal',
            name='latent_layer'
        ),
        layers.BatchNormalization(name='latent_bn'),

        # ============ DECODER =============
        layers.Dense(
            1024,
            activation=layers.LeakyReLU(0.1),
            kernel_initializer='he_normal',
            name='decoder_layer1'
        ),
        layers.BatchNormalization(name='decoder_bn1'),
        layers.Dropout(0.2, name='decoder_dropout1'),
        
        layers.Dense(
            2048, 
            activation=layers.LeakyReLU(0.1),
            kernel_initializer='he_normal',
            name='decoder_layer2'
        ),
        layers.BatchNormalization(name='decoder_bn2'),
        layers.Dropout(0.2, name='decoder_dropout2'),

        # ============ OUTPUT ============
        layers.Dense(
            n_adrs, 
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            bias_initializer=tf.keras.initializers.Constant(bias_factor),
            name='output_layer'
        )
    ])

    return model