import tensorflow as tf
import numpy as np
from tensorflow import keras as k
from tensorflow.keras import layers

def make_model(SIZE=128, LATENT_DIM=9, LATENT_SUB_DIM=15, LR=1e-4, BETA=1.):
    encoder_inputs = layers.Input(shape=(SIZE, SIZE, 1), name='encoder_input')
    e = layers.Conv2D(filters=8,kernel_size=5,padding='SAME',activation='relu',strides=(2,2))(encoder_inputs)
    e = layers.BatchNormalization()(e)
    e = layers.Conv2D(filters=16,kernel_size=5,padding='SAME',activation='relu',strides=(2,2))(e)
    e = layers.BatchNormalization()(e)
    e = layers.Flatten()(e)
    z_mean = layers.Dense(LATENT_DIM*LATENT_SUB_DIM)(e)
    z_mean = layers.Reshape((LATENT_DIM, LATENT_SUB_DIM), name='z_mean')(z_mean)
    z_log_var = layers.Dense(LATENT_DIM*LATENT_SUB_DIM)(e)
    z_log_var = layers.Reshape((LATENT_DIM, LATENT_SUB_DIM), name='z_logvar')(z_log_var)
    encoder = k.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var], name='encoder')


    decoder_inputs = layers.Input(shape=(LATENT_DIM,LATENT_SUB_DIM,), name='decoder_input')
    d = layers.Flatten()(decoder_inputs)
    d = layers.Dense(units=7*7*8,activation='relu')(d)
    d = layers.Reshape((7,7,8))(d)
    d = layers.Conv2DTranspose(filters=8,kernel_size=4,strides=(2, 2), padding="SAME", activation='relu')(d)
    d = layers.Conv2DTranspose(filters=16,kernel_size=4,strides=(2, 2), padding="SAME", activation='relu')(d)
    decoded = layers.Conv2DTranspose(filters=1, kernel_size=3,strides=(1, 1), padding="SAME")(d)
    decoder = k.Model(inputs=decoder_inputs, outputs=decoded, name='decoder')


    def sample(inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    sampler = layers.Lambda(sample)
    z = sampler([z_mean, z_log_var])
    vae = k.Model(inputs=encoder_inputs, outputs=decoder(z), name='vae')

    def compute_kernel(x, y):
        x_shape = tf.shape(x)
        x_size = x_shape[0]
        y_size = tf.shape(y)[0]
        dim = x_shape[1]
        sub_dim = x_shape[2]
        tiled_x = tf.tile(tf.reshape(x, [x_size, 1, dim, sub_dim]), [1, y_size, 1, 1])
        tiled_y = tf.tile(tf.reshape(y, [1, y_size, dim, sub_dim]), [x_size, 1, 1, 1])
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=[2,3]) / tf.cast(dim, tf.float32))


    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    

    true_samples = tf.random_normal(shape=tf.shape(z))
    loss_mmd = compute_mmd(true_samples, z)
    vae.add_loss(loss_mmd*BETA)
        
    vae.compile(loss='mse', optimizer=k.optimizers.Adam(LR), metrics=['mse'])
    return encoder, decoder , vae