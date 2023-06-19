# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:55:23 2023

@author: Houdini69
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size= tf.shape(z_mean)[0]
        z_size= tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean+ tf.exp(0.5 * z_log_var) * epsilon
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder= encoder
        self.decoder= decoder
        self.sampler= Sampler()
        self.total_loss_tracker= keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker= keras.metrics.Mean(
        name="reconstruction_loss")
        self.kl_loss_tracker= keras.metrics.Mean(name="kl_loss")    
    @property
    def metrics(self):
        return [self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker] 
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean,z_log_var=self.encoder(data)
            z=self.sampler(z_mean,z_log_var)
            reconstruction=decoder(z)
            reconstruction_loss=tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction),axis=(1,2)))
            kl_loss=-0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
            total_loss=reconstruction_loss+tf.reduce_mean(kl_loss)
        grads=tape.gradient(total_loss,self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return{
        "total_loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result()}  

print("TensorFlow Version:",tf.__version__)
print("GPU Name:",tf.config.list_physical_devices("GPU"))
print("CPU Name:",tf.config.list_physical_devices("CPU"))
latent_dim=2

# construct encoder
encoder_inputs= keras.Input(shape=(128,128,3))
x=layers.Conv2D(32,3,activation="relu",strides=2,padding="same")(encoder_inputs)
x=layers.Conv2D(32,5,activation="relu",strides=2,padding="same")(x)
x=layers.Conv2D(64,3,activation="relu",strides=2,padding="same")(x)
x=layers.Conv2D(64,5,activation="relu",strides=2,padding="same")(x)
x=layers.Flatten()(x)
x=layers.Dense(8,activation="relu")(x)
z_mean=layers.Dense(latent_dim,name="z_mean")(x)
z_log_var=layers.Dense(latent_dim,name="z_log_var")(x)
encoder=keras.Model(encoder_inputs,[z_mean,z_log_var],name="encoder")
encoder.summary()

# construct decoder
latent_inputs= keras.Input(shape=(latent_dim,))
x=layers.Dense(16*16*128,activation="relu")(latent_inputs)
x=layers.Reshape((16,16,128))(x)
x=layers.Conv2DTranspose(64,5,activation="relu",strides=2,padding="same")(x)
x=layers.Conv2DTranspose(64,3,activation="relu",strides=2,padding="same")(x)
x=layers.Conv2DTranspose(32,5,activation="relu",strides=2,padding="same")(x)
#x=layers.Conv2DTranspose(32,3,activation="relu",strides=2,padding="same")(x)
decoder_outputs=layers.Conv2D(3,3,activation="sigmoid",padding="same")(x)
decoder=keras.Model(latent_inputs,decoder_outputs,name="decoder")
decoder.summary()

dataset=keras.preprocessing.image_dataset_from_directory("./img_align_celeba",label_mode=None,image_size=(128,128),batch_size=32,smart_resize=True)
dataset=dataset.map(lambda x:x/255.)

vae= VAE(encoder,decoder)

vae.compile(optimizer=keras.optimizers.Adam(),run_eagerly=True)
vae.fit(dataset,epochs=40,batch_size=32)
vae.save_weights("./models/van_model/vae",save_format="tf")

x=0
x_range=(-2,2)
y=1
y_range=(-2,2)
plt.figure()
n=20
digit_size=128
figure=np.zeros((digit_size*n,digit_size*n,3))
grid_x=np.linspace(x_range[0],x_range[1],n)
grid_y=np.linspace(y_range[0],y_range[1],n)[::-1]
for i,yi in enumerate(grid_y):
    for j,xi in enumerate(grid_x):
        sample_index=[0,0]
        sample_index[x]=xi
        sample_index[y]=yi
        z_sample=np.array([sample_index])
        x_decoded=vae.decoder.predict(z_sample)
        digit=x_decoded[0].reshape(digit_size,digit_size,3)
        figure[i*digit_size:(i+1)*digit_size:,
        j*digit_size:(j+1)*digit_size:]=digit
plt.figure(figsize=(64,64))
start_range=digit_size//2
end_range=n*digit_size+start_range
pixel_range=np.arange(start_range,end_range,digit_size)
sample_range_x=np.round(grid_x,1)
sample_range_y=np.round(grid_y,1)
plt.xticks(pixel_range,sample_range_x)
plt.yticks(pixel_range,sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.axis("off")
plt.imshow(figure)