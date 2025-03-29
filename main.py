######################## 
# import required libs
########################
import keras.metrics
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot  as plt 
import os
from utils.vae_plotter import VAEPlotter
from vae_train import VAETrainer
from vae_model import Sampling, VAE
import argparse

labels = {0: "T-shirt / top", 
          1: "Trouser",
          2: "Pullover",
          3: "Dress",
          4: "Coat",
          5: "Sandal",
          6: "Shirt",
          7: "Sneaker",
          8: "Bag",
          9: "Ankle boot"
        }


latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
mean = layers.Dense(latent_dim, name="mean")(x)
log_var = layers.Dense(latent_dim, name="log_var")(x)
z = Sampling()([mean, log_var])

encoder = keras.Model(encoder_inputs, [mean, log_var, z], name="encoder")
encoder.summary()

#######################
# define Decoder Block
#######################

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

def main():
    parser = argparse.ArgumentParser(description="Train and visualize a Variational Autoencoder on Fashion MNIST.")
    parser.add_argument("--model_path", type=str, default="vae_fashion_mnist", help="Path to save/load the trained VAE model.")
    args = parser.parse_args()

    # Train or load VAE model
    vae_trainer = VAETrainer(encoder, decoder, model_path=args.model_path)
    vae_trainer.load_or_train()
    
    # Visualize latent space
    vae_plotter = VAEPlotter(encoder, decoder)
    vae_plotter.plot_latent_space()

    print("[+] Done...")

if __name__ == "__main__":
    main()
