import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from vae_model import VAE, Sampling
class VAETrainer:
    def __init__(self, encoder, decoder, model_path="vae_fashion_mnist"):
        self.model_path = model_path
        self.encoder = encoder
        self.decoder = decoder
        self.vae = VAE(self.encoder, self.decoder)

    def load_data(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
        fashion_mnist = np.concatenate([x_train, x_test], axis=0)
        fashion_mnist = np.expand_dims(fashion_mnist, -1).astype("float32") / 255
        return fashion_mnist

    def train(self, epochs=1, batch_size=128):
        print("[+] Model not found, training new model")
        data = self.load_data()
        
        self.vae.build(input_shape=(None, 28, 28, 1))
        self.vae.compile(optimizer=keras.optimizers.Adam())
        self.vae.fit(data, epochs=epochs, batch_size=batch_size)
        # self.vae.save(self.model_path, save_format="tf")
        # print(f"[+] Model saved to {self.model_path}")

    def load_or_train(self):
        if os.path.exists(self.model_path):
            print(f"[+] Loading existing model from {self.model_path}")
            self.vae = keras.models.load_model("vae_fashion_mnist")

        else:
            self.train()
        return self.vae