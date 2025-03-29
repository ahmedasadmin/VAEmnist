# Variational Autoencoder (VAE) for Fashion-MNIST

## Overview

This project implements a Variational Autoencoder (VAE) using Keras and TensorFlow. The VAE is trained on the Fashion-MNIST dataset, which consists of grayscale images of clothing items. The model learns a latent representation of the dataset and can generate new images based on the learned distribution.

## Features

- Implements a convolutional encoder and decoder for the VAE.
- Uses a custom `Sampling` layer to perform reparameterization.
- Computes both reconstruction and KL divergence loss.
- Trains the VAE on the Fashion-MNIST dataset.
- Saves and loads trained models for reuse.
- Provides visualization functions to explore the latent space.
- Modular code structure for better maintainability.

## Dependencies

Ensure you have the following Python packages installed:

```bash
pip install -r requirements.txt
```


## How to Run

1. **Train the Model**

   - Run the script to train a new VAE model if no pre-trained model exists.

   ```bash
   python main.py --train
   ```

   - The trained model will be saved as `vae_fashion_mnist.keras`.


3. **Visualizing the Latent Space**

   - The script includes functions to visualize the learned latent space.

## Model Architecture

- **Encoder**

  - Convolutional layers extract features from the input images.
  - Outputs a latent representation with a mean and log variance.
  - Uses the `Sampling` layer to sample from the latent space.

- **Decoder**

  - Takes sampled latent vectors and reconstructs the input images.
  - Uses transposed convolutional layers to upsample the feature maps.

## Notes

- The dataset is normalized to the range [0, 1] before training.
- The loss function includes both reconstruction loss and KL divergence.
- The model is trained for 10 epochs with a batch size of 128.
- Logging is used to track training progress..

## License
This project is open-source and available for modification and distribution under an open license.

