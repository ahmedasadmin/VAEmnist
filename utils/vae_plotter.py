import matplotlib.pyplot as plt
import numpy as np

class VAEPlotter:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def plot_latent_space(self, n=10, figsize=5):
        img_size = 28
        scale = 0.5
        figure = np.zeros((img_size * n, img_size * n))
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(sample, verbose=0)
                image = x_decoded[0].reshape(img_size, img_size)
                figure[i * img_size: (i + 1) * img_size,
                       j * img_size: (j + 1) * img_size] = image

        plt.figure(figsize=(figsize, figsize))
        start_range = img_size // 2
        end_range = n * img_size + start_range
        pixel_range = np.arange(start_range, end_range, img_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()

    def plot_label_clusters(self, data, test_lab, labels):
        z_mean, _, _ = self.encoder.predict(data)
        plt.figure(figsize=(12, 10))
        sc = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=test_lab)
        cbar = plt.colorbar(sc, ticks=range(10))
        cbar.ax.set_yticklabels([labels.get(i) for i in range(10)])
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()
    