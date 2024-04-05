from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mndata = MNIST('mnist_samples')
images, labels = mndata.load_training()

# Number of images to analyze
num_images = 5

# Define Sobel operator kernels
kernel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

kernel_y = np.array([[-1, -2, -1],
                     [0,  0,  0],
                     [1,  2,  1]])

# Iterate over the selected number of images
for i in range(num_images):
    # Select the image
    image = np.array(images[i]).reshape(28, 28)

    # Perform convolution with Sobel kernels
    gradient_x = np.convolve(image.flatten(), kernel_x.flatten(),
                             mode='same').reshape(image.shape)
    gradient_y = np.convolve(image.flatten(), kernel_y.flatten(),
                             mode='same').reshape(image.shape)

    # Compute magnitude of gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute intensity distribution statistics
    mean_intensity = np.mean(image)
    median_intensity = np.median(image)
    std_intensity = np.std(image)
    skewness_intensity = np.mean(
        (image - mean_intensity) ** 3) / (np.std(image) ** 3)
    kurtosis_intensity = np.mean(
        (image - mean_intensity) ** 4) / (np.std(image) ** 4) - 3

    # Combine features into a feature vector
    feature_vector = np.array([mean_intensity, median_intensity, std_intensity,
                               skewness_intensity, kurtosis_intensity])

    # Plot the original image, the edge-detected image,
    # and the intensity distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Edge-Detected Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.bar(range(len(feature_vector)), feature_vector)
    plt.xticks(range(len(feature_vector)),
               ['Mean', 'Median', 'Std', 'Skewness', 'Kurtosis'])
    plt.title('Intensity Distribution Statistics')
    plt.xlabel('Statistic')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()
