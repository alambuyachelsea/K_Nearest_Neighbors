import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

# Load MNIST dataset
mndata = MNIST('mnist_samples')
train_images, train_labels = mndata.load_training()

# Select only the image at index 2
image_index = 2
images = train_images[:10]

for image in images:

    # Reshape the image to its original shape (28x28)
    image_2d = np.array(image).reshape(28, 28)

    # Get the coordinates of each pixel
    x_coords, y_coords = np.where(image_2d > 0)

    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)

    # Plot the coordinates
    # plt.figure(figsize=(6, 6))
    plt.scatter(mean_x, mean_y, color='black', s=5)


plt.title('Pixel Coordinates of Image')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis()  # Invert y-axis to match image orientation
plt.grid(True)
plt.show()
