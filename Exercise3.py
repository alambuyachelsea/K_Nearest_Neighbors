import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

# Load MNIST dataset
mndata = MNIST('mnist_samples')
train_images, train_labels = mndata.load_training()[0:10000]
test_images, test_labels = mndata.load_testing()[0:1000]

# Display an image
image_index = 209
image = np.array(train_images[image_index], dtype='uint8')
image = image.reshape((28, 28))


plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

print(train_labels[image_index])
print(mndata.display(train_images[image_index]))

# Get the coordinates of points
x_coords, y_coords = np.where(image > 0)

# Plot the points
plt.scatter(x_coords, 28 - y_coords, color='black', s=5)
# Invert y-axis to match image orientation

plt.gca().set_aspect('equal', adjustable='box')
# Set aspect ratio to maintain square pixels

plt.gca().invert_yaxis()  # Invert y-axis to match image orientation
plt.axis('off')
plt.show()

"""
    crying sobbing, why did i choose this life for myself
    Idea to solve
    Use 2 features to create an expection for each figure
    1. Intensity distribution
    2. ?? more info needed
"""
