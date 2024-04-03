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