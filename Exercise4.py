import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# opens the file and retrives the training data
def open_file(file_path):

    training_data = []

    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            line = []
            for value in row:  # info stored as string converted to float
                x = float(value)
                line.append(x)
            training_data.append(line)

    return training_data


# Separate data into x, y, and labels
def separate_data(training_data):

    # Separate data into x, y, and labels
    x = [point[0] for point in training_data]
    y = [point[1] for point in training_data]
    labels = [point[2] for point in training_data]

    return (x, y, labels)


# Gets the test chip's k nearest neighbours to predict it's outcome
def prediction(training_set, test_chip, k):

    training_set = np.array(training_set)
    properties = training_set[:, :2]  # Only get the properties
    labels = training_set[:, 2]  # Only lables

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(properties, labels)

    test_case = np.array([test_chip])

    return knn_model.predict(test_case)


# Compares the given data to the prediction and calculates the error
def countErrors(training_set, k):
    error_counter = 0
    for i in range(len(training_set)):
        # Extracting the true class label from training_set
        true_label = training_set[i, 2]

        # Predicting the class label using part2 method with k
        predicted_label = prediction(training_set,
                                     [training_set[i, 0],
                                      training_set[i, 1]], k)

        # Checking if the predicted label matches the true label
        if predicted_label != true_label:
            error_counter += 1

    return error_counter


# Part 1, opens the csv file and plots the original training data
def part_1(file_path):

    training_data = open_file(file_path)

    seperated_data = separate_data(training_data)

    # Plot the data
    plt.figure(figsize=(5, 4))
    plt.scatter(seperated_data[0], seperated_data[1], c=seperated_data[2],
                cmap='coolwarm', edgecolors='k', alpha=0.8)
    plt.title('Original Microchip Data')
    plt.xlabel('Property 1')
    plt.ylabel('Property 2')
    plt.show()


# Part 2, Predicts the outcome of test chips based on K_nearest neighbours
def part_2(training_data, test_data, k_values):

    seperated_data = separate_data(training_data)

    # Perform predictions for each value of k
    for k in k_values:
        print(f'\nWhen K = {k}')
        count = 1

        for test_chip in test_data:

            vote = prediction(training_data, test_chip, k)

            match vote:
                case 1.0:
                    print('Test Chip {} ===> OK'.format(count))

                case 0.0:
                    print('Test Chip {} ===> Fail'.format(count))

            count += 1

    seperated_data = separate_data(training_data)

    # Plot the original data plus the test chips
    plt.figure(figsize=(5, 4))
    plt.scatter(seperated_data[0], seperated_data[1], c=seperated_data[2],
                cmap='coolwarm', edgecolors='k', alpha=0.8)
    plt.title('Training and Test Data')
    plt.xlabel("Property 1")
    plt.ylabel("Property 2")

    plt.plot(-0.3, 1.0, '*')
    plt.plot(-0.5, -0.1, '*')
    plt.plot(0.6, 0.0, '*')

    plt.show()


# Part 3, plots a 2X2 grid for the decision boundaries for each K value
def part_3(training_data, k_values):

    training_data = np.array(training_data)

    # Create a meshgrid to cover the range of data points
    x_min = training_data[:, 0].min() - 0.1
    x_max = training_data[:, 0].max() + 0.1
    y_min = training_data[:, 1].min() - 0.1
    y_max = training_data[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, k in enumerate(k_values):

        ax = axs[i//2, i % 2]

        # Compute predictions for each point on the meshgrid
        Z = np.array([prediction(training_data, [x, y], k)
                      for x, y in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], alpha=0.4,
                    cmap='coolwarm')

        # Plot training data points
        ax.scatter(training_data[:, 0], training_data[:, 1],
                   c=training_data[:, 2], cmap='coolwarm',
                   edgecolors='k', s=20)

        errors = countErrors(training_data, k)

        # Set plot titles
        ax.set_title(f'Decision Boundary for k = {k}, Errors = {errors}')
        ax.set_xlabel('Property 1')
        ax.set_ylabel('Property 2')

    plt.show()


# File containing the data
file_path = 'A1_datasets/microchips.csv'

# Execute part 1
part_1(file_path)

# Variables for part 2
training_data = open_file(file_path)

# Define the test points
test_data = np.array([
    [-0.3, 1.0],
    [-0.5, -0.1],
    [0.6, 0.0]
])

# Define different values of k
k_values = [1, 3, 5, 7]

# Execute part 2
part_2(training_data, test_data, k_values)

# Execute part 3
part_3(training_data, k_values)
