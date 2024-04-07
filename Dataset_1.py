import csv
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


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
def seperate_data(training_data):

    # Separate data into x, y, and labels
    x = [point[0] for point in training_data]
    y = [point[1] for point in training_data]
    labels = [point[2] for point in training_data]

    return (x, y, labels)


# Computes distance between the test point and
# Every other point in training data
def distance_measurement(trainingset, test_case):
    # point + distance tuple
    distanced_points = []

    # distance calculation
    for train_point in trainingset:
        x_diff = np.square(train_point[0] - test_case[0])
        y_diff = np.square(train_point[1] - test_case[1])

        dist = np.round(np.sqrt(x_diff + y_diff), 4)
        distanced_point = [train_point[0], train_point[1],
                           train_point[2], dist]

        distanced_points.append(distanced_point)

    return distanced_points


# Sort in ascending order and get the k closest points
def k_nearest(distanced_points, k):
    sorted_set = sorted(distanced_points, key=itemgetter(3))
    neighbors = sorted_set[0: k]
    return neighbors


# Neighbours vote and assign prediction
def vote_verdict(neighbors):
    ok_vote = 0
    fail_vote = 0

    for n in neighbors:

        if n[2] == 1.0:  # Compares the labels of neighbours
            ok_vote += 1
        else:
            fail_vote += 1

    return 1.0 if ok_vote > fail_vote else 0.0


# Gets the test point's k nearest neighbours to predict it's outcome
def prediction(training_set, test_point, k):
    distanced_points = distance_measurement(training_set, test_point)

    neighbours = k_nearest(distanced_points, k)
    return vote_verdict(neighbours)


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


# Opens the csv file and plots the original training data
def plot_dataset(file_path):

    training_data = open_file(file_path)

    seperated_data = seperate_data(training_data)

    # Plot the data
    plt.figure(figsize=(5, 4))
    plt.scatter(seperated_data[0], seperated_data[1], c=seperated_data[2],
                cmap='coolwarm', edgecolors='k', alpha=0.8)
    plt.title('Original Dataset')
    plt.xlabel('Property 1')
    plt.ylabel('Property 2')
    plt.show()


# Predicts the outcome of test points based on K_nearest neighbours
def point_prediction(training_data, test_data, k_values):

    # Perform predictions for each value of k
    for k in k_values:
        print('\nWhen K = {}'.format(k))
        count = 1
        for test_point in test_data:

            vote = prediction(training_data, test_point, k)

            match vote:
                case 1.0:
                    print('Test point {} ===> OK'.format(count))

                case 0.0:
                    print('Test point {} ===> Fail'.format(count))

            count += 1


# Plots a 2X2 grid for the decision boundaries for each K value
def decision_boundary(training_data, k_values):

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
file_path = 'Datasets/Dataset_1.csv'

# Plot the dataset
plot_dataset(file_path)

# Store dataset in Array
training_data = open_file(file_path)

# Define the test points
test_data = np.array([
    [-0.3, 1.0],
    [-0.5, -0.1],
    [0.6, 0.0]
])

# Define different values of k
k_values = [1, 3, 5, 7]

# Predict the out come of test labels using different K values
point_prediction(training_data, test_data, k_values)

# Show decision boundary
decision_boundary(training_data, k_values)
