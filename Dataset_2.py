import csv
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


# opens the file and retrives the data
def open_file(file_path):

    dataset = []  # Empty list to store values

    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            line = []

            for value in row:  # info stored as string converted to float

                x = float(value)
                line.append(x)

            dataset.append(line)

    return dataset


# Retrus the k nearest neighbors
def k_nearest(distanced_points, k):

    # Sorts the points by distance and returns an index of the list
    sorted_set = sorted(distanced_points, key=itemgetter(2))
    return sorted_set[0: k]


# Predicts what the y value of a point would be
def predict_y(point, training_set, k):
    # Gets the distance between the point all others in training set
    distanced_points = distance_measurement(training_set, point)

    # Finds the neighbors
    neighbours = k_nearest(distanced_points, k)

    # Gets mean which will be the y value
    return y_verdict(neighbours, k)


# Calculates the distance between the point being tested
# and every other point in the training set
def distance_measurement(training_set, test_case):
    # POint + distance tuple
    distanced_points = []

    # distance calculation
    for train_point in training_set:

        dist = np.round(np.sqrt((train_point[0] - test_case[0])**2), 4)

        # Creates a point with the distance to be sorted and adds it to list
        distanced_point = [train_point[0], train_point[1], dist]
        distanced_points.append(distanced_point)

    return distanced_points


# Calculates the average y values of the neighbours to
# determine the y value to assign
def y_verdict(neighbors, k):

    average = 0

    for item in neighbors:
        average += item[1]

    return average / k


# Calculates the MSE between a given data set and a prediction set
def calculate_MSE(data_set, prediction):

    # Calculates the squared error sum
    squaredErrorSum = sum(np.square(data_set[i][1] - prediction[i][1])
                          for i in range(len(data_set)))

    # Returns the rounded mean
    return round(squaredErrorSum/len(data_set), 2)


# Plots the training and test data sets in 1x2
def plot_training_test(training_set, test_set):

    # Convert to arrays for plotting
    training_set = np.array(training_set)
    test_set = np.array(test_set)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    count = 0  # To determine subplot title
    data = [training_set, test_set]  # Put into list to be looped over

    for item in data:  # First plots training then test data

        axs[count].scatter(item[:, 0], item[:, 1], s=20)
        title = "Training Data" if count == 0 else "Test Data"

        axs[count].set_title(title)
        count = 1

    plt.show()


# Plots KNN regression result and the MSE error for selected k values
def plot_error(dataset, k_values):

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Creates a subplot for every k value
    for i, k in enumerate(k_values):

        ax = axs[i//3, i % 3]  # Specifies which subplot to plot on

        prediction = []
        # make predictions for the training set

        for point in dataset:

            # Use the trainig data to test the training set
            predicted_y = predict_y(point, dataset, k)
            prediction.append([point[0], predicted_y])

        # Sort lists for ploting
        dataset = sorted(dataset, key=itemgetter(0))
        prediction = sorted(prediction, key=itemgetter(0))

        # Calculate the MSE for the training set
        MSE = calculate_MSE(dataset, prediction)

        # Set as array to be plotted
        prediction = np.array(prediction)
        dataset = np.array(dataset)

        # Plot the training set and the points with the predicted y value
        ax.plot(prediction[:, 0], prediction[:, 1], c="Red")
        ax.scatter(dataset[:, 0], dataset[:, 1],
                   edgecolors='k', s=20)

        ax.set_title(f'Training Data MSE = {MSE} when K = {k}')

    plt.tight_layout()
    plt.show()

    return None


# Prints the MSE test error for selected k values
def display_test_set_MSE(test_set, training_set, k):

    prediction = []

    # make predictions for the test set
    for point in test_set:

        # Use training data to test the test set
        predicted_y = predict_y(point, training_set, k)
        prediction.append([point[0], predicted_y])

    # Calculate the MSE and print result
    MSE = calculate_MSE(test_set, prediction)

    print(f'When K = {k} Test Data MSE = {MSE}')


# File containing the data
file_path = 'Datasets/Dataset_2.csv'

# Part 1: Split the givien data
dataset = open_file(file_path)
training_set = dataset[0:100]
test_set = dataset[100:]

# Part 2: Plot training and test data sets in 1x2
plot_training_test(training_set, test_set)

# Part 3: Plot K-NN regression and MSE Training error
k_values = [1, 3, 5, 7, 9, 11]
plot_error(training_set, k_values)

# Part 4: Plot K-NN regression and MSE Test error
plot_error(test_set, k_values)
