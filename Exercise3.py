import numpy as np
import time


# Calculates the distance between two points
def distance(train_point, test_point):
    return np.linalg.norm(train_point - test_point)


# Pedicts the label of a point from k neighbours
def knn_model(X, Y, point, k):
    values = []
    m = X.shape[0]

    # Calculates the distance between a oint in X and the test point
    for i in range(m):
        dist = distance(point, X[i])
        values.append((dist, Y[i]))

    # Sorts the list and returbs the neighbours
    values = sorted(values)
    neighbors = values[:k]
    neighbors = np.array(neighbors)

    new_values = np.unique(neighbors[:, 1], return_counts=True)

    index = new_values[1].argmax()
    prediction = new_values[0][index]

    return prediction


# Returns the number of errors for a given K value
def error_counter(X, Y, x_test, y_test, k):
    counter = 0
    index = 0

    for x in x_test:
        prediction = knn_model(X, Y, x, k)
        if prediction != y_test[index]:  # Increases the error counter
            counter += 1
        index += 1

    return counter


# Loads the CSV file and returns an array list
def load_csv(path):
    return np.loadtxt(path, delimiter=',', dtype="float64")


# Starts classification test data with training set
def start_classifying(X, Y, test_x, test_y, k_values):

    for k in k_values:
        start = time.time()
        error = error_counter(X, Y, test_x, test_y, k)
        end = time.time()

        # Accuracy calculation
        acc = 1 - np.round(error / x_train.shape[0], 3)

        print(f'When k = {k}')
        print(f'Errors = {error} and Accuracy = {acc}')
        print(f'Time taken = {(np.round(end - start))} seconds')


# Files containing the data
train_file_path = 'A1_datasets/mnist_train.csv'
test_file_path = 'A1_datasets/mnist_test.csv'

# Load test data
training_data = load_csv(train_file_path)

X = training_data[:, 1:]
Y = training_data[:, 0]

# Splits the training data to train and test
split = int(0.8 * X.shape[0])
x_train = X[:split, :]
y_train = Y[:split]

# Training test data
x_train_test = X[split:, :]
y_train_test = Y[split:]

# Define different values of k
k_values = [1, 3, 5, 7]

# For training Error
print('For the training error')
start_classifying(x_train, y_train, x_train_test, y_train_test, k_values)

# Loads the test data set
test_data = load_csv(test_file_path)

test_x = test_data[:, 1:]
test_y = test_data[:, 0]

# For Test Error
print('For the test error')
start_classifying(x_train, y_train, test_x, test_y, k_values)
