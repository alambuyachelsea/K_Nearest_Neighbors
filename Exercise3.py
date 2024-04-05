import numpy as np
import time


def distance(train_point, test_pont):
    return np.linalg.norm(train_point - train_point)


def knn_model(X, Y, point, k):
    values = []
    m = X.shape[0]

    for i in range(m):
        dist = distance(point, X[i])
        values.append((dist, Y[i]))

    values = sorted(values)
    neighbors = values[:k]
    neighbors = np.array(neighbors)

    new_values = np.unique(neighbors[:, 1], return_counts=True)

    index = new_values[1].argmax()
    prediction = new_values[0][index]

    return prediction


def error_counter(X, Y, x_test, y_test, k):
    counter = 0
    index = 0

    for x in x_test:
        prediction = knn_model(X, Y, x, k)
        if prediction != y_test[index]:
            counter += 1
        index += 1

    return counter


def load_csv(path):
    return np.loadtxt(path, delimiter=',', dtype="float64")


# Files containing the data
train_file_path = 'A1_datasets/mnist_train.csv'
test_file_path = 'A1_datasets/mnist_test.csv'

# Load test data
training_data = load_csv(train_file_path)

X = training_data[:, 1:]
Y = training_data[:, 0]

# split training data to train and test
split = int(0.8*X.shape[0])
x_train = X[:split, :]
y_train = Y[:split]

# Training test data
x_train_test = X[split:, :]
y_train_test = Y[split:]

# Define different values of k
k_values = [1, 3, 5, 7]

# For training Error
for k in k_values:
    start = time.time()
    error = error_counter(x_train, y_train, x_train_test, y_train_test, k)
    end = time.time()
    print(f'When k = {k} the errors are {error}')
    print(end - start)

test_data = load_csv(test_file_path)

test_x = test_data[:, 1:]
test_y = test_data[:, 0]

# For Test Error
for k in k_values:
    start = time.time()
    error = error_counter(x_train, y_train, test_x, test_y, k)
    end = time.time()
    print(f'When k = {k} the errors are {error}')
    print(end - start)

# Accuracy will be based o the diff between test and train error
