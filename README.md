# K Nearest Neighbours Classification and Regression
The two files take in datasets and perform KNN classification and Regression on them without using external KNN libraries.

# Description

## Dataset_1.py
This program shows a KNN classification on a given dataset:

Takes in Dataset_1.csv which contains 118 rows of data representing 2 properties of some device X and an integer value that represents wheteher the device functioned as expected (1.0 - OK)or if the device failed (0.0 - Fail) and visualizes this on a graph.

Three devices whose functional status is unkown are introduced and the objective it to classify them as either OK or Fail according to the K values [1, 3, 5, 7]

Visualise the decision boundary and the training error for the defined K values [1, 3, 5, 7]

## Dataset_2.py
This program shows a KNN Regression on a given dataset:

The dataset is equally split into training and test set and plots them side by side in 1X2 plot.
The program them calculates the Mean Squared Error for the training error for K values [1, 3, 5, 7, 9, 11] and visualizes the KNN regression result.
The program them calculates the Mean Squared Error for the test error for K values [1, 3, 5, 7, 9, 11] and visualizes the KNN regression result.

# Installation guide
The program uses 3 pyhton libraries specified in requirements.txt run pip3 install -r requirements.txt to install them. Run the files indidually.

