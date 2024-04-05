# machine-learning-a1

First load the requirements packages using pip install


## Exercise 2: K-NN Regression
The MSE training error
K1 = 0
K3 = 18.05
K5 = 22.25
K7 = 23.56
K9 = 24.41
K11 = 26.41

The MSE test error
K1 = 49.24
K3 = 31.58
K5 = 28.48
K7 = 29.23
K9 = 27.7 *
K11 = 30.36

Which is the best K between [1, 3, 5, 7, 9, 11]

I first disqaulify K1 because this creates an overfitted model where the training error is very small but the test error is the largest out of all the K values. I then disqualify K11 beacuse this creates an underfitted model where the both the training and test error are larger.

Amongst the remaining K values [3, 5, 7, 9], K9 has the lowest test error so I think this the K value that gives the best regression for the provided data set.


## Exercise 3: MNIST data set
First I Converted 10,000 train data and 1,000 test MNIST dataset into CSV files which I could work with. I loaded the train data set and split it into [x_train, y_train] which is 80% of the dataset and [x_train_test, y_train_test] which is the remaining 20% of the dataset.

I then used the KNN model to determine the number of errors, the accuracy for each K value and the time it takes to compute it. For the training data set the time taken is about (80 - 90) seconds to compute.

Then I load the test MNIST dataset and use the KNN model to determine the number of errors, the accuracy for each K value and the time it takes to compute it. For the test data set the time taken is about (40 - 50) seconds to compute.

For the full data set, first run the MNIST converte to convert the full dataset and run the Exercise3_Full.py The implementation is the same but due to the much larger dataset the training process time per K value is aproximately 120 minutes and the test processing time is 45 minutes per K value

I think the most suitable K value is K5. It gives a lower test error and higher accuracy than K1 and K3. Since it gives the same error as K7 when we compare the training error we can see that K5 gives a lower error than K7.

# Additional info
Supporting images for exercise 1 and 2 have been included in the images folder to showcase expected results