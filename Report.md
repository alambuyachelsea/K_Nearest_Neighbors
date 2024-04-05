# machine-learning-a1

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
First I Converted the MNIST dataset into a CSV file which i could work with. I loaded the train
