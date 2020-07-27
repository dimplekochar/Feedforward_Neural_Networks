# Regression
Implemented gradient descent using Python. Code accepts three arguments: train.csv, test.csv and a real-valued number for eta (learning rate) It uses train a linear predictor using all the features listed in train.csv. 
python3 lr.py train.csv test.csv 0.001
is the shell command.

Considering only one feature sqft_lot, we run the linear regression algorithm Then, evaluate on the instances in test.csv. The mean absolute error on these instances can be found in report.pdf. 

Creating subsets of the training data containing the first 2000, 5000, 7000, 10000 instances in train.csv, created a plot where the X-axis is the size of the training set (2000, 5000, 7000, 10000, full) and the Y-axis is the mean absolute error on the test.csv instances using only the sqft_lot feature. 
