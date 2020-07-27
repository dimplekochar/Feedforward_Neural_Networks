import os  
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

np.set_printoptions(threshold=np.nan)
THRESH1=9

def importdata():

    path1 = str(sys.argv[1])   
    data1 = pd.read_csv(path1, header=None, names=['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date',])  
    path2 = str(sys.argv[2])   
    data2 = pd.read_csv(path2, header=None, names=['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date',]) 
     
    return data1, data2
 
def mkdata(data1, data2):

    X_train = data1.values[1:, 2:18]
    y_train = data1.values[1:, 1:2]
    X_test = data2.values[1:, 1:17]
     
    return X_train, X_test, y_train
     
def trbyen(X_train, X_test, y_train, depth):
 
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = depth, min_samples_leaf = THRESH1)
    clf_entropy.fit(X_train, y_train)

    return clf_entropy

def prediction(X_test, clf_object):
 
    y_pred = clf_object.predict(X_test)  
    sys.stdout = open('output.txt','wt')
    print(y_pred)
    df = pd.DataFrame(y_pred)
    df.to_csv("output.csv")
    
    return y_pred
     
def main():
     
    data1, data2 = importdata()
    X_train, X_test, y_train = mkdata(data1, data2)
    clf_entropy = trbyen(X_train, X_test, y_train, depth)
    y_pred_entropy = prediction(X_test, clf_entropy)
     
if __name__=="__main__":
    depth = int(sys.argv[3]) 
    if depth == -1:
    	depth=17

    main()