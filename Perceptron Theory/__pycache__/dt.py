import os  
import numpy as np  
import pandas as pd  
import sys


def partition(a):
    np.set_printoptions(threshold=np.nan)

path1 = str(sys.argv[1])   
data1 = pd.read_csv(path1, header=None, names=['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date',
])  
path2 = str(sys.argv[2])   
data2 = pd.read_csv(path2, header=None, names=['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date',
]) 

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mutual_information(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

from pprint import pprint



def recursive_split(x, y):
    # If there could be no split, just return the original set
    

    # We get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y


    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%d = %d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res



cols1 = data1.shape[1]  
X1 = data1.iloc[1:,3:cols1] 
y1 = data1.iloc[1:,2:3]

cols2 = data2.shape[1]  
X2 = data2.iloc[1:,2:cols2]  


X1 = np.matrix(X1.values)  
y1 = np.matrix(y1.values) 

X1[X1 == ''] = 0.0
X1 = X1.astype(np.float)
y1[y1 == ''] = 0.0
y1 = y1.astype(np.float)

X2 = np.matrix(X2.values)  
 
X2[X2 == ''] = 0.0
X2 = X2.astype(np.float)


sys.stdout = open('output.txt','wt')
pprint(recursive_split(X1, y1))   


