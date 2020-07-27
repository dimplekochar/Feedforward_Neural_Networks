import os  
import numpy as np  
import pandas as pd  
import sys

np.set_printoptions(threshold=np.nan)

path1 = str(sys.argv[1])   
data1 = pd.read_csv(path1, header=None, names=['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date',
])  
path2 = str(sys.argv[2])   
data2 = pd.read_csv(path2, header=None, names=['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date',
]) 

def prt(a):
    return {b: (a==b).nonzero()[0] for b in np.unique(a)}
def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mi(y, x):
    res = entropy(y)

    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res
from pprint import pprint

def leaf(s):
    return len(set(s)) == 1

def branch(x, y, depth):
    
    if leaf(y) or len(y) == 0:
        return y

    g = np.array([mi(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(g)

    if np.all(g < 1e-4):
        return y

    sets = prt(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        if (depth==0):
            break
        res["x_%d = %d" % (selected_attr, k)] = branch(x_subset, y_subset, depth-1)

    return res

cols1 = data1.shape[1]  
X11 = data1.iloc[1:,2:3] 
X12 = data1.iloc[1:,3:4] 
X13 = data1.iloc[1:,4:5] 
X14 = data1.iloc[1:,5:6] 
X15 = data1.iloc[1:,6:7] 
X16 = data1.iloc[1:,7:8] 
X17 = data1.iloc[1:,8:9] 
X18 = data1.iloc[1:,9:10] 
X19 = data1.iloc[1:,10:11] 
X110 = data1.iloc[1:,11:12] 
X111 = data1.iloc[1:,12:13] 
X112 = data1.iloc[1:,13:14] 
X113 = data1.iloc[1:,14:15] 
X114 = data1.iloc[1:,15:16] 
X115 = data1.iloc[1:,16:17] 
X116 = data1.iloc[1:,17:18] 
X117 = data1.iloc[1:,18:19] 

y1 = data1.iloc[1:,1:2] 

X11 = np.matrix(X11.values)  
X12 = np.matrix(X12.values)  
X13 = np.matrix(X13.values)  
X14 = np.matrix(X14.values)  
X15 = np.matrix(X15.values)  
X16 = np.matrix(X16.values)  
X17 = np.matrix(X17.values)  
X18 = np.matrix(X18.values)  
X19 = np.matrix(X19.values)  
X110 = np.matrix(X110.values)  
X111 = np.matrix(X111.values)  
X112 = np.matrix(X112.values)  
X113 = np.matrix(X113.values)  
X114 = np.matrix(X114.values)  
X115 = np.matrix(X115.values)  
X116 = np.matrix(X116.values)  
X117 = np.matrix(X117.values)  

y1 = np.matrix(y1.values)  

X11 = X11.astype(np.float)
X12 = X12.astype(np.float)
X13 = X13.astype(np.float) 
X14 = X14.astype(np.float)
X15 = X15.astype(np.float)
X16 = X16.astype(np.float) 
X17 = X17.astype(np.float)
X18 = X18.astype(np.float)
X19 = X19.astype(np.float) 
X110 = X110.astype(np.float)
X111 = X111.astype(np.float)
X112 = X112.astype(np.float) 
X113 = X113.astype(np.float)
X114 = X114.astype(np.float)
X115 = X115.astype(np.float) 
X116 = X116.astype(np.float)
X117 = X117.astype(np.float)

y1 = y1.astype(np.int)

X11 = np.array(X11.T)[0]
X12 = np.array(X12.T)[0]
X13 = np.array(X13.T)[0]
X14 = np.array(X14.T)[0]
X15 = np.array(X15.T)[0]
X16 = np.array(X16.T)[0]
X17 = np.array(X17.T)[0]
X18 = np.array(X18.T)[0]
X19 = np.array(X19.T)[0]
X110 = np.array(X110.T)[0]
X111 = np.array(X111.T)[0]
X112 = np.array(X112.T)[0]
X113 = np.array(X113.T)[0]
X114 = np.array(X114.T)[0]
X115 = np.array(X115.T)[0]
X116 = np.array(X116.T)[0]
X117 = np.array(X117.T)[0]

cols1 = data1.shape[1]  
X21 = data1.iloc[1:,1:2] 
X22 = data1.iloc[1:,2:3] 
X23 = data1.iloc[1:,3:4] 
X24 = data1.iloc[1:,4:5] 
X25 = data1.iloc[1:,5:6] 
X26 = data1.iloc[1:,6:7] 
X27 = data1.iloc[1:,7:8] 
X28 = data1.iloc[1:,8:9] 
X29 = data1.iloc[1:,9:10] 
X210 = data1.iloc[1:,10:11] 
X211 = data1.iloc[1:,11:12] 
X212 = data1.iloc[1:,12:13] 
X213 = data1.iloc[1:,13:14] 
X214 = data1.iloc[1:,14:15] 
X215 = data1.iloc[1:,15:16] 
X216 = data1.iloc[1:,16:17] 
X217 = data1.iloc[1:,17:18] 

X21 = np.matrix(X21.values)  
X22 = np.matrix(X22.values)  
X23 = np.matrix(X23.values)  
X24 = np.matrix(X24.values)  
X25 = np.matrix(X25.values)  
X26 = np.matrix(X26.values)  
X27 = np.matrix(X27.values)  
X28 = np.matrix(X28.values)  
X29 = np.matrix(X29.values)  
X210 = np.matrix(X210.values)  
X211 = np.matrix(X211.values)  
X212 = np.matrix(X212.values)  
X213 = np.matrix(X213.values)  
X214 = np.matrix(X214.values)  
X215 = np.matrix(X215.values)  
X216 = np.matrix(X216.values)  
X217 = np.matrix(X217.values)  

X21 = X21.astype(np.float)
X22 = X22.astype(np.float)
X23 = X23.astype(np.float) 
X24 = X24.astype(np.float)
X25 = X25.astype(np.float)
X26 = X26.astype(np.float) 
X27 = X27.astype(np.float)
X28 = X28.astype(np.float)
X29 = X29.astype(np.float) 
X210 = X210.astype(np.float)
X211 = X211.astype(np.float)
X212 = X212.astype(np.float) 
X213 = X213.astype(np.float)
X214 = X214.astype(np.float)
X215 = X215.astype(np.float) 
X216 = X216.astype(np.float)
X217 = X217.astype(np.float)

X21 = np.array(X21.T)[0]
X22 = np.array(X22.T)[0]
X23 = np.array(X23.T)[0]
X24 = np.array(X24.T)[0]
X25 = np.array(X25.T)[0]
X26 = np.array(X26.T)[0]
X27 = np.array(X27.T)[0]
X28 = np.array(X28.T)[0]
X29 = np.array(X29.T)[0]
X210 = np.array(X210.T)[0]
X211 = np.array(X211.T)[0]
X212 = np.array(X212.T)[0]
X213 = np.array(X213.T)[0]
X214 = np.array(X214.T)[0]
X215 = np.array(X215.T)[0]
X216 = np.array(X216.T)[0]
X217 = np.array(X217.T)[0]

X2 = np.array([X21, X22, X23, X24, X25, X26, X27, X28, X29, X210, X211, X212, X213, X214, X215, X216, X217]).T

y1 = np.array(y1.T)[0]
depth = float(sys.argv[3])  
if depth==-1:
	depth=17
X1 = np.array([X11, X12, X13, X14, X15, X16, X17, X18, X19, X110, X111, X112, X113, X114, X115, X116, X117]).T

sys.stdout = open('output.txt','wt')
pprint(branch(X1, y1, depth))

