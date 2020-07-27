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


def gradientDescent(X, y, w, eta, iterns):  
    temp = np.matrix(np.zeros(w.shape))
    parameters = int(w.ravel().shape[1])
 
    for i in range(iterns):
        err = (X * w.T) - y
        
        for j in range(parameters):
            term = np.multiply(err, X[:,j])
            temp[0,j] = w[0,j] - ((eta / len(X)) * np.sum(term))

        w = temp
        
    return w

data1.insert(0, 'Ones', 1)
data2.insert(0, 'Ones', 1)

cols1 = data1.shape[1]  
X1 = data1.iloc[1:,3:cols1]
X1.insert(0, 'Ones', 1)  
y1 = data1.iloc[1:,2:3]

cols2 = data2.shape[1]  
X2 = data2.iloc[1:,2:cols2]  
X2.insert(0, 'Ones', 1) 

X1 = np.matrix(X1.values)  
y1 = np.matrix(y1.values) 

X1[X1 == ''] = 0.0
X1 = X1.astype(np.float)
y1[y1 == ''] = 0.0
y1 = y1.astype(np.float)

X2 = np.matrix(X2.values)  
 
X2[X2 == ''] = 0.0
X2 = X2.astype(np.float)

w = np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



X1=np.append(X1, np.sqrt(np.absolute(X1[0:,1:20])), axis=1)
X1=np.append(X1, np.square(X1[0:,1:20]), axis=1)
#X1=np.append(X1, 1/(X1[0:,1:20]), axis=1)
row_meansx1 = np.mean(X1, axis=0)
X1=X1/row_meansx1
X1=X1/10

X2=np.append(X2, np.sqrt(np.absolute(X2[0:,1:20])), axis=1)
X2=np.append(X2, np.square(X2[0:,1:20]), axis=1)

X2=X2/row_meansx1
X2=X2/10

X1[1:,1:2]=0
X1[1:,6:9]=0
X1[1:,10:11]=0
X1[1:,12:17]=0
X1[1:,19:20]=0
X1[1:,20:21]=0
X1[1:,25:28]=0
X1[1:,29:30]=0
X1[1:,31:36]=0
X1[1:,38:39]=0
X1[1:,39:40]=0
X1[1:,44:47]=0
X1[1:,48:49]=0
X1[1:,50:55]=0
X2[1:,1:2]=0
X2[1:,6:9]=0
X2[1:,10:11]=0
X2[1:,12:17]=0
X2[1:,19:20]=0
X2[1:,20:21]=0
X2[1:,25:28]=0
X2[1:,29:30]=0
X2[1:,31:36]=0
X2[1:,38:39]=0
X2[1:,39:40]=0
X2[1:,44:47]=0
X2[1:,48:49]=0
X2[1:,50:55]=0

eta = float(sys.argv[3])  
iterns=10000
g = gradientDescent(X1, y1, w, eta, iterns)

sys.stdout = open('output.txt','wt')
print(X1*g.T)   

sys.stdout = open('outputt.txt','wt')
print(X2*g.T) 


df = pd.DataFrame(X2*g.T)
df.to_csv("output.csv")



