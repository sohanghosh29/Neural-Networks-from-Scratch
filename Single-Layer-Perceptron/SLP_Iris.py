# Code: Single Layer Perceptron to classify more than 3 Pairwise Linearly Separable Classes
# Dataset: Iris Dataset (https://www.kaggle.com/arshid/iris-flower-dataset)
# Author: Sohan Ghosh
# Date: 26/09/2020

# Import libraries
import random
import pandas as pd
import numpy as np
from pandas import DataFrame

# Function to multiply two matrices
def matrixMultiply(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(len(A)):  
        for j in range(len(B[0])): 
            for k in range(len(B)): 
                C[i][j] += A[i][k] * B[k][j]
    return C

# Function to train the perceptron
def trainPerceptron(Xtrain, Ytrain, K, Eta):
    misclassifications = 1
    iters = 0
    max_iter = 200

    inp_nodes = Xtrain.shape[1] # no of input nodes
    out_nodes = K # no of output nodes

    W = np.random.rand(inp_nodes, out_nodes)/10    # Initialize weights to small random values

    # Convergences when all points are correctly classified, if not exit after iterating max_iter times
    while misclassifications > 0 and iters < max_iter:
        iters += 1
        misclassifications = 0
        for i in range(Xtrain.shape[0]):
            v = matrixMultiply(W.T, X_train[i].reshape((inp_nodes, 1)))
            maxval = v[0,0]
            maxind = 0
            for j in range(1, K):
                if v[j,0] > maxval:
                    maxval = v[j,0]
                    maxind = j
         
            # Predicted output
            y = np.zeros((K, 1))
            for j in range(K):
                if j == maxind:
                    y[j,0] = 1  # Make the highest yj = 1, rest 0
                else:
                    y[j,0] = 0
        
            # Desired output
            d = np.zeros((K, 1))
            d[int(Y_train[i]),0] = 1
        
            # The predicted and desired outputs do not match
            if (y==d).all() == False:
                misclassifications += 1
                update = eta * X_train[i].reshape((inp_nodes, 1)) # change in weights
                for j in range(K):
                    if j == int(Y_train[i]):
                        for k in range(inp_nodes):
                            W[k,j] += update[k] # Update weights
                    else:
                        for k in range(inp_nodes):
                            W[k,j] -= update[k] # Update weights
            
        print(f'---------- Iteration No: {iters} ----------' )
        print(f'Misclassifications: {misclassifications}')
        print(f'Training Accuracy: {((Xtrain.shape[0]-misclassifications)/Xtrain.shape[0]) * 100}%')
        print()
    return W


# Function to test the perceptron
def testPerceptron(Xtest, Ytest, W):
    misclassifications = 0

    inp_nodes = Xtest.shape[1] # no of input nodes

    for i in range(Xtest.shape[0]):
        v = matrixMultiply(W.T, X_test[i].reshape((inp_nodes, 1)))
        maxval = v[0,0]
        maxind = 0
        for j in range(1, K):
            if v[j,0] > maxval:
                maxval = v[j,0]
                maxind = j
    
        if maxind != int(Y_test[i]):
            misclassifications += 1

    print(f'Test Accuracy = {((Xtest.shape[0]-misclassifications)/Xtest.shape[0]) * 100}%')


# Update the class labels of the dataset
def changeSpecies(x):
    if x == 'Iris-versicolor':
        return 0
    elif x == 'Iris-setosa':
        return 1
    else:
        return 2


if __name__ == "__main__": 

    # Load dataset
    data = pd.read_csv("Iris.csv")
    data = data.sample(frac = 1).reset_index(drop=True)
    data['Species'] = data['Species'].apply(changeSpecies)

    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].copy()
    Y = data[['Species']].copy()
    X = X.to_numpy()
    Y = Y.to_numpy()

    N = X.shape[0] # total number of points
    K = 3 # no of classes
    n = X.shape[1] #no of features

    one = np.ones((N,1))

    # Append x0 = 1 to the dataset to be used as bias
    X = np.concatenate((one, X),axis=1)

    c = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    np.random.shuffle(c)    # Shuuffle data

    X = c[:, :X.size//len(X)].reshape(X.shape) # Shuffled data
    Y = c[:, X.size//len(X):].reshape(Y.shape)  # Shuffled labels

    y = []
    for i in range(len(Y)):
        y.append(Y[i,0])
    Y = np.asarray(y)

    # Use 70% of the data as train set and the rest as test set
    t = int(0.7 * N) + 1
    X_train = X[:t,]
    X_test = X[t:,]
    Y_train = Y[:t]
    Y_test = Y[t:]

    eta = 0.001 #learning rate

    # Train the perceptron model to get a trained set of weights
    Wt = trainPerceptron(X_train, Y_train, K, eta)

    # Test the perceptron model with the trained set of weights
    testPerceptron(X_test, Y_test, Wt)

    print()
