# Code: Single Layer Perceptron to classify 2 Synthetically Generated Linearly Separable Classes
# Problem: Neural Networks Assignment 1 - Problem 4
# Author: Sohan Ghosh
# Date: 26/09/2020

# Import libraries
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pandas import DataFrame

# Function to multiply two matrices
def matrixMultiply(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(len(A)):  
        for j in range(len(B[0])): 
            for k in range(len(B)): 
                C[i][j] += A[i][k] * B[k][j]
    return C[0][0]

# Function to train the perceptron
def trainPerceptron(Xtrain, Ytrain, Eta):
    misclassifications = 1
    iters = 0
    max_iter = 1000

    inp_nodes = Xtrain.shape[1] # no of input nodes
    out_nodes = 1 # no of output nodes

    W = np.random.rand(inp_nodes, out_nodes)/10    # Initialize weights to small random values

    # Convergences when all points are correctly classified, if not exit after iterating max_iter times
    while misclassifications > 0 and iters < max_iter:
        iters += 1
        misclassifications = 0
        for i in range(Xtrain.shape[0]):
            if matrixMultiply(W.T, Xtrain[i].reshape((inp_nodes, 1))) >= 0:
                y = 1
            else:
                y = -1
        
            W = W + Eta * (Ytrain[i] - y) * Xtrain[i].reshape((inp_nodes, 1))   # Update weights rule
        
            if y != Ytrain[i]:
                misclassifications += 1
            
        print(f'---------- Iteration No: {iters} ----------' )
        print(f'Errors: {misclassifications}')
        print(f'Training Accuracy: {((Xtrain.shape[0]-misclassifications)/Xtrain.shape[0]) * 100}%')
        print()
    return W

# Function to test the perceptron
def testPerceptron(Xtest, Ytest, W):
    misclassifications = 0

    inp_nodes = Xtest.shape[1] # no of input nodes

    for i in range(Xtest.shape[0]):
        if matrixMultiply(W.T, Xtest[i].reshape((inp_nodes, 1))) >= 0:
            y = 1
        else:
            y = -1
    
        if y != Ytest[i]:
            misclassifications += 1

    print(f'Test Accuracy = {((Xtest.shape[0]-misclassifications)/Xtest.shape[0]) * 100}%')


if __name__ == "__main__": 

    # Set parameters for creating synthetic data
    N = 2000 # total number of points
    K = 2 # no of classes
    n = 2 # no of features

    # Create synthetic data
    X, Y = make_blobs(n_samples=N, centers=K, n_features=n, random_state=22)

    # Convert labels to -1 and 1 (labels 1 are already present)
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1

    one = np.ones((N,1))

    # Append x0 = 1 to the dataset to be used as bias
    X = np.concatenate((one,X),axis=1)

    c = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    np.random.shuffle(c)    # Shuuffle data

    X = c[:, :X.size//len(X)].reshape(X.shape) # Shuffled data
    Y = c[:, X.size//len(X):].reshape(Y.shape)  # Shuffled labels

    # Use 70% of the data as train set and the rest as test set
    t = int(0.7 * N) + 1
    X_train = X[:t,]
    X_test = X[t:,]
    Y_train = Y[:t]
    Y_test = Y[t:]

    eta = 0.001 #learning rate

    # Train the perceptron model to get a trained set of weights
    Wt = trainPerceptron(X_train, Y_train, eta)

    # Test the perceptron model with the trained set of weights
    testPerceptron(X_test, Y_test, Wt)

    print()

