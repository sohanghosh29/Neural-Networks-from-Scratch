# Code: Single Layer Perceptron to classify 2  Linearly Separable Classes
# Dataset: BankNote Authentication Dataset (https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data)
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
    return C[0][0]


# Function to train the perceptron
def trainPerceptron(Xtrain, Ytrain, Eta):
    misclassifications = 1
    iters = 0
    max_iter = 200

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


# Update the class labels of the dataset
def chnge_negone(x):
    if x == 0:
        return -1
    else:
        return 1


if __name__ == "__main__": 

    # Load dataset
    data = pd.read_csv("BankNote.csv")
    data = data.sample(frac = 1).reset_index(drop=True)
    data['class'] = data['class'].apply(chnge_negone)

    X = data[['variance', 'skewness', 'curtosis', 'entropy']].copy()
    Y = data[['class']].copy()
    X = X.to_numpy()
    Y = Y.to_numpy()

    N = X.shape[0] # total number of points
    K = 2 # no of classes
    n = X.shape[1] # no of features

    one = np.ones((N,1))

    # Append x0 = 1 to the dataset to be used as bias
    X = np.concatenate((one,X),axis=1)

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

    eta = 0.01 #learning rate

    # Train the perceptron model to get a trained set of weights
    Wt = trainPerceptron(X_train, Y_train, eta)

    # Test the perceptron model with the trained set of weights
    testPerceptron(X_test, Y_test, Wt)

    print()
