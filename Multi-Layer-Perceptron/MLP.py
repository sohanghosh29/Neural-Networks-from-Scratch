# Code: Multi-Layer Perceptron (MLP) to Classify Iris and Digits Dataset
# Dataset: 1. Iris Dataset (https://www.kaggle.com/arshid/iris-flower-dataset)
#          2. Handwritten Digits Dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
# Author: Sohan Ghosh
# Date: 12/10/2020

# Import libraries
import random
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.datasets import load_digits


# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid function
def derSigmoid(x):
    return x * (1 - x)

# Update the class labels of the dataset
def changeSpecies(x):
    if x == 'Iris-versicolor':
        return 0
    elif x == 'Iris-setosa':
        return 1
    else:
        return 2


# Check if prediction for a particular data point is correct or not
def checkCorrect(D, Y):
    dt = 0
    yt = 0
    for i in range(D.shape[0]):
        if D[i,0] == 1:
            dt = i
            
    max_y = Y[0,0]
    yt = 0
    for i in range(Y.shape[0]):
        if Y[i,0] > max_y:
            max_y = Y[i,0]
            yt = i
            
    if dt == yt:
        return True
    else:
        return False

# Prepare Iris dataset
def prepareIRISDataset(data):
    K = data['Species'].nunique() # No of classes
    
    data['Species'] = data['Species'].apply(changeSpecies)

    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].copy()
    Y = data[['Species']].copy()
    
    X = X.to_numpy()
    Y = Y.to_numpy()

    N = X.shape[0] # total number of points

    c = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    np.random.shuffle(c)    # Shuuffle data

    X = c[:, :X.size//len(X)].reshape(X.shape) # Shuffled data
    Y = c[:, X.size//len(X):].reshape(Y.shape)  # Shuffled labels

    # Convert Y to one-hot vector
    y = []
    for i in range(len(Y)):
        y.append(Y[i,0])
    Y = np.asarray(y)

    yy = []
    for i in range(N):
        d = [0.0 for k in range(K)]
        d[int(Y[i])] = 1.0
        yy.append(d)
    Y = np.array(yy)

    return X, Y

# Prepare MNIST dataset
def prepareMNISTDataset(X1, Y1):
    X1 = (X1 - X1.mean())/X1.std()
    K = len(np.unique(Y1))
    N = X1.shape[0] # total number of points
    K = 10 # no of classes

    c = np.c_[X1.reshape(len(X1), -1), Y1.reshape(len(Y1), -1)]
    np.random.shuffle(c)    # Shuuffle data

    X1 = c[:, :X1.size//len(X1)].reshape(X1.shape) # Shuffled data
    Y1 = c[:, X1.size//len(X1):].reshape(Y1.shape)  # Shuffled labels

    # Convert Y1 to one-hot vector
    yy = []
    for i in range(N):
        d = [0.0 for k in range(K)]
        d[int(Y1[i])] = 1.0
        yy.append(d)
    Y1 = np.array(yy)

    return X1, Y1

# Split dataset into train and test set
def splitTrainTest(X, Y, t):
    # Use t fraction of the data as train set and the rest as test set
    s = int(t * X.shape[0]) + 1
    X_train = X[:s,]
    X_test = X[s:,]
    Y_train = Y[:s]
    Y_test = Y[s:]
    
    return X_train, X_test, Y_train, Y_test

# Evaluate cots of the model
def evaluateCost(D, Y):
    return (1/2) * np.sum(np.power(D - Y, 2))


# Multi layer perceptron class
class MLP:
    # Initialize MLP
    def __init__(self, L):
        self.l = len(L)
        self.K = 0 # Classes
        
        # Initialize weights, biases and changes in weights and biases
        self.W = [np.random.rand(L[i], L[i+1])/10 for i in range(self.l-1)] # Weights
        self.B = [np.zeros((L[i+1], 1)) for i in range(self.l-1)]   # Biases
        self.DW = [np.zeros((L[i], L[i+1])) for i in range(self.l-1)] # Delta W for previous update
        self.DB = [np.zeros((L[i+1], 1)) for i in range(self.l-1)] # Delta B for previous update
        self.V = [] # Activations
        
        print("MLP Initialized")
        print("---------------------------------------------------------------------------")
        

    # Train MLP   
    def trainMLP(self, Xtrain, Ytrain, eta=0.01, alpha=0.9, max_iter=1000):
        N = Xtrain.shape[0] # No of training data points
        self.K = Ytrain.shape[1]    # No of classes
        
        iters = 0
        
        # Iterate for max_iter times
        while iters < max_iter:
            iters += 1
            correct = 0
            cost = 0
        
            for p in range(N):
                self.V = [Xtrain[p].reshape(Xtrain[p].shape[0],1)]
            
                # Forward propagation
                v = Xtrain[p].reshape(Xtrain[p].shape[0],1)
                for i in range(self.l-1):
                    u = np.matmul(self.W[i].T, v) + self.B[i]
                    v = sigmoid(u)
                    self.V.append(v)
                
            
                d = Ytrain[p].reshape((self.K,1)) #Desired output
            
                # Cost of the model
                cost += evaluateCost(d, v)
            
                # Check if prediction is correct
                if  checkCorrect(d, v) == True:
                    correct += 1
        
        
                # Backpropagation
                E = d - v
                prev_d = 0
                for i in range(1, self.l):
                    if i != 1:
                        E = np.matmul(self.W[-i+1], prev_d)
                
                    slope = derSigmoid(self.V[-i])
                    d_layer = E * slope
                    dW = alpha * self.DW[-i] + eta * np.matmul(self.V[-i-1], d_layer.T) # 1st term is due to the momentum factor
                    self.W[-i] += dW    # Update weight
                    self.B[-i] += eta * np.sum(d_layer) # Update bias
                    self.DW[-i] = dW    # Save delta weight
                    self.DB[-i] = eta * np.sum(d_layer) # Save delta bias
                    
                    prev_d = d_layer
                    

            # Display train loss and accuracy      
            print('[Iter ' + str(iters) + ']: ', end = '')
            print('Loss = ' + str(cost), end= ', ')
            print('Train Accuracy = ' + str((correct/N)*100) + " %")
            
    
    # Test MLP
    def testMLP(self, Xtest, Ytest):
        N = Xtest.shape[0] # No of patterns in the dataset
        l1 = len(self.W) # No of weight matrices
    
        correct = 0
        for p in range(N):
            v = Xtest[p].reshape(Xtest[p].shape[0],1)
        
            # Forward propagation
            for i in range(l1):
                u = np.matmul(self.W[i].T, v) + self.B[i]
                v = sigmoid(u)
                
            
            d = Ytest[p].reshape((self.K,1)) #Desired output
            
        
            # Check if prediction is correct
            if  checkCorrect(d, v) == True:
                correct += 1
        
        # Display test accuracy
        print("---------------------------------------------------------------")
        print('\nTest Accuracy = ' + str((correct/N)*100) + " %")


if __name__ == "__main__":
    data_choice = -1
    while(data_choice != 1 and data_choice!= 2):
        data_choice = int(input('Enter [1] for Iris Dataset or [2] for Digits Dataset:'))
        print(data_choice)

    # Load dataset
    if data_choice == 1:
        iris = pd.read_csv("Iris.csv")
        X, Y = prepareIRISDataset(iris)
    else:
        mnist = load_digits()
        X, Y = prepareMNISTDataset(mnist.data, mnist.target)

    no_hid_layers = int(input('Enter the no of Hidden Layers:'))
    L = []
    for i in range(no_hid_layers):
        L.append(int(input(f'Enter no of nodes in Hidden Layer {i+1}:')))

    
    # Set parameters
    eta = float(input('Enter the learning rate:'))  # Learning rate
    alpha = float(input('Enter momentum factor:'))  # Momentum factor
    epochs = int(input('Enter no of iterations to run:'))   # No of iterations to run
    
    t = 0.7 # Fraction of data to use as train set
    X_train, X_test, Y_train, Y_test = splitTrainTest(X, Y, t)

    L.insert(0, X.shape[1])
    L. append(Y_train.shape[1])

    print('No of nodes in each layer (input, hidden layers, output):' + str(L))
    # Initialize MLP
    mlp = MLP(L)

    # Train MLP
    mlp.trainMLP(X_train, Y_train, eta, alpha, epochs)

    # Test MLP
    mlp.testMLP(X_test, Y_test)
