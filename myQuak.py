'''

2020

Scaffolding code for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''

import sys

assert sys.version_info >= (3, 5)


import sklearn

assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import os


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    #    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [('n10256989', 'James ' 'Norman')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
	- the first field is a ID number
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"

    X = np.genfromtxt(dataset_path, delimiter=',', dtype=str, usecols=range(2, 31))

    y = np.genfromtxt(dataset_path, delimiter=',', dtype=str, usecols=1)

    # finds all the malignant results
    yM = np.argwhere(y == 'M')

    # Find all the benign results
    yB = np.argwhere(y == 'B')

    # Converts all the malignants to 1 and benigns to 0
    y[yM] = 1
    y[yB] = 0

    # turns it into a numpy array
    y1 = np.array(y, dtype=int)
    x1 = np.array(X, dtype=float)

    return x1, y1


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    ##raise NotImplementedError()

    parameters = {'max_depth': range(1, 100)}
    clf = GridSearchCV(DecisionTreeClassifier(max_depth=None, random_state=123456), parameters)
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"

    parameters = {'n_neighbors': range(1, 100)}

    clf = GridSearchCV(KNeighborsClassifier(), parameters)
    clf = clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"

    parameters = {'C': range(1, 100)}

    # removes warning from 0 values
    warnings.filterwarnings('ignore')

    # cross validation using gridsearch adds random state for consistancy
    clf = GridSearchCV(SVC(random_state=20), parameters)

    # Train the model with the training data
    clf.fit(X_training, y_training)

    # Return the automatic cross validated object fitted for the data
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''
        Build a Neural Network classifier (with two dense hidden layers)
        based on the training set X_training, y_training.
        Use the Keras functions from the Tensorflow library

        @param
    	X_training: X_training[i,:] is the ith example
    	y_training: y_training[i] is the class label of X_training[i,:]

        @return
    	clf : the classifier built in this function
        '''
    ##         "INSERT YOUR CODE HERE"

    paramater = {'hidden_layer_sizes': (100,), }  # Parameter list
    iter = 1000
    clf1 = MLPClassifier(max_iter=iter)  # hidden layer will default to 100
    clf = GridSearchCV(clf1, paramater, scoring='accuracy', cv=5)  # Cross-validation
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    ##         "INSERT YOUR CODE HERE"
    print(my_team())

    # path to medical data
    file = './medical_records.data'

    # calls the prepare function
    data, Class = prepare_dataset(file)

    # makes the training data into training and test and training and validation sets at ratio 0.8:0.2
    test_set_size = 0.2
    validation_set_size = 0.2

    # Create initial data sets
    X_train, X_test, y_train, y_test = train_test_split(data, Class, test_size=test_set_size, random_state=5)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_set_size, random_state=5)

    # print (data)
    # print (Class)

    # uncomment the classifier you'd like to test here

    classifier = build_DecisionTree_classifier
    # classifier = build_NearrestNeighbours_classifier
    # classifier = build_SupportVectorMachine_classifier
    # classifier = build_NeuralNetwork_classifier

    # Build Classifier for the training data
    clf = classifier(X_train, y_train)

    # Run the classifier on the training data
    predictions = clf.predict(X_train)

    # Print the best hyper parameter for the data
    print("\nBest Hyperparameter:", clf.best_params_)
    print("-------------------------------------------------------")

    # Print the training score
    print("\nTraining accuracy: ", clf.score(X_train, y_train))
    print("Training error: ", metrics.mean_squared_error(y_train, predictions))

    print("-------------------------------------------------------")

    # Build Classifier for the validation data
    predictions = clf.predict(X_val)

    # Print the validation score
    print("\nValidation accuracy: ", clf.score(X_val, y_val))
    print("Validation error: ", metrics.mean_squared_error(y_val, predictions))

    print("-------------------------------------------------------")

    # Build the Classifier for the test data
    predictions = clf.predict(X_test)

    # Print the testing score
    print("\nTesting accuracy: ", clf.score(X_test, y_test))
    print("Testing error: ", metrics.mean_squared_error(y_test, predictions))

    # Print the confusion matrix to show if the test data was put in the correct class
    print("\nTest Data Confusion Matrix: ")
    print(metrics.confusion_matrix(y_test, predictions))

    print("-------------------------------------------------------")
