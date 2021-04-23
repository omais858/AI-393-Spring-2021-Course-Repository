import os
import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Import libraries for machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, accuracy_score

# Finding the optimal parameter for each classifier
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification

trainingData = pd.read_csv('train.csv')


X = trainingData.drop(labels = ["label"],axis = 1) 
y = trainingData["label"]


def describeDataset(input):
    """
    This function describes the shape of the dataset.  
    """
    print('')
    print("'X' shape: %s."%(X.shape,))
    print('')
    print("'y' shape: %s."%(y.shape,))
    print('')
    print("Unique elements in y: %s"%(np.unique(y)))
    
describeDataset(trainingData)

def histogramVisualization(labels):
    graph = sns.countplot(labels)
    labels.value_counts()
    plt.title("Frequency Histogram of Digits in Training Data")
    plt.xlabel('Number Value')
    plt.ylabel('Frequency')
    
histogramVisualization(y);


def visualizePixels(features):
    # plot the first 20 digits in the training set. 
    f, ax = plt.subplots(4, 4)
    # plot some 4s as an example
    for i in range(21):
        # Create a 1024x1024x3 array of 8 bit unsigned integers
        data = features.iloc[i,0:785].values # This is the first number
        nrows, ncols = 28, 28
        grid = data.reshape((nrows, ncols))
        n=math.ceil(i/5)-1
        m=[0,1,2,3]*5
        ax[m[i-1], n].imshow(grid)
        
visualizePixels(X)

X = trainingData.iloc[0:500,1:] # everything but the first column for the first 50 examples (pixel values)
y = trainingData.iloc[0:500,:1] # first column only for the first 50 examples (label/answer)


X_train = 0
X_test = 0
Y_train = 0
Y_test = 0
xValues = X
yValues = y.values.ravel()
X_train, X_test, Y_train, Y_test = train_test_split(xValues, yValues, test_size=0.2, random_state=20)

def executeNB(a,b,c,d):
    """ Run a NB"""
    classifier = MultinomialNB()
    classifier.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(classifier, c, d, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('SKlearn Naive Bayes - Training set accuracy: {:.2f} ({:.2f})'.format(mean*100, stdev))
    print('')

executeNB(X_train, Y_train, X_test, Y_test)

def selectParametersForNB(a, b, c, d):

    model = MultinomialNB()
   
    model.fit(a, b)
    print('Selected Parameters for NB:')
    print('')
    print(model)
    print('')
    predictions = model.predict(c)
    print('Naive Bayes - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, c, d, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('Naive Bayes Classifier - { (mean : %s) : (std : %s) }' % (mean, stdev))
    print('')
    print('')
    return
selectParametersForNB(X_train, Y_train, X_test, Y_test);

def executeSVC(a,b,c,d):
    classifier = SVC()
    classifier.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(classifier, c, d, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('SKlearn Support Vector Machine - Training set accuracy: {:.2f} ({:.2f})'.format(mean*100, stdev))
    print('')

executeSVC(X_train, Y_train, X_test, Y_test)

def selectParametersForSVM(a, b, c, d):

    model = SVC()
    model.fit(a, b)
    print('Selected Parameters for SVM:')
    print('')
    print(model)
    print('')
    predictions = model.predict(c)
    print('Linear Support Vector Machine - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, c, d, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('Linear Support Vector Machine - { (mean : %s) : (std : %s) }' % (mean, stdev))
    print('')
    print('')
    return

selectParametersForSVM(X_train, Y_train, X_test, Y_test);


def executeKNN(a,b,c,d):
    classifier = KNeighborsClassifier()
    classifier.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(classifier, c, d, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('SKlearn K-Nearest Neighbor Classifier - Training set accuracy: {:.2f} ({:.2f})'.format(mean*100, stdev))
    print('')


executeKNN(X_train, Y_train, X_test, Y_test)

def selectParametersForKNN(a, b, c, d):

    model = KNeighborsClassifier()
   
    model.fit(a, b)
    print('Selected Parameters for KNN:')
    print('')
    print(model)
    print('')
    predictions = model.predict(c)
    print('K-Nearest Neighbors - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, c, d, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('K-Nearest Neighbors Classifier - { (mean : %s) : (std : %s) }' % (mean, stdev))
    print('')
    print('')
    return



selectParametersForKNN(X_train, Y_train, X_test, Y_test);
