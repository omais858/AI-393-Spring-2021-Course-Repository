import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

train.head()

test.head()

train.shape

test.shape

train.info()

test.info()

train.isna().sum()

for i in train.columns:
    if train[i].isna().sum() > 0:
        print(i)

test.isna().sum()

train['label'].value_counts(normalize=True)

plt.hist(train['label'], bins=20, alpha=0.7, color='#603c8e')
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(train[train.columns[1:785]], train[train.columns[0]]
                                                    , test_size=0.2)

# Predictor variables of training set
X_train.shape

# Dependent/outcome variable of training set
y_train.shape

# Predictor variables of validation set
X_test.shape

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

#Testing the model on validation set
mnb_pred = mnb.predict(X_test)

print(f"Accuracy: {round(metrics.accuracy_score(y_test, mnb_pred)*100, 2)}%")

