# AI 393 Spring 2021: Course Repository

### PROJECT MEMBERS
Stdudent ID | Name
------------ | -------------
**62710** | **Muhammad Omais Bin Zaki** 
63623 | Abdullah Tahir
62725 | Fasih Rohan


## Description 
This repository contains assignments and projects submitted to the Artificial Intelligence course offered in Spring 2021 at PafKiet.

# AI Assignment:-

## Introduction:
The goal of the following assignment is to correctly identify handwritten digits. The dataset used is the famous handwritten dataset available on Kaggle.

## Approch:
1. Multinomial Naive Bayes
2. Libraries(matplotlib.plt, multinomialNB, os, numpy, train_test_split, classification_report, metrics)
3. Cross Validation

## Naive Bayes:
Naïve Bayes assumes that each input variable is independent. This is a strong assumption and unrealistic for real data; however, the technique is very effective on a large range of complex problems.

## Description: 
The data contains gray-scale images of hand-written digits 0 to 9. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive. So each row represents an image, with the individual pixel values in each of the 784 positions along with (or without) the label.
The training dataset has 785 columns, the first column contains the label of the digit and the other 784 columns contain the pixel values of the associated image. There are 42000 rows in the training dataset. The test data is the same as the training data, except the images are unlabeled. Hence, 784 columns and 28000 rows.
The aim is to correctly identify the unlabeled digits. Using Naive Bayes Classifier




