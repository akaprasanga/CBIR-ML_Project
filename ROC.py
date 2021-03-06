# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:21 2020

@author: lbrice1
"""


import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("index_file_path", help="For evaluation of feature extraction please run ROC.py with path of created image index file")
args = parser.parse_args()
index_file_path = args.index_file_path

#Load dataset (image)index)
dataset = pd.read_csv(index_file_path)

#Drop non feature/label columns
dataset= dataset.drop(dataset.columns[0], axis = 1)
dataset= dataset.drop(dataset.columns[0], axis = 1)
dataset= dataset.drop(dataset.columns[1], axis = 1)

# cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Binarize lables
labels = label_binarize(dataset.iloc[: , 0], classes = ['animal', 'electronics', 'flower', 'indoor', 'music', 'nature'])
# labels = label_binarize(dataset.iloc[: , 0], classes = cifar_labels)

n_classes = labels.shape[1]

#Scale data
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset.iloc[:, 1:])

#Set training features and labels
training_features = dataset[:, 1:]
training_labels = labels

training_features = (training_features - np.min(training_features)) / (np.max(training_features) - np.min(training_features))
X_train, X_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=0.3, shuffle=True)

#Peform One vs All classifier
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, verbose=True))
classifier = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1))

y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area under curve (auc) for each class
fpr = dict() #false positive rate
tpr = dict() #true positive rate
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot a ROC for a specific class
plt.figure(figsize=(12,8), dpi=160)
lw = 3
plt.plot(fpr[3], tpr[3], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[3])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Using Color Features')
plt.legend(loc="lower right")
plt.show()

#Plot all ROC curves

#Aggregate all fpr
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

#Interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

#Average and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#Plot
# Plot all ROC curves
plt.figure(figsize=(12,8), dpi=160)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Multi-class Using COlorFeatures')
plt.legend(loc="lower right")
plt.show()