# Imports
from cmath import e
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
import numpy as np
import os
from tuning import *
from train_score import *


# prepare features directories
dir = os.getcwd()+'/Gold_Hinge'
hingeDir = dir+"/FeaturesOutput/hinge_features.npy"
coldDir = dir+"/FeaturesOutput/cold_features.npy"
icdarCold = dir+"/FeaturesOutput/icdar_cold_features.npy"
icdarHinge = dir+"/FeaturesOutput/icdar_hinge_features.npy"
icdarLabels = dir+"/FeaturesOutput/icdar_labels.npz"
labelsDir = dir+"/FeaturesOutput/labels.npz"


# Load features
print("Loading features...")
X_hinge = np.load(hingeDir)
X_cold = np.load(coldDir)
Y_train = np.load(labelsDir)['label']
label_names = np.load(labelsDir)['label_name']

X_icdar_hinge = np.load(icdarHinge)
X_icdar_cold = np.load(icdarCold)
Y_icadar_labels = np.load(icdarLabels)['label']
label_names_icdar = np.load(icdarLabels)['label_name']

X_cold = np.concatenate((X_cold, X_icdar_cold), axis=0)
X_hinge = np.concatenate((X_hinge, X_icdar_hinge), axis=0)

# prepare icdar labels
icdar_classes = pd.read_csv('train_answers.csv')['male'].values
icdar_labels = []
for i in range(len(Y_icadar_labels)):
    icdar_labels.append(icdar_classes[int(Y_icadar_labels[i])-1])

X_train = np.concatenate((X_cold, X_hinge), axis=1)
Y_train = np.concatenate((Y_train, icdar_labels), axis=0)


# tunning parameters using GridSearchCV
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train, Y_train, test_size=0.1)

print("Tunning parameters for SVM (Both)...")
clf = getBestParamsForSVM(X_train, Y_train, 'scaler_both.joblib')
dump(clf, 'svm_both.joblib')


print("Trainig&Predict SVM (Both)...")
predictions_cold_hinge = trainAndPredict(X_train, Y_train, X_test, Y_test,
                                         clf)

print("Tunning parameters for SVM (COLD)...")
clf_cold = getBestParamsForSVM(
    X_train[:, np.arange(0, 420)], Y_train, 'scaler_cold.joblib')
dump(clf_cold, 'svm_cold.joblib')

print("Trainig&Predict SVM (COLD)...")
predictions_cold = trainAndPredict(X_train[:, np.arange(0, 420)], Y_train, X_test[:, np.arange(
    0, 420)], Y_test, clf_cold)

print("Tunning parameters for SVM (Hinge)...")
clf_hinge = getBestParamsForSVM(
    X_train[:, np.arange(420, 1200)], Y_train, 'scaler_hinge.joblib')
dump(clf_hinge, 'svm_hinge.joblib')

print("Trainig&Predict SVM (Hinge)...")
predictions_hinge = trainAndPredict(X_train[:, np.arange(420, 1200)], Y_train, X_test[:, np.arange(
    420, 1200)], Y_test, clf_hinge)


finalPredictions = votingClassifier(
    predictions_cold, predictions_hinge, predictions_cold_hinge)

accuracy = sum(finalPredictions == Y_test)/len(Y_test)
print("\nAccuracy on test set using voting classifier: {:.2f}".format(
    accuracy))
