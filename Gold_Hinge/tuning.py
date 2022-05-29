from sklearn.svm import SVC
from sklearn import preprocessing
from time import time
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, cross_val_score
import numpy as np


def getBestParamsForSVM(X_train, Y_train):
    scaler = preprocessing.MinMaxScaler().fit(X_train)

    GridSearchCV_parameters = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale'],
        'kernel': ['rbf', 'linear']
    }

    t0 = time()
    clf = GridSearchCV(SVC(class_weight='balanced'),
                       GridSearchCV_parameters,  refit=True)

    clf = clf.fit(scaler.transform(X_train), Y_train)
    print("Best estimator found by grid search:")
    print("\nBest parameters: ", clf.best_params_)
    print("Mean Cross Validation Score: %0.2f" % clf.best_score_)
    print("Training time: %.3f" % (time() - t0))
    return clf
