from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def trainAndPredict(X_train, Y_train, X_test, Y_test,  clf):
    # Train SVM with best parameters for COLD features
    scaler = preprocessing.MinMaxScaler().fit(X_train)

    clf.fit(scaler.transform(X_train), Y_train)

    predictions_cold = clf.predict(
        scaler.transform(X_test))

    print("Classification Report:\n")

    score_train_gold = clf.score(scaler.transform(
        X_train), Y_train)
    print("Accuracy on training set: {:.2f}".format(
        score_train_gold))

    score_test_gold = clf.score(scaler.transform(
        X_test), Y_test)
    print("Accuracy on test set: {:.2f}".format(
        score_test_gold))

    return predictions_cold


def votingClassifier(predictions_cold, predictions_hinge, predictions_cold_hinge):

    # Voting classifier
    final_predictions = []
    for i in range(len(predictions_cold)):
        male = 0
        female = 0

        female = (female+1) if predictions_cold[i] == 0 else female
        male = (male+1) if predictions_cold[i] == 1 else male

        female = (female+1) if predictions_cold_hinge[i] == 0 else female
        male = (male+1) if predictions_cold_hinge[i] == 1 else male

        female = (female+1) if predictions_hinge[i] == 0 else female
        male = (male+1) if predictions_hinge[i] == 1 else male

        final_predictions.append(0 if female > male else 1)
    return final_predictions
