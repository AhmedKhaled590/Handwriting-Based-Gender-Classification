from cmath import e
from time import time
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy as Entropy
from sklearn import svm
from sklearn.svm import SVC
import utils
import os
import pandas as pd
import numpy as np


def extractGLCM(filename, outputFileName):
    img = cv2.imread(filename)
    # Extract RGB channels
    img = img[:, :, 0]

    step = [1]  # step size
    step = np.asarray(step)
    angle = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles (0, 45, 90, 135)

    coOccuranceMat = graycomatrix(
        img, step, angle, levels=256, symmetric=True, normed=True)

    # calculate the GLCM properties
    contrast = graycoprops(coOccuranceMat, prop='contrast')
    correlation = graycoprops(coOccuranceMat, prop='correlation')
    energy = graycoprops(coOccuranceMat, prop='energy')
    homogeneity = graycoprops(coOccuranceMat, prop='homogeneity')
    entropy = []
    entropy.insert(0, Entropy(coOccuranceMat[0, 0, :, :]))
    entropy.insert(1, Entropy(coOccuranceMat[0, 1, :, :]))
    entropy.insert(2, Entropy(coOccuranceMat[1, 0, :, :]))
    entropy.insert(3, Entropy(coOccuranceMat[1, 1, :, :]))
    entropy = np.array(entropy)

    # append all features to a numpy array
    features = np.array([contrast.flatten(), homogeneity.flatten(),
                        energy.flatten(), correlation.flatten(), entropy.flatten()])

    features = features.flatten()
    features = features.reshape(1, -1)

    with open(outputFileName+'.csv', 'a') as csvfile:
        np.savetxt(csvfile, features, fmt='%f', delimiter=',')
        csvfile.close()
    return features


if __name__ == '__main__':
    print("Extracting GLCM features...")
    # check if file exists delete it
    if os.path.isfile('female.csv'):
        os.remove('female.csv')

    if os.path.isfile("male.csv"):
        os.remove('male.csv')

    if os.path.isfile("test.csv"):
        os.remove('test.csv')
    if os.path.isfile("icdar.csv"):
        os.remove('icdar.csv')

    with open('female.csv', 'a') as csvfile:
        np.savetxt(csvfile, [], delimiter=',',
                   header='Contrast1,Contrast2,Contrast3,Contrast4,homogeneity1,homogeneity2,homogeneity3,homogeneity4,energy1,energy2,energy3,energy4,correlation1,correlation2,correlation3,correlation4,entropy1,entropy2,entropy3,entropy4')
        csvfile.close()

    with open('male.csv', 'a') as csvfile:
        np.savetxt(csvfile, [], delimiter=',',
                   header='Contrast1,Contrast2,Contrast3,Contrast4,homogeneity1,homogeneity2,homogeneity3,homogeneity4,energy1,energy2,energy3,energy4,correlation1,correlation2,correlation3,correlation4,entropy1,entropy2,entropy3,entropy4')

        csvfile.close()

    with open('test.csv', 'a') as csvfile:
        np.savetxt(csvfile, [], delimiter=',',
                   header='Contrast1,Contrast2,Contrast3,Contrast4,homogeneity1,homogeneity2,homogeneity3,homogeneity4,energy1,energy2,energy3,energy4,correlation1,correlation2,correlation3,correlation4,entropy1,entropy2,entropy3,entropy4')

        csvfile.close()

    with open('icdar.csv', 'a') as csvfile:
        np.savetxt(csvfile, [], delimiter=',',
                   header='Contrast1,Contrast2,Contrast3,Contrast4,homogeneity1,homogeneity2,homogeneity3,homogeneity4,energy1,energy2,energy3,energy4,correlation1,correlation2,correlation3,correlation4,entropy1,entropy2,entropy3,entropy4')

        csvfile.close()
    features = []
    maleFeatures = []
    train_classes = []
    test = []
    test_classes = []
    # time to extract features
    start_time = time()

    # read csv file
    df = pd.read_csv('train_answers.csv')
    # get the labels
    icdar_classes = df['male'].values
    icdar_classes_train = np.array([])

    i = 0
    j = 0
    class0 = 0
    for filename in os.listdir('images_gender/images/train'):
        try:
            features.append(extractGLCM(
                'images_gender/images/train/'+filename, 'icdar'))
            if i % 2 == 0:
                class0 = icdar_classes[j]
                j = j + 1
            icdar_classes_train = np.append(icdar_classes_train, class0)
            i = i + 1
        except Exception as e:
            print(e)
            continue

    # append last element in icdar_classes to icdar_classes_train
    icdar_classes_train = np.append(icdar_classes_train, 0)
    icdar_classes_train = np.append(icdar_classes_train, 0)

    icdar_classes = icdar_classes_train

    for filename in os.listdir("Females/Females"):
        try:
            features.append(extractGLCM(
                'Females/Females/'+filename, 'female'))
            train_classes.append(0)
        except Exception as e:
            print(e)
            continue

    for filename in os.listdir("Males/Males"):
        try:
            features.append(extractGLCM(
                'Males/Males/'+filename, 'male'))
            train_classes.append(1)
        except Exception as e:
            print(e)
            continue

    for filename in os.listdir('Unspecified/Unspecified'):
        if filename.find('F'):
            test_classes.append(0)
        else:
            test_classes.append(1)
        test.append(extractGLCM('Unspecified/Unspecified/'+filename, 'test'))
    end_time = time()
    print("Time taken to extract features: ", end_time-start_time, " seconds")

    features = np.array(features)
    train_classes = np.array(train_classes)
    # append icdar classes to the train classes as elements
    train_classes = np.append(train_classes, icdar_classes)
    train_classes = train_classes.reshape(train_classes.shape[0], -1)
    train_classes = train_classes.flatten()
    test = np.array(test)
    features = features.reshape(features.shape[0], -1)

    # train the classifier and predict the test data
    print("Training the classifier...")
    start_time = time()
    clf = SVC(C=50000.0, class_weight='balanced', gamma=0.0001, kernel='rbf')
    # print(train_classes.shape)
    # svmTuner(features, train_classes)
    clf.fit(features, train_classes)

    end_time = time()
    print("Time taken to train the classifier: ",
          end_time-start_time, " seconds")

    # predict the test data
    test = test.reshape(test.shape[0], -1)
    print("Predicting the test data...")
    start_time = time()
    predicted_classes = clf.predict(test)
    end_time = time()
    print("Time taken to predict the test data: ",
          end_time-start_time, " seconds")

    # calculate the accuracy
    print("Calculating the accuracy...")
    print("Accuracy: ", utils.get_accuracy(test_classes, predicted_classes))
