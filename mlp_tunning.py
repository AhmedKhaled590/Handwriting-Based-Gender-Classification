from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def MLPTunning(X_train, Y_train, X_Test, Y_Test):
    mlp = MLPClassifier(max_iter=50000)
    scaler = preprocessing.StandardScaler().fit(X_train)
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (15,), (50,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05, 0.001],
        'learning_rate': ['constant', 'adaptive'],
    }

    clf = GridSearchCV(mlp, parameter_space, cv=5)
    clf.fit(scaler.transform(X_train), Y_train)
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    y_true, y_pred = Y_Test, clf.predict(scaler.transform(X_Test))

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))
