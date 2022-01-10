import os
from pathlib import Path
import platform
import pickle
from librosa.util.utils import normalize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump, load, parallel_backend
from svm_features_classification import check_data, dataset_path
from cnn_classification import plot_cm, choose_path


def confidence_interval(accuracy_all):
    acc = np.mean(accuracy_all)
    print(accuracy_all)
    std = np.std(accuracy_all)
    ci_range = 1.96*(std/np.sqrt(5)) 
    print('Mean accuracy: ' + str(acc))
    print('Confidence interval range: ' + str(ci_range))


def get_classifier(fold_no, dataset, train_data=None, train_labels=None, hyp_par_opt=False):
    """Trains the classifier or loads it, if already existent"""
    os.chdir(os.path.join(dataset_path, '..', 'Results/Classification', dataset))
    print('Getting the Classifier')
    if hyp_par_opt:
        print('Optimizing Hyperparameters')
        gamma_list = [2**exp for exp in range(-14, -1)]
        C_list = [2**exp for exp in range(-2, 5)]

        tuned_parameters = [{'gamma': gamma_list, 'C': C_list}]
        clf = GridSearchCV(SVC(), tuned_parameters, cv=3, verbose=1)

        clf.fit(train_data, train_labels)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)  # gamma = 2**(-10), C = 16
        print()
        return clf

    else:
        if not Path('SVC' +  str(fold_no) + '.joblib').exists():
            print('Training')
            clf = SVC(probability=True, C=16, gamma=0.00048828125)
            with parallel_backend('threading', n_jobs=-1):
                clf.fit(train_data, train_labels)
            print(train_data.shape)
        else:
            clf = load('SVC' +  str(fold_no) + '.joblib')
        return clf


def scale_data(train_data, test_data):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data


def fold_prediction(clf, test_data, test_labels, y_true_all, pred_all, accuracy_all):
    pred_labels = clf.predict(test_data)
    print('Prediction: ' + str(pred_labels))
    print('True: ' + str(test_labels))
    print('Accuracy: ' + str(accuracy_score(test_labels, pred_labels)))
    y_true_all.extend(test_labels)
    pred_all.extend(pred_labels)
    cm = confusion_matrix(test_labels, pred_labels, normalize='true')
    print(cm)
    acc_sum = np.sum(np.diagonal(cm))
    acc_sum = acc_sum + cm[2, 8] + cm[8, 2]
    acc_fold = acc_sum/11
    accuracy_all.append(acc_fold)

    return y_true_all, pred_all, accuracy_all


def concatenate_fx_data():
    X, y = [], []
    for dr in os.listdir(os.getcwd()):
        os.chdir(dr)
        print(dr)
        X_currentfx, _ = check_data(dr)
        n_smp = X_currentfx.shape[0]
        y_currentfx = np.array([dr]*n_smp)
        X.extend(list(X_currentfx))
        y.extend(list(y_currentfx))
        os.chdir('..')
    X = np.array(X)
    y = np.array(y)

    X = np.nan_to_num(X, 0.0)

    return X, y


def train_svc(dataset):
    """Extracts/loads feature data and trains Support Vector Machine Classifier"""
    os.chdir(os.path.join(dataset_path, dataset))

    data, labels = concatenate_fx_data()

    n_splits=5
    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all, pred_all, accuracy_all = [], [], []
    fold_no = 1

    for train_index, val_index in kf.split(data):
        print('Fold ' + str(fold_no))
        train_data, test_data = data[train_index], data[val_index]
        train_labels, test_labels = labels[train_index], labels[val_index]
        train_data, test_data = scale_data(train_data, test_data)

        clf = get_classifier(fold_no, dataset, train_data=train_data, train_labels=train_labels, hyp_par_opt=False)
        y_true_all, pred_all, accuracy_all = fold_prediction(clf, test_data, test_labels, y_true_all, pred_all, accuracy_all)

        choose_path(os.path.join(dataset_path, '../..', 'Results/Classification/SVM', dataset))
        dump(clf, 'SVC' + str(fold_no) + '.joblib')
        print('SVC saved')
        fold_no +=1
    y_true_all = np.array(y_true_all)
    pred_all = np.array(pred_all)
    confidence_interval(accuracy_all)

    choose_path(os.path.join(dataset_path, '../..', 'Results/Classification/SVM', dataset))
    plot_cm(np.array(y_true_all), np.array(pred_all), feat='SVM')

    print('done')

if __name__ == '__main__':
    for dataset in ['Classification', 'IDMT']:
        train_svc(dataset)

