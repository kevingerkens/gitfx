import os
import platform
import joblib
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import gc
from cnnfeatextr import DATA_PATH, check_for_feature_data, param_names
from cnn_param import scale_data, split_labels, fold_prediction, create_dataframe
import plots

TEST_PATH = os.path.join(os.path.dirname(__file__), '../..', 'Datasets/GEPE-GIM Pitch Changes')

model_params = {
    'conv_layers': [2],
    'conv_filters': [6],
    'dense_layers': [3],
    'dense_neurons': [64],
    'kernel_size': [3]
}


"""parameters for nn architecture"""
kernel_size = (3, 3)
n_conv = 2
n_full = 3
n_nodes = 64
n_filters = 6
batch_size = 128

n_splits = 5        #number of kfold splits for cross validation


def get_test_data(dr, feat):
    os.chdir(TEST_PATH)
    os.chdir(dr)
    print('Loading feature data and labels')
    labels = np.load('CNNLabels.npz')['arr_0']
    print(feat)

    data = np.load(str(feat) + '.npz')['arr_0']
    print('Finished loading feature data and labels')
    return np.array(data), np.array(labels)


def load_and_scale_test_data(scalers, no_params):
    X_test, y_test = get_test_data(fx, feat)
    for i in range(X_test.shape[1]):
        X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])
    X_test = np.expand_dims(X_test, axis=3) 
    label_test = np.delete(y_test, np.s_[3:(3+no_params)], axis=1)
    y_test = y_test[:, 3:(3+no_params)].astype(float)

    return X_test, y_test, label_test


def estimate(fx, feat):
    print(fx)
    par_names = param_names(fx)
    no_params = len(par_names)
    os.chdir(DATA_PATH)
    os.chdir(fx)
    X, y = check_for_feature_data(fx, feat)
    y, _= split_labels(y, no_params)
    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_' + str(batch_size)

    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1

    all_pred, all_error, all_y, all_label = [], [], [], []

    for train, val in kf.split(X):
        print(fold_no)
        X_train= X[train]
        y_train= y[train]

        X_train, _, scalers = scale_data(X_train, X_test=None)

        os.chdir(os.path.join(DATA_PATH, '../..', 'Results/Parameter Estimation'))

        print("Loading model")
        file_name = 'CNNModel' + nn_setting + str(fold_no)
        my_model = models.load_model(file_name)
        print('Model loaded')

        X_test, y_test, label_test = load_and_scale_test_data(scalers, no_params)
        all_pred, all_error, all_y, all_label = fold_prediction(my_model, X_test, all_pred, y_test, all_error, all_y, all_label, label_test)
        fold_no += 1

    create_dataframe(all_pred, all_error, all_y, all_label, fx, nn_setting + '_Scale', os.path.join(DATA_PATH, '../..' + 'Results/Parameter Estimation'))

    del X, y, X_train, X_test, y_train, y_test
    gc.collect()

if __name__ == '__main__':
    feats = ['MFCC40', 'Chroma', 'Spec', 'GFCC40']
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay']
    for folder in fx:
        for feat in feats:
            estimate(folder, feat)
