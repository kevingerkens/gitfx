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
from cnn_parameter_estimation import scale_data, split_labels, fold_prediction, choose_path
import plots

TEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'Datasets/Parameter Estimation Pitch Changes')


"""parameters for nn architecture"""
kernel_size = (3, 3)
n_conv = 2
n_full = 3
n_nodes = 64
n_filters = 6
batch_size = 128

n_splits = 5        #number of kfold splits for cross validation


def create_dataframe(all_pred, all_error, all_y, all_label, fx, nn_setting, path):
    df = pd.DataFrame(zip(all_pred, all_error, all_y, all_label))
    choose_path(os.path.join(path, fx, 'Scale'))
    df_name = 'df_' + nn_setting + '.pickle'
    df.to_pickle(df_name)


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
    test_data, test_labels = get_test_data(fx, feat)
    for i in range(test_data.shape[1]):
        test_data[:, i, :] = scalers[i].transform(test_data[:, i, :])
    test_data = np.expand_dims(test_data, axis=3) 
    label_test = np.delete(test_labels, np.s_[3:(3+no_params)], axis=1)
    test_labels = test_labels[:, 3:(3+no_params)].astype(float)

    return test_data, test_labels, label_test


def estimate(fx, feat):
    print(fx)
    par_names = param_names(fx)
    no_params = len(par_names)
    os.chdir(DATA_PATH)
    os.chdir(fx)
    data, labels = check_for_feature_data(fx, feat)
    labels, _= split_labels(labels, no_params)
    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_' + str(batch_size)

    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1

    all_pred, all_error, all_y, all_label = [], [], [], []

    for train_index, val_index in kf.split(data):
        print(fold_no)
        train_data= data[train_index]
        train_labels= labels[train_index]

        train_data, _, scalers = scale_data(train_data, test_data=None)

        choose_path(os.path.join(DATA_PATH, '../..', 'Results/Parameter Estimation', fx))

        print("Loading model")
        file_name = 'CNNModel' + nn_setting + str(fold_no)
        my_model = models.load_model(file_name)
        print('Model loaded')

        test_data, test_labels, label_test = load_and_scale_test_data(scalers, no_params)
        all_pred, all_error, all_y, all_label = fold_prediction(my_model, test_data, all_pred, test_labels, all_error, all_y, all_label, label_test)
        fold_no += 1

    create_dataframe(all_pred, all_error, all_y, all_label, fx, nn_setting + '_Scale', os.path.join(DATA_PATH, '../..' + 'Results/Parameter Estimation'))

    del data, labels, train_data, test_data, train_labels, test_labels
    gc.collect()

if __name__ == '__main__':
    feats = ['MFCC40', 'Chroma', 'Spec', 'GFCC40']
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay']
    for folder in fx:
        for feat in feats:
            estimate(folder, feat)