import os
import joblib
import pickle
from pathlib import Path
import numpy as np
from tensorflow.keras import models, layers, optimizers, utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import gc
from cnnfeatextr import DATA_PATH, check_for_feature_data, param_names
from cnn_parameter_estimation import choose_path
import plots



"""parameters for nn architecture"""
kernel_size = (3, 3)
n_conv = 2
n_full = 3
n_nodes = 64
n_filters = 6
batch_size = 128


n_splits = 5        #number of kfold splits for cross validation

def split_labels(y, no_params):
    labels = np.delete(y, np.s_[3:(3+no_params)], axis=1)
    y = y[:, 3:(3+no_params)].astype(float)

    return y, labels


def scale_data(train_data, test_data):
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
    if not test_data == None:
        for i in range(test_data.shape[1]):
            test_data[:, i, :] = scalers[i].transform(test_data[:, i, :]) 

    train_data, test_data = np.expand_dims(train_data, axis=3), np.expand_dims(test_data, axis=3)

    return train_data, test_data, scalers


def get_model(n_conv, kernel_size, n_full, n_nodes, n_filters, batch_size, fold_no, feat):
    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_' + str(batch_size)

    print("Loading model")
    file_name = 'CNNModel' + nn_setting + str(fold_no)
    my_model = models.load_model(file_name)
    print('Model loaded')

    return my_model


def create_dataframe(all_pred, all_error, all_y, all_label, fx, nn_setting, path):
    df = pd.DataFrame(zip(all_pred, all_error, all_y, all_label))
    choose_path(os.path.join(path, fx, 'Noise'))
    df_name = 'df_' + nn_setting + '.pickle'
    df.to_pickle(df_name)


def fold_prediction(my_model, test_data, all_pred, test_labels, all_error, all_y, all_label, label_test):  
    pred = my_model.predict(test_data)
    all_pred.append(pred)
    error = np.abs(pred - np.array(test_labels))
    all_error.append(error)
    all_y.append(test_labels)
    all_label.append(label_test)
    mean_error = np.mean(error, axis=0)

    return all_pred, all_error, all_y, all_label


def noise_plots(feat, fold_no, example, label, noise_factor):
    if fold_no ==5:
        if feat == 'MFCC40' or feat == 'GFCC40':
            fig, ax = plt.subplots()
            sr = 44100
            sr_gfcc = sr*1.98/1.73
            if feat == 'GFCC40':
                img = librosa.display.specshow(example, x_axis='time', ax=ax, sr=sr_gfcc)
            else:
                img = librosa.display.specshow(example, x_axis='time', ax=ax, sr=sr)
            plt.xlabel('Time in s')
            plt.ylabel('Coefficients')
            plt.yticks([*range(0, 45 , 5)])
            fig.colorbar(img, ax=ax)
        elif feat == 'Spec':
            fig, ax = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(example,
                                                    ref=np.max),
                            y_axis='log', x_axis='time', ax=ax, sr=44100)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            plt.xlabel('Time in s')
            plt.ylabel('Frequency in Hz')
        elif feat == 'Chroma':
            fig, ax = plt.subplots()
            img = librosa.display.specshow(example, y_axis='chroma', x_axis='time', ax=ax, sr=44100)
            plt.xlabel('Time in s')
            plt.ylabel('Pitch Class')
            fig.colorbar(img, ax=ax)
        plt.tight_layout()
        file_name = '_'.join([str(elem) for elem in label])  + '_' + feat  + '_'  + 'alpha=' + str(noise_factor)+ '.pdf'
        plt.savefig(file_name)
        plt.clf()        


def estimate(fx, feat, noise_factor):
    print(fx)
    print(feat)
    print(str(noise_factor))
    par_names = param_names(fx)
    no_params = len(par_names)
    os.chdir(DATA_PATH)
    os.chdir(fx)
    data, labels = check_for_feature_data(fx, feat)
    labels, label_other = split_labels(labels,  no_params)
    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_' + str(batch_size)

    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1

    all_pred, all_error, all_y, all_label = [], [], [], []

    for train_index, val_index in kf.split(data):  	            
        print(fold_no)
        train_data, test_data = data[train_index], data[val_index]
        train_labels, test_labels = labels[train_index], labels[val_index]
        label_train, label_test = label_other[train_index], label_other[val_index]

        train_data, test_data, _ = scale_data(train_data, test_data)

        choose_path(os.path.join(DATA_PATH, '../..', 'Results/Parameter Estimation', fx))
        noise = np.random.normal(0, np.amax(test_data)*noise_factor, test_data.shape)
        test_data = test_data + noise
        example = test_data[2500]
        noise_plots(feat, fold_no, example, test_labels[2500], noise_factor)

        my_model = get_model(n_conv, kernel_size, n_full, n_nodes, n_filters, batch_size, fold_no, feat)
        all_pred, all_error, all_y, all_label = fold_prediction(my_model, test_data, all_pred, test_labels, all_error, all_y, all_label, label_test)

        fold_no += 1

    create_dataframe(all_pred, all_error, all_y, all_label, fx, nn_setting + '_noise_' + str(noise_factor), os.path.join(DATA_PATH, '../..' + 'Results/Parameter Estimation'))

    del data, labels, train_data, test_data, train_labels, test_labels              #   clear memory to save resources
    gc.collect()

if __name__ == '__main__':
    feats = ['Chroma', 'Spec', 'GFCC40', 'MFCC40']
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay']
    noise = [0.0, 0.001, 0.01, 0.05]
    for feat in feats:
        for folder in fx:
            for amplitude in noise:
                estimate(folder, feat, amplitude)
