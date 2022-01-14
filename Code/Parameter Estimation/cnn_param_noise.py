import os
import joblib
from pathlib import Path
import numpy as np
import keras
from keras import models, layers, optimizers, utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import gc
from cnnfeatextr import DATA_PATH, check_data, param_names
import plots

plt.rcParams["figure.figsize"] = (20.1,9.5)


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

n_splits = 5        #number of kfold splits for cross validation

def add_conv(model, i, kernel_size, n_filters):
    """adds a convolutional layer, batch normalization and max pooling to the model"""
    model.add(layers.Conv2D(n_filters*(i+1), kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    return model

def add_full(model, n_nodes):
    """adds a dense layer with batch normalization and dropout to the model"""
    model.add(layers.Dense(n_nodes, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    return model

def create_model(input_dim, output_dim, kernel_size, n_conv, n_full, n_nodes, n_filters):
    """creates nn model with architecture corresponding to parameters"""
    keras.backend.clear_session() 
    model = []
    model = models.Sequential()
    """add first convolutional layer"""
    model.add(layers.Conv2D(n_filters, kernel_size=kernel_size, activation='relu', input_shape=input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    """add additional conv layers to match specified number"""
    for i in range(n_conv-1):
        model = add_conv(model, i+1, kernel_size, n_filters)
    model.add(layers.Flatten())
    """add fully connected layers to match specified number"""
    for _ in range(n_full-1):
        model = add_full(model, n_nodes)
    """add output layer with sigmoid activation so output will be between 0.0 and 1.0"""
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    model.compile(loss='mean_squared_error', 
                  optimizer=optimizers.Adam(learning_rate=0.001), 
                  metrics=['mse'])

    print(model.summary())
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """trains the model for the estimation"""
    utils.normalize(X_train)
    print(X_train.shape)
    print(y_train.shape)
    history = model.fit(X_train, y_train, epochs=70, batch_size=128, verbose=2, validation_data=(X_test, y_test))
    plots.learning_curve(history)

    return history.history

def estimate(fx, feat, noise_factor):
    print(fx)
    par_names = param_names(fx)
    no_params = len(par_names)
    os.chdir(DATA_PATH)
    os.chdir(fx)
    X, y = check_data(fx, feat)
    print(X.shape)
    print(y.shape)
    label = np.delete(y, np.s_[3:(3+no_params)], axis=1)
    #y = y[:, 3:(3+no_params)].astype(float)
    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_128'

    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1

    all_pred, all_error, all_y, all_label = [], [], [], []

    for train, val in kf.split(X):
        print(fold_no)
        X_train, X_test = X[train], X[val]
        y_train, y_test = y[train], y[val]
        #label_train, label_test = label[train], label[val]

        scalers = {}
        for i in range(X_train.shape[1]):
            scalers[i] = StandardScaler()
            X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
        for i in range(X_test.shape[1]):
            X_test[:, i, :] = scalers[i].transform(X_test[:, i, :]) 

        #X_train, X_test = np.expand_dims(X_train, axis=3), np.expand_dims(X_test, axis=3)

        os.chdir(DATA_PATH)
        os.chdir(fx)
        os.chdir('Results')
        print("Loading model")
        # file_name = 'CNNModel' + nn_setting + str(fold_no)
        # my_model = models.load_model(file_name)
        # print('Model loaded')
        noise = np.random.normal(0, np.amax(X_test)*noise_factor, X_test.shape)
        X_test = X_test + noise
        example = X_test[2500]
        print(y_test[2500])
        if (feat == 'MFCC40' or feat == 'GFCC40') and fold_no == 5:
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
        file_name = '_'.join([str(elem) for elem in y_test[2500]])  + '_' + feat  + '_'  + 'alpha=' + str(noise_factor)+ '.pdf'
        plt.savefig(file_name)
        plt.clf()




        fold_no += 1

    # df = pd.DataFrame(zip(all_pred, all_error, all_y, all_label))
    # print(df)
    # os.chdir(DATA_PATH)
    # os.chdir(fx)
    # if noise == True:
    #     df_name = str(DATA_PATH) + '/' + fx  + '/'  + 'Results' + '/' + 'df_' + nn_setting + '_noise_' + str(TEST_NOISE_FACTOR) + '.pickle'
    # else: 
    #     df_name = str(DATA_PATH) + '/' + fx  + '/'  + 'Results' + '/' + 'df_' + nn_setting + '.pickle'
    # df.to_pickle(df_name)

    del X, y, X_train, X_test, y_train, y_test
    gc.collect()

if __name__ == '__main__':
    feats = ['Chroma', 'Spec', 'GFCC40', 'MFCC40']
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay']
    noise = [0.0, 0.001, 0.01, 0.05]
    for feat in feats:
        for folder in fx:
            for amplitude in noise:
                estimate(folder, feat, amplitude)
