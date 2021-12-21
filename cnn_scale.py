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
from cnnfeatextr import DATA_PATH, check_data, param_names
import plots

TEST_PATH = Path('C:/Desktop/Testsets/Scale')    \
    if platform.system() == 'Windows' else os.path.join(os.path.expanduser('~'), 'tmp/Testsets/Scale')


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

def get_test_data(dr, feat):
    os.chdir(TEST_PATH)
    os.chdir(dr)
    print('Loading feature data and labels')
    labels = np.load('CNNLabels.npz')['arr_0']
    print(feat)

    data = np.load(str(feat) + '.npz')['arr_0']
    print('Finished loading feature data and labels')
    return np.array(data), np.array(labels)

def estimate(fx, feat):
    print(fx)
    par_names = param_names(fx)
    no_params = len(par_names)
    os.chdir(DATA_PATH)
    os.chdir(fx)
    X, y = check_data(fx, feat)
    print(X.shape)
    print(y.shape)
    label = np.delete(y, np.s_[3:(3+no_params)], axis=1)
    y = y[:, 3:(3+no_params)].astype(float)
    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_128'

    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1

    all_pred, all_error, all_y, all_label = [], [], [], []

    for train, val in kf.split(X):
        print(fold_no)
        X_train= X[train]
        y_train= y[train]

        scalers = {}
        for i in range(X_train.shape[1]):
            scalers[i] = StandardScaler()
            X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
        os.chdir(DATA_PATH)
        os.chdir(fx)
        os.chdir('Results')


        print("Loading model")
        file_name = 'CNNModel' + nn_setting + str(fold_no)
        my_model = models.load_model(file_name)
        print('Model loaded')
        X_test, y_test = get_test_data(fx, feat)
        for i in range(X_test.shape[1]):
            X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])
        X_train, X_test = np.expand_dims(X_train, axis=3), np.expand_dims(X_test, axis=3) 
        label_test = np.delete(y_test, np.s_[3:(3+no_params)], axis=1)
        y_test = y_test[:, 3:(3+no_params)].astype(float)
        pred = my_model.predict(X_test)
        all_pred.append(pred)
        error = np.abs(pred - np.array(y_test))
        all_error.append(error)
        all_y.append(y_test)
        all_label.append(label_test)
        mean_error = np.mean(error, axis=0)
        print(mean_error)

        fold_no += 1

    df = pd.DataFrame(zip(all_pred, all_error, all_y, all_label))
    print(df)
    os.chdir(TEST_PATH)
    os.chdir(fx)

    df_name = str(TEST_PATH) + '/' + fx  + '/'  + 'Results' + '/' + 'df_' + nn_setting + '_Scale' + '.pickle'
    df.to_pickle(df_name)

    del X, y, X_train, X_test, y_train, y_test
    gc.collect()

if __name__ == '__main__':
    feats = ['MFCC40', 'Chroma', 'Spec', 'GFCC40']
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay']
    for folder in fx:
        for feat in feats:
            estimate(folder, feat)