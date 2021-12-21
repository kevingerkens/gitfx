import os
import pickle
from pathlib import Path
from librosa.util.utils import normalize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import gc
import tensorflow.keras
#import seaborn as sn
from tensorflow.keras import models, layers, optimizers, utils, metrics
from tensorflow.keras.utils import to_categorical	
from cnnfeat_clf import check_data, DATA_PATH
#import plots
#from cnn_param import create_model, train_model, add_full, add_conv

folders = ["Chorus", "Distortion", "FeedbackDelay", "Flanger", "NoFX", "Overdrive", "Phaser", "Reverb", "SlapbackDelay", "Tremolo", "Vibrato"]
mapping = {}
for i in range(len(folders)):
  mapping[folders[i]] = i

def add_conv(model, i, kernel_size, n_filters):
    """adds a convolutional layer, batch normalization and max pooling to the model"""
    model.add(layers.Conv2D(n_filters*(i+1), kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    return model

def add_full(model, n_nodes):
    """adds a dense layer with batch normalization and dropout to the model"""
    model.add(layers.Dense(n_nodes, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    return model

def create_model(input_dim, output_dim, kernel_size, n_conv, n_full, n_nodes, n_filters):
    """creates nn model with architecture corresponding to parameters"""
    tensorflow.keras.backend.clear_session() 
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
    """add output layer with softmax activation"""
    model.add(layers.Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=0.001), 
                  metrics=[metrics.CategoricalAccuracy()])

    print(model.summary())
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """trains the model for the estimation"""
    utils.normalize(X_train)
    print(X_train.shape)
    print(y_train.shape)
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=2, validation_data = (X_test, y_test))
    #plots.learning_curve(history)
    return history.history

def plot_cm(y_test, pred, feat):
    y_test = np.argmax(y_test, axis=1)
    pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(y_test, pred, normalize='true')
    file_name = str(DATA_PATH) + 'cm'  + '_' + feat + '.pickle'
    print(cm)
    #with open(file_name, 'wb') as handle:
    #    pickle.dump(cm, handle)
    print('\n')
    #sn.heatmap(cm, annot=True, vmin=0.0, vmax=1.0, xticklabels=folders, yticklabels=folders, cbar=False, square=True, fmt='.2f', cmap=plt.cm.Blues)
    #plt.show()

def acc_ki(acc_all):
    acc = np.mean(acc_all)
    print(acc_all)
    std = np.std(acc_all)
    ki = 1.96*(std/np.sqrt(5)) 
    print('Mean accuracy: ' + str(acc))
    print('Confidence interval: ' + str(ki))



def classification(feat):
    print(feat)
    os.chdir(DATA_PATH)
    n_splits = 5
    kernel_size = (3, 3)
    n_conv = 2
    n_full = 3
    n_nodes = 64
    n_filters = 32
    X, y= [], []
    for fx in os.listdir(os.getcwd()):
        fxdata, fxlabels = check_data(fx, feat)
        #vst = fxlabels[:, 0]
        #mix = fxlabels[:, 1]
        n_smp = fxdata.shape[0]
        labels = np.array([fx]*n_smp)
        X.extend(list(fxdata))
        y.extend(list(labels))
    X = np.array(X)
    y = np.array(y)
    for i in range(len(y)):
        y[i] = mapping[y[i]]
    labels_onehot = to_categorical(y)
    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_64'


    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all = []
    pred_all = []
    acc_all = []
    fold_no = 1
    for train, val in kf.split(X):
        print('Fold ' + str(fold_no))
        X_train, X_test = X[train], X[val]
        y_train, y_test = labels_onehot[train], labels_onehot[val]
        scalers = {}
        for i in range(X_train.shape[1]):
            scalers[i] = StandardScaler()
            X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
        for i in range(X_test.shape[1]):
            X_test[:, i, :] = scalers[i].transform(X_test[:, i, :]) 
        
        X_train, X_test = np.expand_dims(X_train, axis=3), np.expand_dims(X_test, axis=3)
        os.chdir(DATA_PATH)
        os.chdir('..')
        os.chdir('Results')

        if not Path('CNNModel' + nn_setting + str(fold_no)).exists():
            print("Creating Model")
            my_model = create_model(X_train.shape[1:], y_train.shape[1], kernel_size, n_conv, n_full, n_nodes, n_filters)
            print("Training Model")
            history = train_model(my_model, X_train, y_train, X_test, y_test)

            print("Saving Model")
            file_name = 'CNNModel' + nn_setting + str(fold_no)
            hist_name = 'History' + nn_setting + str(fold_no) + '.pickle'
            with open(hist_name, 'wb') as handle:
                pickle.dump(history, handle)
            my_model.save(file_name)
            print('Model saved')
        else:
            print("Loading model")
            file_name = 'CNNModel' + nn_setting + str(fold_no)
            my_model = models.load_model(file_name)
            print('Model loaded')

        pred = my_model.predict(X_test)
        scores = my_model.evaluate(X_test, y_test, verbose = 0)
        print('Test Data Accuracy for fold '+ str(fold_no) + ': ') 
        print(scores)
        y_true_all.extend(list(y_test))
        pred_all.extend(list(pred))
        y_test = np.argmax(np.array(y_test), axis=1)
        pred = np.argmax(np.array(pred), axis=1)
        cm = confusion_matrix(y_test, pred, normalize='true')
        print(cm)
        acc_sum = np.sum(np.diagonal(cm))
        acc_sum = acc_sum + cm[2, 8] + cm[8, 2]
        acc_fold = acc_sum/len(folders)
        acc_all.append(acc_fold)


        fold_no += 1

    acc_ki(acc_all)
    plot_cm(np.array(y_true_all), np.array(pred_all), feat)
    del X, y, X_train, X_test, y_train, y_test
    gc.collect()



if __name__ == "__main__":
    feats = ['Chroma', 'MFCC40', 'GFCC40']    
    for feat in feats:
        classification(feat)