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
from tensorflow.keras import models, layers, optimizers, utils, metrics, backend
from tensorflow.keras.utils import to_categorical	
from cnn_features_classification import check_for_feature_data, DATA_PATH



def onehot_mapping(folders):
    mapping = {}
    for i in range(len(folders)):
        mapping[folders[i]] = i

    return mapping


def choose_path(path):
    isdir = os.path.isdir(path)
    abs_path = os.path.abspath(path)
    if isdir == False:
        os.makedirs(abs_path, exist_ok=True)
    os.chdir(abs_path)


def load_features_and_labels(dataset, feat):
    X, y= [], []
    folders = os.listdir(os.getcwd())
    for fx in folders:
        fxdata, fxlabels = check_for_feature_data(fx, dataset, feat)
        n_smp = fxdata.shape[0]
        labels = np.array([fx]*n_smp)
        X.extend(list(fxdata))
        y.extend(list(labels))
    X = np.array(X)
    y = np.array(y)
    mapping = onehot_mapping(folders)
    for i in range(len(y)):
        y[i] = mapping[y[i]]
    labels_onehot = to_categorical(y)

    return X, labels_onehot


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


def train_model(model, train_data, train_labels, test_data, test_labels):
    """trains the model for the estimation"""
    utils.normalize(train_data)
    print(train_data.shape)
    print(train_labels.shape)
    history = model.fit(train_data, train_labels, epochs=100, batch_size=64, verbose=2, validation_data = (test_data, test_labels))
    return history.history


def plot_cm(test_labels, pred, feat):
    if not feat == 'SVM': 
        test_labels = np.argmax(test_labels, axis=1)
        pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(test_labels, pred, normalize='true')
    file_name = 'cm'  + '_' + feat + '.pickle'
    print(cm)
    with open(file_name, 'wb') as handle:
        pickle.dump(cm, handle)
    print('\n')


def confidence_interval(accuracy_all):
    accuracy = np.mean(accuracy_all)
    std = np.std(accuracy_all)
    bounds = 1.96*(std/np.sqrt(5)) 
    print('95 \% Confidence interval: ' + str(accuracy) + ' +- ' +  str(bounds))


def scale_features(train_data, test_data):
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])
    for i in range(test_data.shape[1]):
        test_data[:, i, :] = scalers[i].transform(test_data[:, i, :]) 
    train_data, test_data = np.expand_dims(train_data, axis=3), np.expand_dims(test_data, axis=3)

    return train_data, test_data
    

def get_model(n_conv, kernel_size, n_full, n_nodes, n_filters, batch_size, fold_no, train_data, test_data, train_labels, test_labels, feat):
    backend.clear_session()

    nn_setting = feat + '_' + str(n_conv) + '_' + str(kernel_size[0]) + '_' + str(n_full) + '_' + str(n_nodes) + '_' + str(n_filters) + '_' + str(batch_size)
    if not Path('CNNModel' + nn_setting + str(fold_no)).exists():
        print("Creating Model")
        my_model = create_model(train_data.shape[1:], train_labels.shape[1], kernel_size, n_conv, n_full, n_nodes, n_filters)
        print("Training Model")
        history = train_model(my_model, train_data, train_labels, test_data, test_labels)

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

    return my_model


def fold_accuracy(y_true_all, pred_all, test_labels, pred, accuracy_all, fb_sd_all, folders):
    y_true_all.extend(list(test_labels))
    pred_all.extend(list(pred))
    test_labels = np.argmax(np.array(test_labels), axis=1)
    pred = np.argmax(np.array(pred), axis=1)
    cm = confusion_matrix(test_labels, pred, normalize='true')
    accuracy_diagonal = np.sum(np.diagonal(cm))
    if accuracy_diagonal.size >=11:
        accuracy_diagonal_fb_sd = accuracy_diagonal + cm[2, 8] + cm[8, 2]
    else:
        accuracy_diagonal_fb_sd = accuracy_diagonal
    accuracy_fold = accuracy_diagonal/len(folders) 
    fb_sd_fold = accuracy_diagonal_fb_sd/len(folders) 
    accuracy_all.append(accuracy_fold)
    fb_sd_all.append(fb_sd_fold)

    return y_true_all, pred_all, accuracy_all, fb_sd_all


def prediction(my_model, test_data, test_labels, fold_no):
    pred = my_model.predict(test_data)
    scores = my_model.evaluate(test_data, test_labels, verbose = 0)
    print('Test Data Loss and Accuracy for fold '+ str(fold_no) + ': ') 
    print(scores)

    return pred, scores


def classification(feat, dataset):
    print(feat)
    print(dataset)
    os.chdir(os.path.join(DATA_PATH, dataset))
    folders = os.listdir(os.getcwd())
    n_splits = 5
    kernel_size = (3, 3)
    n_conv = 2
    n_full = 3
    n_nodes = 64
    n_filters = 32
    batch_size = 64

    data, labels_onehot = load_features_and_labels(dataset, feat)
    
    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all, pred_all, accuracy_all, fb_sd_all = [], [], [], []
    fold_no = 1
    for train_index, val_index in kf.split(data):
        print('Fold ' + str(fold_no))
        train_data, test_data = data[train_index], data[val_index]
        train_labels, test_labels = labels_onehot[train_index], labels_onehot[val_index]
        train_data, test_data = scale_features(train_data, test_data)
        
        choose_path(os.path.join(DATA_PATH, '..', 'Results/Classification/CNN', dataset))

        my_model = get_model(n_conv, kernel_size, n_full, n_nodes, n_filters, batch_size, fold_no, train_data, test_data, train_labels, test_labels, feat)

        pred, scores = prediction(my_model, test_data, test_labels, fold_no)
        y_true_all, pred_all, accuracy_all, fb_sd_all = fold_accuracy(y_true_all, pred_all, test_labels, pred, accuracy_all, fb_sd_all, folders)
        fold_no += 1

    confidence_interval(accuracy_all)
    print('Confidence interval when FeedbackDelay = SlapbackDelay: ')
    confidence_interval(fb_sd_all)
    choose_path(os.path.join(DATA_PATH, '..', 'Results/Classification/CNN', dataset))
    plot_cm(np.array(y_true_all), np.array(pred_all), feat)
    
    del data, labels_onehot, train_data, test_data, train_labels, test_labels
    gc.collect()



if __name__ == "__main__":
    feats = ['Spec', 'Chroma', 'MFCC40', 'GFCC40']  
    for dataset in ['GEC-GIM', 'IDMT']:  
        for feat in feats:
            classification(feat, dataset)
