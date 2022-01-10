"""Estimates the parameters of the used Audio Effect"""

import os
from pathlib import Path
import pickle
import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
import joblib
from keras import models, layers, optimizers, utils
from cnn_parameter_estimation import choose_path
from cnnfeatextr import check_dataset

DATA_PATH = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'Datasets/Parameter Estimation'))


def get_dist_feat(y_cut, sr):
    """Extracts features for Distortion parameter estimation"""
    v_features = []
    mfcc = librosa.feature.mfcc(y=y_cut, sr=sr, n_mfcc=3)
    mfcc_delta = librosa.feature.delta(mfcc)
    m_features_mfcc = np.concatenate((mfcc, mfcc_delta))

    for feat in m_features_mfcc:
        lin_coeff, lin_residual, _, _, _ = np.polyfit(np.arange(len(feat)), feat, 1, full=True)
        v_features.extend(lin_coeff)
        # v_features.append(lin_residual)

    return v_features


def get_trem_feat(y_cut, sr):
    """Extracts features for Tremolo parameter estimation"""
    rms = librosa.feature.rms(S=librosa.core.stft(y_cut))
    rms_delta = librosa.feature.delta(rms)
    m_features_rms = np.concatenate((rms, rms_delta))

    v_features = []

    for feat in m_features_rms:

        feat_cut = feat - np.average(feat)

        feat_windowed = feat_cut * np.hanning(len(feat_cut))
        feat_int = np.pad(feat_windowed, (0, 1024 - len(feat_windowed) % 1024), 'constant')

        rfft = np.fft.rfft(feat_int)
        rfft_norm = np.abs(rfft) * 4 / 1024
        rfft_max = np.max(rfft_norm)
        rfft_max_ind = np.argmax(rfft_norm)
        low_limit = rfft_max_ind - 32 if rfft_max_ind - 32 >= 0 else 0
        high_limit = rfft_max_ind + 32 if rfft_max_ind + 32 <= len(rfft_norm) else len(rfft_norm)
        rfft_norm[low_limit:high_limit] = np.zeros(high_limit - low_limit)

        rfft_max2_ind = np.argmax(rfft_norm)
        if rfft_max_ind < rfft_max2_ind:
            v_features.extend([rfft_max, rfft_max_ind,
                               np.max(rfft_norm), rfft_max2_ind])
        else:
            v_features.extend([np.max(rfft_norm), rfft_max2_ind,
                               rfft_max, rfft_max_ind])

    return v_features


def get_dly_feat(y_cut, sr, y):
    """Extracts features for Delay parameter estimation"""
    onset_strength = librosa.onset.onset_strength(y=y_cut, sr=sr)
    onset_strength = np.reshape(onset_strength, [1, len(onset_strength)])
    v_features = []

    dly_onsets = librosa.onset.onset_detect(y=y_cut, sr=sr, units='frames', backtrack=False)

    dtype = [('onset_strength', float), ('onset', int)]
    all_onsets_strength = [(onset_strength[0, onset], onset) for onset in dly_onsets]
    all_onsets_strength_np = np.array(all_onsets_strength, dtype=dtype)
    onsets_sorted = np.sort(all_onsets_strength_np, order='onset_strength')
    strongest_onset = onsets_sorted[-1]
    if len(onsets_sorted) > 1:
        print('More than one onset candidate found')
        strongest_onset_2 = onsets_sorted[-2]
    else:
        strongest_onset_2 = np.array((0, 0), dtype=dtype)

    mfcc_delta = librosa.feature.delta(librosa.feature.mfcc(y_cut, sr=sr, n_mfcc=1)
                                       )[:, strongest_onset['onset']-5:strongest_onset['onset']+3]
    if len(onsets_sorted) > 1:
        mfcc_delta_2 = librosa.feature.delta(librosa.feature.mfcc(y_cut, sr=sr, n_mfcc=1)
                                             )[:, strongest_onset_2['onset']-5:strongest_onset_2['onset']+3]
    else:
        mfcc_delta_2 = np.zeros((1, 8))
    mfcc_delta_sum = np.sum(mfcc_delta, axis=1)
    mfcc_delta_sum_2 = np.sum(mfcc_delta_2, axis=1)
    rms = librosa.amplitude_to_db(librosa.feature.rms(y_cut)).T
    v_features.extend([mfcc_delta_sum, strongest_onset['onset'], rms[strongest_onset['onset']],
                       mfcc_delta_sum_2, strongest_onset_2['onset'], rms[strongest_onset_2['onset']]])

    return v_features


def read_data(path_folder):
    """Reads sample data from files and extracts features"""
    os.chdir(DATA_PATH)
    train_data = []
    train_labels = []

    os.chdir(path_folder)
    check_dataset()
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith(".wav"):
            print(file_name)
            y, sr = librosa.load(file_name, sr=44100)
            label = file_name[:-4].split('_')[3:5].astype(np.float)
            train_labels.append(label)
            y = np.insert(y, 0, np.zeros(1023))
            y = librosa.util.normalize(y)

            onset_frame = librosa.onset.onset_detect(y=y, sr=sr, units='frames',
                                                        pre_max=20000, post_max=20000,
                                                        pre_avg=20000, post_avg=20000, delta=0, wait=1000)
            offset_frame = librosa.samples_to_frames(samples=y.shape[0])
            onset_sample = librosa.core.frames_to_samples(onset_frame[0])
            offset_sample = librosa.core.frames_to_samples(offset_frame)
            y_cut = y[onset_sample:offset_sample]

            v_features = []
            if path_folder == 'Distortion':
                v_features = get_dist_feat(y_cut=y_cut, sr=sr)
            elif path_folder == 'Tremolo':
                v_features = get_trem_feat(y_cut=y_cut, sr=sr)
            elif path_folder == 'SlapbackDelay':
                v_features = get_dly_feat(y_cut=y_cut, sr=sr, y=y)
            else:
                print('Sample folder for feature extraction not found')

            train_data.append(np.hstack(v_features))

    train_data = np.array(train_data)
    print(train_data.shape)
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)

    train_labels = np.array(train_labels)
    os.chdir(DATA_PATH)
    return train_data, train_labels


def create_model(input_dim, output_dim):
    """Creates the Neural Network for the estimation"""
    model = models.Sequential()
    model.add(layers.Dense(32, input_dim=input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(output_dim, activation='linear'))
    model.summary()
    model.compile(optimizer=optimizers.Adam(),
                  loss='mean_squared_error',
                  metrics=['mse'])
    return model

def getfilename(dr):
    os.chdir(DATA_PATH)
    os.chdir(dr)
    filenames = np.load('Labels.npz')['arr_0'] 
    filenames = np.delete(filenames, np.s_[3:5], axis=1)   
    return filenames


def train_model(model, train_data, train_labels, test_data, test_labels):
    """Trains the model for the estimation"""
    utils.normalize(train_data)
    history = model.fit(train_data, train_labels, epochs=1000, verbose=2, validation_data = (test_data, test_labels))
    # plots.learning_curve(history)


def check_for_features_and_labels(folder_path):
    if not Path('ParamEst.npz').exists():
        feat_data, feat_labels = read_data(path_folder=folder_path)
        os.chdir(DATA_PATH)
        os.chdir(folder_path)
        np.savez('ParamEst.npz', X = feat_data, y = feat_labels)
        print('Data Saved')
    else:
        print('Loading feature data and labels...')
        data = np.load('ParamEst.npz')
        feat_data = data['X']
        feat_labels = data['y'].astype(np.float)

    return feat_data, feat_labels


def get_model(fold_no, train_data, train_labels, folder_path, test_data, test_labels):
    os.chdir(os.path.join(DATA_PATH, '../..', 'Results/Parameter Estimation', folder_path))
    if not Path('ParamEstModel' + str(fold_no) + '.pickle').exists():

        my_model = create_model(train_data.shape[1], train_labels.shape[1])

        train_model(my_model, train_data, train_labels, test_data, test_labels)
        with open('ParamEstModel' + str(fold_no) + '.pickle', 'wb') as handle:
            joblib.dump(my_model, handle)
        print('Model Saved')
    else:
        with open('ParamEstModel' + str(fold_no) + '.pickle', 'rb') as handle:
            my_model = joblib.load(handle)
        print('Model Loaded')

    return my_model

def fold_prediction(my_model, X_test, all_pred, y_test, all_error, all_y, all_label, label_test):  
    pred = my_model.predict(X_test)
    all_pred.append(pred)
    error = np.abs(pred - np.array(y_test))
    all_error.append(error)
    all_y.append(y_test)
    all_label.append(label_test)
    mean_error = np.mean(error, axis=0)

    return all_pred, all_error, all_y, all_label


def create_dataframe(all_pred, all_error, all_y, all_label, fx, nn_setting, path):
    df = pd.DataFrame(zip(all_pred, all_error, all_y, all_label))
    choose_path(os.path.join(path, fx, 'Juergens'))
    df_name = 'df_j' + '.pickle'
    df.to_pickle(df_name)


def estimate(folder_path):
    """Reads the data from folder path, trains the model, and estimates on test data"""
    os.chdir(DATA_PATH)
    os.chdir(folder_path)
    print(folder_path)
    feat_data, feat_labels = check_for_features_and_labels(folder_path)
    filenames = getfilename(folder_path)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    all_pred, all_error, all_y, all_label = [], [], [], []

    for train_index, val_index in kf.split(feat_data):
        print(fold_no)
        train_data, train_labels = feat_data[train_index], feat_labels[train_index]
        test_data, test_labels = feat_data[val_index], feat_labels[val_index]
        label_train, label_test = filenames[train_index], filenames[val_index]

        my_model = get_model(fold_no, train_data, train_labels, folder_path, test_data, test_labels)
        all_pred, all_error, all_y, all_label = fold_prediction(my_model, test_data, all_pred, test_labels, all_error, all_y, all_label, label_test)

        fold_no += 1

    create_dataframe(all_pred, all_error, all_y, all_label, folder_path, nn_setting='')

if __name__ == '__main__':
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay']
    for folder in fx:
        estimate(folder)
