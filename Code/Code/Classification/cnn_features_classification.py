import os
import pickle
import platform
from pathlib import Path
import numpy as np
import librosa
from skimage.transform import rescale
from spafe.features.gfcc import gfcc as sgfcc
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..', 'Datasets')  


def check_dataset():
    wav_counter = 0
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith(".wav"):
            wav_counter += 1
        if wav_counter > 0:
            break
    if wav_counter == 0:
        print('Cannot find dataset. Please follow the instructions provided at https://github.com/kevingerkens/gitfx.')


def get_feat_str(feat_string, spec, chroma, mfcc, gfcc):
    feat_dict = {
        'Spec': spec,
        'Chroma' : chroma,
        'MFCC40' : mfcc,
        'GFCC40' : gfcc
    }
    feat = feat_dict[feat_string]

    return feat


def extract_features(y, sr, y1, sr1):
    """extracts spectogram, mfcc, chromagram and gfcc from audio file"""
    spectogram = np.abs(librosa.stft(y))
    spectogram = rescale(spectogram, scale=(0.25, 1.0))
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y, sr=sr)
    gfcc = sgfcc(y1, num_ceps=40, nfilts=80)

    return spectogram, mfcc, chroma, gfcc


def get_features_and_labels(dr, dataset):
    """extracts labels and features from all audio files in directory"""
    os.chdir(os.path.join(DATA_PATH, dataset))
    os.chdir(dr)
    check_dataset()
    print(dr)
    print('Extracting Features and Labels')
    all_specs, all_mfcc, all_chroma, all_gfcc = [], [], [], []
    label_files = []
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith(".wav"):
            print(file_name)
            label = file_name[:-4].split(sep='_')
            y, sr = librosa.load(file_name, sr=None)
            y = librosa.util.normalize(y)
            y1, sr1 = librosa.load(file_name, sr=16000)
            y1 = librosa.util.normalize(y1)
            spectogram, mfcc, chroma, gfcc = extract_features(y, sr, y1, sr1)
            all_specs.append(spectogram)
            all_mfcc.append(mfcc)
            all_chroma.append(chroma)
            all_gfcc.append(gfcc)
            label_files.append(label)
    all_specs = np.array(all_specs)
    all_mfcc = np.array(all_mfcc)
    all_chroma = np.array(all_chroma)
    all_gfcc = np.swapaxes(np.array(all_gfcc), 1, 2)
    label_files = np.array(label_files)

    return  all_specs, all_mfcc, all_chroma, all_gfcc, label_files


def check_for_feature_data(dr, dataset, feat=None):
    os.chdir(os.path.join(DATA_PATH, dataset, dr))
    print(dr)
    if not Path('CNNLabels.npz').exists():
        spec, mfcc, chroma, gfcc, labels = get_features_and_labels(dr, dataset) 
        np.savez('Spec.npz', spec)
        np.savez('MFCC40.npz', mfcc)
        np.savez('Chroma.npz', chroma)
        np.savez('GFCC40.npz', gfcc)
        np.savez('CNNLabels.npz', labels)
        print('All Data saved')
        if feat in ['Spec', 'Chroma', 'MFCC40', 'GFCC40']:
            data = get_feat_str(feat, spec, chroma, mfcc, gfcc)
        else:
            data = [0]
    else:
        print('Loading feature data and labels')
        if type(feat) == str:
            labels = np.load('CNNLabels.npz')['arr_0']
            print(feat)
            data = np.load(str(feat) + '.npz')['arr_0']
            print('Finished loading feature data and labels')
        else:
            'Feature data has already been extracted. Please select a valid feature: Spec, Chroma, MFCC40 or GFCC40.'
            data = [0]
            labels = [0]
    return np.array(data), np.array(labels)


if __name__ == "__main__":
    for dataset in ['GEC-GIM', 'IDMT']:
        os.chdir(os.path.join(DATA_PATH, dataset))
        for folder in os.listdir(os.getcwd()):
            check_for_feature_data(folder, dataset)
