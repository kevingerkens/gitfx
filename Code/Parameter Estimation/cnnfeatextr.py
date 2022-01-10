import os
import pickle
import platform
from pathlib import Path
import numpy as np
import librosa
from skimage.transform import rescale
from spafe.features.gfcc import gfcc as sgfcc
import matplotlib.pyplot as plt

DATA_PATH = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'Datasets/Parameter Estimation'))

def param_names(dr):
    #returns variable parameters for the chosen effect
    names = {
        'Chorus' : ['Rate', 'Depth'],
        'Distortion' : ['Gain', 'Tone'],
        'FeedbackDelay' : ['Time', 'Feedback', 'Fx Mix'],
        'Flanger' : ['Rate', 'Depth', 'Feedback'],
        'NoFX' : ['Low', 'Mid', 'High'],
        'Overdrive' : ['Gain', 'Tone'],
        'Phaser' : ['Rate', 'Depth'],
        'Reverb' : ['Room Size', 'Fx Mix'],
        'SlapbackDelay' : ['Time', 'Fx Mix'],
        'Tremolo' : ['Depth', 'Frequency'],
        'Vibrato' : ['Rate', 'Depth'],
        'DistortionTremolo' : ['Gain', 'Tone', 'Depth', 'Frequency'],
        'DistortionSlapbackDelay' : ['Gain', 'Tone', 'Time', 'Fx Mix'],
        'TremoloSlapbackDelay' : ['Depth', 'Frequency', 'Time', 'Fx Mix'],
        'DistortionTremoloSlapbackDelay' : ['Gain', 'Tone', 'Depth', 'Frequency', 'Time', 'Fx Mix'],
        }
    
    par_names = names[dr]
    return par_names


def extract_features(y, sr, y1, sr1):
    """extracts spectogram, mfcc, chromagram and gfcc from audio file"""
    spectogram = np.abs(librosa.stft(y))
    spectogram = rescale(spectogram, scale=(0.25, 1.0))
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y, sr=sr)
    gfcc = sgfcc(y1, num_ceps=40, nfilts=80)

    return spectogram, mfcc, chroma, gfcc

def append_features_and_labels(all_specs, all_mfcc, all_chroma, all_gfcc, label_files, spectogram, mfcc, chroma, gfcc, label):
    all_specs.append(spectogram)
    all_mfcc.append(mfcc)
    all_chroma.append(chroma)
    all_gfcc.append(gfcc)
    label_files.append(label)

    return all_specs, all_mfcc, all_chroma, all_gfcc, label_files


def get_features_and_labels(dr):
    """extracts labels and features from all audio files in directory"""
    os.chdir(DATA_PATH)
    os.chdir(dr)
    print(dr)
    print('Extracting Data')
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
            all_specs, all_mfcc, all_chroma, all_gfcc, label_files = append_features_and_labels(all_specs, all_mfcc, all_chroma, 
                                                                all_gfcc, label_files, spectogram, mfcc, chroma, gfcc, label)
    all_specs = np.array(all_specs)
    all_mfcc = np.array(all_mfcc)
    all_chroma = np.array(all_chroma)
    all_gfcc = np.swapaxes(np.array(all_gfcc), 1, 2)
    label_files = np.array(label_files)
    # print(all_specs.shape)
    # print(all_mfcc.shape)
    # print(all_chroma.shape)
    # print(all_gfcc.shape)
    return  all_specs, all_mfcc, all_chroma, all_gfcc, label_files

def check_for_feature_data(dr, feat=None):
    os.chdir(os.path.join(DATA_PATH, dr))
    print(dr)
    if not Path('CNNLabels.npz').exists():
        spec, mfcc, chroma, gfcc, labels = get_features_and_labels(dr) 
        np.savez('Spec.npz', spec)
        np.savez('MFCC40.npz', mfcc)
        np.savez('Chroma.npz', chroma)
        np.savez('GFCC40.npz', gfcc)
        np.savez('CNNLabels.npz', labels)
        print('All Data saved')
        data=None
    elif Path('CNNLabels.npz').exists() and feat is not None:
        print('Loading feature data and labels')
        labels = np.load('CNNLabels.npz')['arr_0']
        print(feat)
        data = np.load(feat + '.npz')['arr_0']
        print('Finished loading feature data and labels')

        return np.array(data), np.array(labels)

    else:
        print('Features and labels have already been extracted.')
    os.chdir('..')
    os.chdir(DATA_PATH)
        
    


if __name__ == "__main__":
    os.chdir(DATA_PATH)
    for folder in os.listdir(os.getcwd()):
        check_for_feature_data(folder)