import os
import pickle
import platform
from pathlib import Path
import numpy as np
import librosa
from skimage.transform import rescale
from spafe.features.gfcc import gfcc as sgfcc
import matplotlib.pyplot as plt

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
        'Reverb' : ['Room Size'],
        'SlapbackDelay' : ['Time', 'Fx Mix'],
        'Tremolo' : ['Depth', 'Frequency'],
        'Vibrato' : ['Rate', 'Depth'],
        'DistortionTremolo' : ['Gain', 'Tone', 'Depth', 'Frequency'],
        'DistortionSlapbackDelay' : ['Gain', 'Tone', 'Time', 'Fx Mix'],
        'TremoloSlapbackDelay' : ['Depth', 'Frequency', 'Time', 'Fx Mix'],
        'DistortionTremoloSlapBackDelay' : ['Gain', 'Tone', 'Depth', 'Frequency', 'Time', 'Fx Mix'],
        }
    
    par_names = names[dr]
    return par_names

#DATA_PATH = Path('C:/Desktop/uni/Studienarbeit/Audio/NewDataset/G+K+B+D') \
DATA_PATH = Path('C:/Desktop/uni/Studienarbeit/Audio/MixRand/Gitarre monophon')    \
    if platform.system() == 'Windows' else os.path.join(os.path.expanduser('~'), 'tmp/Clf/Gitarre monophon/Samples')

def extract_feats(y, sr, y1, sr1):
    """extracts spectogram, mfcc, chromagram and gfcc from audio file"""
    #spectogram, _, _, _ = plt.specgram(y, Fs=sr, nfft = 256, noverlap=64)
    spectogram = np.abs(librosa.stft(y))
    spectogram = rescale(spectogram, scale=(0.25, 1.0))
    #print(spectogram.shape)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    #print(mfcc.shape)
    chroma = librosa.feature.chroma_stft(y, sr=sr)
    #print(chroma.shape)
    gfcc = sgfcc(y1, num_ceps=40, nfilts=80)

    return spectogram, mfcc, chroma, gfcc

def get_data(dr):
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
            spectogram, mfcc, chroma, gfcc = extract_feats(y, sr, y1, sr1) #, gfcc spectogram, mfcc, chroma
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
    print(all_specs.shape)
    print(all_mfcc.shape)
    print(all_chroma.shape)
    print(all_gfcc.shape)
    return  all_specs, all_mfcc, all_chroma, all_gfcc, label_files

def check_data(dr, feat=None):
    os.chdir(DATA_PATH)
    os.chdir(dr)
    print(dr)
    if not Path('CNNLabels.npz').exists():
        spec, mfcc, chroma, gfcc, labels = get_data(dr) 
        np.savez('Spec.npz', spec)
        np.savez('MFCC40.npz', mfcc)
        np.savez('Chroma.npz', chroma)
        np.savez('GFCC40.npz', gfcc)
        np.savez('CNNLabels.npz', labels)
        print('All Data saved')
        data=None
    else:
        print('Loading feature data and labels')
        labels = np.load('CNNLabels.npz')['arr_0']
        print(feat)
        data = np.load(str(feat) + '.npz')['arr_0']
        #print('Please select a valid feature.')
        #  data = None
        print('Finished loading feature data and labels')
    return np.array(data), np.array(labels)


if __name__ == "__main__":
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay']
    os.chdir(DATA_PATH)
    for folder in os.listdir(os.getcwd()):
        check_data(folder)