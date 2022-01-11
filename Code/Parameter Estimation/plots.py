import os
import math
import librosa
import joblib
import librosa.display as ld
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
from spafe.features import gfcc as sgfcc
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from pathlib import Path
from cnnfeatextr import DATA_PATH

plt.rcParams["font.size"] = "30"

def learning_curve(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])  # Loss is mean squared error (=/= mean absolute error)
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(0.0, 0.2)
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def plot_cnn_features(data, feature):
    xlabel = 'Time in s'
    sr = 44100
    fig, ax = plt.subplots()
    if feature == 'Spec':
        data = librosa.amplitude_to_db(data, ref=np.max)
        ylabel = 'Frequency in Hz'
        y_axis= 'log'
        format = '%+2.0f dB'

    elif feature == 'MFCC40':
        y_axis=None
        ylabel = 'Coefficients'
        format = None
        plt.yticks([*range(0, 45, 5)])

    elif feature == 'GFCC40':
        y_axis=None
        ylabel = 'Coefficients'
        format = None
        plt.yticks([*range(0, 45, 5)])
        sr = sr*1,1445

    elif feature == 'Chroma':
        y_axis = 'chroma'
        ylabel = 'Pitch Class'

    img = librosa.display.specshow(data=data, x_axis='time', y_axis=y_axis, ax=ax, sr=sr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.colorbar(img, ax=ax, format=format)
    plt.show()


def feature_examples(file_name=os.path.join(os.path.dirname(__file__), '../..', 'Datasets/GEPE-GIM/Distortion/ag_G+K+B+D_Distortion_0.7_0.85_52_bp12_3.wav')):
    y, sr = librosa.load(file_name, sr=None)
    y_gfcc, sr_gfcc = librosa.load(file_name, sr=16000)
    spec = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    gfcc = sgfcc(sig=y_gfcc, fs=sr_gfcc, num_ceps=40, nfilts=80)

    for index, data in enumerate([spec, mfcc, chroma, gfcc]):
        feature_list = ['Spec', 'MFCC40', 'Chroma', 'GFCC40']
        plot_cnn_features(data, feature_list[index])


def noise_feature_plots(path = os.path.join(os.path.dirname(__file__), '../..', 'Datasets/GEPE-GIM')):
    fold_no = 5
    os.chdir(path)
    for fx in ['Distortion', 'Tremolo', 'SlapbackDelay']:
        os.chdir(fx)
        print(fx)
        for feature in ['Spec', 'MFCC40', 'Chroma', 'GFCC40']:
            print(feature)
            for factor in ('0.0', '0.001', '0.01', '0.05'):
                print('Noise Factor alpha = ' + factor)
                feature_data = joblib.load(feature + '_' + factor + '_' + str(fold_no) +'.pickle')
                plot_cnn_features(feature_data, feature)


def waveform(sample1=os.path.join(os.path.dirname(__file__), '../..', 'Datasets/GEPE-GIM/Tremolo/ag_G+K+B+D_Tremolo_1.0_0.15_52_bp12_-36.wav'), 
    sample2=os.path.join(os.path.dirname(__file__),'../..', 'Datasets/GEPE-GIM/Tremolo/ag_G+K+B+D_Tremolo_1.0_0.15_52_bp12_3.wav')):
    plt.rcParams["lines.linewidth"] = 0.2
    plt.rcParams["font.size"] = "22"
    os.chdir(DATA_PATH)
    os.chdir('Tremolo')
    y, sr = librosa.load(sample1, sr=None)
    y1, _ = librosa.load(sample2, sr=None)
    y = librosa.util.normalize(y)
    y1 = librosa.util.normalize(y1)
    time = np.linspace(0, 2, sr*2)
    for wav in [y, y1]:
        plt.xlabel('Time in s') 
        plt.ylabel('Amplitude')
        plt.plot(time, wav)
        plt.show()

waveform()
