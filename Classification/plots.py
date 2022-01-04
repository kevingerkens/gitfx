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
from cnn_features_classification import DATA_PATH

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


def feature_examples(file_name=os.path.join(os.path.dirname(__file__), '../..', 'Datasets/Parameter Estimation/Distortion/ag_G+K+B+D_Distortion_0.7_0.85_52_bp12_3.wav')):
    y, sr = librosa.load(file_name, sr=None)
    y_gfcc, sr_gfcc = librosa.load(file_name, sr=16000)
    spec = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    gfcc = sgfcc(sig=y_gfcc, fs=sr_gfcc, num_ceps=40, nfilts=80)

    for index, data in enumerate([spec, mfcc, chroma, gfcc]):
        feature_list = ['Spec', 'MFCC40', 'Chroma', 'GFCC40']
        plot_cnn_features(data, feature_list[index])


def noise_feature_plots(path = os.path.join(os.path.dirname(__file__), '../..', 'Datasets/Parameter Estimation')):
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

  


def chroma_pickle_noise():
    data_path = 'C:/Desktop/Uni/Studienarbeit/Audio/Distortion'
    os.chdir(data_path)
    files = ['chroma1.pickle', 'noise_chroma_0.0011.pickle', 'noise_chroma_0.011.pickle']
    sr= 44100
    S = joblib.load(files[0])
    S = np.reshape(S, (12, 173))
    print(np.amin(S))
    print(np.amax(S))
    #S = S - np.amin(S)
    S_noise = joblib.load(files[1])
    S_noise = np.reshape(S_noise, (12, 173))
    S_noise2 = joblib.load(files[2])
    S_noise2 = np.reshape(S_noise2, (12, 173))
    #S_noise = S_noise - np.amin(S_noise)
    fig, ax = plt.subplots(nrows=3, sharex=True)
    img1 = librosa.display.specshow(S, x_axis='time', y_axis='chroma', ax=ax[0], sr=sr)
    #ax[0].set(title='Spectogram Without Noise')
    ax[0].set(xlabel='Time in s')  
    ax[0].set(ylabel='Pitch Class')  
    fig.colorbar(img1, ax=[ax[0]])
    img2 = librosa.display.specshow(S_noise,
                                                     x_axis='time', y_axis='chroma', ax=ax[1], sr=sr)
    #ax[1].set(title='Spectogram With Noise')
    ax[1].set(xlabel='Time in s')  
    ax[1].set(ylabel='Pitch Class')  
    fig.colorbar(img2, ax=[ax[1]])
    img3 = librosa.display.specshow(S_noise2,
                                                     x_axis='time', y_axis='chroma', ax=ax[2], sr=sr)
    #ax[1].set(title='Spectogram With Noise')
    ax[2].set(xlabel='Time in s')  
    ax[2].set(ylabel='Pitch Class')  
    fig.colorbar(img2, ax=[ax[2]])
    plt.show()


def spec_pickle_noise():
    data_path = 'C:/Desktop/Uni/Studienarbeit/Audio/Distortion'
    os.chdir(data_path)
    files = ['spec1.pickle', 'noise_spec1.pickle']
    sr= 44100
    S = joblib.load(files[0])
    S = np.reshape(S, (256, 173))
    S_noise = joblib.load(files[1])
    S_noise = np.reshape(S_noise, (256, 173))
    fig, ax = plt.subplots(nrows=2, sharex = True, sharey=True)
    img1 = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max), x_axis='time', y_axis='log', ax=ax[0], sr=sr)
    #ax[0].set(title='Spectogram Without Noise')
    ax[0].set(xlabel='Time in s')  
    ax[0].set(ylabel='Frequency in Hz')  
    fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
    img2 = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max), x_axis='time', y_axis='log', ax=ax[1], sr=sr)
    #ax[1].set(title='Spectogram With Noise')
    ax[1].set(xlabel='Time in s')  
    ax[1].set(ylabel='Frequency in Hz')  
    fig.colorbar(img2, ax=[ax[1]], format='%+2.0f dB')
    plt.show()
    # for file in files:
    #     print(file)
    #     spec = joblib.load(file)
    #     spec = np.reshape(spec, (256, 173))
    #     #spec = np.abs(spec)
    #     print(spec.shape)
    #     sr = 44100
    #     fig, ax = plt.subplots()
    #     img = librosa.display.specshow(librosa.amplitude_to_db(spec,
    #                                                 ref=np.max), x_axis='time', y_axis='log', ax=ax, sr=sr) 
    #     plt.xlabel('Time in s')  
    #     plt.ylabel('Frequency in Hz')
    #     fig.colorbar(img, ax=ax, format='%+2.0f dB')
    #     plt.show()  



def spec_noise(factor):
    plt.rcParams["font.size"] = 20
    file_name = 'C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/Distortion/ag_G+K+B+D_Distortion_0.7_0.75_52_bp12_0.wav'
    y, sr = librosa.load(file_name, sr=None)
    y = librosa.util.normalize(y)
    S = librosa.stft(y)
    #mfcc = mfcc/np.max(np.abs(mfcc))
    noise = np.random.normal(0, np.amax(np.abs(S))*factor, S.shape)
    S_noise = S + noise
    print(S_noise)
    y_noise = librosa.istft(S_noise)
    noise_file = 'C:/Desktop/distnoise_' + str(factor) +  '.wav'
    #write(noise_file, sr, y_noise)
    fig, ax = plt.subplots(nrows=2)
    img1 = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max), x_axis='time', y_axis='log', ax=ax[0], sr=sr)
    #ax[0].set(title='Spectogram Without Noise')
    ax[0].set(xlabel='Time in s')  
    ax[0].set(ylabel='Frequency in Hz')  
    fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
    img2 = librosa.display.specshow(librosa.amplitude_to_db(S_noise,
                                                       ref=np.max), x_axis='time', y_axis='log', ax=ax[1], sr=sr)
    #ax[1].set(title='Spectogram With Noise')
    ax[1].set(xlabel='Time in s')  
    ax[1].set(ylabel='Frequency in Hz')  
    fig.colorbar(img2, ax=[ax[1]], format='%+2.0f dB')
    plt.show()


def mfcc_noise():
    plt.rcParams["font.size"] = 20
    file_name = 'C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/Distortion/ag_G+K+B+D_Distortion_0.7_0.75_52_bp12_0.wav'
    y, sr = librosa.load(file_name, sr=None)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=10)
    #mfcc = mfcc/np.max(np.abs(mfcc))
    noise = np.random.normal(0, np.amax(np.abs(mfcc))*0.01, mfcc.shape)
    mfcc_noise = mfcc + noise
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img1 = librosa.display.specshow(mfcc, x_axis='time', ax=ax[0], sr=sr)
    ax[0].set(title='MFCC Without Noise')
    fig.colorbar(img1, ax=[ax[0]])
    img2 = librosa.display.specshow(mfcc_noise, x_axis='time', ax=ax[1], sr=sr)
    ax[1].set(title='MFCC With Noise')
    fig.colorbar(img2, ax=[ax[1]])
    plt.show()


def waveform(sample1=os.path.join(os.path.dirname(__file__), '../..', 'Datasets/Parameter Estimation/Tremolo/ag_G+K+B+D_Tremolo_1.0_0.15_52_bp12_-36.wav'), 
    sample2=os.path.join(os.path.dirname(__file__),'../..', 'Datasets/Parameter Estimation/Tremolo/ag_G+K+B+D_Tremolo_1.0_0.15_52_bp12_3.wav')):
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