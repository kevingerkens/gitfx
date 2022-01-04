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


def chroma_trem():
    file1 = 'C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/G+K+B+D/Tremolo/ag_G+K+B+D_Tremolo_1.0_0.4_52_bp12_-36.wav'
    file2 = 'C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/G+K+B+D/Tremolo/ag_G+K+B+D_Tremolo_1.0_0.4_52_bp12_-3.wav'
    y1, sr = librosa.load(file1, sr=None)
    y2, _ = librosa.load(file2, sr=sr)
    chroma1 = librosa.feature.chroma_stft(y=y1, sr=sr)
    chroma2 = librosa.feature.chroma_stft(y=y2, sr=sr)
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img1 = librosa.display.specshow(chroma1, y_axis='chroma', x_axis='time', ax=ax[0], sr=sr)
    img2 = librosa.display.specshow(chroma2, y_axis='chroma', x_axis='time', ax=ax[1], sr=sr)
    fig.colorbar(img1, ax=ax)
    ax[0].set(xlabel='Time in s')
    ax[1].set(xlabel='Time in s')
    #ax[0].set(title='Chromagram')
    plt.show()

def chroma_plot():
    plt.rcParams["font.size"] = "20"
    file_name = ('C:/Desktop/chromapiano.wav')
    y, sr = librosa.load(file_name, sr=44100)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax, sr=sr)
    fig.colorbar(img, ax=ax)
    plt.show()

def mel_scale():
    f = np.array([*range(1, 10000, 1)])
    mel = 2595*np.log10(1+f/700)
    plt.xscale('log')
    plt.xlabel('Frequenz (Hz)')
    plt.ylabel('Mel')
    plt.plot(mel)
    plt.show()

def window():
    plt.rcParams["font.size"] = "20"
    time = np.linspace(0, 10, 5000)
    y = np.sin(10*time)
    hann = np.hanning(2000)
    square = signal.boxcar(2000)
    w_square = np.concatenate((square, np.zeros(3000)))
    w_hann = np.concatenate((hann, np.zeros(3000)))
    for win in [w_hann, w_square]:
        y_win = y*win
        plt.title('Hann Window and Windowed Sine Wave')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.plot(time, y_win, label='Windowed Sine Wave')
        plt.plot(time, win, label='Hann Window')
        plt.legend()
        plt.show()

def mel_fb():
    #plt.rcParams["font.size"] = "20"
    melfb = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=10)
    x = np.array([*range(0, 10250, 10)])
    ymax = np.max(melfb[0])
    for filter in melfb:
        filter = (filter*ymax/np.max(filter))/ymax
        plt.plot(x, filter, color='k')
    #plt.title('Mel Filterbank')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('$X^{mel}$')
    plt.show()

def feat_ex():
    os.chdir(DATA_PATH)
    os.chdir('Distortion')
    sr = 44100
    feats = ['Spec', 'MFCC40', 'GFCC40', 'Chroma']
    spec = joblib.load(feats[0] + '_example.pickle')
    mfcc = joblib.load(feats[1] + '_example.pickle')
    gfcc = joblib.load(feats[2] + '_example.pickle')
    chroma = joblib.load(feats[3] + '_example.pickle')
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(spec,
                                                       ref=np.max), x_axis='time', y_axis='log', ax=ax, sr=sr)
    plt.xlabel('Time in s')  
    plt.ylabel('Frequency in Hz')    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.show()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, sr=sr)     
    fig.colorbar(img, ax=ax)
    plt.yticks([*range(0, 45, 5)])
    plt.xlabel('Time in s')  
    plt.ylabel('Coefficients') 
    plt.show()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax, sr=sr)     
    fig.colorbar(img, ax=ax)
    plt.xlabel('Time in s')  
    plt.ylabel('Pitch Class') 
    plt.show()    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(gfcc, x_axis='time', ax=ax, sr=sr*1.98/1.73)
    plt.yticks([*range(0, 45, 5)])
    plt.xlabel('Time in s')  
    plt.ylabel('Coefficients') 
    fig.colorbar(img, ax=ax)       
    plt.show()    

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

def gamma_fb():
    #plt.rcParams["font.size"] = "18"
    gammafb = gammatone_filter_banks(nfilts=10)
    ymax = np.max(gammafb[0])
    x = np.array([*range(0, 8224, 32)])
    for filter in gammafb:
        filter = (filter*ymax/np.max(filter))/ymax
        plt.plot(x, filter, color='k')
    #plt.title('Gammatone Filterbank')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('$G$')
    plt.show()

def plot_rms():
    #plt.rcParams["font.size"] = "20"
    file_name = "C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/G+K+B+D/Distortion/ag_G+K+B+D_Distortion_0.7_0.3_52_bp12_3.wav"
    y, sr = librosa.load(file_name, sr=None)
    y = librosa.util.normalize(y)
    rms = librosa.feature.rms(y)[0]
    rms_db = 10*np.log(np.where(rms < 0.005, 0.005, rms))
    plt.ylabel('RMS Energy in dB')
    plt.xlabel('Time s')
    x_axis = librosa.frames_to_time(range(0, len(rms_db)), sr=sr)
    #plt.title('RMS Energy of a Full Mix Sample with Distortion Effect')
    plt.plot(x_axis, rms_db)
    plt.show()

def plot_cqt():
    plt.rcParams["font.size"] = "28"
    file_name = "C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/G+K+B+D/Distortion/ag_G+K+B+D_Distortion_0.7_0.3_52_bp12_3.wav"
    y, sr = librosa.load(file_name, sr=None)
    y = librosa.util.normalize(y)
    fig, ax = plt.subplots()
    C = np.abs(librosa.cqt(y, sr=sr))
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
    #ax.set_title('Constant-Q power spectrum')
    plt.xlabel("Time in s")
    plt.ylabel("Note")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

def plot_feats():
    plt.rcParams["font.size"] = "36"
    os.chdir(DATA_PATH)
    os.chdir('Distortion')
    file_name = 'ag_G+K+B+D_Distortion_0.7_0.85_52_bp12_3.wav'
    y, sr = librosa.load(file_name, sr=None)
    y1, sr1 = librosa.load(file_name, sr=16000)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y, sr=sr)
    spec = np.abs(librosa.stft(y))
    gfcc = sgfcc.gfcc(y1, fs=sr1, nfilts=80, num_ceps=40)
    gfcc = np.transpose(gfcc)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(spec,
                                                       ref=np.max), x_axis='time', y_axis='log', ax=ax, sr=sr)
    plt.title('Spectogram')   
    plt.xlabel('Time in s')  
    plt.ylabel('Frequency in Hz')    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.show()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, sr=sr)     
    fig.colorbar(img, ax=ax)
    plt.title('MFCC')
    plt.yticks([*range(0, 45, 5)])
    plt.xlabel('Time in s')  
    plt.ylabel('Coefficients') 
    plt.show()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax, sr=sr)     
    fig.colorbar(img, ax=ax)
    plt.xlabel('Time in s')  
    plt.ylabel('Pitch Class') 
    plt.title('Chromagram')  
    plt.show()    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(gfcc, x_axis='time', ax=ax, sr=sr*1.98/1.73)
    plt.title('GFCC')
    plt.yticks([*range(0, 45, 5)])
    plt.xlabel('Time in s')  
    plt.ylabel('Coefficients') 
    fig.colorbar(img, ax=ax)       
    plt.show()                  

def spec_noise(factor):
    plt.rcParams["font.size"] = 20
    file_name = 'C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/G+K+B+D/Distortion/ag_G+K+B+D_Distortion_0.7_0.75_52_bp12_0.wav'
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
    file_name = 'C:/Desktop/Uni/Studienarbeit/Audio/NewDataset/G+K+B+D/Distortion/ag_G+K+B+D_Distortion_0.7_0.75_52_bp12_0.wav'
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

def actfunct():
    x1 = np.linspace(-7.0, 7.1, 500)
    x2 = np.linspace(-3.0, 3.1, 500)
    sig = 1/(1+np.exp(-x1))
    relu = np.maximum(0, x2)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 9))
    axes[0].set_title(r'$\varphi_{sig}$ (x) = $\frac{1}{1+e^{-x}}$', fontsize=25)
    axes[1].set_title(r'$\varphi_{ReLU}$ (x) = $max(0, x)$', fontsize=25)
    for ax in axes.flat:
        ax.set(xlabel='x', ylabel=r'$\varphi$')
    axes[0].plot(x1, sig)
    axes[1].plot(x2, relu)
    fig.tight_layout()
    plt.show()

def clipped():
    plt.rcParams["font.size"] = "20"
    plt.rcParams['lines.linewidth'] = 2.5
    t = np.linspace(-1, 1, 1000)
    x = np.sin(2*np.pi*t)
    y_soft = 0.9*np.tanh(1.5*x)
    y_hard = np.clip(x, a_min=-0.7, a_max=0.7)
    plt.plot(t, x, label='Input Signal')
    plt.plot(t, y_soft, '--',  label = 'Soft-Clipped Output Signal')
    plt.plot(t, y_hard, ':', label = 'Hard-Clipped Output Signal')
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.show()

def spectogram():
    #plt.rcParams["font.size"] = "18"
    os.chdir(DATA_PATH)
    os.chdir('Distortion')
    file_name = 'ag_G+K+B+D_Distortion_1.0_0.7_52_bp12_3.wav'
    y, sr = librosa.load(file_name, sr=None)
    spec = np.abs(librosa.stft(y))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax, sr=sr)
    #ax.set_title('Power spectrogram')
    plt.xlabel('Time in s')
    plt.ylabel('FRequency in Hz')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.show()

def specfeats():
    #plt.rcParams["font.size"] = "18"
    os.chdir(DATA_PATH)
    os.chdir('Distortion')
    file_name = 'ag_G+K+B+D_Distortion_0.7_0.85_52_bp12_3.wav'
    y, sr = librosa.load(file_name, sr=None)
    scontr = librosa.feature.spectral_contrast(y, sr=sr, n_bands=6)
    scentr = librosa.feature.spectral_centroid(y, sr=sr)
    fig, ax = plt.subplots()
    times = librosa.times_like(scentr)
    S, phase = librosa.magphase(librosa.stft(y=y))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax, sr=sr)
    ax.plot(times, scentr.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    ax.set(xlabel='Time in s')
    ax.set(ylabel='Frequency in Hz')
    #ax.set(title='log Power spectrogram with Spectral Centroid')
    plt.show()
    fig, ax = plt.subplots()
    times = librosa.times_like(scontr)
    img1 = librosa.display.specshow(scontr, x_axis='time', sr=sr)
    fig.colorbar(img1, ax=ax, format='%+2.0f dB')
    #ax.set(title='Spectral Contrast with 7 Frequency Bands')
    ax.set(xlabel='Time in s')
    ax.set(ylabel='Frequency Bands')
    plt.show()

def waveform():
    plt.rcParams["lines.linewidth"] = 0.2
    plt.rcParams["font.size"] = "22"
    os.chdir(DATA_PATH)
    os.chdir('Tremolo')
    y, sr = librosa.load('ag_G+K+B+D_Tremolo_1.0_0.15_52_bp12_-36.wav', sr=None)
    y1, sr = librosa.load('ag_G+K+B+D_Tremolo_1.0_0.15_52_bp12_3.wav', sr=None)
    y = librosa.util.normalize(y)
    y1 = librosa.util.normalize(y1)
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 9))
    #axes[0].set_title('Tremolo Sample with Lowest Mix Volume', fontsize=25)
    #axes[1].set_title('Tremolo Sample with Highest Mix Volume', fontsize=25)

    time = np.linspace(0, 2, 88200)
    #for ax in axes.flat:
    for wav in [y, y1]:
        plt.xlabel('Time in s') 
        plt.ylabel('Amplitude')
        plt.plot(time, wav)
        plt.show()


if __name__ == '__main__':
    waveform()