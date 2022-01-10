import pickle
import os
import platform
import numpy as np
import pandas as pd
import librosa
import librosa.display
import parselmouth
from parselmouth import praat
from parselmouth.praat import call
from pathlib import Path
from sklearn import preprocessing
from cnn_features_classification import check_dataset

dataset_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..', 'Datasets')

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
        }
    
    par_names = names[dr]
    return par_names
    

def get_label(file_name):
    label_file = file_name[:-4] + '.pickle'
    with open(label_file, 'rb') as handle:
        label = pickle.load(handle)

    return label


def extract_functionals(features):
    functionals = []
    for feat in features:
        functionals.append(feat.max())
        functionals.append(feat.min())
        functionals.append(feat.mean())
        functionals.append(feat.std())

        lin_coeff, lin_residual, _, _, _ = np.polyfit(np.arange(len(feat)), feat, 1, full=True)
        functionals.extend(lin_coeff)
        functionals.append(lin_residual)

        quad_coeff, quad_residual, _, _, _ = np.polyfit(np.arange(len(feat)), feat, 2, full=True)
        functionals.extend(quad_coeff)
        functionals.append(quad_residual)

        feat_no_offset = feat - np.average(feat)
        feat_windowed = feat_no_offset * np.hanning(len(feat_no_offset))
        feat_int = np.pad(feat_windowed, (0, 1024 - len(feat_windowed) % 1024), 'constant')

        rfft = np.fft.rfft(feat_int)
        rfft_norm = np.abs(rfft) * 4 / 1024
        rfft_norm[:16] = np.zeros(16)
        rfft_max = np.max(rfft_norm)
        functionals.append(rfft_max)

    functionals = np.hstack(functionals)
    
    return functionals


def phase_fmax(sig):
    """Analyses phase error of frequency bin with maximal amplitude
        compared to pure sine wave"""
    D = librosa.stft(y=sig, hop_length=256)[20:256]
    S, P = librosa.core.magphase(D)
    phase = np.angle(P)
    # plots.phase_spectrogram(phase)

    spec_sum = S.sum(axis=1)
    max_bin = spec_sum.argmax()
    phase_freq_max = phase[max_bin]
    # plots.phase_fmax(phase_freq_max)

    S_max_bin_mask = S[max_bin]
    thresh = S[max_bin].max()/8
    phase_freq_max = np.where(S_max_bin_mask > thresh, phase_freq_max, 0)
    phase_freq_max_t = np.trim_zeros(phase_freq_max)  # Using only phase with strong signal

    phase_fmax_straight_t = np.copy(phase_freq_max_t)
    diff_mean_sign = np.mean(np.sign(np.diff(phase_freq_max_t)))
    if diff_mean_sign > 0:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i-1]) > np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] += 2*np.pi
    else:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i - 1]) < np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] -= 2 * np.pi

    x_axis_t = np.arange(0, len(phase_fmax_straight_t))
    coeff = np.polyfit(x_axis_t, phase_fmax_straight_t, 1)
    linregerr_t = np.copy(phase_fmax_straight_t)
    linregerr_t -= (coeff[0] * x_axis_t + coeff[1])
    linregerr_t = np.reshape(linregerr_t, (1, len(linregerr_t)))

    return linregerr_t


def librosa_features(y, sr):

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_pos = (mfcc - np.min(mfcc))
    mfcc_norm = mfcc_pos / np.max(mfcc_pos) - np.mean(mfcc_pos)
    mfcc_delta = librosa.feature.delta(mfcc_norm)

    spec_contr = librosa.feature.spectral_contrast(y=y, sr=sr)

    zero_cr = librosa.feature.zero_crossing_rate(y=y)
    zero_cr_delta = librosa.feature.delta(zero_cr)

    rms = librosa.feature.rms(y=y)
    rms *= 1/rms.max()
    rms_delta = librosa.feature.delta(rms)



    l_features = np.concatenate((mfcc_norm, mfcc_delta, spec_contr,
                                zero_cr, zero_cr_delta, rms, rms_delta))

    l_functionals = extract_functionals(l_features)

    return l_functionals


def praat_features(y, file_name, path):
    snd = parselmouth.Sound(os.path.join(path, file_name))

    phase_res = phase_fmax(y)

    pitch = snd.to_pitch().to_array()
    pitch_curve, voice_prob = zip(*pitch[0][:])
    pitch_curve = np.array(pitch_curve)
    voice_prob = np.array(voice_prob)

    pitch_curve = np.reshape(pitch_curve, [1, pitch_curve.shape[0]])
    voice_prob = np.reshape(voice_prob, [1, voice_prob.shape[0]])

    harmonicity = praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = praat.call(harmonicity, "Get mean", 0, 0)

    p_functionals = extract_functionals(phase_res)
    p_functionals = np.append(p_functionals, extract_functionals(pitch_curve))
    p_functionals = np.append(p_functionals, extract_functionals(voice_prob))
    p_functionals = np.append(p_functionals, hnr)

    return p_functionals


def extract_features(file_name, path):
    #extracts librosa and praat features, then calculates functionals
    y, sr = librosa.load(file_name, sr=None)
    y = librosa.util.normalize(y)[:sr*2]
    #extracting librosa features
    l_functionals = librosa_features(y, sr)
    #extracting praat features and phase error of frequency bin with maximal amplitudecompared to pure sine wave
    p_functionals = praat_features(y, file_name, path)
    #join features sets
    functionals = np.append(l_functionals, p_functionals)

    return functionals

       
def read_data(dr):
    check_dataset()
    print('Extracting feature data and labels')
    train_data, train_labels = [], []
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith(".wav"):
            #label = get_label(file_name)
            #print(file_name)
            label = np.array(file_name[:-4].split('_'))
            print(label)
            train_labels.append(np.hstack(label))
            path = os.getcwd()
            functionals = extract_features(file_name, path)
            train_data.append(np.hstack(functionals))
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    return train_data, train_labels


def check_data(dr):
    #extracts features and labels or loads data, if already existent
    os.chdir(dr)
    print(dr)
    if not Path('Data_unsc.npz').exists():
        feats, labels = read_data(dr)
        np.savez('Data_unsc.npz', X = feats, y = labels)
    else:
        print('Loading feature data and labels...')
        data = np.load('Data_unsc.npz')
        feats = data['X']
        labels = data['y']

    return feats, labels
        

if __name__ == '__main__':
    for dataset in ['Classification', 'IDMT']: 
        os.chdir(dataset_path)
        os.chdir(dataset)
        print(dataset)
        for dr in os.listdir(os.getcwd()):
            check_data(dr)
            os.chdir('..')
