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

DATA_PATH = Path('C:/Desktop/uni/Studienarbeit/Audio/NewDataset/G+K+B+D')    \
    if platform.system() == 'Windows' else os.path.join(os.path.expanduser('~'), 'tmp/Clf/Gitarre monophon/Samples')
TEST_PATH = Path('C:/Desktop/uni/Studienarbeit/Audio/NewDataset/G+K+B+D')    \
    if platform.system() == 'Windows' else os.path.join(os.path.expanduser('~'), 'tmp/Clf/Gitarre monophon/Samples')
#PARAM_STEPS = 21

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
        # plots.lin_regression(feat, lin_coeff)

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
        # plots.rfft(rfft_norm)
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
    # plots.phase_error_unwrapped(phase_fmax_straight_t, coeff, x_axis_t)

    return linregerr_t

def librosa_features(y, sr):
    #mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_pos = (mfcc - np.min(mfcc))
    mfcc_norm = mfcc_pos / np.max(mfcc_pos) - np.mean(mfcc_pos)
    mfcc_delta = librosa.feature.delta(mfcc_norm)
    #spectral features
    spec_contr = librosa.feature.spectral_contrast(y=y, sr=sr)
    #spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    #zero crossing rate
    zero_cr = librosa.feature.zero_crossing_rate(y=y)
    zero_cr_delta = librosa.feature.delta(zero_cr)
    #rms energy
    rms = librosa.feature.rms(y=y)
    rms *= 1/rms.max()
    rms_delta = librosa.feature.delta(rms)
    #chroma features
    #chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    #chroma_delta = librosa.feature.delta(chroma)


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
    #pitch_onset = int((onset_sample/sample.sig.shape[0]) * pitch_curve.shape[0])
    #pitch_curve = pitch_curve[pitch_onset:]
    #voice_prob = voice_prob[pitch_onset:]
    # plots.pitch_voiced_curve(pitch_curve, voice_prob)

    pitch_curve = np.reshape(pitch_curve, [1, pitch_curve.shape[0]])
    # plots.pitch(pitch_curve)
    voice_prob = np.reshape(voice_prob, [1, voice_prob.shape[0]])

    harmonicity = praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = praat.call(harmonicity, "Get mean", 0, 0)
    #p_features = np.concatenate((pitch_curve, voice_prob, hnr))

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
       
def read_data(dr, test=False):
    #extract data and labels
    if test==False:
        os.chdir(DATA_PATH)
    else:
        os.chdir(TEST_PATH)
    os.chdir(dr)
    print(dr)
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

def check_data(dr, test=False):
    #extracts features and labels or loads data, if already existent
    if test==False:
        os.chdir(DATA_PATH)
    else:
        os.chdir(TEST_PATH)
    os.chdir(dr)
    if not Path('Data_unsc.npz').exists():
        feats, labels = read_data(dr, test=test)
        np.savez('Data_unsc.npz', X = feats, y = labels)
    else:
        print('Loading feature data and labels...')
        data = np.load('Data_unsc.npz')
        feats = data['X']
        labels = data['y']

    return feats, labels
        
def make_dataframes():
    #creates pandas dataframe containing labels and features of all samples in each fx folder
    os.chdir(DATA_PATH)
    for dr in os.listdir(os.getcwd()):
        os.chdir(dr)
        print(dr)
        par_names = param_names(dr)
        feats, labels = check_data(dr)
        print('Labels shape: ', labels.shape)
        print('Feature shape: ', feats.shape)
        joined_data = np.concatenate((labels, feats), axis=1)
        print(joined_data.shape)
        new_cols = ['Mix', 'Effect']                                                    #column names for dataframe
        new_cols.extend(par_names)
        new_cols.append('Varied Parameter')
        new_cols.append('Pitch')
        data_df = pd.DataFrame(data = joined_data)                                      #convert data to pandas dataframe
        new_names_map = {data_df.columns[i]:new_cols[i] for i in range(len(new_cols))}  #assign column names
        data_df.rename(new_names_map, axis=1, inplace=True)
        os.chdir(DATA_PATH)
        os.chdir(dr)
        data_df.to_csv('data_unsc_df.csv', sep=';')                                          #save as csv file
        os.chdir(DATA_PATH)

if __name__ == '__main__':
    os.chdir(DATA_PATH)
    for dr in os.listdir(os.getcwd()):
        check_data(dr, test=False)
        os.chdir(DATA_PATH)
