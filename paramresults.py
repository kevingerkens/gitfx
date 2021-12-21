import os
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
#from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
from spafe.features import gfcc as sgfcc
from cnnfeatextr import DATA_PATH, param_names

plt.rcParams["font.size"] = 28
plt.rcParams["figure.figsize"] = (8, 6)

NOISE_PATH = Path("C:/Desktop/Uni/Studienarbeit/Test/Noise")
SCALE_PATH = Path("C:/Desktop/Uni/Studienarbeit/Audio/Scaletest")
EXAMPLE_PATH = Path("C:/Desktop/Uni/Studienarbeit/Audio/Examples")

pitch_dict = {
    '40' : 'Low E',
    '52' : 'Middle E',
    '0' : 'Low',
    '12' : 'High'
}

mult_dict = {
    'DistortionTremolo' : ['Distortion', 'Tremolo'],
    'DistortionSlapbackDelay' : ['Distortion', 'SlapbackDelay'],
    'TremoloSlapbackDelay' : ['Tremolo', 'SlapbackDelay'],
    'DistortionTremoloSlapbackDelay' : ['Distortion', 'Tremolo', 'SlapbackDelay']
}

bp_dict = {
    0 : 'E',
    1 : 'F',
    2 : 'F#',
    3 : 'G',
    4 : 'G#', 
    5 : 'A',
    6 : 'A#',
    7 : 'B', 
    8 : 'C',
    9 : 'C#',
    10 : 'D',
    11 : 'D#', 
    12 : 'High E'
}

def spec(dr):
    os.chdir(EXAMPLE_PATH)
    os.chdir(dr)
    #os.chdir(dr + ' Samples')
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith('.wav') and 'REF' in file_name:
            print(file_name)
            y, sr = librosa.load(file_name, sr=None)
            spec = np.abs(librosa.stft(y))
            fig, ax = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
                                    y_axis='log', x_axis='time', ax=ax, sr=sr)
            plt.xlabel('Time in s')
            plt.ylabel('Frequency in Hz')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

        plt.show()   

def mfcc(dr):
    os.chdir(EXAMPLE_PATH)
    os.chdir(dr)
    #os.chdir(dr + ' Samples')
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith('.wav') and 'REF' in file_name:
            print(file_name)
            y, sr = librosa.load(file_name, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, sr=sr)
            plt.xlabel('Time in s')
            plt.ylabel('Coefficients')
            plt.yticks([*range(0, 45, 5)])
            fig.colorbar(img, ax=ax)

        plt.show()   

def chroma(dr):
    os.chdir(EXAMPLE_PATH)
    os.chdir(dr)
    #os.chdir(dr + ' Samples')
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith('.wav') and 'REF' in file_name:
            print(file_name)
            y, sr = librosa.load(file_name, sr=None)
            mfcc = librosa.feature.chroma_stft(y=y, sr=sr)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mfcc, x_axis='time', y_axis='chroma', ax=ax, sr=sr)
            plt.xlabel('Time in s')
            #plt.ylabel('Coefficients')
            #plt.yticks([*range(0, 45, 5)])
            fig.colorbar(img, ax=ax)

        plt.show()   

def gfcc(dr):
    os.chdir(EXAMPLE_PATH)
    os.chdir(dr)
    #os.chdir(dr + ' Samples')
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith('.wav') and 'REF' in file_name:
            print(file_name)
            y, sr = librosa.load(file_name, sr=16000)
            gfcc = sgfcc.gfcc(y, fs=sr, nfilts=80, num_ceps=40)
            gfcc = gfcc.T
            fig, ax = plt.subplots()
            img = librosa.display.specshow(gfcc, x_axis='time', ax=ax, sr=50.5/16*sr)
            plt.xlabel('Time in s')
            plt.ylabel('Coefficients')
            plt.yticks([*range(0, 45, 5)])
            fig.colorbar(img, ax=ax)

        plt.show()   

def examples(dr, feat):
    print(dr)
    print(feat)
    os.chdir(EXAMPLE_PATH)
    os.chdir(dr)
    os.chdir('Results')
    df = get_df('df_' + feat + '_2_3_3_64_6_128_Examples.pickle')
    df['Labels'] = df['Labels'].astype(str)
    grouped = df.groupby('Labels')
    for name, group in grouped:
        print(name)
        print(group['Mix Volume'].tolist()[0])
        print(group['Prediction'].mean())
        print(group['Error'].mean())
        print(group['Pitch'])
        print(group['BP'])
        print(group['VST'])
        print('\n')


def get_df(file_name):
    old_cols = ['Prediction', 'Error', 'Labels', 'Mix Volume']
    new_cols = ['Prediction', 'Error', 'Labels', 'VST', 'Mix', 'Fx', 'Pitch', 'BP', 'Mix Volume']
    df = pd.read_pickle(file_name)
    df.columns = old_cols
    df = df.explode(['Prediction', 'Error', 'Labels', 'Mix Volume'])
    df['Mix Volume'].str.split(expand=True)
    df = pd.concat([df[old_cols[:-1]], df['Mix Volume'].apply(pd.Series)], axis=1)
    df.columns = new_cols
    df.index = [*range(len(df.index))]
    df['Mix Volume'] = df['Mix Volume'].astype(float)

    return df


def get_nn_values(file_name):
    nn_setting = file_name[:-7].split('_')[1:]
    if len(nn_setting) > 6:
        batch_size = nn_setting[-1]
        if len(batch_size) > 3:
            batch_size = batch_size[:-1]
    else:
        batch_size = 64

    nn_dict = { 'Feature': nn_setting[0],
                'Number of Conv2D Layers': nn_setting[1],
                'Kernel Size': nn_setting[2],
                'Number of Dense Layers': nn_setting[3],
                'Number of Dense Layer Neurons': nn_setting[4],
                'Number of Conv2D  1st Layer Filters': nn_setting[5],
                'Number of Conv2D  2nd Layer Filters': str(int(nn_setting[5])*2),
                'Batch Size': batch_size}
    for item in nn_dict.items():
        print(item)
    print('\n')
    return nn_dict


def j_byvol(dr, par_names):
    old_cols = ['Prediction', 'Error', 'Labels', 'Mix Volume']
    new_cols = ['Prediction', 'Error', 'Labels', 'VST', 'Mix', 'Fx', 'Pitch', 'BP', 'Mix Volume']
    df = pd.read_pickle('df_j.pickle')
    df.columns = old_cols
    df = df.explode(['Prediction', 'Error', 'Labels', 'Mix Volume'])
    df['Mix Volume'].str.split(expand=True)
    df = pd.concat([df[old_cols[:-1]], df['Mix Volume'].apply(pd.Series)], axis=1)
    df.columns = new_cols
    df.index = [*range(len(df.index))]
    df['Mix Volume'] = df['Mix Volume'].astype(float)
    me_byvol = []
    for vol, df_group in df.groupby('Mix Volume'):
        error_vol = np.array(df_group['Error'].tolist())
        me_vol = list(np.mean(error_vol, axis=0))
        me_vol.append(vol)
        me_byvol.append(me_vol)
    me_byvol = np.array(me_byvol)
    plot_error_by_vol(me_byvol, par_names, dr, 'Jürgens et al.')


def get_pca_error(dr):
    os.chdir(DATA_PATH)
    os.chdir(dr)
    with open('NNAbsoluteErrorReluSig10.pickle', 'rb') as handle:
        error = np.array(joblib.load(handle))
    shape = error.shape
    error = np.reshape(error, (shape[0]*shape[1], shape[2]), order='F')
    return np.array(error)


def get_j_error(dr):
    os.chdir(DATA_PATH)
    os.chdir(dr)
    with open('NNAbsoluteErrorJuergens.pickle', 'rb') as handle:
        error = joblib.load(handle)
    error = np.array(error)
    shape = error.shape
    abs_error = np.reshape(error, (shape[0]*shape[1], shape[2]), order='F')
    fold_errors = []
    for index in range(5):
        fold_mean = np.mean(abs_error[int(index*len(abs_error)/5):int(((index+1)*len(abs_error)/5)-1)], axis=0)
        fold_errors.append(fold_mean)
    std = np.std(fold_errors, axis=0)
    ki = 1.96*(std/np.sqrt(5))

    return abs_error, ki


def plot_error_over_pitch(df, dr, feat):
    df_pitch = df.groupby('Pitch')
    par_names = param_names(dr)
    error = []
    pitches = []
    for pitch, group in df_pitch:
        group_error = np.transpose(np.array(group['Error'].tolist()))
        pitches.append(pitch)
        error.extend(list(group_error))
    mean_error = np.mean(np.array(error), axis=1)
    print('Mean Error by Guitar Pitch: ' + str(mean_error))
    print(np.array(error).shape)
    plt.boxplot(error, whis=1.5)
    #plt.title('Absolute Error for ' + dr + ' by Guitar Pitch, Feature: ' + feat)
    xlabels = []    
    for pitch in pitches:
        for par in par_names:
            xlabels.append(pitch_dict[pitch] + ' ' + par)
    plt.xticks([*range(1, len(par_names)*len(pitches)+1)], xlabels)
    xmax = len(par_names) * len(pitches) + 0.5
    plt.hlines(y=0.05, xmin=0.5, xmax=xmax, colors='red', linestyles='--')
    plt.ylabel('Absolute FP-Error')
    plt.text(1.5, 0.06, 'Estimated Human\nSetup Error', fontsize=14, horizontalalignment='center')
    plt.show()


def plot_error_over_bppitch(df, dr, feat):
    df_pitch = df.groupby('BP')
    par_names = param_names(dr)
    error = []
    pitches = []
    for pitch, group in df_pitch:
        group_error = np.transpose(np.array(group['Error'].tolist()))
        pitches.append(pitch)
        error.extend(list(group_error))
    mean_error = np.mean(np.array(error), axis=1)
    print('Mean Error by Bass and Piano Pitch: ' + str(mean_error))
    print(np.array(error).shape)
    plt.boxplot(error, whis=1.5)
    #plt.title('Absolute Error for ' + dr + ' by Piano and Bass Pitch, Feature: ' + feat)
    xlabels = []    
    for pitch in pitches:
        for par in par_names:
            xlabels.append(pitch_dict[pitch[2:]] + ' Pitch ' + ' ' + par)
    plt.xticks([*range(1, len(par_names)*len(pitches)+1)], xlabels)
    xmax = len(par_names) * len(pitches) + 0.5
    plt.hlines(y=0.05, xmin=0.5, xmax=xmax, colors='red', linestyles='--')
    plt.ylabel('Absolute FP-Error')
    plt.text(1.5, 0.06, 'Estimated Human\nSetup Error', fontsize=14, horizontalalignment='center')
    plt.show()


def plot_est_error_over_multparams(df, dr, feat):
    os.chdir("C:/Desktop/Uni/Studienarbeit/Bilder/Mult Error over Params")
    plt.rcParams['figure.figsize'] = (19, 9.2)
    df_group = df.groupby(df['Labels'].apply(tuple))
    labels, errors = [], []
    for name, group in df_group:
        error = group['Error'].apply(tuple).max()
        labels.append(list(name))
        errors.append(error)
    par_names = param_names(dr)
    x=np.array(labels)[:, 0]
    y=np.array(labels)[:, 1]
    for param_index, param in enumerate(par_names):
        x=np.array(labels)[:, param_index]
        c=np.array(errors)[:, param_index]
        error_column_name = param + ' Absolute Error'
        for param2_index, param2 in enumerate(par_names):
            if not param == param2:
                y=np.array(labels)[:, param2_index]
                plt.scatter(x=x, y=y, s=70, c=c, cmap='magma_r')
                plt.xlabel('True ' + param + ' Setting')
                plt.ylabel('True ' + param2 + ' Setting')
                plt.colorbar()
                plt.tight_layout()
                file_name = dr + '_' + param + '_' + param2 + '_' + feat + '.pdf'
                plt.savefig(fname=file_name)
                plt.clf()
                #plt.show()


def plot_est_error_over_params(df, dr, feat):
    df_group = df.groupby(df['Labels'].apply(tuple))
    labels, errors = [], []
    for name, group in df_group:
        error = group['Error'].apply(tuple).max()
        labels.append(list(name))
        errors.append(error)
    par_names = param_names(dr)
    plt.xlabel('True ' + par_names[0] + ' Setting')
    plt.ylabel('True ' + par_names[1] + ' Setting')
    x=np.array(labels)[:, 0]
    y=np.array(labels)[:, 1]
    for index, par in enumerate(par_names):
        c=np.array(errors)[:, index]
        plt.xlabel('True ' + par_names[0] + ' Setting')
        plt.ylabel('True ' + par_names[1] + ' Setting')
        plt.scatter(x=x, y=y, s=70, c=c , cmap='magma_r')
        plt.colorbar()
        plt.show()


def plot_error_by_vol(error_vol, par_names, dr, feat):
    line_shapes = ['-', ':', '--', '-.', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10))]
    markers = ['o', 's', '^', 'v', 'd', 'P']
    error = error_vol[:, :-1]
    vol = error_vol[:, -1]
    plt.xticks(range(len(vol)), vol)
    plt.xlabel('Volume Ratio ' + r'$\beta$' + ' in dB')
    plt.ylabel('Mean Absolute Error')    
    for index, par_error in enumerate(error.T):
        plt.plot(par_error, label=par_names[index], linestyle=line_shapes[index], marker=markers[index], linewidth=3, markersize=12)
    if np.amax(error) >= 0.05:
        plt.hlines(y=0.05, xmin=0.01, xmax=len(vol)-1, label='Estimated Human\nSetup Error', colors='red', linestyles='--', linewidth=3)
    plt.legend(loc=2)
    plt.show()


def multcomp(dr):
    os.chdir(DATA_PATH)
    par_names = param_names(dr)
    all_errors, folder_list = [], []
    dr_list = os.listdir(os.getcwd())
    dr_list.sort(key=len)
    for folder in dr_list:
        if dr in folder:
            print(folder)
            os.chdir(folder)
            os.chdir('Results')
            file_name =  'df_Spec_2_3_3_64_6_128.pickle'
            df = get_df(file_name)
            abs_error = np.array(df['Error'].tolist())
            if not dr == folder:
                par_indices = []
                mult_par_names = param_names(folder)
                for par in par_names:
                    index = mult_par_names.index(par)
                    par_indices.append(index)
                abs_error = abs_error[:, par_indices]
                folder = mult_dict[folder]
                folder = "\n".join(folder)
            folder_list.append(folder)
            all_errors.extend(list(abs_error.T))
            os.chdir(DATA_PATH)
    for index, par in enumerate(par_names):
        print(par)
        plt.boxplot(all_errors[index::2], whis=1.5)
        plt.xticks(np.array([1, 2, 3, 4]), np.array(folder_list))
        xmax = 4.5
        plt.hlines(y=0.05, xmin=0.5, xmax=xmax, colors='red', linestyles='--', linewidth=3, label='Estimated Human\nSetup Error')
        plt.ylabel('Absolute Error')
        plt.legend(loc=2)
        plt.show()       
    

def plot_multbp(error, dr, par_names, feat):
    fx = mult_dict[dr]
    plt.boxplot(error, whis=1.5)
    plt.xticks([*range(1, len(par_names)+1)], par_names)
    xmax = len(par_names) + 0.5
    plt.hlines(y=0.05, xmin=0.5, xmax=xmax, colors='red', linestyles='--', label='Estimated Human\nSetup Error')
    plt.ylabel('Absolute Error')
    plt.legend(loc=2)
    plt.show()  


def plot_errorbp(error, par_names):
    plt.boxplot(error, whis=1.5)
    xlabels = par_names
    plt.xticks([*range(1, len(par_names)+1)], xlabels)
    xmax = len(par_names) * 1 + 0.5
    plt.hlines(y=0.05, xmin=0.5, xmax=xmax, colors='red', linestyles='--', linewidth=3, label='Estimated Human\nSetup Error')
    plt.ylabel('Absolute Error')
    plt.legend(loc=2)
    plt.show()


def plot_featbp(feats, error, j_error, par_names, dr):
    error.append(list(j_error))
    feats.append('Jürgens et al.')
    for index, par in enumerate(par_names):
        plt.boxplot(np.array(error)[:, :, index].T, whis=1.5)
        plt.xticks([1, 2, 3, 4, 5], feats)
        xmax = len(par_names) * 2 + 1.5
        plt.hlines(y=0.05, xmin=0.5, xmax=xmax, colors='red', linestyles='--', linewidth=3, label='Estimated Human\nSetup Error')
        plt.ylabel('Absolute Error')
        plt.legend(loc=2)
        plt.show()


def plot_history(dr):
    os.chdir(DATA_PATH)
    os.chdir(dr)
    os.chdir('History')
    feats = ['MFCC40', 'GFCC40', 'Spec', 'Chroma']
    for feat in feats:
        history = joblib.load('History' + feat + '_2_3_3_64_6_1281.pickle')
        plt.plot(history['loss'])  # Loss is mean squared error (=/= mean absolute error)
        plt.plot(history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim(0.0, 0.1)
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.show()


def plot_over_noise(dr, feat_list, error):
    """plots mean error across different levels of noise on data"""
    factors = np.array([0.0, 0.001, 0.01, 0.05])
    line_shapes = ['-', ':', '--', '-.']
    markers = ['o', 's', '^', 'v']
    par_names = param_names(dr)
    os.chdir(NOISE_PATH)
    os.chdir(dr)
    error = np.array(error)
    print(error)
    for j, par_error in enumerate(error.T):
        for i, feat in enumerate(feat_list):
            pf_error = par_error[i*len(factors):i*len(factors)+len(factors)]
            plt.xticks(range(len(factors)), factors.astype(str))
            plt.plot(pf_error, label=feat + ' ' + par_names[j], linestyle=line_shapes[i], marker = markers[i], linewidth=3, markersize=12)
        plt.xlabel('Noise Factor ' + r'$\alpha$')
        plt.ylabel('Mean Absolute Error')
        plt.hlines(y=0.05, xmin=0.01, xmax=len(list(factors))-1, label='Estimated Human\nSetup Error', 
                        colors='red', linestyles='--', linewidth=3)
        plt.legend()
        plt.show()


def scale_error():
    line_shapes = ['-', ':', '--', '-.']
    os.chdir(SCALE_PATH)
    for dr in os.listdir(os.getcwd()):
        print(dr)
        par_names = param_names(dr)
        feat_list = []
        all_error = []
        os.chdir(dr)
        for file_name in os.listdir(os.getcwd()):
            df = get_df(file_name)
            grouped = df.groupby('BP')
            df = df.drop(grouped.get_group('bp0').index)
            feat = file_name[:-7].split('_')[1]
            feat_list.append(feat)
            abs_error = np.array(df['Error'].tolist())
            all_error.append(abs_error)
        for index, par in enumerate(par_names):
            plt.boxplot(np.array(all_error)[:, :, index].T, whis=1.5)
            plt.xticks([1, 2, 3, 4], feat_list)
            xmax = 4.5
            plt.hlines(y=0.05, xmin=0.5, xmax=xmax, colors='red', linestyles='--')
            plt.ylabel('Absolute Error')
            plt.text(1.5, 0.06, 'Estimated Human\nSetup Error', fontsize=14, horizontalalignment='center')

            plt.show()
        os.chdir(SCALE_PATH)

def transform_scale_df(file_name, vol):
    df = get_df(file_name)
    grouped = df.groupby('Mix Volume')
    group = grouped.get_group(vol)
    grouped = group.groupby('BP')
    df_grouped = group.drop(grouped.get_group('bp0').index)

    print(str(vol))

    return df_grouped


def transform_scale_df_train(file_name, vol):
    df_train = get_df(file_name[:-13] + '.pickle')
    grouped_train = df_train.groupby(['VST', 'Pitch', 'Mix Volume'])
    df_train = grouped_train.get_group(('ag', '40', vol))

    return df_train


def scale_comp():
    """plots results for pitch shifted bass and piano test dataset"""
    line_shapes, markers = ['-', ':', '--', '-.'], ['o', 's', '^', 'v']
    os.chdir(SCALE_PATH)
    for dr in os.listdir(os.getcwd()):
        os.chdir(dr)
        print(dr)
        par_names = param_names(dr)
        os.chdir(dr)
        vol_list = [-12.0, -3.0, 3.0]
        for vol in vol_list:
            feat_list, all_labels, all_errors = [], [], []
            for file_name in os.listdir(os.getcwd()):
                df_grouped = transform_scale_df(file_name, vol)
                feat = file_name[:-7].split('_')[1]
                vol = float(vol)
                feat_list.append(feat)
                os.chdir(os.path.join(DATA_PATH, dr, 'Results'))
                df_train = transform_scale_df_train(file_name, vol)
                df_grouped = pd.concat([df_grouped, df_train])
                error, labels = [], []
                for name, group in df_grouped.groupby('BP'):
                    labels.append(int(name[2:]))
                    error.append(group['Error'].mean())
                all_labels.append(labels)
                all_errors.append(error)
                os.chdir(os.path.join(SCALE_PATH, dr))
            all_errors = np.array(all_errors)
            for i, par in enumerate(par_names):
                for j, feat in enumerate(feat_list):
                    x, y = zip(*sorted(zip(all_labels[j], all_errors[j, :, i])))
                    plt.xticks(x, list(bp_dict.values()))
                    plt.plot(y, label = feat + ' ' + par, linestyle = line_shapes[j], marker = markers[j], linewidth=3, markersize=12)
                    print(feat)
                    print(par)
                plt.xlabel('Bass and Piano Pitch')
                plt.ylabel('Mean Absolute Error')
                plt.legend(loc=2)
                plt.show()
        os.chdir(SCALE_PATH)


def get_juergens_data(dr, par_names):
    j_error, j_ki = get_j_error(dr)
    j_byvol(dr, par_names)
    j_me = np.mean(j_error, axis=0)

    return j_error, j_ki, j_me


def conf_interval(fold_errors):
    std = np.std(fold_errors, axis=0)
    ki = 1.96*(std/np.sqrt(5))

    return ki


def vol_error(grouped):
    me_byvol = []
    for vol, df_group in grouped:
        error_vol = np.array(df_group['Error'].tolist())
        me_vol = list(np.mean(error_vol, axis=0))
        me_vol.append(vol)
        me_byvol.append(me_vol)
        
    return me_byvol


def noise_error(dr, all_noise_errors, file_name):
    os.chdir(NOISE_PATH)
    os.chdir(dr)
    for factor in ['0.001', '0.01', '0.05']:
        noise_df = get_df(file_name[:-7] + '_noisetest_' + factor + '.pickle')
        noise_error = np.array(noise_df['Error'].tolist())
        noise_mean_error = np.mean(noise_error, axis=0)
        all_noise_errors.append(list(noise_mean_error))


def getdf(dr):
    print(dr)
    basefx = ['Distortion', 'Tremolo', 'SlapbackDelay'] #list of fx for which all single fx plots can be created
    par_names = param_names(dr)
    all_mean_errors, all_conf, feat_list, comp_error, all_noise_errors = [], [], [], [], []
    if dr in basefx:
         j_error, j_ki, j_me = get_juergens_data(dr, par_names)
    os.chdir(os.path.join(DATA_PATH, dr, 'Results'))
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith('.pickle'):
            nn_dict = get_nn_values(file_name)
            df = get_df(file_name)
            abs_error = np.array(df['Error'].tolist())
            mean_error = np.mean(abs_error, axis=0)
            fold_errors = []
            for index in range(5):
                fold_mean = np.mean(abs_error[int(index*len(df)/5):int(((index+1)*len(df)/5)-1)], axis=0)
                fold_errors.append(fold_mean)
            all_conf.append(conf_interval(fold_errors))
            feat_list.append(nn_dict['Feature'])
            comp_error.append(abs_error)
            all_mean_errors.append(mean_error)
            me_byvol = np.array(vol_error(df.groupby('Mix Volume')))
            if dr in list(mult_dict.keys()):
                plot_multbp(list(abs_error.T), dr, par_names, nn_dict['Feature'])
                plot_est_error_over_multparams(df, dr, nn_dict['Feature'])
            else:
                plot_est_error_over_params(df, dr, nn_dict['Feature'])
                if dr in basefx:
                    all_noise_errors.append(list(mean_error))
                    all_noise_errors = noise_error(dr, all_noise_errors, file_name)
                else:
                    plot_errorbp(list(abs_error.T), par_names)
            plot_error_by_vol(me_byvol, par_names, dr, nn_dict['Feature'])
            print('\n' + '='*60 + '\n')
        os.chdir(os.path.join(DATA_PATH, dr, 'Results'))
    plot_over_noise(dr, feat_list, all_noise_errors)
    all_mean_errors.append(j_me)
    all_conf.append(j_ki)
    print(feat_list)
    print('Mean Error and confidence interval of all models:')
    print(np.array(all_mean_errors))
    print(np.array(all_conf))
    plot_featbp(feat_list, comp_error, j_error, par_names, dr)
    os.chdir(DATA_PATH)


if __name__ == '__main__':  
    fx = ['Distortion', 'Tremolo', 'SlapbackDelay', 'Chorus', 'Phaser', 'Reverb', 'Overdrive',
                'DistortionTremolo', 'DistortionSlapbackDelay', 'TremoloSlapbackDelay', 'DistortionTremoloSlapbackDelay'] 

    for folder in fx:
        getdf(folder)
    for folder in fx:
        if folder in list(mult_dict.keys()):
            multcomp(folder)
    scale_comp()


            