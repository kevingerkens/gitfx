import seaborn as sns
from pathlib import Path
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..', 'Results/Classification')


def get_cmlabels(dr):
    mix_labels = ['Chorus', 'Distortion', 'FeedbackDelay', 'Flanger', 'NoFX', 'Overdrive', 'Phaser', 'Reverb', 'SlapbackDelay', 'Tremolo', 'Vibrato']
    idmt_labels = ['Chorus', 'Distortion', 'EQ', 'FeedbackDelay', 'Flanger', 'NoFX', 'Overdrive', 'Phaser', 'Reverb', 'SlapbackDelay', 'Tremolo', 'Vibrato']

    if dr == 'Classification':
        return mix_labels
    if dr == 'IDMT':
        return idmt_labels


def get_feature(file_name):
    feat = file_name[:-7].split('_')[1]
    if feat == 'mix':
        feat = 'SVM'

    return feat
        

def plot_cm(dr, classifier):
    plt.rcParams["font.size"] = "14"
    os.chdir(DATA_PATH)
    os.chdir(classifier)
    os.chdir(dr)
    print(dr)
    print(classifier)
    labels = get_cmlabels(dr)
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith('.pickle') and 'cm' in file_name:
            with open(file_name, 'rb') as handle:
                cfm = pickle.load(handle)
            cfm = cfm.astype(float)
            cfm = np.round_(cfm, decimals = 3)
            accuracy = np.mean(np.diagonal(cfm))
            feat = get_feature(file_name)
            print(feat +  ' Accuracy ' + str(accuracy))
            ax = sns.heatmap(cfm, annot=True, cmap='Blues',  xticklabels=labels, yticklabels=labels, fmt=".2g", square=True)
            plt.ylabel("True Class")
            plt.xlabel("Predicted Class")
            plt.show()

if __name__ == '__main__':
    datasets = ['GEC-GIM', 'IDMT']
    classifiers = ['CNN', 'SVM']
    for classifier in classifiers:
        for set in datasets:
            plot_cm(set, classifier)


