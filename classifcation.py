import os
from pathlib import Path
import platform
import pickle
from librosa.util.utils import normalize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump, load, parallel_backend
#from getfeats_unscaled import DATA_PATH
from getfeats_unscaled import check_data

DATA_PATH = Path('C:/Desktop/uni/Studienarbeit/Audio/NewDataset/G+K+B+D')    \
    if platform.system() == 'Windows' else os.path.join(os.path.expanduser('~'), 'tmp/Clf/Gitarre monophon/Samples')

def get_data():
    os.chdir(DATA_PATH)
    all_feats, all_labels = np.empty((0, 949)), []
    for dr in os.listdir(os.getcwd()):
        if os.path.isdir(os.path.join(DATA_PATH, dr)):
            os.chdir(dr)
            feats, _ = check_data(dr) 
            labels = [dr] * len(feats)
            all_feats = np.vstack((all_feats, feats))
            all_labels.extend(labels)
            os.chdir(DATA_PATH)
    all_labels = np.array(all_labels)
    print(all_feats.shape)
    print(all_labels.shape)
    return all_feats, all_labels

def acc_ki(acc_all):
    acc = np.mean(acc_all)
    print(acc_all)
    std = np.std(acc_all)
    ki = 1.96*(std/np.sqrt(5)) 
    print('Mean accuracy: ' + str(acc))
    print('Confidence interval: ' + str(ki))


def get_classifier(fold_no, X_train=None, y_train=None, hyp_par_opt=False):
    """Trains the classifier or loads it, if already existent"""
    os.chdir(DATA_PATH)
    os.chdir('..')
    if hyp_par_opt:
        print('Optimizing Hyperparameters')
        gamma_list = [2**exp for exp in range(-14, -1)]
        C_list = [2**exp for exp in range(-2, 5)]

        tuned_parameters = [{'gamma': gamma_list, 'C': C_list}]
        clf = GridSearchCV(SVC(), tuned_parameters, cv=3, verbose=1)

        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)  # gamma = 2**(-10), C = 16
        print()
        return clf

    else:
        if not Path('SVC' +  str(fold_no) + '.joblib').exists():
            print('Training')
            clf = SVC(probability=True, C=16, gamma=0.00048828125)
            with parallel_backend('threading', n_jobs=-1):
                clf.fit(X_train, y_train)
            print(X_train.shape)
        else:
            clf = load('SVC' +  str(fold_no) + '.joblib')
        return clf

def train_svc():
    """Extracts/loads feature data and trains Support Vector Machine Classifier"""
    os.chdir(DATA_PATH)
    # wavtoarray.read_data() # for profiling
    X, y = [], []

    os.chdir(DATA_PATH)
    for dr in os.listdir(os.getcwd()):
        os.chdir(dr)
        print(dr)
        X_fx, _ = check_data(dr)
        n_smp = X_fx.shape[0]
        y_fx = np.array([dr]*n_smp)
        X.extend(list(X_fx))
        y.extend(list(y_fx))
        os.chdir(DATA_PATH)
    X = np.array(X)
    y = np.array(y)

    X = np.nan_to_num(X, 0.0)
    n_splits=5
    print('Splitting Dataset into ' + str(n_splits) + ' folds for cross validation')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all = []
    pred_all = []
    acc_all = []
    fold_no = 1
    for train, val in kf.split(X):
        print('Fold ' + str(fold_no))
        X_train, X_test = X[train], X[val]
        y_train, y_test = y[train], y[val]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        # plots.feat_importance(X_train, y_train, X_test, y_test)
        print('Getting the Classifier')
        clf = get_classifier(fold_no, X_train=X_train, y_train=y_train, hyp_par_opt=False)

        # plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix')
        # plt.show()

        y_pred = clf.predict(X_test)
        print('Prediction: ' + str(y_pred))
        print('True: ' + str(y_test))
        print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
        y_true_all.extend(y_test)
        pred_all.extend(y_pred)
        #y_test = np.argmax(np.array(y_test), axis=1)
        #pred = np.argmax(np.array(pred), axis=1)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        print(cm)
        acc_sum = np.sum(np.diagonal(cm))
        acc_sum = acc_sum + cm[2, 8] + cm[8, 2]
        acc_fold = acc_sum/11
        acc_all.append(acc_fold)

        os.chdir(DATA_PATH)
        os.chdir('..')
        dump(clf, 'SVC' + str(fold_no) + '.joblib')
        print('SVC saved')
        fold_no +=1
    y_true_all = np.array(y_true_all)
    pred_all = np.array(pred_all)
    print(pred_all.shape)
    acc_ki(acc_all)

    m_cf = confusion_matrix(y_true_all, pred_all, normalize='true').astype(float)
    m_cf_norm = m_cf
    print(m_cf_norm)
    print(np.diagonal(m_cf_norm))
    print('Chorus, Distortion, FeedbackDelay, Flanger, NoFX, Overdrive')
    print('Phaser, Reverb, SlapbackDelay, Tremolo, Vibrato')
    os.chdir(DATA_PATH)
    os.chdir('..')
    file_name = "cf_mix.pickle" 
    with open(file_name, 'wb') as handle:
        pickle.dump(m_cf_norm, handle)    

    # m_cf_df = pd.DataFrame(m_cf_norm)
    # m_cf_df.index = pd.Index(['Chorus', 'Distortion', 'FeedbackDelay', 'Flanger',
    #                           'NoFX', 'Overdrive', 'Phaser', 'Reverb', 'SlapbackDelay',
    #                           'Tremolo', 'Vibrato'])
    # m_cf_df.columns = ['Chorus', 'Distortion', 'FeedbackDelay',
    #                    'Flanger', 'NoFX', 'Overdrive', 'Phaser',
    #                    'Reverb', 'SlapbackDelay', 'Tremolo', 'Vibrato']

    # m_cf_df.to_csv('Confusion_Matrix.csv', sep=';')

    

    # test_excluding_feature(get_feature_list(X_train, y_train, X_test, y_test, y_pred))
    # test_excluding_functionals(X_train, y_train, X_test, y_test, y_pred)

    print('done')

if __name__ == '__main__':
    train_svc()

