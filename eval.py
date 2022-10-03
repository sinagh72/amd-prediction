import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras_preprocessing.sequence import pad_sequences
import os
import matplotlib.pyplot as plt
from data_prepration import testing_data, testing_data2, testaugmentation
import keras as k
from ModelTraining import create_model

BASE_DIR = 'exact_post_wcc_bslen_halfpyramid_fold10_n5020/p-0.0/'
DATA_DIR = './data/'
f = 0
# TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'Imaging_clinical_feature_set_folds_outcomes_07_25_2018.xls')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'BPEI_feature_set_folds_outcomes_06_10_2019 (1).xls')

# df_cov = pd.read_excel(TRAIN_DATA_DIR)
# df_cov = df_cov.fillna('N/A')

df_miami = pd.read_excel(TEST_DATA_DIR)
df_miami = df_miami.fillna('N/A')
print(len(df_miami))

# mon = [6] # if testing only one month

mon = [3, 6, 9, 12, 15, 18, 21]
# mon = [3]
NN = [1]
# NN = [5, 10, 20, 25, 30, 50]
# NN = [25, 30, 50]
# FOLDS = [1]
# FOLDS = [1, 2, 3, 4, 5]
FOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

auc_matrix = np.zeros((len(FOLDS), len(NN)))

for m in mon:
    print('***********************************************************************')
    print('Results for ', str(m), ' months outcome')

    strm = 'Outcome at ' + str(m) + ' months'
    test = df_miami[df_miami[strm] != 'N/A'].copy()
    test = test.replace('N/A', 0, regex=True)
    # test = df_miami

    test = test.reset_index(drop=True)

    PP = pd.read_pickle(
        r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_prob_' + str(f) + '.pickle')
    ROC_AUC = pd.read_pickle(
        r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_ROC_AUC_' + str(f) + '.pickle')
    IP = pd.read_pickle(r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_IP_' + str(f) + '.pickle')
    IY = pd.read_pickle(r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_IY_' + str(f) + '.pickle')
    SLEN = pd.read_pickle(r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_SLEN_' + str(f) + '.pickle')
    aucMesh = np.array(ROC_AUC).reshape(len(NN), len(FOLDS))

    print('AUC values:')
    print(aucMesh)

    # AUC for NN configs over folds
    AUC_NN = []
    for i in range(0, len(NN) * len(FOLDS), len(FOLDS)):
        ipall = np.vstack((IP[i], IP[i + 1], IP[i + 2], IP[i + 3], IP[i + 4]))
        # print(ipall.shape)
        iyall = np.vstack((IY[i], IY[i + 1], IY[i + 2], IY[i + 3], IY[i + 4]))
        # print(iyall.shape)
        fpr, tpr, thresholds = roc_curve(iyall[iyall[:, 2] == 0, 1], ipall[iyall[:, 2] == 0, 1])
        roc_auc = auc(fpr, tpr)
        AUC_NN.append(roc_auc)
        # print(roc_auc)

    print('AUC over all folds for different NN configs', AUC_NN)

    ind = np.argmax(AUC_NN)
    best_NN = NN[ind]
    #
    best_fold = np.argmax(ROC_AUC[ind * len(FOLDS):ind * len(FOLDS) + len(FOLDS)])
    slen = SLEN[ind * len(FOLDS):ind * len(FOLDS) + len(FOLDS)][best_fold]
    #
    best_combo = [best_NN, FOLDS[best_fold], slen]
    print('[NN, fold, slen]: ', best_combo)

    # test on test dataset
    patients_vec_test, patients_label_test, Seq_len_test = testing_data(test, strm, slen)
    # slen = 57
    # slen = max(Seq_len_test)
    # print('max #visit: ', slen)
    # new_patients_vec, new_patients_label = testaugmentation(patients_vec_test, patients_label_test, 0)

    X_test = pad_sequences(patients_vec_test, slen, padding='post', truncating='post', value=0, dtype='float32')
    Y_test = pad_sequences(patients_label_test, slen, padding='post', truncating='post', value=2.)
    # X_test = np.asarray(patients_vec_test)
    # Y_test = np.asarray(patients_label_test)

    Y_categorical_test = k.utils.np_utils.to_categorical(Y_test, 3)
    Y_categorical_test = Y_categorical_test.reshape(Y_test.shape[0], Y_test.shape[1], 3)

    y_test = Y_categorical_test

    # bestmodel = model
    latest_file = BASE_DIR + 'models/OCT_model_with_weights_' + str(m) + '_' + str(best_combo[0]) + '_' + str(
        best_combo[1]) + '.h5'
    print(latest_file)
    num_features = X_test.shape[2]

    bestmodel = create_model(slen, num_features, best_combo[0])
    bestmodel.load_weights(latest_file)

    batch_size = slen
    preds = bestmodel.predict(X_test, batch_size=batch_size)
    # convert to the (Batch*Visit, one_hot)
    y_pred = preds.reshape(X_test.shape[0] * slen, 3)
    y_true = y_test.reshape(X_test.shape[0] * slen, 3)
    # select those that index 2 is 0 (in one-hot encoding the y is 0) and
    y_true_categorical = y_true[y_true[:, 2] == 0, 1]
    y_pred_score = y_pred[y_true[:, 2] == 0, 1]

    ## ROC with out padding
    fpr, tpr, thresholds = roc_curve(y_true_categorical, y_pred_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    lr_precision, lr_recall, _ = precision_recall_curve(y_true_categorical, y_pred_score)
    lr_auc = auc(lr_recall, lr_precision)

    print('AUC and lr_AUC: ', roc_auc, lr_auc)

    ## Plot teh ROC curve

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve for ' + str(m) + 'months (area = %0.4f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for short and long term prediction')
    plt.legend(loc="lower right")
    plt.show()
