import numpy as np
import pandas as pd
import pickle
from ModelTraining import model_training, model_using
from os.path import exists
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from itertools import combinations

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = './data/'
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'Imaging_clinical_feature_set_folds_outcomes_07_25_2018.xls')
# TEST_DATA_DIR = os.path.join(BASE_DIR, 'BPEI_feature_set_folds_outcomes_06_10_2019 (1).xls')

df_cov = pd.read_excel(TRAIN_DATA_DIR)
df_cov = df_cov.fillna('N/A')
df = df_cov
# df_cov['diff'] = df_cov['Elapsed time since first imaging'].shift(-1) - df_cov['Elapsed time since first imaging']
# avg = df_cov[df_cov['diff'] > 0]['diff'].mean()
# print('avg diff: ',avg)
#
# df_cov['diff'] = df_cov['diff'].fillna(1)
#
# df = df_cov[df_cov['diff'].abs() > 0.5]
#
# df.loc[df['Fold number'] == 2, 'Fold number'] = 1
#
# df.loc[df['Fold number'] == 3, 'Fold number'] = 2
# df.loc[df['Fold number'] == 4, 'Fold number'] = 2
#
# df.loc[df['Fold number'] == 5, 'Fold number'] = 3
# df.loc[df['Fold number'] == 6, 'Fold number'] = 3
#
# df.loc[df['Fold number'] == 7, 'Fold number'] = 4
# df.loc[df['Fold number'] == 8, 'Fold number'] = 4
#
# df.loc[df['Fold number'] == 10, 'Fold number'] = 5
# df.loc[df['Fold number'] == 9, 'Fold number'] = 5

# df_miami = pd.read_excel(TEST_DATA_DIR)
# df_miami = df_miami.fillna('N/A')

mon = [3, 6, 9, 12, 15, 18, 21]  # if testing only one month
# mon = [3]  # if testing only one month
# NN = [5]
NN = [5, 10, 20, 25, 30, 50]
# NN = [5]

FOLDS = [1, 2, 3, 4, 5]
# FOLDS = [1]


val_flag = 1  # if using split val data, use 1, if using test data during training, use 0

file_exists = exists("./5/models")
if not file_exists:
    os.mkdir("./5/models")

file_exists = exists("./5/CV_resultsv2")
if not file_exists:
    os.mkdir("./5/CV_resultsv2")

file_exists = exists("weights3")
if not file_exists:
    os.mkdir("weights3")

for m in mon:

    FPR = []
    TPR = []
    ROC_AUC = []
    PREDS_prob = []
    IP = []
    IY = []
    SLEN = []

    print('month:', m)
    fpr_CV = dict()
    tpr_CV = dict()
    roc_auc_CV = dict()
    prediction = dict()
    gt = dict()
    precision = dict()
    recall = dict()
    roc_pr_CV = dict()

    strm = 'Outcome at ' + str(m) + ' months'
    df_train = df[df[strm] != 'N/A']
    df_train = df_train.replace('N/A', 0, regex=True)

    # df_test = df_miami[df_miami[strm] != 'N/A'].copy()
    # df_test = df_test.replace('N/A', 0, regex=True)

    print("no of patient:", len(df_train['Patient number'].unique()))
    # flg = [0,1]
    # the flag was 0, which is meaningless, because the test data set is used as validating data set.
    f = 1

    #     for f in flg:
    #         print('f: ', f)

    for n in NN:
        for fold in FOLDS:
            # fold = random.randint(1,10)
            print('fold: ' + str(fold))
            print('NN: ', n)
            path = 'OCT_model_with_weights_' + str(m) + '_' + str(n) + '_' + str(fold) + '.h5'
            if os.path.isfile(path):
                fpr, tpr, roc_auc, preds, y_pred, y_true, lr_precision, lr_recall, lr_auc, slen = \
                    model_using(df_train, m, fold, n, strm, path)
            #     PP = pd.read_pickle(
            #         r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_prob_' + str(f) + '.pickle')
            #     ROC_AUC = pd.read_pickle(
            #         r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_ROC_AUC_' + str(f) + '.pickle')
            #     IP = pd.read_pickle(r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_IP_' + str(f) + '.pickle')
            #     IY = pd.read_pickle(r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_IY_' + str(f) + '.pickle')
            #     SLEN = pd.read_pickle(r'' + BASE_DIR + 'CV_resultsv2/HARBOR' + str(m) + 'mon_SLEN_' + str(f) + '.pickle')

            else:
                fpr, tpr, roc_auc, preds, y_pred, y_true, lr_precision, lr_recall, lr_auc, slen = \
                    model_training(
                        df_train, None,
                        m,
                        fold,
                        n,
                        f,
                        strm,
                        val_flag)

            print('roc_auc: ', roc_auc)
            print('===================')
            FPR.append(fpr)
            TPR.append(tpr)
            ROC_AUC.append(roc_auc)
            PREDS_prob.append(preds)
            IP.append(y_pred)
            IY.append(y_true)
            SLEN.append(slen)

    with open('./5/CV_resultsv2/HARBOR' + str(m) + 'mon_predictionFPR_' + str(f) + '.pickle', 'wb') as handle:
        pickle.dump(FPR, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./5/CV_resultsv2/HARBOR' + str(m) + 'mon_predictionTPR_' + str(f) + '.pickle', 'wb') as handle:
        pickle.dump(TPR, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./5/CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_prob_' + str(f) + '.pickle', 'wb') as handle:
        pickle.dump(PREDS_prob, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./5/CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_ROC_AUC_' + str(f) + '.pickle', 'wb') as handle:
        pickle.dump(ROC_AUC, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./5/CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_IP_' + str(f) + '.pickle', 'wb') as handle:
        pickle.dump(IP, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./5/CV_resultsv2/HARBOR' + str(m) + 'mon_prediction_IY_' + str(f) + '.pickle', 'wb') as handle:
        pickle.dump(IY, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./5/CV_resultsv2/HARBOR' + str(m) + 'mon_SLEN_' + str(f) + '.pickle', 'wb') as handle:
        pickle.dump(SLEN, handle, protocol=pickle.HIGHEST_PROTOCOL)
