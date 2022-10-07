import math
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler


def get_seq_len(df_cov):
    n_patient = len(df_cov['Patient number'].unique())
    patient_ids = df_cov['Patient number'].unique()
    df_cov.sort_index(axis=1, inplace=True)
    Seq_len = []
    for i in range(n_patient):
        p = df_cov[df_cov['Patient number'] == patient_ids[i]]
        Seq_len.append(len(p))
    return Seq_len


def training_data2(df_cov, outcomestring, srlen):
    n_patient = len(df_cov['Patient number'].unique())
    patient_ids = df_cov['Patient number'].unique()
    df_cov.sort_index(axis=1, inplace=True)

    patients_vec = []
    patients_label = []
    Seq_len = []
    # print(n_patient)
    c = 0
    for i in range(n_patient):
        p = df_cov[df_cov['Patient number'] == patient_ids[i]]
        p = p.sort_values('Elapsed time since first imaging', ascending=True)
        p = p.reset_index(drop=True)
        max_visit = p['Elapsed time since first imaging'].max()
        Seq_len.append(len(p))
        itr = int(max_visit / srlen) + 1
        cnt = 0
        t = p['Elapsed time since first imaging'] + 0.1
        # print('patient: ', patient_ids[i], 'itr:', itr)
        # print('t:', t.round())
        for j in range(itr):
            patients_vec.append([])
            patients_label.append([])
            for v in range(j * srlen + 1, (j + 1) * srlen + 1):
                if v in t.round():
                    # print(p.iloc[cnt]['Elapsed time since first imaging'])
                    temp = p.iloc[cnt]
                    temp = temp.fillna(0)
                    patients_label[c].append(p.iloc[cnt][outcomestring])
                    temp = temp.drop("Patient number")
                    temp = temp.drop("Fold number")
                    temp = temp.drop("Outcome at 6 months")
                    temp = temp.drop("Outcome at 3 months")
                    temp = temp.drop("Outcome at 9 months")
                    temp = temp.drop("Outcome at 12 months")
                    temp = temp.drop("Outcome at 15 months")
                    temp = temp.drop("Outcome at 18 months")
                    temp = temp.drop("Outcome at 21 months")
                    temp = temp.drop("Outcome at 24 months")  # added later
                    temp = temp.drop("Elapsed time since first imaging")
                    temp = temp.drop("Progression during study")
                    temp = temp.drop("Max. months remain dry")
                    temp = temp.drop("Min. months to wet")
                    temp = temp.drop("diff")
                    patients_vec[c].append(temp.tolist())
                    cnt += 1
                    # print('cnt: ', cnt)
                else:
                    patients_label[c].append(2)
                    patients_vec[c].append([0] * 53)
            c += 1
    return patients_vec, patients_label, Seq_len


def training_data(df_cov, outcomestring, srlen):
    n_patient = len(df_cov['Patient number'].unique())
    patient_ids = df_cov['Patient number'].unique()
    df_cov.sort_index(axis=1, inplace=True)

    patients_vec = []
    patients_label = []
    Seq_len = []
    # print('no of patient: ', n_patient)
    for i in range(n_patient):
        patients_vec.append([])
        patients_label.append([])
        p = df_cov[df_cov['Patient number'] == patient_ids[i]]
        # print('for this patient',len(p))
        Seq_len.append(len(p))
        # store vector and labels for
        # pre_visit = 0
        p = p.sort_values('Elapsed time since first imaging', ascending=True)
        p = p.reset_index(drop=True)

        for j in range(len(p)):
            # stroring vectors and labels
            temp = p.iloc[j]
            # if abs(temp['Elapsed time since first imaging'] - pre_visit) < 0.5:
            #     pre_visit = temp['Elapsed time since first imaging']
            #     print('Error. This should have been removed before!')
            #     continue
            # pre_visit = temp['Elapsed time since first imaging']
            temp = temp.fillna(0)
            patients_label[i].append(p.iloc[j][outcomestring])
            # if p.iloc[j]['Event'] in  [True]:
            # if p.iloc[j]['Event'] in  [True]:
            #   patients_label[i].append(1)
            # if p.iloc[j]['Event'] in [False]:
            #   patients_label[i].append(0)
            temp = temp.drop("Patient number")
            # temp = temp.drop("Event")
            temp = temp.drop("Fold number")
            # temp = temp.drop("stop")
            # temp = temp.drop("start")
            temp = temp.drop("Outcome at 6 months")
            temp = temp.drop("Outcome at 3 months")
            temp = temp.drop("Outcome at 9 months")
            temp = temp.drop("Outcome at 12 months")
            temp = temp.drop("Outcome at 15 months")
            temp = temp.drop("Outcome at 18 months")
            temp = temp.drop("Outcome at 21 months")
            temp = temp.drop("Outcome at 24 months")  # added later
            temp = temp.drop("Elapsed time since first imaging")
            temp = temp.drop("Progression during study")
            temp = temp.drop("Max. months remain dry")
            temp = temp.drop("Min. months to wet")
            # temp = temp.drop("diff")

            # print(temp.shape)

            patients_vec[i].append(temp.tolist())
    return patients_vec, patients_label, Seq_len


def dataaugmentation(patients_vec, patients_label, percentage):
    new_patients_vec = []
    new_patients_label = []
    dummy = [0] * len(patients_vec[0][0])
    dummy_label = 2
    for i in range(len(patients_vec)):  ## patient-wise
        # len(patients_vec[i]) - 1 to make sure the last index will never be selected
        # find the percentage values of the dataset
        p = math.floor((len(patients_vec[i]) - 1) * percentage)
        # select p random values between 0 and to len(patient[i]) - 2 for the first half of augmentation
        rnd = np.random.choice(range(len(patients_vec[i]) - 1), p, replace=False)
        # select p random values between 0 and to len(patient[i]) - 2 for the second half of augmentation
        # rnd_inv = np.random.choice(range(len(patients_vec[i]) - 1), p, replace=False)
        # print('patient id: ', i, ' rnd:', rnd, ' rnd_inv: ', rnd_inv, ', max visit:', len(patients_vec[i]))
        for j in range(len(patients_vec[i])):  ## patient-visit  ## pyramid
            # new_patients_vec.append([])
            # new_patients_label.append([])
            T = []
            L = []
            if j in rnd:
                for k in range(j + 1):
                    T.append(dummy)
                    L.append(dummy_label)
            else:
                for k in range(j + 1):
                    T.append(patients_vec[i][k])
                    L.append(patients_label[i][k])

            new_patients_vec.append(T)
            new_patients_label.append(L)

        # for j in range(len(patients_vec[i]) - 1):  ## inverse pyramid
        #     l = len(patients_vec[i]) - j
        #     T = []
        #     L = []
        #
        #     if j in rnd_inv:
        #         for k in range(l):
        #             T.append(dummy)
        #             L.append(dummy_label)
        #     else:
        #         for k in range(l):
        #             T.append(patients_vec[i][k])
        #             L.append(patients_label[i][k])
        #
        #     new_patients_vec.append(T)
        #     new_patients_label.append(L)

    return new_patients_vec, new_patients_label


def testing_data2(df_cov, outcomestring, srlen=100):
    n_patient = len(df_cov['Patient name and eye'].unique())
    patient_ids = df_cov['Patient name and eye'].unique()
    df_cov.sort_index(axis=1, inplace=True)
    patients_vec = []
    patients_label = []
    Seq_len = []
    # print(n_patient)
    c = 0
    for i in range(n_patient):
        p = df_cov.loc[df_cov['Patient name and eye'] == patient_ids[i]]
        p = p.sort_values('Elapsed time since first imaging', ascending=True)
        p = p.reset_index(drop=True)
        max_visit = p['Elapsed time since first imaging'].max()
        Seq_len.append(len(p))
        itr = int(max_visit / srlen) + 1
        cnt = 0
        t = p['Elapsed time since first imaging'] + 0.1
        # print('patient: ', patient_ids[i], 'itr:', itr)
        # print('t:', t.round())
        for j in range(itr):
            patients_vec.append([])
            patients_label.append([])
            for v in range(j * srlen + 1, (j + 1) * srlen + 1):
                if v in t.round():
                    # print(p.iloc[cnt]['Elapsed time since first imaging'])
                    temp = p.iloc[cnt]
                    temp = temp.fillna(0)
                    patients_label[c].append(p.iloc[cnt][outcomestring])
                    temp = temp.drop('Patient name and eye')
                    temp = temp.drop("Patient number")
                    temp = temp.drop("Fold number")
                    temp = temp.drop("Elapsed time since first imaging")
                    temp = temp.drop("Outcome at 6 months")
                    temp = temp.drop("Outcome at 3 months")
                    temp = temp.drop("Outcome at 9 months")
                    temp = temp.drop("Outcome at 12 months")
                    temp = temp.drop("Outcome at 15 months")
                    temp = temp.drop("Outcome at 18 months")
                    temp = temp.drop("Outcome at 21 months")
                    temp = temp.drop("Outcome at 24 months")  # added later
                    temp = temp.drop("Contralateral eye status")  # added later
                    temp = temp.drop("Contralateral eye moths wet")  # added later
                    temp = temp.drop("Number averaged scans")  # added later
                    temp = temp.drop("Number previous visits")  # added later
                    temp = temp.drop("Progression during study")
                    temp = temp.drop("Max. months remain dry")
                    temp = temp.drop("Min. months to wet")
                    patients_vec[c].append(temp.tolist())
                    cnt += 1
                    # print('cnt: ', cnt)
                else:
                    patients_label[c].append(2)
                    patients_vec[c].append([0] * 53)
            c += 1
    return patients_vec, patients_label, Seq_len


def testing_data(df_cov, outcomestring, srlen=100):
    n_patient = len(df_cov['Patient name and eye'].unique())
    patient_ids = df_cov['Patient name and eye'].unique()
    df_cov.sort_index(axis=1, inplace=True)
    patients_vec = []
    patients_label = []
    Seq_len = []
    # print(n_patient)
    for i in range(n_patient):
        patients_vec.append([])
        patients_label.append([])
        p = df_cov.loc[df_cov['Patient name and eye'] == patient_ids[i]]
        p = p.sort_values('Elapsed time since first imaging', ascending=True)
        p = p.reset_index(drop=True)
        # p = p.sort_values(['start'], ascending=[True])
        # print(len(p))
        Seq_len.append(len(p))
        dif = 0 if srlen > len(p) else len(p) - srlen
        # store vector and labels for
        for j in range(len(p)):
            if dif > 0:
                dif -= 1
                continue
            # stroring vectors and labels
            temp = p.iloc[j]
            temp = temp.fillna(0)
            patients_label[i].append(p.iloc[j][outcomestring])
            temp = temp.drop('Patient name and eye')
            temp = temp.drop("Patient number")
            temp = temp.drop("Fold number")
            temp = temp.drop("Elapsed time since first imaging")
            temp = temp.drop("Outcome at 6 months")
            temp = temp.drop("Outcome at 3 months")
            temp = temp.drop("Outcome at 9 months")
            temp = temp.drop("Outcome at 12 months")
            temp = temp.drop("Outcome at 15 months")
            temp = temp.drop("Outcome at 18 months")
            temp = temp.drop("Outcome at 21 months")
            temp = temp.drop("Outcome at 24 months")  # added later
            temp = temp.drop("Contralateral eye status")  # added later
            temp = temp.drop("Contralateral eye moths wet")  # added later
            temp = temp.drop("Number averaged scans")  # added later
            temp = temp.drop("Number previous visits")  # added later
            temp = temp.drop("Progression during study")
            temp = temp.drop("Max. months remain dry")
            temp = temp.drop("Min. months to wet")

            # print(temp.shape)

            # row.append(temp["Row"])
            # temp = temp.drop("Row")

            patients_vec[i].append(temp.tolist())

    return patients_vec, patients_label, Seq_len


def testaugmentation(X_test, y_test, no_visit=5):
    new_patients_vec = []
    new_patients_label = []
    for i in range(len(X_test)):  ## patient-wise
        T = X_test[i]
        # T  = T[y_test[i][:,2]==0]
        P = y_test[i]
        # P = P[y_test[i][:,2]==0]
        if len(T) >= no_visit:
            K = []
            L = []
            for j in range(len(T) - no_visit, len(T)):
                K.append(T[j])
                L.append(P[j])
            new_patients_vec.append(K)
            new_patients_label.append(L)
    return new_patients_vec, new_patients_label


def normalizing(df, norm_type):
    cnt = 0
    result = df.dtypes
    if norm_type == 0:
        trans = MinMaxScaler()
    elif norm_type == 1:
        trans = StandardScaler()
    elif norm_type == 2:
        trans = QuantileTransformer(n_quantiles=100, output_distribution='normal')

    for col_name in df.columns:
        # df[col_name].fillna(0, inplace=True)
        if result[cnt] not in ['int',
                               'float'] or 'Outcome' in col_name or 'Elapsed time since first imaging' in col_name or 'Race' in col_name or 'Smoking' in col_name or 'Gender' in col_name:
            cnt += 1
            continue
        # Select the column
        num_df = df[col_name]

        col_values = num_df.values.reshape(-1, 1)

        col_values_norm = trans.fit_transform(col_values)

        df[col_name] = col_values_norm
        cnt += 1
    return df


import pandas as pd


def load_data():
    BASE_DIR = './data/'
    TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'Imaging_clinical_feature_set_folds_outcomes_07_25_2018.xls')
    TEST_DATA_DIR = os.path.join(BASE_DIR, 'BPEI_feature_set_folds_outcomes_06_10_2019 (1).xls')
    # TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'Imaging_clinical_feature_set_folds_outcomes_07_25_2018.csv')
    # TEST_DATA_DIR = os.path.join(BASE_DIR, 'BPEI_feature_set_folds_outcomes_06_10_2019.csv')

    df_miami = pd.read_excel(TEST_DATA_DIR)
    # df_miami = pd.read_csv(TRAIN_DATA_DIR)
    df_miami = df_miami.fillna('N/A')

    df_harbor = pd.read_excel(TRAIN_DATA_DIR)
    # df_miami = pd.read_csv(TRAIN_DATA_DIR)

    df_harbor = df_harbor.fillna('N/A')
    df_harbor.loc[df_harbor['Fold number'] == 1, 'Fold number'] = 1
    df_harbor.loc[df_harbor['Fold number'] == 6, 'Fold number'] = 1

    df_harbor.loc[df_harbor['Fold number'] == 2, 'Fold number'] = 2
    df_harbor.loc[df_harbor['Fold number'] == 7, 'Fold number'] = 2

    df_harbor.loc[df_harbor['Fold number'] == 3, 'Fold number'] = 3
    df_harbor.loc[df_harbor['Fold number'] == 8, 'Fold number'] = 3

    df_harbor.loc[df_harbor['Fold number'] == 4, 'Fold number'] = 4
    df_harbor.loc[df_harbor['Fold number'] == 9, 'Fold number'] = 4

    df_harbor.loc[df_harbor['Fold number'] == 5, 'Fold number'] = 5
    df_harbor.loc[df_harbor['Fold number'] == 10, 'Fold number'] = 5
    return df_harbor, df_miami


def preprocess(df_harbor, df_miami, month, fold, percentage):
    print('month:', month)
    strm = 'Outcome at ' + str(month) + ' months'
    df_train = df_harbor[df_harbor[strm] != 'N/A']
    df_train = df_train.replace('N/A', 0, regex=True)
    df_train["Patient number"] = df_train["Patient number"].astype(str)

    print("Harbor #patient:", len(df_train['Patient number'].unique()))
    train = df_train[df_train['Fold number'] != fold]
    train = train.reset_index(drop=True)
    train = normalizing(train, 2)
    seq_train = get_seq_len(train)

    val = df_train[df_train['Fold number'] == fold]
    val = val.reset_index(drop=True)
    val = normalizing(val, 2)
    seq_val = get_seq_len(val)

    slen = max(max(seq_train), max(seq_val))

    patients_vec_train, patients_label_train, Seq_len = training_data(train, strm, slen)
    print("#train: ", len(patients_vec_train))

    patients_vec_val, patients_label_val, Seq_len_val = training_data(val, strm, slen)
    print("#val: ", len(patients_vec_val))
    x_train_aug, y_train_aug = dataaugmentation(patients_vec_train, patients_label_train, percentage)

    df_test = df_miami[df_miami[strm] != 'N/A']
    df_test = df_test.replace('N/A', 0, regex=True)
    result = df_test.dtypes
    pd.options.mode.chained_assignment = None
    len([x for x in result if x == 'bool'])
    df_test = df_test.replace('N/A', 0, regex=True)
    df_test["Gender: (0) Male, (1) Female"] = df_test["Gender: (0) Male, (1) Female"].astype(int)
    # df_test.drop(columns=["Number previous visits", "Contralateral eye moths wet", "Number averaged scans",
    #                       "Contralateral eye status", "Progression during study"], inplace=True)
    df_test["Patient name and eye"] = df_test["Patient name and eye"].astype(str)
    df_test["Patient number"] = df_test["Patient number"].astype(str)
    df_test = df_test.reset_index(drop=True)
    df_test = normalizing(df_test, 2)

    patients_vec_test, patients_label_test, Seq_len_test = testing_data(df_test, strm, slen)
    print("#test: ", len(patients_vec_test))

    return x_train_aug, y_train_aug, Seq_len, patients_vec_val, patients_label_val, Seq_len_val, patients_vec_test, \
           patients_label_test, Seq_len_test
