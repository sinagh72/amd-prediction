import math

import numpy as np


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


def training_data(df_cov, outcomestring):
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
        # for j in range(len(p)):
        #     # stroring vectors and labels
        #     temp = p.iloc[j]
        #     temp = temp.fillna(0)
        #     patients_label[i].append(p.iloc[j][outcomestring])
        #     # if p.iloc[j]['Event'] in  [True]:
        #     # if p.iloc[j]['Event'] in  [True]:
        #     #   patients_label[i].append(1)
        #     # if p.iloc[j]['Event'] in [False]:
        #     #   patients_label[i].append(0)
        #     temp = temp.drop('Patient name and eye')
        #     temp = temp.drop("Patient number")
        #     # temp = temp.drop("Event")
        #     # temp = temp.drop("stop")
        #     # temp = temp.drop("start")
        #     temp = temp.drop("Fold number")
        #
        #     temp = temp.drop("Elapsed time since first imaging")
        #     temp = temp.drop("Outcome at 6 months")
        #     temp = temp.drop("Outcome at 3 months")
        #     temp = temp.drop("Outcome at 9 months")
        #     temp = temp.drop("Outcome at 12 months")
        #     temp = temp.drop("Outcome at 15 months")
        #     temp = temp.drop("Outcome at 18 months")
        #     temp = temp.drop("Outcome at 21 months")
        #     temp = temp.drop("Outcome at 24 months")  # added later
        #     temp = temp.drop("Contralateral eye status")  # added later
        #     temp = temp.drop("Contralateral eye moths wet")  # added later
        #     temp = temp.drop("Number averaged scans")  # added later
        #     temp = temp.drop("Number previous visits")  # added later
        #     temp = temp.drop("Progression during study")
        #     temp = temp.drop("Max. months remain dry")
        #     temp = temp.drop("Min. months to wet")
        #
        #     # print(temp.shape)
        #
        #     # row.append(temp["Row"])
        #     # temp = temp.drop("Row")
        #
        #     patients_vec[i].append(temp.tolist())

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
