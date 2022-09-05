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
        p = df_cov.loc[df_cov['Patient number'] == patient_ids[i]]
        # print('for this patient',len(p))
        Seq_len.append(len(p))
        # store vector and labels for
        for j in range(len(p)):
            # stroring vectors and labels
            temp = p.iloc[j]
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

        # print(temp.shape)
            patients_vec[i].append(temp.tolist())
    return patients_vec, patients_label, Seq_len


def dataaugmentation(patients_vec, patients_label):
    new_patients_vec = []
    new_patients_label = []
    flg = 0
    for i in range(len(patients_vec)):  ## patient-wise
        for j in range(len(patients_vec[i])): ## patient-visit
            # new_patients_vec.append([])
            # new_patients_label.append([])
            T = []
            L = []
            for k in range(j + 1):
                T.append(patients_vec[i][k])
                L.append(patients_label[i][k])
            new_patients_vec.append(T)
            new_patients_label.append(L)
        for j in range(len(patients_vec[i]) - 1):
            l = len(patients_vec[i]) - j
            T = []
            L = []
            for k in range(l):
                T.append(patients_vec[i][k])
                L.append(patients_label[i][k])
            new_patients_vec.append(T)
            new_patients_label.append(L)
    return new_patients_vec, new_patients_label


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
            # if p.iloc[j]['Event'] in  [True]:
            # if p.iloc[j]['Event'] in  [True]:
            #   patients_label[i].append(1)
            # if p.iloc[j]['Event'] in [False]:
            #   patients_label[i].append(0)
            temp = temp.drop('Patient name and eye')
            temp = temp.drop("Patient number")
            # temp = temp.drop("Event")
            # temp = temp.drop("stop")
            # temp = temp.drop("start")
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
        if len(T) > no_visit or len(T) == no_visit:
            K = []
            L = []
            for j in range(len(T) - no_visit, len(T)):
                K.append(T[j])
                L.append(P[j])
            new_patients_vec.append(K)
            new_patients_label.append(L)
    return new_patients_vec, new_patients_label
