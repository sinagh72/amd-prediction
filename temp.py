import pandas as pd
import pickle
from ModelTraining import model_training
from data_prepration import training_data, testing_data, dataaugmentation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = './data/'
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'Imaging_clinical_feature_set_folds_outcomes_07_25_2018.xls')

df_cov = pd.read_excel(TRAIN_DATA_DIR)
df_cov = df_cov.fillna('N/A')

m = 3
strm = 'Outcome at ' + str(m) + ' months'
patients_vec_train, patients_label_train, Seq_len = training_data(df_cov, strm)
