import pickle

import torch
# import argparse
import os
import pandas as pd
import torch.nn.functional as F
from data_prepration import training_data, dataaugmentation
from keras_preprocessing.sequence import pad_sequences
import numpy as np


def load_data():
    base_dir = './data/'
    train_data_dir = os.path.join(base_dir, 'Imaging_clinical_feature_set_folds_outcomes_07_25_2018.xls')
    df_cov = pd.read_excel(train_data_dir)
    df_cov = df_cov.fillna('N/A')
    return df_cov


def preprocess(df, month, fold):
    cuda0 = torch.device('cuda:0')
    print('month:', month)
    strm = 'Outcome at ' + str(month) + ' months'
    df_train = df[df[strm] != 'N/A']
    df_train = df_train.replace('N/A', 0, regex=True)
    print("no of patient:", len(df_train['Patient number'].unique()))
    train = df_train[df_train['Fold number'] % 5 != fold - 1]
    train = train.reset_index(drop=True)
    patients_vec_train, patients_label_train, Seq_len = training_data(train, strm)
    print("#train: ", len(patients_vec_train))
    val = df_train[df_train['Fold number'] % 5 == fold - 1]
    val = val.reset_index(drop=True)
    patients_vec_val, patients_label_val, Seq_len_val = training_data(val, strm)
    print("#val: ", len(patients_vec_val))

    slen = max(max(Seq_len), max(Seq_len_val))

    print('Slen: ' + str(slen))

    x_train_aug, y_train_aug = dataaugmentation(patients_vec_train, patients_label_train)
    x_train = torch.tensor([np.pad(x, [(0, slen - len(x)), (0, 0)], mode='constant', constant_values=0.0)
                            for x in x_train_aug], dtype=torch.float32, device=cuda0)
    y_train = torch.tensor([np.pad(x, (0, slen - len(x)), mode='constant', constant_values=2.0)
                            for x in y_train_aug], dtype=torch.float32, device=cuda0)
    y_train = F.one_hot(y_train, num_classes=-1)

    x_val = torch.as_tensor([np.pad(x, [(0, slen - len(x)), (0, 0)], mode='constant', constant_values=0.0)
                             for x in patients_vec_val], dtype=torch.float32, device=cuda0)
    y_val = torch.as_tensor([np.pad(x, (0, slen - len(x)), mode='constant', constant_values=2.0)
                             for x in patients_label_val], dtype=torch.float32, device=cuda0)
    y_val = F.one_hot(y_val, num_classes=-1)

    # print(X_train.shape)
    # print(Y_train.shape)
    #
    # X_val = torch.nn.utils.rnn.pad_sequence(patients_vec_val, slen, padding='pre', truncating='pre', value=0,
    #                                         dtype='float32')
    # Y_val = torch.nn.utils.rnn.pad_sequence(patients_label_val, slen, padding='pre', truncating='pre', value=2.)
    #
    # Y_categorical_train = k.utils.np_utils.to_categorical(Y_train, 3)
    # Y_categorical_train = Y_categorical_train.reshape(Y_train.shape[0], Y_train.shape[1], 3)
    # Y_categorical_val = k.utils.np_utils.to_categorical(Y_val, 3)
    # Y_categorical_val = Y_categorical_val.reshape(Y_val.shape[0], Y_val.shape[1], 3)
    #
    # y_train = Y_categorical_train
    # y_val = Y_categorical_val


# if __name__ == "__main__":
# parser = argparse.ArgumentParser()
# parser.parse_args()
# parser.add_argument("-d", "--d_model", default=512, type=int,
#                     help="the number of expected features in the encoder/decoder inputs")
#
# parser.add_argument("-nh", "--n_head", default=8, type=int,
#                     help="the number of heads in the multiheadattention models (default=8)")
#
# parser.add_argument("-ne", "--num_encoder_layers", default=6, type=int,
#                     help="the number of sub-encoder-layers in the encoder (default=6)")
#
# parser.add_argument("-nd", "--num_decoder_layers", default=6, type=int,
#                     help="the number of sub-decoder-layers in the decoder (default=6)")
#
# parser.add_argument("-f", "--dim_feedforward", default=2048, type=int,
#                     help="the dimension of the feedforward network model (default=2048)")
#
# parser.add_argument("-dr", "--dropout", default=0.1, type=float,
#                     help="the dropout value (default=0.1)")
#
# parser.add_argument("-a", "--activation", default="relu", type=str, choices=["relu", "gelu"],
#                     help="the activation function of encoder/decoder intermediate layer, "
#                          "can be a string (“relu” or “gelu”) or a unary callable. Default: relu")
#
# parser.add_argument("-le", "--layer_norm_eps", default=1e-5, type=float,
#                     help="the eps value in layer normalization components (default=1e-5)")
#
# parser.add_argument("-b", "--batch_first", action="store_true",
#                     help="If True, then the input and output tensors are provided as (batch, seq, feature). "
#                          "Default: False (seq, batch, feature)")
#
# parser.add_argument("-nf", "--norm_first", action="store_true",
#                     help="if True, encoder and decoder layers will perform LayerNorms before other attention and "
#                          "feedforward operations, otherwise after. Default: False (after)")
#
# args = parser.parse_args()

# d_model = args.d_model
# nhead = args.nhead
# num_encoder_layers = args.num_encoder_layers
# num_decoder_layers = args.num_decoder_layers
# dim_feedforward = args.dim_feedforward
# dropout = args.dropout
# activation = args.activation
# custom_encoder = args.custom_encoder
# custom_decoder = args.custom_decoder
# layer_norm_eps = args.layer_norm_eps
# batch_first = args.batch_first
# norm_first = args.norm_first

# tf_model = torch.nn.Transformer(d_model=d_model,
#                                 nhead=nhead,
#                                 num_encoder_layers=num_encoder_layers,
#                                 num_decoder_layers=num_decoder_layers,
#                                 dim_feedforward=dim_feedforward,
#                                 dropout=dropout,
#                                 activation=activation,
#                                 custom_encoder=custom_encoder,
#                                 custom_decoder=custom_decoder,
#                                 layer_norm_eps=layer_norm_eps,
#                                 batch_first=batch_first,
#                                 norm_first=norm_first)


# X_train = torch.tensor(X_train)
# print(X_train.shape)

#
df = load_data()
folds = [1, 2, 3, 4, 5]
mon = [3, 6, 9, 12, 15, 18, 21]
preprocess(df, mon[0], folds[0])
#  the sequence to the encoder (required)
#     src = None
#     # the sequence to the decoder (required).
#     tgt = None
#
#     src, tgt = load_daata()
#
#     out = tf_model(src, tgt)
