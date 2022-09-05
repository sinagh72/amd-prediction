from sklearn.utils import class_weight
from keras_preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Input, LSTM, Dense, TimeDistributed
from keras.layers.convolutional import ZeroPadding2D
from keras import optimizers
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.layers import BatchNormalization
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import random
from preprocessing import training_data, testing_data, testaugmentation, dataaugmentation
import numpy as np
import keras_metrics as km
import tensorflow as tf
import glob
import os

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy np.array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ?  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def create_model(slen, num_features, n):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation='sigmoid', stateful=False, input_shape=(slen, num_features),
                   name='LSTM_1'))
    model.add(BatchNormalization())
    model.add(LSTM(20, return_sequences=True, activation='sigmoid', stateful=False, name='LSTM_2'))
    model.add(TimeDistributed(Dense(3, activation='softmax', name='Softmax'), name='TimeDis_main_output'))
    return model


def model_training(df_cov9, df_miami, m, fold, n, flag, strm):
    train = df_cov9[df_cov9['Fold number'] != fold]
    train = train.reset_index(drop=True)
    test = df_miami
    # test= df_cov9[df_cov9['Fold number']==fold]
    test = test.reset_index(drop=True)
    # print (l#Read Data
    temp, patients_vec_train, patients_label_train, Seq_len = training_data(train, strm)
    temp, patients_vec_test, patients_label_test, Seq_len_test, row = testing_data(test, strm)
    slen = max(max(Seq_len), max(Seq_len_test))
    # print('Slen'+str(slen))
    X_train_aug, y_train_aug = dataaugmentation(patients_vec_train, patients_label_train)
    X_train = pad_sequences(X_train_aug, slen, padding='post', truncating='post', value=0, dtype='float32')
    Y_train = pad_sequences(y_train_aug, slen, padding='post', truncating='post', value=2.)
    X_test = pad_sequences(patients_vec_test, slen, padding='post', truncating='post', value=0, dtype='float32')
    Y_test = pad_sequences(patients_label_test, slen, padding='post', truncating='post', value=2.)
    Y_categorical_train = tf.keras.utils.to_categorical(Y_train, 3)
    Y_categorical_train = Y_categorical_train.reshape(Y_train.shape[0], Y_train.shape[1], 3)
    Y_categorical_test = tf.keras.utils.to_categorical(Y_test, 3)
    Y_categorical_test = Y_categorical_test.reshape(Y_test.shape[0], Y_train.shape[1], 3)
    y_train = Y_categorical_train
    y_test = Y_categorical_test
    filepath = "./weights/Miami" + str(m) + "monweights-improvement-{epoch:02d}-{val_precision_1:.3f}.h5py"
    checkpoint = ModelCheckpoint(filepath, monitor='val_precision_1', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    num_features = X_test.shape[2]
    print('num features: ')
    print(num_features)
    model = create_model(slen, num_features, n)
    model.save('OCT_model.h5')
    print('Model saved!!')
    try:
        wei = list(Y_test.reshape(X_test.shape[0] * slen))
        print(len(wei))
        class_weights = class_weight.compute_class_weight('balanced', np.unique(wei), wei)
        weights = np.array([class_weights[0], class_weights[1], class_weights[2]])
    except:
        weights = np.array([1, 50, 0.1])
    print(weights)
    loss = weighted_categorical_crossentropy(weights)
    if flag == 1:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)],
                      metrics=[km.categorical_precision(label=0), km.categorical_precision(label=1),
                               km.categorical_recall(label=0), km.categorical_recall(label=1)],
                      optimizer=optimizers.RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=1e-6))
    else:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)],
                      metrics=[km.categorical_precision(label=0, name="percision 0"), km.categorical_precision(label=1, name="percision 1"),
                               km.categorical_recall(label=0, name="recall 0"), km.categorical_recall(label=1, name="recall 1")],
                      optimizer=optimizers.Adam(learning_rate=0.001, decay=1e-6))
    history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=callbacks_list, shuffle=True)
    list_of_files = glob.glob('./weights/*.h5py')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    bestmodel = create_model(slen, num_features, n)
    bestmodel.load_weights(latest_file)
    batch_size = 50
    preds_prob3mon = bestmodel.predict_proba(X_test, batch_size=batch_size)
    print(preds_prob3mon.shape)
    ind_preds3mon = preds_prob3mon.reshape(X_test.shape[0] * slen, 3)
    ind_Y_test3mon = y_test.reshape(X_test.shape[0] * slen, 3)
    fpr, tpr, thresholds = roc_curve(np.array(ind_Y_test3mon[ind_Y_test3mon[:, 2] == 0, 1]),
                                     np.array(ind_preds3mon[ind_Y_test3mon[:, 2] == 0, 1]))
    roc_auc = auc(fpr, tpr)
    lr_precision, lr_recall, _ = precision_recall_curve(np.array(ind_Y_test3mon[ind_Y_test3mon[:, 2] == 0, 1]),
                                                        np.array(ind_preds3mon[ind_Y_test3mon[:, 2] == 0, 1]))
    lr_auc = auc(lr_recall, lr_precision)

    return fpr, tpr, roc_auc, ind_preds3mon, ind_Y_test3mon, lr_precision, lr_recall, lr_auc
