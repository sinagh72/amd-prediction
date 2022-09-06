import tensorflow as tf
from sklearn.utils import class_weight
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM, Dense, TimeDistributed
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from data_prepration import training_data, testing_data, dataaugmentation
import keras as k
import numpy as np
import os, glob


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

    model.add(LSTM(n, return_sequences=True, activation='sigmoid', stateful=False, input_shape=(slen, num_features),
                   name='lstm_1'))
    # model.add(BatchNormalization(mode=0))
    model.add(BatchNormalization())
    # model.add(LSTM(1000, return_sequences=True, activation='sigmoid', name='LSTM_2'))
    model.add(Dropout(0.2, name='dropout'))
    model.add(LSTM(n, return_sequences=True, activation='sigmoid', stateful=False, name='lstm_2'))
    # model.add(Dropout(0.2))
    # model.add(LSTM(25, return_sequences=True, activation='sigmoid', name='LSTM_3'))
    # model.add(Activation("relu", name = 'Relu_activation'))
    # model.add(LeakyReLU(alpha=0.3, name = 'LeakyRelu_activation'))
    # model.add(TimeDistributed(Dense(1, activation='relu', name='dense_sigmoid'),name ='TimeDis_main_output'))
    model.add(TimeDistributed(Dense(3, activation='softmax', name='Softmax'), name='TimeDis_main_output'))
    # model.add(TimeDistributed(Dense(2, activation='relu', name='dense_sigmoid'),name ='TimeDis_main_output'))
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001), sample_weight_mode='temporal')

    return model


def model_training(df_train, df_test, m, fold, n, flag, strm, val_flag):
    train = df_train[df_train['Fold number'] != fold]
    # train = df_train[df_train['Fold number'] % 5 != fold - 1]
    train = train.reset_index(drop=True)
    patients_vec_train, patients_label_train, Seq_len = training_data(train, strm)
    print("#train: ", len(patients_vec_train))
    # if val_flag == 0:
    #     test = df_miami
    #     test = test.reset_index(drop=True)
    #     #print (list(test))
    #     patients_vec_test, patients_label_test, Seq_len_test, row = testing_data(test, strm)
    # elif val_flag == 1:
    #     test = df_cov9[df_cov9['Fold number'] % 5 == fold-1]
    #     test = test.reset_index(drop=True)
    #     #print (list(test))
    #     patients_vec_test, patients_label_test, Seq_len_test = training_data(test, strm)

    val = df_train[df_train['Fold number'] == fold]
    val = val.reset_index(drop=True)
    patients_vec_val, patients_label_val, Seq_len_val = training_data(val, strm)
    print("#val: ", len(patients_vec_val))

    slen = max(max(Seq_len), max(Seq_len_val))

    print('Slen: ' + str(slen))

    X_train_aug, y_train_aug = dataaugmentation(patients_vec_train, patients_label_train)
    X_train = pad_sequences(X_train_aug, slen, padding='post', truncating='post', value=0, dtype='float32')
    Y_train = pad_sequences(y_train_aug, slen, padding='post', truncating='post', value=2.)

    X_val = pad_sequences(patients_vec_val, slen, padding='post', truncating='post', value=0, dtype='float32')
    Y_val = pad_sequences(patients_label_val, slen, padding='post', truncating='post', value=2.)

    Y_categorical_train = k.utils.np_utils.to_categorical(Y_train, 3)
    Y_categorical_train = Y_categorical_train.reshape(Y_train.shape[0], Y_train.shape[1], 3)
    Y_categorical_val = k.utils.np_utils.to_categorical(Y_val, 3)
    Y_categorical_val = Y_categorical_val.reshape(Y_val.shape[0], Y_val.shape[1], 3)

    y_train = Y_categorical_train
    y_val = Y_categorical_val

    filepath = "./weights/Harbor" + str(m) + "monweights-improvement-{epoch:02d}-{val_precision:.3f}.h5py"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_precision_1', verbose=1, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_precision', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_precision', mode='max', verbose=1, patience=25)
    callbacks_list = [checkpoint, es]
    num_features = X_train.shape[2]
    model = create_model(slen, num_features, n)
    print('slen+nfeatures+nn = ', slen, num_features, n)
    #     model.save('OCT_model.h5')
    #     print('Model saved!!')
    try:
        wei = list(Y_train.reshape(X_train.shape[0] * slen))
        print('len weight: ', len(wei))
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(wei),
                                                          wei)
        weights = np.array([class_weights[0], class_weights[1], class_weights[2]])
    except:
        weights = np.array([1, 50, 0.1])
    print(weights)
    loss = weighted_categorical_crossentropy(weights)
    if flag == 1:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)],
                      metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
                      optimizer=optimizers.RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=1e-6))
    else:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)],
                      metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
                      optimizer=optimizers.Adam(learning_rate=0.001, decay=1e-6))
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=100,
                        validation_data=(X_val, y_val), callbacks=callbacks_list, shuffle=True)
    # list_of_files = glob.glob('./weights/*.h5py')  # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # print('latest file', latest_file)
    # bestmodel = create_model(slen, num_features, n)
    # bestmodel.load_weights(latest_file)

    bestmodel = model
    model_filename = 'models/OCT_model_with_weights_' + str(m) + '_' + str(n) + '_' + str(fold) + '.h5'

    bestmodel.save(model_filename)
    print('Model saved!!: ', model_filename)

    batch_size = 50

    preds = bestmodel.predict(X_val, batch_size=batch_size)
    y_pred = preds.reshape(X_val.shape[0] * slen, 3)
    y_true = y_val.reshape(X_val.shape[0] * slen, 3)

    # y_true_categorical = np.argmax(y_true[y_true[:, 2] == 0], axis=1)
    # y_pred_categorical = np.argmax(y_pred[y_true[:, 2] == 0], axis=1)
    y_true_categorical = y_true[y_true[:, 2] == 0, 1]
    y_pred_score = y_pred[y_true[:, 2] == 0, 1]

    fpr, tpr, thresholds = roc_curve(y_true_categorical, y_pred_score)
    roc_auc = auc(fpr, tpr)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true_categorical, y_pred_score)
    lr_auc = auc(lr_recall, lr_precision)

    return fpr, tpr, roc_auc, preds, y_pred, y_true, lr_precision, lr_recall, lr_auc, slen


def model_using(df_train, m, fold, n, strm, model_path):
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
    # X_train_aug, y_train_aug = dataaugmentation(patients_vec_train, patients_label_train)
    # X_train = pad_sequences(X_train_aug, slen, padding='pre', truncating='pre', value=0, dtype='float32')
    # Y_train = pad_sequences(y_train_aug, slen, padding='pre', truncating='pre', value=2.)
    #
    X_val = pad_sequences(patients_vec_val, slen, padding='pre', truncating='pre', value=0, dtype='float32')
    Y_val = pad_sequences(patients_label_val, slen, padding='pre', truncating='pre', value=2.)

    # Y_categorical_train = k.utils.np_utils.to_categorical(Y_train, 3)
    # Y_categorical_train = Y_categorical_train.reshape(Y_train.shape[0], Y_train.shape[1], 3)
    Y_categorical_val = k.utils.np_utils.to_categorical(Y_val, 3)
    Y_categorical_val = Y_categorical_val.reshape(Y_val.shape[0], Y_val.shape[1], 3)

    # y_train = Y_categorical_train
    y_val = Y_categorical_val

    num_features = X_val.shape[2]

    model = create_model(slen, num_features, n)
    model.load_weights(model_path)
    print('model loaded: ', model_path)
    print('slen+nfeatures+nn = ', slen, num_features, n)
    batch_size = 50

    preds = model.predict(X_val, batch_size=batch_size)
    y_pred = preds.reshape(X_val.shape[0] * slen, 3)
    y_true = y_val.reshape(X_val.shape[0] * slen, 3)

    y_true_categorical = y_true[y_true[:, 2] == 0, 1]
    y_pred_categorical = y_pred[y_true[:, 2] == 0, 1]

    fpr, tpr, thresholds = roc_curve(y_true_categorical, y_pred_categorical)
    roc_auc = auc(fpr, tpr)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true_categorical, y_pred_categorical)
    lr_auc = auc(lr_recall, lr_precision)

    return fpr, tpr, roc_auc, preds, y_pred, y_true, lr_precision, lr_recall, lr_auc, slen
