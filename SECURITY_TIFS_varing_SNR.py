from __future__ import absolute_import, division, print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import struct
import math
from tensorflow.keras import layers
from tensorflow import keras
np.set_printoptions(precision=3)
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
#tf.disable_v2_behavior()
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
import pickle
#import imageio
import PIL


import glob
import os

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam



## Change byte information to NP array

def byte_to_float(fi, num_bytes):
    LP_vector = []
    i = 0
    while i < num_bytes:
        data = fi.read(4)
        i += 4
        unpacked = struct.unpack("<f", data)
        LP_vector.append(np.abs(unpacked))
    return np.array(LP_vector)



## Slice since by four consecutive waves

def slice_sine_t4(sample):
    period = 48
    period_4 = period*4
    sine_collected = []
    thr = 0.00001
    total_idx = sample.shape[0]

    idx = 0

    while idx < (total_idx - period_4):
        max_val = np.max(sample[idx:idx+period])
        min_val = np.min(sample[idx:idx+period])

        if (max_val - min_val) > thr:
            if min([sample[idx+int(period/2)+period*k] for k in range(4)]) - max([sample[idx+period*k] for k in range(5)]) > thr/2:
                if sample[idx] < sample[idx+1] < sample[idx+2] < sample[idx+3] < sample[idx+4]:
                    data = sample[idx:idx+period_4]
                    data = (data - min(data)) / (max(data)-min(data))
                    sine_collected.append(data)
                    idx += period_4
                else:
                    idx += 1
            else:
                idx += 1
        else:
            idx += 1

    print("success_num = ", len(sine_collected))
    return sine_collected


def read_data(file_name):
    fi = open(file_name,'rb')
    fi.seek(0,2)
    num_bytes = fi.tell()/10
    fi.seek(0)
    fi_val = byte_to_float(fi, num_bytes)
    generated_sine = slice_sine_t4(fi_val)
    return generated_sine


from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print(cm)

    length_change = len(classes)

    fig, ax = plt.subplots(figsize=(11, 22))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ########################
    ##Changed here!!!!!!!!
    plt.ylim([length_change - 0.5, -.5])
    ########################

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax







def cal_precision(prediction, label_val, threshold=0.0):
    # Find the value of last label
    last_val = label_val.shape[1] - 1
    # Find the predicted value by finding maximum
    prediction_max = np.max(prediction, axis=1)
    # If the predicted value is less than threshold, determine that this is unknown tag
    unknown_mat = np.where(prediction_max > threshold, 0, 2).reshape(-1, 1)

    prediction_rev = np.hstack((prediction, unknown_mat))

    prediction_rev_dec = np.argmax(prediction_rev, axis=1)

    # Transform one-hot coded label to decimal value
    label_val_dec = np.argmax(label_val, axis=1)

    # Find the correctly found labels
    true_val = np.equal(label_val_dec, prediction_rev_dec)

    # Generate label data for know_unknown detection
    label_for_unknown_known_detection = np.where(label_val_dec != last_val, 0, 2).reshape(-1, 1)

    # Find the correctly found labels
    known_unknown_val = np.equal(unknown_mat, label_for_unknown_known_detection).astype(float)

    PD = np.sum((label_for_unknown_known_detection / 2) * known_unknown_val) / (
        np.sum(label_for_unknown_known_detection / 2))
    FA = np.sum((1 - label_for_unknown_known_detection / 2) * (1 - known_unknown_val)) / (
        np.sum(1 - label_for_unknown_known_detection / 2))

    return prediction_rev_dec, label_val_dec, true_val.mean(), known_unknown_val.mean(), PD, FA


def cal_precision_2(prediction, label_val, threshold=0.9):
    # Find the value of last label
    last_val = label_val.shape[1] - 1
    # Find the predicted value by finding maximum
    prediction_max = np.max(prediction, axis=1)
    # If the predicted value is less than threshold, determine that this is unknown tag
    unknown_mat = np.where(prediction_max > threshold, 0, 2).reshape(-1, 1)

    prediction_rev = np.hstack((prediction, unknown_mat))

    prediction_rev_dec = np.argmax(prediction_rev, axis=1)
    prediction_rev_dec = (prediction_rev_dec == last_val).astype('float')

    label_val_dec = np.argmax(label_val, axis=1)
    label_val_dec = (label_val_dec == last_val).astype('float')

    return prediction_rev_dec, label_val_dec


def cal_precision_GAN(DNN_model, data_val, label_val, GAN_data_val, GAN_label_val):
    _, val_accu = DNN_model.evaluate(data_val, label_val, verbose=0)
    _, GAN_val_accu = DNN_model.evaluate(GAN_data_val, GAN_label_val, verbose=0)

    print("")
    print("RESULT: VALIDATION SET = %0.2f, GAN SET = %0.2f," % (100 * val_accu, 100 * GAN_val_accu))
    print("")

    return val_accu, GAN_val_accu


def generate_and_save_images(model, epoch, test_input, test_label, n_class):
    predictions = model([test_input, test_label], training=False)

    fig = plt.figure(figsize=(50,4))

    for i in range(predictions.shape[0]):
        plt.subplot(2, n_class/2, i+1)
        plt.plot(predictions[i])

    plt.savefig('GAN_image/image_at_epoch_{:04d}.png'.format(epoch))



def create_data_label(data_name, tag_count, tag_count_thr):

    ## Read tag data whose number of data is larger than this value

    ## Number of tag data to be used during training (Large)
    large_data_tr_num = 1000

    List_known_tr = []
    List_known_val = []
    List_unknown = []

    Label_data_tr = []
    Label_data_val = []

    used_tag_index = []

    ## read precollected tag data
    with open(data_name,"rb") as fr:
        test_list = pickle.load(fr)

    ## Find the minimum number of tag data
    min_val = int(1e10)
    for i in range(tag_count):
        cur_val = np.array(test_list[i]).shape[0]
        if (cur_val < min_val) & (cur_val > tag_count_thr):
            min_val = cur_val


    ## In the following code, only part of data will be used
    index_val = 0
    temp_unknown_list = []
    for i in range(tag_count):
        cur_val = np.array(test_list[i])
        cur_size = cur_val.shape[0]
        if cur_size > tag_count_thr:
            idx_0 = np.random.choice(cur_size, tag_count_thr)
            idx_1 = idx_0[:int(0.9*tag_count_thr)]

            idx_2 = idx_0[-(int(0.1*tag_count_thr)):]

            List_known_tr.append(cur_val[idx_1])
            List_known_val.append(cur_val[idx_2])

            Label_data_tr.append(index_val * np.ones(int(0.9*tag_count_thr)))
            Label_data_val.append(index_val * np.ones(int(0.1*tag_count_thr)))

            index_val += 1
            used_tag_index.append(i)
        else:
            temp_unknown_list.append(i+1)
            List_unknown.append(cur_val)
    print("unknown index")

    print(temp_unknown_list)



    List_known_tr = np.array(List_known_tr)
    List_known_val = np.array(List_known_val)
    List_unknown = np.array(List_unknown)

    Label_data_tr = np.array(Label_data_tr)
    Label_data_val = np.array(Label_data_val)

    used_tag_size = len(used_tag_index)

    ### Code to concanate list_unknown
    List_unknown_temp = np.array([])
    for i in range(List_unknown.shape[0]):
        a = List_unknown[i]
        if i == 0:
            List_unknown_temp = a
        else:
            List_unknown_temp = np.vstack((List_unknown_temp, a))

    List_unknown = List_unknown_temp

    ### Code to concanate list_unknown
    List_known_tr_temp = np.array([])
    List_known_val_temp = np.array([])

    Label_known_tr_temp = np.array([])
    Label_known_val_temp = np.array([])

    for i in range(List_known_tr.shape[0]):
        a = List_known_tr[i]
        b = List_known_val[i]
        c = Label_data_tr[i]
        d = Label_data_val[i]

        if i == 0:
            List_known_tr_temp = a
            List_known_val_temp = b
            Label_known_tr_temp = c
            Label_known_val_temp = d
        else:
            List_known_tr_temp = np.vstack((List_known_tr_temp, a))
            List_known_val_temp = np.vstack((List_known_val_temp, b))
            Label_known_tr_temp = np.hstack((Label_known_tr_temp, c))
            Label_known_val_temp = np.hstack((Label_known_val_temp, d))


    List_known_tr = List_known_tr_temp
    List_known_val = List_known_val_temp
    Label_data_tr = Label_known_tr_temp
    Label_data_val = Label_known_val_temp

    List_known_tr = np.squeeze(List_known_tr, axis=2)
    List_known_val = np.squeeze(List_known_val, axis=2)
    List_unknown = np.squeeze(List_unknown, axis=2)

    Label_data_tr = to_categorical(Label_data_tr, used_tag_size)
    Label_data_val = to_categorical(Label_data_val, used_tag_size)

    perm_1 = np.random.permutation(List_known_tr.shape[0])

    data_tr = List_known_tr[perm_1]
    label_tr = Label_data_tr[perm_1]


    perm_2 = np.random.permutation(List_known_val.shape[0])

    data_val = List_known_val[perm_2]
    label_val = Label_data_val[perm_2]

    perm_3 = np.random.permutation(List_unknown.shape[0])
    unknown_data = List_unknown[perm_3]

    return used_tag_size, data_tr, label_tr, data_val, label_val, unknown_data





def GEN_FNN_MODEL(n_class, n_hidden, n_layer, lr):

    model_dnn = tf.keras.Sequential()
    for i in range(n_layer):
        model_dnn.add(layers.Dense(n_hidden, activation='relu'))
        model_dnn.add(layers.BatchNormalization())
        model_dnn.add(layers.Dropout(0.1))


    model_dnn.add(layers.Dense(n_class, activation='softmax'))

    model_dnn.compile(optimizer=tf.keras.optimizers.Adam(lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    return model_dnn



def GEN_CNN_MODEL(n_class, n_hidden, n_layer, lr):

    model_cnn = tf.keras.Sequential()
    for i in range(n_layer):
        model_cnn = tf.keras.Sequential()
        model_cnn.add(layers.Conv1D(filters=128, kernel_size=6, padding='same', activation='relu'))
        model_cnn.add(layers.Dropout(0.1))


    model_cnn.add(layers.Flatten())
    model_cnn.add(layers.Dense(n_hidden, activation='relu'))
    model_cnn.add(layers.Dense(n_class, activation='softmax'))
    model_cnn.compile(optimizer=tf.keras.optimizers.Adam(lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    return model_cnn



def GEN_RNN_MODEL(n_class, n_hidden, n_layer, lr):

    model_rnn = tf.keras.Sequential()
    model_rnn.add(layers.SimpleRNN(units=100, input_shape=(48, 4), activation="relu"))
    model_rnn.add(layers.Dense(500, activation="relu"))
    model_rnn.add(layers.Dense(n_class, activation='softmax'))
    model_rnn.compile(optimizer=tf.keras.optimizers.Adam(lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])


    return model_rnn



def basic_predict(DNN, data, n_class, thr):
    chan_samples = np.ones((data.shape[0], 1)) * thr
    predict_val = DNN.predict(data)
    predict_val = np.hstack((predict_val, chan_samples))
    return_val = np.argmax(predict_val, axis=1)
    return return_val





def cal_accu(predict_known, true_known, predict_unknown, n_class):
    accu_known = np.equal(predict_known, np.argmax(true_known, axis=1))
    accu_known = accu_known.astype(float)
    accu_known_val = np.mean(accu_known)

    accu_unknown = np.equal(predict_unknown, n_class)
    accu_unknown = accu_unknown.astype(float)
    accu_unknown_val = np.mean(accu_unknown)


    return accu_known_val, accu_unknown_val



def cal_accu_multi(predict_known, true_known, predict_unknown, n_class, n_average, thr):

    found_idx_mat = np.zeros((n_class, ))
    found_val_mat = np.zeros((n_class, n_class))
    found_val_unknown_mat = np.zeros((n_class, ))
    accu_known_tot = 0
    accu_known_true = 0
    accu_unknown_tot = 0
    accu_unknown_true = 0

    for i in range(predict_known.shape[0]):
        found_idx_mat = found_idx_mat + true_known[i]
        true_index = np.argmax(true_known[i])
        found_val_mat[true_index] = found_val_mat[true_index] + predict_known[i]/n_average


        if found_idx_mat[true_index] == n_average:
            found_idx_mat[true_index] = 0
            found_idx_max = np.argmax(found_val_mat[true_index])


            if (true_index == found_idx_max) & (found_val_mat[true_index, found_idx_max] > thr):
                accu_known_true = accu_known_true + 1
            found_val_mat[true_index] = 0
            accu_known_tot = accu_known_tot + 1


    for i in range(predict_unknown.shape[0]):
        found_val_unknown_mat = found_val_unknown_mat + predict_unknown[i]/n_average
        if (i+1) % n_average == 0:
            found_idx_max = np.max(found_val_unknown_mat)
            if (found_idx_max < thr):
                accu_unknown_true = accu_unknown_true + 1
            found_val_unknown_mat = np.zeros((n_class,))
            accu_unknown_tot = accu_unknown_tot + 1

    accu_known_val = accu_known_true/accu_known_tot
    accu_unknown_val = accu_unknown_true / accu_unknown_tot

    return accu_known_val, accu_unknown_val




def print_accu_no_ensemble(DNN_module, data_val, label_val, unknown_data, n_class, multi_num=5, thr = 0.9):
    dnn_a = basic_predict(DNN_module, data_val, n_class, thr)
    dnn_b = basic_predict(DNN_module, unknown_data, n_class, thr)

    accu_known_val, accu_unknown_val = cal_accu(dnn_a, label_val, dnn_b, n_class)

    dnn_known_predict = DNN_module.predict(data_val)
    dnn_unknown_predict = DNN_module.predict(unknown_data)

    accu_known_val_mul, accu_unknown_val_mul = cal_accu_multi(dnn_known_predict, label_val, dnn_unknown_predict, n_class, multi_num, thr)

    return accu_known_val, accu_unknown_val, accu_known_val_mul, accu_unknown_val_mul





def print_accu_ensemble(model_DNN_ESB_1, model_CNN_ESB, model_RNN_ESB, data_val, label_val, unknown_data, CNN_data_val, CNN_data_unknown, RNN_data_val, RNN_data_unknown, n_class, n_unit_sample, multi_num=5, thr = 0.9):
    dnn_a = model_DNN_ESB_1.predict(data_val)
    dnn_b = model_DNN_ESB_1.predict(unknown_data)

    cnn_a = model_CNN_ESB.predict(CNN_data_val)
    cnn_b = model_CNN_ESB.predict(CNN_data_unknown)

    rnn_a = model_RNN_ESB.predict(RNN_data_val)
    rnn_b = model_RNN_ESB.predict(RNN_data_unknown)


    predict_val_a = (dnn_a +  cnn_a + rnn_a) / 3
    predict_val_b = (dnn_b + cnn_b + rnn_b) / 3

    chan_samples_1 = np.ones((data_val.shape[0], 1)) * thr
    chan_samples_2 = np.ones((unknown_data.shape[0], 1)) * thr

    predict_val_a_new = np.hstack((predict_val_a, chan_samples_1))
    predict_val_b_new = np.hstack((predict_val_b, chan_samples_2))

    predict_val_a_new = np.argmax(predict_val_a_new, axis=1)
    predict_val_b_new = np.argmax(predict_val_b_new, axis=1)


    accu_known_val, accu_unknown_val = cal_accu(predict_val_a_new, label_val, predict_val_b_new, n_class)


    accu_known_val_mul, accu_unknown_val_mul = cal_accu_multi(predict_val_a, label_val, predict_val_b, n_class, multi_num,
                                                      thr)

    return accu_known_val, accu_unknown_val, accu_known_val_mul, accu_unknown_val_mul


def print_by_changing_thr_ensemble(model_DNN, data_val, label_val, unknown_data, n_class, type_of_DNN, step_num):
    thr_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    accu_known_val_mat = []
    accu_unknown_val_mat = []
    accu_known_val_mul_mat = []
    accu_unknown_val_mul_mat = []

    for i in range(10):
        thr_temp = thr_list[i]
        accu_known_val, accu_unknown_val, accu_known_val_mul, accu_unknown_val_mul = print_accu_no_ensemble(
            model_DNN, data_val, label_val, unknown_data, n_class, multi_num=step_num, thr=thr_temp)
        accu_known_val_mul_mat.append(accu_known_val_mul)
        accu_unknown_val_mul_mat.append(accu_unknown_val_mul)

    accu_known_val_mul_mat = np.array(accu_known_val_mul_mat)
    accu_unknown_val_mul_mat = np.array(accu_unknown_val_mul_mat)


    print("*" * 50)
    print("")

    print(type_of_DNN)
    print("KNOWN ACCU")
    print(accu_known_val_mul_mat)
    print("UNKNOWN ACCU")
    print(accu_unknown_val_mul_mat)


    return accu_known_val_mul_mat, accu_unknown_val_mul_mat



#########
# 옵션 설정
######
tag_count = 50
total_epoch = 100
batch_size = 128
n_hidden = 256
n_unit_sample = 48
n_input = n_unit_sample*4
n_noise = 32
n_layer_DNN = 6
lr_FNN = 0.0005
MIX_PORTION = 5
gan_gen_num = 10000


used_tag_size, data_tr, label_tr, data_val, label_val, unknown_data = create_data_label("data_large.pickle", tag_count, 40000)
n_class = used_tag_size


unknown_data = unknown_data[:int(16*4000)]

CNN_data_tr = np.reshape(data_tr, (data_tr.shape[0], 1, data_tr.shape[1]))
CNN_data_val = np.reshape(data_val, (data_val.shape[0], 1, data_val.shape[1]))
CNN_data_unknown = np.reshape(unknown_data, (unknown_data.shape[0], 1, unknown_data.shape[1]))


RNN_data_tr = np.reshape(data_tr, (data_tr.shape[0], n_unit_sample, 4))
RNN_data_val = np.reshape(data_val, (data_val.shape[0], n_unit_sample, 4))
RNN_data_unknown = np.reshape(unknown_data, (unknown_data.shape[0], n_unit_sample, 4))


RTA_enn_mat = []
UTA_enn_mat = []
thr_enn_mat = []

for loop_for_tset in range(1):
    training_set_size = data_tr.shape[0]
    batch_size_val = int(training_set_size / 50)
    print(training_set_size)
    print(unknown_data.shape)
    print(data_val.shape)

    model_DNN_BASIC = GEN_FNN_MODEL(n_class, n_hidden, n_layer_DNN, lr_FNN)

    model_CNN_BASIC = GEN_CNN_MODEL(n_class, n_hidden, n_layer_DNN, lr_FNN)

    model_RNN_BASIC = GEN_RNN_MODEL(n_class, n_hidden, n_layer_DNN, lr_FNN)



    model_DNN_BASIC.fit(data_tr[:training_set_size], label_tr[:training_set_size], epochs=2000, batch_size=batch_size_val,
                        validation_data=(data_val, label_val), verbose=2)

    model_CNN_BASIC.fit(CNN_data_tr[:training_set_size], label_tr[:training_set_size], epochs=2000, batch_size=batch_size_val,
              validation_data=(CNN_data_val, label_val), verbose=2)

    model_RNN_BASIC.fit(RNN_data_tr[:training_set_size], label_tr[:training_set_size], epochs=500, batch_size=batch_size_val,
              validation_data=(RNN_data_val, label_val), verbose=2)


    model_DNN_BASIC.save_weights('./weights/model_DNN_BASIC')
    model_CNN_BASIC.save_weights('./weights/model_CNN_BASIC')
    model_RNN_BASIC.save_weights('./weights/model_RNN_BASIC')



    model_DNN_BASIC.load_weights('./weights/model_DNN_BASIC')
    model_CNN_BASIC.load_weights('./weights/model_CNN_BASIC')
    model_RNN_BASIC.load_weights('./weights/model_RNN_BASIC')

    SNR_MAT = np.array([0.11, 0.06, 0.035, 0.01988, 0.011, 0.005])

    for iii in range(6):

        print("#"*50)
        print("#" * 50)
        print("#" * 50)
        print("#" * 50)
        noise_term = SNR_MAT[iii]
        print("ERROR = ",  iii)

        print(data_val.shape)
        data_val_err = np.random.randn(*data_val.shape)*noise_term
        unknown_data_err = np.random.randn(*unknown_data.shape) *noise_term

        CNN_data_val_err = np.random.randn(*CNN_data_val.shape)*noise_term
        CNN_unknown_data_err = np.random.randn(*CNN_data_unknown.shape) * noise_term

        RNN_data_val_err = np.random.randn(*RNN_data_val.shape)*noise_term
        RNN_unknown_data_err = np.random.randn(*RNN_data_unknown.shape) * noise_term


        multi_num_val = 16
        print("DNN")

        print("data_val", data_val.shape)
        print("unkowndata", unknown_data.shape)

        RTA_dnn, UTA_dnn = print_by_changing_thr_ensemble(model_DNN_BASIC, data_val+data_val_err, label_val, unknown_data+unknown_data_err,
                                                                   n_class, "DNN", step_num=multi_num_val)

        print("CNN")
        RTA_cnn, UTA_cnn = print_by_changing_thr_ensemble(model_CNN_BASIC, CNN_data_val+CNN_data_val_err, label_val,
                                                                   CNN_data_unknown+CNN_unknown_data_err, n_class, "CNN", step_num=multi_num_val)

        print("RNN")
        RTA_rnn, UTA_rnn = print_by_changing_thr_ensemble(model_RNN_BASIC, RNN_data_val+RNN_data_val_err, label_val,
                                                                   RNN_data_unknown+RNN_unknown_data_err, n_class, "RNN", step_num=multi_num_val)


        thr_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        accu_known_val_mul_mat = []
        accu_unknown_val_mul_mat = []

        for i in range(10):
            thr_temp = thr_list[i]
            _, _, accu_known_val_mul, accu_unknown_val_mul = print_accu_ensemble(model_DNN_BASIC,  model_CNN_BASIC, model_RNN_BASIC, data_val+data_val_err, label_val, unknown_data+unknown_data_err, CNN_data_val+CNN_data_val_err, CNN_data_unknown+CNN_unknown_data_err, RNN_data_val+RNN_data_val_err, RNN_data_unknown+RNN_unknown_data_err, n_class, n_unit_sample, multi_num=multi_num_val, thr=thr_temp)
            accu_known_val_mul_mat.append(accu_known_val_mul)
            accu_unknown_val_mul_mat.append(accu_unknown_val_mul)

        accu_known_val_mul_mat = np.array(accu_known_val_mul_mat)
        accu_unknown_val_mul_mat = np.array(accu_unknown_val_mul_mat)

        print("ensemble")
        print("KNOWN ACCU")
        print(accu_known_val_mul_mat)
        print("UNKNOWN ACCU")
        print(accu_unknown_val_mul_mat)



