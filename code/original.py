import pandas as pd
import csv
import numpy as np
import os

from random import shuffle
from keras.models import Model
from keras.layers import Activation, Input, Concatenate, Reshape, MaxPooling2D, Conv2DTranspose, MaxPooling1D, Lambda
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

import tensorflow as tf
import keras as K

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: tf.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)
    return x

"""
    Main entry start >>>
"""
def main():

    # train()
    predict()

    return

def predict():

    # data_eval_path = './data/eval/'

    DATA_TEST_PATH = os.path.join('ai/data/test_sample/')
    # data_eval_path = os.path.join('../data/test_sample/')
    X_EVAL, X_EVAL_raw = read_data(DATA_TEST_PATH)
    # print(X_EVAL_raw)

    _MODEL_ = cnn()
    _MODEL_.load_weights(os.path.join('ai/models/fin/aiocp_24.h5'))

    """
        autoencoded threshold [12.689298069851413, 0.8731070181658912]
    """
    # result = _MODEL_.evaluate(X_EVAL,X_EVAL)
    # print(result)
    # print(X_EVAL[0:1,:,:])
    for i in range(1):
        result = _MODEL_.evaluate(X_EVAL[i:i+1,:,:],X_EVAL[i:i+1,:,:])
        predicted = _MODEL_.predict(X_EVAL[i:i+1,:,:])
        print(i, X_EVAL[i:i+1,:,:50], predicted[i][:50])
        if result[0] > 12.689298069851413 and result[1] < 0.8731070181658912:
            print("Abnormal OCP Command Detected at row:", i)
            print(X_EVAL_raw[i][:50])
        else:
            print("Normal OCP Command")
        # pred_res = _MODEL_.predict(X_EVAL[i:i+1,:,:])
        # print(result)
        # print(pred_res)


    return


def train():
    _MODEL_ = cnn()
    # _MODEL_.load_weights('./models/nf4/aiocp_49.h5')

    os.system('pause')
    
    data_train_path = './data/train/'
    data_eval_path = './data/eval/'
    # data_train_path = './data2/train/'
    # data_eval_path = './data2/eval/'
    X_TRAIN, _ = read_data(data_train_path)
    X_EVAL, _ = read_data(data_eval_path)

        
    BATCH_SIZE = 512
    EPOCHS = 25
    STEPS = int(X_TRAIN.shape[0]/BATCH_SIZE)
    print('STEPS PER EPOCH:', STEPS)


    for epoch in range(1, EPOCHS):
        # global CONFUSION_METRIC
        np.random.shuffle(X_TRAIN)
        # Initialize CONFUSION_METRIC before training epoch
        for step in range(STEPS):
            # os.system("cls"); 
            print("EPOCHS : {} ************** step : {}/{}".format(epoch, step, STEPS))
            X_Train_Batch = X_TRAIN[step * BATCH_SIZE : step * BATCH_SIZE + BATCH_SIZE,:,:]
            result = _MODEL_.fit(X_Train_Batch, X_Train_Batch, batch_size=BATCH_SIZE)


        # eval_res = _MODEL_.evaluate(Xs_VAL, Y_VAL)

        ########## Save model ######################
        if epoch // 50 == 0:
            _MODEL_.save('./models/aiocp_' + str(epoch) + '.h5')

    return 
"""
    Main entry end <<<
"""




alpha=0.84
def mixed_cost(y_true, y_pred):
    # MSE
    mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y_true, y_pred), 1))
    # MS SSIM
    ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(y_true, y_pred, 1))
    # Mixed cost
    cost = alpha*ssim + (1 - alpha)*mse
    # return cost
    # return cost
    return ssim


# Model Architecture
def cnn(input_shape=(2304, 1)):

    # epoch number, 150 is default value following the published paper
    epochs=150
    # train steps, 800 is default value following the published paper
    train_steps=800
    # init learning rate, 1e-3 is default value following the published paper
    learning_rate=1e-6
    # number of epochs that learning rate will be decreased
    epochs_to_reduce_lr=100
    # decrease factor. After epochs_to_reduce_lr, the learning_rate will be decreased by reduce_lr
    reduce_lr=0.25

    inputs = Input(input_shape)

    x = Convolution1D(16, (36), activation='relu', padding='same')(inputs)
    x = Convolution1D(16, (36), activation='relu', padding='same')(x)
    x = Convolution1D(16, (36), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(4))(x)
    x = Convolution1D(32, (36), activation='relu', padding='same')(x)
    x = Convolution1D(32, (36), activation='relu', padding='same')(x)
    x = Convolution1D(32, (36), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(4))(x)
    x = Convolution1D(64, (36), activation='relu', padding='same')(x)
    x = Convolution1D(64, (36), activation='relu', padding='same')(x)
    x = Convolution1D(64, (36), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(4))(x)
    x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(4))(x)
    # x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    # x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    # x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    # x = MaxPooling1D(pool_size=(2))(x)
    # x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    # x = Convolution1D(128, (36), activation='relu', padding='same')(x)
    encoded = x

    # upconv3 = Conv1DTranspose(64, 2, strides=2, padding='same')(conv3)
    y = Conv1DTranspose(encoded, 128, 4, strides=4, padding='same')
    y = Convolution1D(128, (36), activation='relu', padding='same')(y)
    y = Convolution1D(128, (36), activation='relu', padding='same')(y)
    y = Convolution1D(128, (36), activation='relu', padding='same')(y)
    # upconv3 = Conv1DTranspose(64, 2, strides=2, padding='same')(conv3)
    # y = Conv1DTranspose(y, 128, 4, strides=2, padding='same')
    # y = Convolution1D(128, (36), activation='relu', padding='same')(y)
    # y = Convolution1D(128, (36), activation='relu', padding='same')(y)
    # y = Convolution1D(128, (36), activation='relu', padding='same')(y)
    # upconv3 = Conv1DTranspose(64, 2, strides=2, padding='same')(conv3)
    y = Conv1DTranspose(y, 128, 4, strides=4, padding='same')
    y = Convolution1D(64, (36), activation='relu', padding='same')(y)
    y = Convolution1D(64, (36), activation='relu', padding='same')(y)
    y = Convolution1D(64, (36), activation='relu', padding='same')(y)
    # upconv3 = Conv1DTranspose(64, 2, strides=2, padding='same')(conv3)
    y = Conv1DTranspose(y, 64, 4, strides=4, padding='same')
    y = Convolution1D(32, (36), activation='relu', padding='same')(y)
    y = Convolution1D(32, (36), activation='relu', padding='same')(y)
    y = Convolution1D(32, (36), activation='relu', padding='same')(y)
    # upconv4 = Conv1DTranspose(32, 2, strides=2, padding='same')(conv4)
    y = Conv1DTranspose(y, 32, 4, strides=4, padding='same')
    y = Convolution1D(16, (36), activation='relu', padding='same')(y)
    y = Convolution1D(16, (36), activation='relu', padding='same')(y)
    y = Convolution1D(16, (36), activation='relu', padding='same')(y)
    # upconv5 = Conv1DTranspose(16, 2, strides=2, padding='same')(conv5)
    # y = Conv1DTranspose(y, 16, 2, strides=2, padding='same')
    decoded = Convolution1D(1, (36), activation='sigmoid', padding='same')(y)

    outputs = decoded
    
    print("Built model..")
    cnn = Model(inputs=inputs, outputs=outputs, name="CNN6")
    cnn.summary()
    cnn.compile(
        optimizer=Adam(lr=learning_rate),
        loss='mean_absolute_percentage_error',
        metrics=['accuracy'])

    return cnn

def read_data_file(inputfile):
    
    index = 0
    max_data_size = 2048
    sck_col_index = 5
    ocp_col_index = 17
    data_col_index = 33
    sck_size = ocp_col_index - sck_col_index
    ocp_size = data_col_index - ocp_col_index
    dividend_padding_size = 228
    
    with open(inputfile, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_file = []
        data_file_raw = []

        for lines in csv_reader:
            index += 1
            end_col_index = len(lines)
            shape = (data_col_index - sck_col_index + max_data_size + dividend_padding_size)
            data_record = np.ones(shape, dtype=float, order='C')

            if index == 1 or index == 2:
                continue
            try:
                # print('debug', index)
                if not lines[sck_col_index] == '':
                    for i in range(sck_size):
                        data_record[i] = int(lines[sck_col_index + i], 16)
                if not lines[ocp_col_index] == '':
                    for i in range(ocp_size):
                        data_record[i + sck_size] = int(lines[ocp_col_index + i], 16)
                if end_col_index > data_col_index:
                    for i in range(end_col_index - data_col_index):
                        data_record[i + sck_size + ocp_size] = int(lines[data_col_index + i], 16)

                # print(end_col_index, len(data_record), data_col_index)

                # print (data_record[0:end_col_index - sck_col_index])
                # print (data_record.shape)
                data_file.append(data_record)
                # print('sck', lines[sck_col_index:ocp_col_index])
                # print('ocp', lines[ocp_col_index:data_col_index])
                # print('data', lines[data_col_index:end_col_index])
                # print (int(lines[sck_col_index], 16))

                data_file_raw.append(lines)

            except Exception as e: 
                print(e)
        
    
    data_file = np.asarray(data_file)
    data_file = np.expand_dims(data_file, axis=2)
    print(data_file.shape)

    return data_file, data_file_raw

def read_data(data_location):
    
    # data_location = './data/'

    full_data = None

    for filename in os.listdir(data_location):
        print(filename)
        file_fullpath = data_location + filename 
        if full_data is None:
            full_data, full_data_raw = read_data_file(file_fullpath)
        else:
            data, data_raw = read_data_file(file_fullpath)
            print("debug",data_raw)
            full_data = np.concatenate((full_data, data_raw), axis=0)
            full_data_raw.extends(data_raw)

        
    print (full_data.shape)

    return full_data, full_data_raw


if __name__ == '__main__':
    main()
    