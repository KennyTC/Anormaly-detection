import pandas as pd
import csv
import numpy as np
import os
import pickle
import random
from matplotlib import pyplot as plt
from random import shuffle
from keras.models import Model, Sequential
from keras.layers import Activation, Input, Concatenate, Reshape, MaxPooling2D, Conv2DTranspose, MaxPooling1D, Lambda, concatenate, BatchNormalization, Dense
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dropout
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time
from keras.callbacks import CSVLogger

DATA_BASE_PATH = os.path.join('ai/data/')
DATA_TRAIN_PATH = os.path.join('ai/data/train/')
DATA_EVAL_PATH = os.path.join('ai/data/eval/')
DATA_TEST_PATH = os.path.join('ai/data/test/')
# RESULT_FOLDER = "../deploy/result"
# DATA_TEST_PATH = os.path.join('ai/data/test/')
RESULT_FOLDER = "ai/models/unet_datacommand26"
os.makedirs(RESULT_FOLDER, exist_ok = True)

def plot_loss(loss, val_loss):        
    plt.figure()
    plt.plot(loss)    
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["train", "val"], loc='upper right')            
    plt.savefig(os.path.join(RESULT_FOLDER, "loss.png"))

def main():
    config = ConfigProto()    
    session = InteractiveSession(config=config)
    
    X_TRAIN, X_TRAIN_raw, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw = createdata(create=True)
    _MODEL_ = lstm3()       
    train(_MODEL_,X_TRAIN, X_EVAL)        
    return




def train(_MODEL_,X_TRAIN, X_EVAL):    

    BATCH_SIZE = 512
    EPOCHS, START =1600, 1
    STEPS = int(X_TRAIN.shape[0]/BATCH_SIZE)
    print('STEPS PER EPOCH:', STEPS)
    
    csv_logger = CSVLogger(os.path.join(RESULT_FOLDER, 'log.csv'), append=True, separator=';')    
    df_tracking = pd.DataFrame(columns=["epoch", "step", "actual", "predicted", "loss", "acc", "val_loss", "val_acc"])
    df_tracking_detail = pd.DataFrame(columns=["loss", "acc"])
    for epoch in range(START, START+EPOCHS):
        for step in range(STEPS):
            print("EPOCHS : {} ************** step : {}/{}".format(epoch, step, STEPS))
            X_Train_Batch = X_TRAIN[step * BATCH_SIZE : step * BATCH_SIZE + BATCH_SIZE,:,:]
            result = _MODEL_.fit(X_Train_Batch, X_Train_Batch, batch_size= BATCH_SIZE, callbacks=[csv_logger],verbose = 0)
            df_tracking_detail = df_tracking_detail.append({
                        "loss": result.history["loss"][0], 
                        "acc": result.history["loss"][0]}, ignore_index = True)

        eval_res = _MODEL_.evaluate(X_EVAL, X_EVAL)
        index = random.randint(0, X_Train_Batch.shape[0])
        predicted = _MODEL_.predict(tf.expand_dims(X_Train_Batch[index], axis=0)) 
                       
        df_tracking = df_tracking.append({"epoch": epoch, 
                        "step": step,                             
                        "actual":X_Train_Batch[index,:25,:], 
                        "predicted": predicted[:,:25],
                        "loss": result.history["loss"][0], 
                        "acc": result.history["loss"][0],
                        "val_loss":eval_res[0], 
                        "val_acc": eval_res[1]}, ignore_index = True)


        ########## Save model ######################
        if epoch % 5 == 0 or epoch == START+EPOCHS - 1 or epoch == START+EPOCHS - 2:
            _MODEL_.save(os.path.join(RESULT_FOLDER, str(epoch) + '.h5'))            
    df_tracking.set_index("epoch", inplace=True)    
    df_tracking.to_csv(os.path.join(RESULT_FOLDER, "tracking.csv"), index = False)    
    plot_loss(df_tracking["loss"], df_tracking["val_loss"])

def fcn(input_shape=(512,)):
    input_layer = Input(input_shape)
    # encoding architecture    
    encode_layer1 = Dense(256, activation='relu')(input_layer)    
    encode_layer2 = Dense(128, activation='relu')(encode_layer1)
    encode_layer3 = Dense(64, activation='relu')(encode_layer2)
    # latent view
    latent_view = Dense(32, activation='sigmoid')(encode_layer3)
    # decoding architecture
    decode_layer1 = Dense(64, activation='relu')(latent_view)
    decode_layer2 = Dense(128, activation='relu')(decode_layer1)
    decode_layer3 = Dense(256, activation='relu')(decode_layer2)
    # output layer
    output_layer = Dense(512)(decode_layer3)
    fcn = Model(inputs=input_layer, outputs=output_layer, name="CNN6")
    fcn.summary()
    fcn.compile(
        optimizer=Adam(lr=1e-6),
        loss='mse',
        metrics=['mse'])
    return fcn    
def lstm3(timesteps=512, input_dim=1):    
    
    model = Sequential()
    model.add(LSTM(64, activation="relu", input_shape=(timesteps, input_dim)))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(512))
    model.add(LSTM(units=64, activation="relu", return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(units=1)))
    model.compile(loss='mse', optimizer=Adam(lr=1e-6), metrics=['mse'])
    model.summary()
    return model


def createdata(create=True):
    X_TRAIN, X_TRAIN_raw, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw = None, None, None, None, None, None
    if create:
        X_TRAIN, X_TRAIN_raw = read_data(DATA_TRAIN_PATH)
        with open(os.path.join(DATA_BASE_PATH, "train_agg_datacommand.pkl"), 'wb') as f:
            pickle.dump(X_TRAIN, f)
        with open(os.path.join(DATA_BASE_PATH, "train_raw_agg_datacommand.pkl"), 'wb') as f:
            pickle.dump(X_TRAIN_raw, f)

        X_EVAL, X_EVAL_raw = read_data(DATA_EVAL_PATH)    
        with open(os.path.join(DATA_BASE_PATH,"eval_agg_datacommand.pkl"), 'wb') as f:
            pickle.dump(X_EVAL, f)
        with open(os.path.join(DATA_BASE_PATH,"eval_raw_agg_datacommand.pkl"), 'wb') as f:
            pickle.dump(X_EVAL_raw, f)

        X_TEST, X_TEST_raw = read_data(DATA_TEST_PATH)    
        with open(os.path.join(DATA_BASE_PATH,"test_agg_datacommand.pkl"), 'wb') as f:
            pickle.dump(X_TEST, f)
        with open(os.path.join(DATA_BASE_PATH,"test_raw_agg_datacommand.pkl"), 'wb') as f:
            pickle.dump(X_TEST_raw, f)
    else:
        with open(os.path.join(DATA_BASE_PATH,"train_agg_datacommand.pkl"), "rb") as f:
            X_TRAIN = pickle.load(f)
        with open(os.path.join(DATA_BASE_PATH,"train_raw_agg_datacommand.pkl"), "rb") as f:
            X_TRAIN_raw = pickle.load(f)
        with open(os.path.join(DATA_BASE_PATH, "eval_agg_datacommand.pkl"), "rb") as f:
            X_EVAL = pickle.load(f)
        with open(os.path.join(DATA_BASE_PATH, "eval_raw_agg_datacommand.pkl"), "rb") as f:
            X_EVAL_raw = pickle.load(f)
        with open(os.path.join(DATA_BASE_PATH,"test_agg_datacommand.pkl"), "rb") as f:
            X_TEST = pickle.load(f)
        with open(os.path.join(DATA_BASE_PATH,"test_raw_agg_datacommand.pkl"), "rb") as f:
            X_TEST_raw = pickle.load(f)
    return X_TRAIN, X_TRAIN_raw, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw
def read_data(data_location):
    
    # data_location = './data/'

    full_data = None

    for filename in os.listdir(data_location):
        print(filename)
        file_fullpath = data_location + filename 
        if full_data is None:
            full_data, full_data_raw = read_data_file3(file_fullpath)
        else:
            data, data_raw = read_data_file3(file_fullpath)
            # print("debug",data_raw)
            # print(full_data.shape, data.shape)
            full_data = np.concatenate((full_data, data), axis=0)            
            full_data_raw.extend(data_raw)

        
    print (full_data.shape)   
    
    return full_data, full_data_raw
def read_data_file3(inputfile):
    
    index = 0
    max_data_size = 512
    data_col_index = 33    
    # dividend_padding_size = 64
    
    with open(inputfile, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_file = []
        data_file_raw = []

        for lines in csv_reader:
            index += 1
            end_col_index = len(lines)
            shape = (max_data_size)
            data_record = np.full(shape, fill_value=1, dtype=float, order='C')
            # data_record = np.ones(shape, dtype=float, order='C')
            if index == 1 or index == 2:
                continue
            try:                
                if end_col_index > data_col_index:
                    for i in range(end_col_index - data_col_index):                                                
                        data_record[i] = int(lines[data_col_index + i], 16)
                data_file.append(data_record)
                data_file_raw.append(lines)

            except Exception as e: 
                print(e)
        
    
    data_file = np.asarray(data_file)
    print("before", data_file.shape)
    data_file = np.expand_dims(data_file, axis=2)
    print("after", data_file.shape)

    return data_file, data_file_raw



if __name__ == '__main__':
    main()
    