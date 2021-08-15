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
DATA_BASE_PATH = os.path.join('../data/')
#DATA_TRAIN_PATH = os.path.join('../data/train/')
DATA_EVAL_PATH = os.path.join('../data/eval/')
DATA_TEST_PATH = os.path.join('../data/test/')
RESULT_FOLDER = "../deploy/result"
# DATA_TEST_PATH = os.path.join('ai/data/test/')
# RESULT_FOLDER = "ai/models/unet_datacommand25"
os.makedirs(RESULT_FOLDER, exist_ok = True)

def main():
    config = ConfigProto()
    session = InteractiveSession(config=config)
    X_TRAIN, X_TRAIN_raw, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw = createdata(create=True)
    _MODEL_ = fcn()   
    _MODEL_.load_weights(os.path.join(RESULT_FOLDER,'1200.h5'))      
    validate(_MODEL_, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw)    
    return

def validate(_MODEL_, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw):    
    eval_res = _MODEL_.evaluate(X_EVAL,X_EVAL)
    
    df_abnormal = pd.DataFrame(columns=["ret", "row", "sck", "ocp", "data", "data_hex", "predicted", "loss", "acc"])
    abnormal, normal = [], []
    
    start = time.time()
    for i in range(X_TEST.shape[0]):
        result = _MODEL_.evaluate(X_TEST[i:i+1,:,:],X_TEST[i:i+1,:,:],)        
        if result[0] > eval_res[0]:     
            print("abnormal")                           
            abnormal.append(str(i+2))
            df_abnormal = df_abnormal.append({"ret":"abnormal",
                        "row": i+2, 
                        "sck": X_TEST_raw[i][5:17],
                        "ocp": X_TEST_raw[i][17:33],
                        "data": X_TEST_raw[i][33:],
                        "data_hex":"",
                        "predicted": "",
                        "loss": "{:.2}".format(result[0]),
                        "acc": result[1]}, ignore_index = True)                    
        else:
            normal.append(str(i+2))
            df_abnormal = df_abnormal.append({"ret":"normal",
                        "row": i+2, 
                        "sck": X_TEST_raw[i][5:17],
                        "ocp": X_TEST_raw[i][17:33],
                        "data": X_TEST_raw[i][33:],
                        "data_hex":"",
                        "predicted": "",
                        "loss": "{:.2}".format(result[0]),
                        "acc": result[1]}, ignore_index = True)        
    
    end = time.time()    
    df_abnormal = df_abnormal.append(pd.Series(), ignore_index=True)
    print("******************************************* Results*******************************************")
    print("Number of records: {}".format(X_TEST.shape[0]))
    print("Number of abnormal: {}".format(len(abnormal)))
    print("Number of normal: {}".format(len(normal)))
    print("Number of processing time: {:.0f}s".format(end-start))
    print("Average processing time: {:.2f}s/record".format((end-start)/X_TEST.shape[0]))
    df_abnormal[["row","ret","data","loss"]].to_csv(os.path.join(RESULT_FOLDER, "Result.csv"), index = False)
    return



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
    