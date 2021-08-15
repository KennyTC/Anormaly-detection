import pandas as pd
import csv
import numpy as np
import os
import pickle
import random
from matplotlib import pyplot as plt
from random import shuffle
from keras.models import Model
from keras.layers import Activation, Input, Concatenate, Reshape, MaxPooling2D, Conv2DTranspose, MaxPooling1D, Lambda, concatenate, BatchNormalization, Dense
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

import tensorflow as tf
import keras as K

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

DATA_TRAIN_PATH = os.path.join('ai/data/train/')
DATA_EVAL_PATH = os.path.join('ai/data/eval/')
DATA_TEST_PATH = os.path.join('../deploy/test/')
RESULT_FOLDER = "../deploy/result"
os.makedirs(RESULT_FOLDER, exist_ok = True)


# tf.test.is_gpu_available(
#     cuda_only=False, min_cuda_compute_capability=None
# )
def plot_loss(loss, val_loss):        
    plt.figure()
    plt.plot(loss)    
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["train", "val"], loc='upper right')            
    plt.savefig(os.path.join(RESULT_FOLDER, "loss.png"))

def plot_loss_detail(loss):        
    plt.figure()
    plt.plot(loss)    
    plt.title('Model loss')
    plt.xlabel('No. of interations')
    plt.ylabel('Loss')            
    plt.savefig(os.path.join(RESULT_FOLDER, "loss_detail.png"))

def plot_acc(acc, val_acc):        
    plt.figure()
    plt.plot(acc)    
    plt.plot(val_acc)
    plt.title('Model acc')
    plt.xlabel('Epoch')        
    plt.ylabel('Accuracy')    
    plt.legend(["train", "val"], loc='upper right')            
    plt.savefig(os.path.join(RESULT_FOLDER, "accuracy.png"))

def plot_acc_detail(acc):        
    plt.figure()
    plt.plot(acc)    
    plt.title('Model acc')
    plt.xlabel('No. of interations')     
    plt.ylabel('Accuracy')    
    plt.savefig(os.path.join(RESULT_FOLDER, "accuracy_detail.png"))

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
    

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

        
    
    # X_TRAIN, X_TRAIN_raw, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw = createdata(create=False)

    

    # _MODEL_ = fcn(input_shape=(512,))   
    # _MODEL_.load_weights(os.path.join(RESULT_FOLDER,'1300.h5'))  
    # train(_MODEL_,X_TRAIN, X_EVAL)    
    
    # _MODEL_.load_weights(os.path.join(RESULT_FOLDER,'1600.h5'))  
    # validate(_MODEL_, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw)
    # validate2(_MODEL_,X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw)    

    inference()
    return

def inference():
    X_TEST, X_TEST_raw = read_data(DATA_TEST_PATH)    
    _MODEL_ = fcn(input_shape=(512,))   
    _MODEL_.load_weights(os.path.join(RESULT_FOLDER,'1600.h5'))  
    
    # test
    df_abnormal = pd.DataFrame(columns=["ret", "row", "sck", "ocp", "data", "data_hex", "predicted", "loss", "acc"])

    abnormal, normal = [], []
    import time
    start = time.time()

    for i in range(X_TEST.shape[0]):
        result = _MODEL_.evaluate(X_TEST[i:i+1,:,:],X_TEST[i:i+1,:,:],)        
        if result[0] > 2.536500930786133:     
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

def createdata(create=True):
    if create:
        X_TRAIN, X_TRAIN_raw = read_data(DATA_TRAIN_PATH)
        with open("ai/data/train_agg_datacommand512_1.pkl", 'wb') as f:
            pickle.dump(X_TRAIN, f)
        with open("ai/data/train_raw_agg_datacommand512_1.pkl", 'wb') as f:
            pickle.dump(X_TRAIN_raw, f)

        X_EVAL, X_EVAL_raw = read_data(DATA_EVAL_PATH)    
        with open("ai/data/eval_agg_datacommand512_1.pkl", 'wb') as f:
            pickle.dump(X_EVAL, f)
        with open("ai/data/eval_raw_agg_datacommand512_1.pkl", 'wb') as f:
            pickle.dump(X_EVAL_raw, f)

        X_TEST, X_TEST_raw = read_data(DATA_TEST_PATH)    
        with open("ai/data/test_agg_datacommand512_1.pkl", 'wb') as f:
            pickle.dump(X_TEST, f)
        with open("ai/data/test_raw_agg_datacommand512_1.pkl", 'wb') as f:
            pickle.dump(X_TEST_raw, f)
    else:
        with open(os.path.join("ai/data/train_agg_datacommand512_1.pkl"), "rb") as f:
            X_TRAIN = pickle.load(f)
        with open(os.path.join("ai/data/train_raw_agg_datacommand512_1.pkl"), "rb") as f:
            X_TRAIN_raw = pickle.load(f)
        with open(os.path.join("ai/data/eval_agg_datacommand512_1.pkl"), "rb") as f:
            X_EVAL = pickle.load(f)
        with open(os.path.join("ai/data/eval_raw_agg_datacommand512_1.pkl"), "rb") as f:
            X_EVAL_raw = pickle.load(f)
        with open(os.path.join("ai/data/test_agg_datacommand512_1.pkl"), "rb") as f:
            X_TEST = pickle.load(f)
        with open(os.path.join("ai/data/test_raw_agg_datacommand512_1.pkl"), "rb") as f:
            X_TEST_raw = pickle.load(f)
    return X_TRAIN, X_TRAIN_raw, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw


def train(_MODEL_,X_TRAIN, X_EVAL):
    
    BATCH_SIZE = 512
    EPOCHS, START =300, 1301
    STEPS = int(X_TRAIN.shape[0]/BATCH_SIZE)
    print('STEPS PER EPOCH:', STEPS)
    

    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger(os.path.join(RESULT_FOLDER, 'log.csv'), append=True, separator=';')    

    df_tracking = pd.DataFrame(columns=["epoch", "step", "actual", "predicted", "loss", "acc", "val_loss", "val_acc"])
    df_tracking_detail = pd.DataFrame(columns=["loss", "acc"])
    for epoch in range(START, START+EPOCHS):
        # global CONFUSION_METRIC
        # Initialize CONFUSION_METRIC before training epoch
        for step in range(STEPS):
            print("EPOCHS : {} ************** step : {}/{}".format(epoch, step, STEPS))
            X_Train_Batch = X_TRAIN[step * BATCH_SIZE : step * BATCH_SIZE + BATCH_SIZE,:,:]
            result = _MODEL_.fit(X_Train_Batch, X_Train_Batch, batch_size= BATCH_SIZE, callbacks=[csv_logger],verbose = 0)
            df_tracking_detail = df_tracking_detail.append({
                        "loss": result.history["loss"][0], 
                        "acc": result.history["loss"][0]}, ignore_index = True)

        eval_res = _MODEL_.evaluate(X_EVAL, X_EVAL)
        index = random.randint(0, 100)
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
    df_tracking.to_csv(os.path.join(RESULT_FOLDER, "tracking.csv"), index = False)    
    
    df_tracking.set_index("epoch", inplace=True)    
    plot_loss(df_tracking["loss"], df_tracking["val_loss"])
    # plot_acc(df_tracking["acc"], df_tracking["val_acc"])

    # df_tracking_detail.reset_index(inplace=True)    
    # plot_loss_detail(df_tracking_detail["loss"])
    # plot_acc_detail(df_tracking_detail["acc"])

def validate(_MODEL_, X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw):    
    
    # validate
    eval_res = _MODEL_.evaluate(X_EVAL,X_EVAL)

    # test
    df_abnormal = pd.DataFrame(columns=["ret", "row", "sck", "ocp", "data", "data_hex", "predicted", "loss", "acc"])

    abnormal, normal = [], []
    import time
    start = time.time()

    for i in range(5):
        result = _MODEL_.evaluate(X_TEST[i:i+1,:,:],X_TEST[i:i+1,:,:],)
        predicted = _MODEL_.predict(X_TEST[i:i+1,:,:])
        if result[0] > eval_res[0]:     
            print("abnormal")                           
            abnormal.append(str(i+2))
            df_abnormal = df_abnormal.append({"ret":"abnormal",
                        "row": i+2, 
                        "sck": X_TEST_raw[i][5:17],
                        "ocp": X_TEST_raw[i][17:33],
                        "data": X_TEST_raw[i][33:],
                        "data_hex":"",
                        "predicted": predicted[0].astype("int"),
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
                        "predicted": predicted[0].astype("int"),
                        "loss": "{:.2}".format(result[0]),
                        "acc": result[1]}, ignore_index = True)        
    
    end = time.time()

    df_abnormal = df_abnormal.append(pd.Series(), ignore_index=True)
    # df_abnormal = df_abnormal.append({"ret":"Information     ---------------------------------------------"}, ignore_index=True)
    # df_abnormal = df_abnormal.append({"ret":"No. of records: {}".format(X_TEST.shape[0])}, ignore_index=True)

    # df_abnormal = df_abnormal.append({"ret":"Abnormal: {}".format(len(abnormal)), 
    #                                   "data ": "{0}".format(", ".join(abnormal))}, ignore_index=True)
    
    # df_abnormal = df_abnormal.append({"ret":"Normal: {}".format(len(normal)), 
    #                                   "data": "{0}".format(", ".join(normal))}, ignore_index=True)
    # df_abnormal = df_abnormal.append({"ret":"Processing time: {:.2}s".format(end-start)}, ignore_index=True)

    print("******************** Information************************")
    print("No. of records: {}".format(X_TEST.shape[0]))
    print("No. of abnormal: {}".format(len(abnormal)))
    print("No. of normal: {}".format(len(normal)))
    print("No. of processing time: {:.2}s".format(end-start))
    df_abnormal[["row","ret","data","loss"]].to_csv(os.path.join(RESULT_FOLDER, "Result.csv"), index = False)


    print(eval_res)

def validate2(_MODEL_,X_EVAL, X_EVAL_raw, X_TEST, X_TEST_raw):

    epochs = [i for i in range(75, 201) if i % 5 == 0]     
    epochs = [1600]   
    df_debug = pd.DataFrame(columns=["epoch", "percentile", "val_loss", "ab_count", "normal_count"])

    for epoch in epochs:
        _MODEL_.load_weights(os.path.join(RESULT_FOLDER,'{0}.h5'.format(epoch)))   
        

        # val_loss = []        
        # for i in range(X_EVAL.shape[0]):
        #     result = _MODEL_.evaluate(X_EVAL[i:i+1,:,:],X_EVAL[i:i+1,:,:],verbose=0)
        #     val_loss.append(result[0])
            
        
        from sklearn.metrics import mean_squared_error        
        X_EVAL =  X_EVAL.reshape(X_EVAL.shape[0], 512)
        X_EVAL_tp = tf.transpose(X_EVAL)
        predicted = _MODEL_.predict(X_EVAL)
        predicted_tp = tf.transpose(predicted)
        print("X_EVAL", X_EVAL.shape, "X_EVAL_tf", X_EVAL_tp.shape) 
        print("predicted", predicted.shape, "predicted_tp", predicted_tp.shape) 
        errors = mean_squared_error(X_EVAL_tp, predicted_tp , multioutput = "raw_values")
        print(errors.shape) #(, )
        val_loss = errors.tolist()
        # with open(os.path.join(RESULT_FOLDER, "{}_val_loss.pkl".format(epoch)), 'wb') as f:
        #     pickle.dump(val_loss, f)
    

        test_loss = []
        test_acc = []
        for i in range(X_TEST.shape[0]):
            result = _MODEL_.evaluate(X_TEST[i:i+1,:,:],X_TEST[i:i+1,:,:], verbose=0)
            test_loss.append(result[0])
            test_acc.append(result[1])
        
        # find the percentile of val s.t < min(test_loss[:78])
        percentiles = [i for i in range(100)]
        threshold_loss = 0
        for percentile in range(100):
            if np.percentile(val_loss, percentile) <  min(test_loss[:78]):                                
                threshold_loss = np.percentile(val_loss, percentile)
                break
        
        normal_count, ab_count = 0, 0
        for i in range(X_TEST.shape[0]):
            result = _MODEL_.evaluate(X_TEST[i:i+1,:,:],X_TEST[i:i+1,:,:],verbose=0)
            if result[0] > threshold_loss:    
                ab_count += 1
            else:
                normal_count +=1
        
        print(epoch, percentile, threshold_loss, ab_count, normal_count)
        df_debug = df_debug.append({"epoch": epoch, 
                "percentile": percentile, 
                "val_loss": threshold_loss,
                "ab_count": ab_count,
                "normal_count": normal_count}, ignore_index=True)

    df_debug.to_csv(os.path.join(RESULT_FOLDER, "debug.csv"), index = False)    

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

def cnn2(input_shape=(512, 1)):

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
    decoded = Convolution1D(1, (36), padding='same')(y)

    outputs = decoded
    
    print("Built model..")
    cnn = Model(inputs=inputs, outputs=outputs, name="CNN6")
    cnn.summary()
    cnn.compile(
        optimizer=Adam(lr=learning_rate),
        loss='mse',
        metrics=['accuracy'])

    return cnn


def fcn(input_shape=(512,)):
    input_layer = Input(input_shape)
    # encoded = Dense(128, activation='relu')(input_layer)
    # encoded = Dense(64, activation='relu')(encoded)    
    # encoded = Dense(32, activation='relu')(encoded)
    # decoded = Dense(64, activation='relu')(encoded)
    # decoded = Dense(128, activation='relu')(decoded)
    # decoded = Dense(512, activation='linear')(decoded)
    # output_layer = decoded


    
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

def unet(input_shape=(2304,1), filters=16, kernel_size=36, batchnom = True):    
    
    inputs = Input(input_shape)
    conv1 = Convolution1D(filters * 1, kernel_size, activation="relu", padding="same")(inputs)
    conv1 = Convolution1D(filters * 1, kernel_size, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1) if batchnom else conv1
    pool1 = MaxPooling1D(pool_size=(4))(conv1)
    
    conv2 = Convolution1D(filters * 2, kernel_size, activation="relu", padding="same")(pool1)
    conv2 = Convolution1D(filters * 2, kernel_size, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2) if batchnom else conv2
    pool2 = MaxPooling1D(pool_size=(4))(conv2)

    conv3 = Convolution1D(filters * 4, kernel_size, activation="relu", padding="same")(pool2)
    conv3 = Convolution1D(filters * 4, kernel_size, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3) if batchnom else conv3
    pool3 = MaxPooling1D(pool_size=(4))(conv3)

    conv4 = Convolution1D(filters * 8, kernel_size, activation="relu", padding="same")(pool3)
    conv4 = Convolution1D(filters * 8, kernel_size, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4) if batchnom else conv4
    pool4 = MaxPooling1D(pool_size=(4))(conv4)
   

    # Middle
    convm = Convolution1D(filters * 16, kernel_size, activation="relu", padding="same")(pool4)
    convm = Convolution1D(filters * 16, kernel_size, activation="relu", padding="same")(pool4)
    convm = BatchNormalization()(convm) if batchnom else convm

    # Expansive
    deconv4 = Conv1DTranspose(convm, filters * 8, kernel_size, strides=4, padding="same")
    uconv4  = concatenate([deconv4, conv4])
    uconv4  = Convolution1D(filters * 8, kernel_size, activation="relu", padding ="same")(uconv4)
    uconv4  = Convolution1D(filters * 8, kernel_size, activation="relu", padding ="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4) if batchnom else uconv4

    deconv3 = Conv1DTranspose(uconv4, filters * 4, kernel_size, strides=4, padding="same")
    uconv3  = concatenate([deconv3, conv3])
    uconv3  = Convolution1D(filters * 4, kernel_size, activation="relu", padding ="same")(uconv3)
    uconv3  = Convolution1D(filters * 4, kernel_size, activation="relu", padding ="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3) if batchnom else uconv3

    deconv2 = Conv1DTranspose(uconv3, filters * 2, kernel_size, strides=4, padding="same")
    uconv2  = concatenate([deconv2, conv2])
    uconv2  = Convolution1D(filters * 2, kernel_size, activation="relu", padding ="same")(uconv2)
    uconv2  = Convolution1D(filters * 2, kernel_size, activation="relu", padding ="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2) if batchnom else uconv2
    
    deconv1 = Conv1DTranspose(uconv2, filters * 1, kernel_size, strides=4, padding="same")
    uconv1  = concatenate([deconv1, conv1])
    uconv1  = Convolution1D(filters * 1, kernel_size, activation="relu", padding ="same")(uconv1)
    uconv1  = Convolution1D(filters * 1, kernel_size, activation="relu", padding ="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1) if batchnom else uconv1

    decoded = Convolution1D(1, kernel_size, activation='sigmoid', padding='same')(uconv1)

    outputs = decoded
    
    print("Built model..")
    unet = Model(inputs=inputs, outputs=outputs, name="CNN6")
    unet.summary()
    unet.compile(
        optimizer=Adam(lr=1e-6),
        loss='mean_absolute_percentage_error',
        metrics=['accuracy'])

    return unet

# using elu instead of relu
def unet2(input_shape=(2304,1), filters=16, kernel_size=36, batchnom = True):        
    inputs = Input(input_shape)
    conv1 = Convolution1D(filters * 1, kernel_size, activation="elu", padding="same")(inputs)
    conv1 = Convolution1D(filters * 1, kernel_size, activation="elu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1) if batchnom else conv1
    pool1 = MaxPooling1D(pool_size=(4))(conv1)
    
    conv2 = Convolution1D(filters * 2, kernel_size, activation="elu", padding="same")(pool1)
    conv2 = Convolution1D(filters * 2, kernel_size, activation="elu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2) if batchnom else conv2
    pool2 = MaxPooling1D(pool_size=(4))(conv2)

    conv3 = Convolution1D(filters * 4, kernel_size, activation="elu", padding="same")(pool2)
    conv3 = Convolution1D(filters * 4, kernel_size, activation="elu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3) if batchnom else conv3
    pool3 = MaxPooling1D(pool_size=(4))(conv3)

    conv4 = Convolution1D(filters * 8, kernel_size, activation="elu", padding="same")(pool3)
    conv4 = Convolution1D(filters * 8, kernel_size, activation="elu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4) if batchnom else conv4
    pool4 = MaxPooling1D(pool_size=(4))(conv4)
   

    # Middle
    convm = Convolution1D(filters * 16, kernel_size, activation="elu", padding="same")(pool4)
    convm = Convolution1D(filters * 16, kernel_size, activation="elu", padding="same")(pool4)
    convm = BatchNormalization()(convm) if batchnom else convm

    # Expansive
    deconv4 = Conv1DTranspose(convm, filters * 8, kernel_size, strides=4, padding="same")
    uconv4  = concatenate([deconv4, conv4])
    uconv4  = Convolution1D(filters * 8, kernel_size, activation="elu", padding ="same")(uconv4)
    uconv4  = Convolution1D(filters * 8, kernel_size, activation="elu", padding ="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4) if batchnom else uconv4

    deconv3 = Conv1DTranspose(uconv4, filters * 4, kernel_size, strides=4, padding="same")
    uconv3  = concatenate([deconv3, conv3])
    uconv3  = Convolution1D(filters * 4, kernel_size, activation="elu", padding ="same")(uconv3)
    uconv3  = Convolution1D(filters * 4, kernel_size, activation="elu", padding ="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3) if batchnom else uconv3

    deconv2 = Conv1DTranspose(uconv3, filters * 2, kernel_size, strides=4, padding="same")
    uconv2  = concatenate([deconv2, conv2])
    uconv2  = Convolution1D(filters * 2, kernel_size, activation="elu", padding ="same")(uconv2)
    uconv2  = Convolution1D(filters * 2, kernel_size, activation="elu", padding ="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2) if batchnom else uconv2
    
    deconv1 = Conv1DTranspose(uconv2, filters * 1, kernel_size, strides=4, padding="same")
    uconv1  = concatenate([deconv1, conv1])
    uconv1  = Convolution1D(filters * 1, kernel_size, activation="elu", padding ="same")(uconv1)
    uconv1  = Convolution1D(filters * 1, kernel_size, activation="elu", padding ="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1) if batchnom else uconv1

    decoded = Convolution1D(1, kernel_size, padding='same')(uconv1)

    outputs = decoded
    
    print("Built model..")
    unet = Model(inputs=inputs, outputs=outputs, name="CNN6")
    unet.summary()
    unet.compile(
        optimizer=Adam(lr=1e-6),
        loss='mse',
        metrics=['accuracy'])

    return unet

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
            data_record = np.full(shape, fill_value=256, dtype=float, order='C')

            if index == 1 or index == 2:
                continue
            try:
                # print('debug', index)
                # if not lines[sck_col_index] == '':
                #     for i in range(sck_size):
                #         data_record[i] = int(lines[sck_col_index + i], 16)
                # if not lines[ocp_col_index] == '':
                #     for i in range(ocp_size):
                #         data_record[i + sck_size] = int(lines[ocp_col_index + i], 16)

                data_record[:sck_size + ocp_size] = 256
                if end_col_index > data_col_index:
                    if lines[data_col_index:data_col_index+7]==['0','1','1','0','0','0','0']: #response command
                        end_col_index = data_col_index + 7
                    for i in range(end_col_index - data_col_index):                        
                        # print(i + sck_size + ocp_size, lines[data_col_index + i], int(lines[data_col_index + i], 16))                        
                        data_record[i + sck_size + ocp_size] = int(lines[data_col_index + i], 16)

                # print(end_col_index, len(data_record), data_col_index)

                # print (data_record[0:end_col_index - sck_col_index])
                # print (data_record.shape)
                data_file.append(data_record)
                # print('sck', lines[sck_col_index:ocp_col_index])
                # print('ocp', lines[ocp_col_index:data_col_index])
                # if lines[data_col_index:end_col_index]==[]:
                #     print('data', lines[data_col_index:end_col_index])
                # print (int(lines[sck_col_index], 16))    

                data_file_raw.append(lines)

            except Exception as e: 
                print(e)
        
    
    data_file = np.asarray(data_file)
    data_file = np.expand_dims(data_file, axis=2)
    print(data_file.shape)

    return data_file, data_file_raw

def read_data_file2(inputfile):
    
    index = 0
    max_data_size = 2048
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
            data_record = np.full(shape, fill_value=256, dtype=float, order='C')

            if index == 1 or index == 2:
                continue
            try:                
                if end_col_index > data_col_index:
                    if lines[data_col_index:data_col_index+7]==['0','1','1','0','0','0','0']: #response command
                        end_col_index = data_col_index + 7
                    for i in range(end_col_index - data_col_index):                                                
                        data_record[i] = int(lines[data_col_index + i], 16)
                data_file.append(data_record)
                data_file_raw.append(lines)

            except Exception as e: 
                print(e)
        
    
    data_file = np.asarray(data_file)
    data_file = np.expand_dims(data_file, axis=2)
    print(data_file.shape)

    return data_file, data_file_raw

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
            full_data, full_data_raw = read_data_file3(file_fullpath)
        else:
            data, data_raw = read_data_file3(file_fullpath)
            # print("debug",data_raw)
            # print(full_data.shape, data.shape)
            full_data = np.concatenate((full_data, data), axis=0)            
            full_data_raw.extend(data_raw)

        
    print (full_data.shape)   
    
    return full_data, full_data_raw


if __name__ == '__main__':
    main()
    