# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:14:20 2023

@author: Houdini69
"""

import csv
import time
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib 
import warnings
warnings.filterwarnings("ignore")
print("TensorFlow Version:",tf.__version__)
print("GPU Name:",tf.config.list_physical_devices("GPU"))
print("CPU Name:",tf.config.list_physical_devices("CPU"))
with open('data.csv', newline='', encoding = "utf-8-sig") as csvfile:
    rows = csv.reader(csvfile)
    datalist= list(rows)
    del datalist[0]
    y=[]
    y_label=[]
    X=[[],[],[],[],[],[]]
    for i in range(2418):
        
        if int(datalist[i][3])<=50:
            y.append(0)
            #y_label.append("良好")
        elif int(datalist[i][3])<=100:
            y.append(1)
            #y_label.append("普通")
        elif int(datalist[i][3])<=150:
            y.append(2)
            #y_label.append("對敏感族群不健康")
        elif int(datalist[i][3])<=200:
            y.append(3)
            #y_label.append("對所有族群不健康")
        elif int(datalist[i][3])<=300:
            y.append(4)
            #y_label.append("非常不健康")
        else:
            y.append(5)
            #y_label.append("危害")
            
        if datalist[i][4]!='':
            X[0].append(float(datalist[i][4]))
        else:
            X[0].append(4.54)#avg
            
        if datalist[i][5]!='':
            X[1].append(float(datalist[i][5]))
        else:
            X[1].append(4.35)
            
        if datalist[i][7]!='':
            X[2].append(float(datalist[i][7]))
        else:
            X[2].append(36.86)
            
        if datalist[i][8]!='':
            X[3].append(float(datalist[i][8]))
        else:
            X[3].append(30.34)
            
        if datalist[i][9]!='':
            X[4].append(float(datalist[i][9]))
        else:
            X[4].append(40.69)
        if datalist[i][10]!='':
            X[5].append(float(datalist[i][10]))
        else:
            X[5].append(58.12)

    X=np.array(X).T
    y=np.array(y)
    
    #前處理(標準化至-1到1並經過z-score轉換)
    min_max_scaler_n1_to_1=preprocessing.MinMaxScaler(feature_range=(-1,1))
    X=min_max_scaler_n1_to_1.fit_transform(X)
    z_scaler = preprocessing.StandardScaler()
    X= z_scaler.fit_transform(X)
    
    with open("result.txt",'w' , encoding = "utf-8-sig") as f:
        print("結果會保存至資料夾中的result.txt,完成後會顯示完成,共108種結果,若要減少,請從下方參數適當做刪除,但各參數至少要剩一個")
        print("自己測試的結果在資料夾中，檔名為old result")
        #資料切割
        X_train_n = int( 0.6 * len( X ) )
        X_valid_n = int( 0.2 * len( X ) )
        X_test_n = len( X ) - X_train_n - X_valid_n
        print("training_set = " + str( X_train_n  ) + " validation_set = " + str( X_valid_n ) + " testing_set = " + str( X_test_n ) + "\n")
        sampling_rate=6
        
        
        #模型建置及預測
        sequence_length=[3,4,5]
        #sequence_length=[3]
        epochs=[20,30,40]
        #epochs=[2,1]
        optimizers=[tf.keras.optimizers.Adam(learning_rate=0.001),tf.keras.optimizers.Adam(learning_rate=0.0001),tf.keras.optimizers.legacy.SGD(learning_rate=0.001,momentum=0.0),tf.keras.optimizers.legacy.SGD(learning_rate=0.0001,momentum=0.5)]
        optimizers_str=['tf.keras.optimizers.Adam(learning_rate=0.001)','tf.keras.optimizers.Adam(learning_rate=0.0001)','tf.keras.optimizers.SGD(learning_rate=0.001)','tf.keras.optimizers.SGD(learning_rate=0.0001)']
        #optimizers=['Adam']
        batch_sizes=[32,64,128]
        t=0
        total=len(sequence_length)*len(epochs)*len(optimizers)*len(batch_sizes)
        for s in sequence_length:
            for e in epochs:
                for b in batch_sizes:
                    for o in optimizers:
                        t+=1
                        print(t,"/",total)
                        delay=sampling_rate * s
                        batch_size=b
                        X_shape=X.shape
                        value_dic={"sampling_rate":sampling_rate,"sequence_length":s,"delay":delay,"batch_size":batch_size,"X_shape":X_shape}
                        train_dataset=keras.utils.timeseries_dataset_from_array(X[:-delay],targets = y[delay:],sampling_rate=sampling_rate,sequence_length=s,shuffle=True,batch_size=batch_size,start_index = 0,end_index=X_train_n)
                        valid_dataset=keras.utils.timeseries_dataset_from_array(X[:-delay],targets=y[delay:],sampling_rate=sampling_rate,sequence_length=s,shuffle=True,start_index=X_train_n,end_index=X_train_n+X_valid_n)
                        test_dataset=keras.utils.timeseries_dataset_from_array(X[:-delay],targets=y[delay:],sampling_rate=sampling_rate,sequence_length=s,shuffle=True,batch_size=batch_size,start_index=X_train_n+X_valid_n)
                        callbacks=[keras.callbacks.ModelCheckpoint(filepath=".\Checkpoint",save_best_only=True)]
                        inputs=keras.Input(shape=(s,X.shape[-1]))
                        x = layers.LSTM(64,recurrent_dropout=0.25,return_sequences=True)(inputs)
                        x = layers.LSTM(64,recurrent_dropout=0.25)(x)
                        """....."""
                        x=layers.Dense(32)(x)
                        x=layers.Dense(16)(x)
                        outputs=layers.Dense(4)(x)#max(y)=3 => 0~3
                        model=keras.Model(inputs,outputs)
                        model.compile(optimizer=o,loss="mse",metrics=["mae","accuracy"])
                        model.fit(train_dataset,epochs=e,validation_data=valid_dataset,callbacks=callbacks)
                        print()
                        model.summary()
                        print()
                        model = keras.models.load_model(".\Checkpoint")
                        pre_mae=model.evaluate(test_dataset)[1]
                        pre_acc=model.evaluate(test_dataset)[2]
                        print('\n\n')
                        print("Test MAE of sequence_length=",s,",epochs=",e,",batch size =",b,"and optimizers=",optimizers_str[optimizers.index(o)],"is:"+str(pre_mae),"and the accuracy is ",pre_acc)
                        print("Test MAE of sequence_length=",s,",epochs=",e,",batch size =",b,"and optimizers=",optimizers_str[optimizers.index(o)],"is:"+str(pre_mae),"and the accuracy is ",pre_acc,"\n",file=f)
                        print("\n\n")
        ypred=model.predict(test_dataset)
        ypred1=np.argmax(ypred,axis=1)

        f.close()
    print('完成')
    csvfile.close()
    
    
    
    
    
   