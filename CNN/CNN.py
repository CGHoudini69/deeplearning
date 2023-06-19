# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:29:07 2023

@author: Houdini69
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib 
import fastai 
from fastbook import *
from fastai.vision.widgets import *

print("TensorFlow version:",tf.__version__)
print("fast.ai version:",fastai.__version__)
print("GPU name:",tf.config.list_physical_devices("GPU"))
print("CPU name:",tf.config.list_physical_devices("CPU"))


'''
重要 請先讀我
在執行程式的過程中可能出現Graph execution error
經查閱資料後發現此為tensorflow本身的bug 目前並未找到解決方式
重新執行程式就有機會避免
'''


#人工智慧概論老師提供的網路抓圖片爬蟲及路徑驗證
keywords={"Macaroni":"Macaroni","Spaghetti":"Spaghetti","Rotini":"Rotini","Penne":"Penne"}
path_train=Path('./Data/Pasta/train')
path_test=Path('./Data/Pasta/test')
array=keywords.items()
if not path_train.exists() :
    os.makedirs(path_train)
    for key,value in array:
        print("\n",key,value)
        dest=(path_train/key)
        dest.mkdir(exist_ok=True)
        urls=search_images_ddg(f"{value}",max_images=200)
        download_images(dest,urls=urls,n_workers=0)

fns=get_image_files(path_train) 
def verify_images(fns) :
    "Find images in `fns` that can't be opened"
    return L(fns[i]for i,o in enumerate(fns.map(verify_image)) if not o)
failed=verify_images(fns)
print(failed.map(Path.unlink))
with open("result.txt",'w' , encoding = "utf-8-sig") as f:
    print("結果會保存至資料夾中的result.txt,完成後會顯示完成,共32種結果(需要約一天),若要減少,請從下方參數適當做刪除,但各參數至少要剩一個")
    print("自己測試的結果在資料夾中，檔名為old result")
    epochs=[20,30]
    #epochs=[2,1]
    optimizers=[tf.keras.optimizers.Adam(learning_rate=0.001),tf.keras.optimizers.Adam(learning_rate=0.0001),tf.keras.optimizers.legacy.SGD(learning_rate=0.001,momentum=0.05),tf.keras.optimizers.legacy.SGD(learning_rate=0.0001,momentum=0.5)]
    optimizers_str=['tf.keras.optimizers.Adam,learning_rate=0.001','tf.keras.optimizers.Adam,learning_rate=0.0001,','tf.keras.optimizers.SGD,learning_rate=0.001,','tf.keras.optimizers.SGD,learning_rate=0.0001']
    #optimizers=['Adam']
    batch_sizes=[16,32]
    classifier=[1,2]
    t=1
    total=len(classifier)*len(epochs)*len(optimizers)*len(batch_sizes)
    for c in classifier:
        for e in epochs:
            for b in batch_sizes:
                for o in optimizers:
                    print(t,"/",total)
                    t+=1
                    #資料轉換
                    datagenerator1=ImageDataGenerator(
                        featurewise_center=True,
                        samplewise_center=False,
                        zca_epsilon=1e-06,
                        rotation_range=45,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        brightness_range=None,
                        shear_range=0.3,
                        zoom_range=0.2,
                        channel_shift_range=0.0,
                        fill_mode='nearest',
                        horizontal_flip=False,
                        vertical_flip=True,
                        rescale=1./255,
                        validation_split=0.0,
                        interpolation_order=1,
                        dtype=None
                    )
                    
                    training_set=datagenerator1.flow_from_directory(
                        path_train,
                        target_size=(256, 256),
                        color_mode='rgb',
                        classes=None,
                        class_mode="binary",
                        batch_size=b,
                        shuffle=True,
                        seed=0,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='jpg',
                        follow_links=False,
                        subset=None,
                        interpolation='nearest',
                        keep_aspect_ratio=False
                    )
                    
                    datagenerator2=ImageDataGenerator(
                        rescale=1./255 
                    )
                    
                    testing_set=datagenerator2.flow_from_directory(
                        path_test ,
                        target_size=(256,256),
                        batch_size=int(b/4),
                        class_mode="binary",
                        save_format='jpg',
                        seed=0,
                    )
                    
                    if c==1:
                        #設定模型
                        c1=keras.models.Sequential()
                        c1.add(layers.Conv2D(32,(5,5),input_shape=(256,256,3),activation="relu",padding="same"))
                        c1.add(layers.MaxPooling2D(pool_size=(2,2)))
                        c1.add(layers.Conv2D(64,(5,5),activation="relu",padding="same"))
                        c1.add(layers.Conv2D(32,(3,3),activation="relu",padding="same"))
                        c1.add(layers.MaxPooling2D(pool_size=(2,2)))
                        c1.add(layers.Conv2D(64,(2,2),activation="relu",padding="same"))
                        c1.add(layers.Flatten())
                        c1.add(layers.Dense(64,activation="relu"))
                        c1.add(layers.Dropout(0.2))
                        c1.add(layers.BatchNormalization())
                        c1.add(layers.Dense(64,activation="relu"))
                        c1.add(layers.Dense(32,activation="softmax"))
                        c1.add(layers.Dropout(0.2))
                        c1.add(layers.BatchNormalization())
                        c1.add(layers.Dense(16,activation="relu"))
                        c1.add(layers.Dense(4,activation="sigmoid"))
                        c1.compile(optimizer=o,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
                        c1.summary()
                        c1.fit(training_set,validation_data=testing_set,batch_size=b,epochs=e)
                        
                        test_loss , test_acc = c1.evaluate( testing_set )
                        print('\n\n')
                        print("Test Accuracy of epochs=",e,",batch size =",b,"and optimizers=",optimizers_str[optimizers.index(o)],"with classifier1 is:"+str(test_acc))
                        print("Test Accuracy of epochs=",e,",batch size =",b,"and optimizers=",optimizers_str[optimizers.index(o)],"with classifier1 is:"+str(test_acc),"\n",file=f)
                        print("\n\n")
                    
                    else:
                        c2=keras.models.Sequential()
                        c2.add(layers.Conv2D(32,(5,5),input_shape=(256,256,3),activation="relu",padding="same"))
                        c2.add(layers.Conv2D(32,(5,5),activation="relu",padding="same"))
                        c2.add(layers.MaxPooling2D(pool_size=(2,2)))
                        c2.add(layers.Conv2D(64,(5,5),activation="relu",padding="same"))
                        c2.add(layers.Conv2D(32,(3,3),activation="relu",padding="same"))
                        c2.add(layers.AveragePooling2D(pool_size=(2,2)))
                        c2.add(layers.Conv2D(64,(2,2),activation="relu",padding="same"))
                        c2.add(layers.Conv2D(64,(2,2),activation="relu",padding="same"))
                        c2.add(layers.AveragePooling2D(pool_size=(2,2)))
                        c2.add(layers.Conv2D(32,(3,3),activation="relu",padding="same"))
                        c2.add(layers.Conv2D(32,(3,3),activation="relu",padding="same"))
                        c2.add(layers.MaxPooling2D(pool_size=(2,2)))
                        c2.add(layers.Flatten())
                        c2.add(layers.Dense(64,activation="relu"))
                        c2.add(layers.Dropout(0.2))
                        c2.add(layers.BatchNormalization())
                        c2.add(layers.Dense(64,activation="relu"))
                        c2.add(layers.Dense(32,activation="softmax"))
                        c2.add(layers.Dropout(0.2))
                        c2.add(layers.BatchNormalization())
                        c2.add(layers.Dense(16,activation="relu"))
                        c2.add(layers.Dense(4,activation="sigmoid"))
                        c2.compile(optimizer=o,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
                        c2.summary()
                        c2.fit(training_set,validation_data=testing_set,batch_size=b,epochs=e)
                        
                        test_loss,test_acc=c2.evaluate(testing_set)
                        print('\n\n')
                        print("Test Accuracy of epochs=",e,",batch size =",b,"and optimizers=",optimizers_str[optimizers.index(o)],"with classifier2 is:"+str(test_acc))
                        print("Test Accuracy of epochs=",e,",batch size =",b,"and optimizers=",optimizers_str[optimizers.index(o)],"with classifier2 is:"+str(test_acc),"\n",file=f)
                        print("\n\n")
    f.close()