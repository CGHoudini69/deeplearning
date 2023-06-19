# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:26:33 2023

@author: Houdini69
"""
import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

start=time.process_time()
with open('data.csv', newline='') as csvfile:
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
    X=np.array(X).T.tolist()
    X= pd.DataFrame(X)
    test_size=[0.2,0.3]
    layer_size=[(750,750),(500,500,500),(250,250,250,250,250,250)]
    solver=['adam','sgd']
    for s in test_size:
        xtrain,xtest,ytrain,ytest= train_test_split(X,y,test_size=s,random_state=1)# normalize data
        scaler=MinMaxScaler()
        xtrain_transform=scaler.fit_transform(xtrain)
        xtest_tansform=scaler.fit_transform(xtest)
        for j in layer_size:
            for k in solver:
                print('start')
                model=MLPClassifier(hidden_layer_sizes=j,activation='relu', max_iter=3000,random_state=1,solver=k)# Training Model
                model.fit(xtrain_transform,ytrain)
                ypred=model.predict(xtest_tansform)
                accuracy =accuracy_score(ytest,ypred)
                print('The accuracy of test_size='+str(s)+' and hidden_layer_sizes=',j,' with solver=',k,' is ',accuracy)
                print('end \n')
                cm=confusion_matrix(ytest,ypred)
                cr=classification_report(ytest,ypred)
                print(cr)
end=time.process_time()
print('\nTotal process time:'+str(end-start))