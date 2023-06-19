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
    X=np.array(X).T.tolist()
    X= pd.DataFrame(X)
    test_size=[0.2,0.3]
    layer_size=[(75,75),(50,50,50),(25,25,25,25,25,25)]
    solver=['adam','lbfgs']
    activation=['identity', 'logistic', 'tanh', 'relu']
    alpha=[0.0001,0.0002,0.0003]
    momentum=[0.7,0.8,0.9]
    learning_rate=['constant', 'invscaling', 'adaptive']
    with open("result.txt",'w' , encoding = "utf-8-sig") as f:
        print("結果會保存至資料夾中的result.txt,完成後會顯示完成,共1080種結果(實測約要1.5小時),若要減少,請從上方參數適當做刪除,但各參數至少要剩一個")
        print("自己測試的結果在資料夾中,檔名為\"old result\"")
        print("若要即時顯示結果，請將第103、112、114、138、140行的註解刪掉並重新執行")
        print("在自己測試中 accuracy最高的為test_size=0.2、 hidden_layer_sizes=(25, 25, 25, 25, 25, 25)、activation=identity、alpha=0.0001、learning rate=constant、 solver=adam,其值為0.8450413223140496")
        start=time.process_time()
        mode=[]
        acc=[]
        for s in test_size:
            xtrain,xtest,ytrain,ytest= train_test_split(X,y,test_size=s,random_state=1)
            scaler=MinMaxScaler()
            xtrain_transform=scaler.fit_transform(xtrain)# normalize data
            xtest_tansform=scaler.fit_transform(xtest)
            for j in layer_size:
                for k in solver:
                    for a in activation:
                        for i in alpha:
                            for l in learning_rate:
                                print('start',file=f)
                                #print('start')
                                model=MLPClassifier(hidden_layer_sizes=j,activation=a, max_iter=3000,random_state=1,solver=k,alpha=i,learning_rate=l)# Training Model
                                model.fit(xtrain_transform,ytrain)
                                ypred=model.predict(xtest_tansform)
                                accuracy =accuracy_score(ytest,ypred)
                                st='test_size='+str(s)+' and hidden_layer_sizes='+str(j)+'and activation='+str(a)+'and alpha='+str(i)+'and learning rate='+str(l)+' with solver='+str(k)
                                mode.append(st)
                                acc.append(accuracy)
                                print('The accuracy of test_size='+str(s)+' and hidden_layer_sizes=',j,'and activation=',a,'and alpha=',i,'and learning rate=',l,' with solver=',k,' is ',accuracy,file=f)
                                #print('The accuracy of test_size='+str(s)+' and hidden_layer_sizes=',j,'and activation=',a,'and alpha=',i,'and learning rate=',l,' with solver=',k,' is ',accuracy)
                                print('end \n',file=f)
                                #print('end \n')
        #因為momentum只有在solver為sgd時有用，因此分成兩部分處理
        for s in test_size:
            xtrain,xtest,ytrain,ytest= train_test_split(X,y,test_size=s,random_state=1)# normalize data
            scaler=MinMaxScaler()
            xtrain_transform=scaler.fit_transform(xtrain)
            xtest_tansform=scaler.fit_transform(xtest)
            for j in layer_size:
                for a in activation:
                    for i in alpha:
                        for l in learning_rate:
                            for m in momentum:
                                print('start',file=f)
                                model=MLPClassifier(hidden_layer_sizes=j,activation=a, max_iter=3000,random_state=1,solver='sgd',alpha=i,learning_rate=l,momentum=m)# Training Model
                                model.fit(xtrain_transform,ytrain)
                                ypred=model.predict(xtest_tansform)
                                accuracy =accuracy_score(ytest,ypred)
                                st='test_size='+str(s)+',and hidden_layer_sizes='+str(j)+',and activation='+str(a)+',and alpha='+str(i)+',and momentum='+str(m)+',and learning rate='+str(l)+' with solver='
                                mode.append(st)
                                acc.append(accuracy)
                                print('The accuracy of test_size='+str(s)+',and hidden_layer_sizes=',j,',and activation=',a,',and alpha=',i,',and momentum=',m,',and learning rate=',l,' with solver=sgd is ',accuracy,file=f)
                                print('end \n',file=f)
        end=time.process_time()
        print('\nTotal process time:'+str(end-start),file=f)
        #print('\nTotal process time:'+str(end-start))
        print("此次測試中最高分的是",mode[acc.index(max(acc))],"其準確率為",max(acc),file=f)
        #print("此次測試中最高分的是",mode[acc.index(max(acc))],"其準確率為",max(acc))
    f.close()
    print('完成')
csvfile.close()