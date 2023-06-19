# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:14:02 2023

@author: Houdini69
"""

import warnings
warnings.filterwarnings("ignore")
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report

with open('hypertension_gene.csv',encoding="utf-8") as file:
    data=[]
    rows=csv.reader(file)
    for index, row in enumerate(rows):
        if index==0:
            continue
        row_f=[]
        for value in row:
            if value=="":
                value=0
            else:
                value=float(value)
            row_f.append(value)
        data.append(list(row_f))
data=np.array(data)
X=np.array(data[1:]).transpose()[:100].transpose()
y=np.array(data[1:]).transpose()[100].transpose()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,test_size=0.25,random_state=0)

MLP=MLPClassifier(hidden_layer_sizes=(200,200,200,200,100,100,100,100),activation='relu', max_iter=5000,random_state=0,solver="adam")

min_max_scaler_0_1=preprocessing.MinMaxScaler(feature_range=(0,1))
min_max_scaler_n1_1=preprocessing.MinMaxScaler(feature_range=(-1,1))
z_scaler=preprocessing.StandardScaler()
'''

#使用valid校正隱藏層大小
print("原始資料:")
MLP.fit(X_train,y_train)
y_pred=MLP.predict(X_valid)
print("accuracy="+str(accuracy_score(y_valid,y_pred)))
report=classification_report(y_valid,y_pred)
print(report)
cm=confusion_matrix(y_valid,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

'''

print("原始資料:")
MLP.fit(X_train,y_train)
y_pred=MLP.predict(X_test)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('原始資料+標準化至[0,1]')
X_train_normal,X_test_normal=min_max_scaler_0_1.fit_transform(X_train),min_max_scaler_0_1.fit_transform(X_test)
MLP.fit(X_train_normal,y_train)
y_pred=MLP.predict(X_test_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('原始資料+標準化至[0,1]+z-score轉換:')
X_train_normal_zscore,X_test_normal_zscore=z_scaler.fit_transform(X_train_normal),z_scaler.fit_transform(X_test_normal)
MLP.fit(X_train_normal_zscore,y_train)
y_pred=MLP.predict(X_test_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('原始資料+標準化至[-1,1]')
X_train_normal,X_test_normal=min_max_scaler_n1_1.fit_transform(X_train),min_max_scaler_n1_1.fit_transform(X_test)
MLP.fit(X_train_normal,y_train)
y_pred=MLP.predict(X_test_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('原始資料+標準化至[-1,1]+z-score轉換:')
X_train_normal_zscore,X_test_normal_zscore=z_scaler.fit_transform(X_train_normal),z_scaler.fit_transform(X_test_normal)
MLP.fit(X_train_normal_zscore,y_train)
y_pred=MLP.predict(X_test_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('-----------------------------------------------------')

print('倒數變換:')
#若數值有0則將其平移後取倒數
try:
    X_train_reciprocal,X_test_reciprocal=1/(X_train),1/(X_test)
except:
    add=min([np.min(X_train),np.min(X_test)])+0.1
    X_train_reciprocal,X_test_reciprocal=1/(X_train+add),1/(X_test+add)
MLP.fit(X_train_reciprocal,y_train)
y_pred=MLP.predict(X_test_reciprocal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('倒數變換+標準化至[0,1]:')
X_train_reciprocal_normal,X_test_reciprocal_normal=min_max_scaler_0_1.fit_transform(X_train_reciprocal),min_max_scaler_0_1.fit_transform(X_test_reciprocal)
MLP.fit(X_train_reciprocal_normal,y_train)
y_pred=MLP.predict(X_test_reciprocal_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('倒數變換+標準化至[0,1]+z-score轉換:')
X_train_reciprocal_normal_zscore,X_test_reciprocal_normal_zscore=z_scaler.fit_transform(X_train_reciprocal_normal),z_scaler.fit_transform(X_test_reciprocal_normal)
MLP.fit(X_train_reciprocal_normal_zscore,y_train)
y_pred=MLP.predict(X_test_reciprocal_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('倒數變換+標準化至[-1,1]:')
X_train_reciprocal_normal,X_test_reciprocal_normal=min_max_scaler_n1_1.fit_transform(X_train_reciprocal),min_max_scaler_n1_1.fit_transform(X_test_reciprocal)
MLP.fit(X_train_reciprocal_normal,y_train)
y_pred=MLP.predict(X_test_reciprocal_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('倒數變換+標準化至[-1,1]+z-score轉換:')
X_train_reciprocal_normal_zscore,X_test_reciprocal_normal_zscore=z_scaler.fit_transform(X_train_reciprocal_normal),z_scaler.fit_transform(X_test_reciprocal_normal)
MLP.fit(X_train_reciprocal_normal_zscore,y_train)
y_pred=MLP.predict(X_test_reciprocal_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('-----------------------------------------------------')

print('平方根變換:')
#先記錄負數之index，將其絕對值開根號後依照index還原成負數
negative_index1=[]
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        if X_train[i][j]<0:
            negative_index1.append([i,j])
negative_index2=[]
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if X_test[i][j]<0:
            negative_index2.append([i,j])
X_train_abs,X_test_abs=np.abs(X_train),np.abs(X_test)
X_train_sqrt,X_test_sqrt=np.sqrt(X_train_abs),np.sqrt(X_test_abs)
for i in negative_index1:
    X_train_sqrt[i[0],i[1]]=-X_train_sqrt[i[0],i[1]]
for i in negative_index2:
    X_test_sqrt[i[0],i[1]]=-X_test_sqrt[i[0],i[1]]
MLP.fit(X_train_sqrt,y_train)
y_pred=MLP.predict(X_test_sqrt)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('平方根變換+標準化至[0,1]:')
X_train_sqrt_normal,X_test_sqrt_normal=min_max_scaler_0_1.fit_transform(X_train_sqrt),min_max_scaler_0_1.fit_transform(X_test_sqrt)
MLP.fit(X_train_sqrt_normal,y_train)
y_pred=MLP.predict(X_test_sqrt_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('平方根變換+標準化至[0,1]+z-score轉換:')
X_train_sqrt_normal_zscore,X_test_sqrt_normal_zscore=z_scaler.fit_transform(X_train_sqrt_normal),z_scaler.fit_transform(X_test_sqrt_normal)
MLP.fit(X_train_sqrt_normal_zscore,y_train)
y_pred=MLP.predict(X_test_sqrt_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('平方根變換+標準化至[-1,1]:')
X_train_sqrt_normal,X_test_sqrt_normal=min_max_scaler_n1_1.fit_transform(X_train_sqrt),min_max_scaler_n1_1.fit_transform(X_test_sqrt)
MLP.fit(X_train_sqrt_normal,y_train)
y_pred=MLP.predict(X_test_sqrt_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('平方根變換+標準化至[-1,1]+z-score轉換:')
X_train_sqrt_normal_zscore,X_test_sqrt_normal_zscore=z_scaler.fit_transform(X_train_sqrt_normal),z_scaler.fit_transform(X_test_sqrt_normal)
MLP.fit(X_train_sqrt_normal_zscore,y_train)
y_pred=MLP.predict(X_test_sqrt_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('-----------------------------------------------------')

print('對數變換:')
#先記錄負數之index，將其絕對值取對數後依照index還原成負數
negative_index1=[]
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        if X_train[i][j]<0:
            negative_index1.append([i,j])
negative_index2=[]
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if X_test[i][j]<0:
            negative_index2.append([i,j])
X_train_abs,X_test_abs=np.abs(X_train),np.abs(X_test)
try:
    X_train_log,X_test_log=np.log(X_train_abs),np.log(X_test_abs)
except:
    X_train_log,X_test_log=np.log(X_train_abs+0.1),np.log(X_test_abs+0.1)
for i in negative_index1:
    X_train_log[i[0],i[1]]=-X_train_log[i[0],i[1]]
for i in negative_index2:
    X_test_log[i[0],i[1]]=-X_test_log[i[0],i[1]]

MLP.fit(X_train_log,y_train)
y_pred=MLP.predict(X_test_log)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('對數變換+標準化至[0,1]:')
X_train_log_normal,X_test_log_normal=min_max_scaler_0_1.fit_transform(X_train_log),min_max_scaler_0_1.fit_transform(X_test_log)
MLP.fit(X_train_log_normal,y_train)
y_pred=MLP.predict(X_test_log_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('對數變換+標準化至[0,1]+z-score轉換:')
X_train_log_normal_zscore,X_test_log_normal_zscore=z_scaler.fit_transform(X_train_log_normal),z_scaler.fit_transform(X_test_log_normal)
MLP.fit(X_train_log_normal_zscore,y_train)
y_pred=MLP.predict(X_test_log_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('對數變換+標準化至[-1,1]:')
X_train_log_normal,X_test_log_normal=min_max_scaler_n1_1.fit_transform(X_train_log),min_max_scaler_n1_1.fit_transform(X_test_log)
MLP.fit(X_train_log_normal,y_train)
y_pred=MLP.predict(X_test_log_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('對數變換+標準化至[-1,1]+z-score轉換:')
X_train_log_normal_zscore,X_test_log_normal_zscore=z_scaler.fit_transform(X_train_log_normal),z_scaler.fit_transform(X_test_log_normal)
MLP.fit(X_train_log_normal_zscore,y_train)
y_pred=MLP.predict(X_test_log_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('-----------------------------------------------------')

print('指數變換:')
X_train_exponential,X_test_exponential=np.exp(X_train),np.exp(X_test)
MLP.fit(X_train_exponential,y_train)
y_pred=MLP.predict(X_test_exponential)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('指數變換+標準化至[0,1]:')
X_train_exponential_normal,X_test_exponential_normal=min_max_scaler_0_1.fit_transform(X_train_exponential),min_max_scaler_0_1.fit_transform(X_test_exponential)
MLP.fit(X_train_exponential_normal,y_train)
y_pred=MLP.predict(X_test_exponential_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('指數變換+標準化至[0,1]+z-score轉換:')
X_train_exponential_normal_zscore,X_test_exponential_normal_zscore=z_scaler.fit_transform(X_train_exponential_normal),z_scaler.fit_transform(X_test_exponential_normal)
MLP.fit(X_train_exponential_normal_zscore,y_train)
y_pred=MLP.predict(X_test_exponential_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('指數變換+標準化至[-1,1]:')
X_train_exponential_normal,X_test_exponential_normal=min_max_scaler_n1_1.fit_transform(X_train_exponential),min_max_scaler_n1_1.fit_transform(X_test_exponential)
MLP.fit(X_train_exponential_normal,y_train)
y_pred=MLP.predict(X_test_exponential_normal)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('指數變換+標準化至[-1,1]+z-score轉換:')
X_train_exponential_normal_zscore,X_test_exponential_normal_zscore=z_scaler.fit_transform(X_train_exponential_normal),z_scaler.fit_transform(X_test_exponential_normal)
MLP.fit(X_train_exponential_normal_zscore,y_train)
y_pred=MLP.predict(X_test_exponential_normal_zscore)
print("accuracy="+str(accuracy_score(y_test,y_pred)))
report=classification_report(y_test,y_pred)
print(report)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:')
print(str(cm[0]))
print(str(cm[1]))

print('-----------------------------------------------------')
print('results:')
print('以上所有變換中,accuracy最高者為「平方根變換+標準化至[-1,1]」')
print('以上所有變換中,recall最高者為「原始資料+標準化至[-1,1]+z-score轉換」')
print('以上所有變換中,precision最高者為「平方根變換+標準化至[-1,1]」')






