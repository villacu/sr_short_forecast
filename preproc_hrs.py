# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:18:00 2020
prepara ventana de D días


@author: villacuPC
"""

# -*- coding: utf-8 -*-
"""
prepara los vectores para entrenamiento
ESTÁ HECHO A LA CARRERA Y FALTA OPTIMIZAR MUCHO

@author: villacuPC
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

#definimos r2 de artículo
from sklearn.metrics import mean_squared_error
def rsquare(y,yhat):
    ybar = np.mean(y)
    tmp = np.sum((yhat-y)**2)
    return (1-tmp/np.sum((y-ybar)**2))


def get_data(yrs_train,yr_val,yr_test,datadir,WINDOW=24):
    #numyrs = len(yrs)
    df = pd.read_csv(datadir+yrs_train.pop(0)+'.csv',header=2,dtype=np.float32)
    for i in yrs_train:
        tmp_df = pd.read_csv(datadir+i+'.csv',header=2,dtype=np.float32)
        df=df.append(tmp_df)
        
    #dataframe de validacio n
    df_validation = pd.read_csv(datadir+yr_val+'.csv',header=2,dtype=np.float32)

    df_test = pd.read_csv(datadir+yr_test+'.csv',header=2,dtype=np.float32)
    
    feature_lbls = ['GHI','Clearsky GHI','Precipitable Water',
               'Wind Direction','Hour','Month']
    predict_lbls=['GHI']

    y_train = df[predict_lbls].to_numpy()
    x_train = df[feature_lbls].to_numpy()
    
    y_val = df_validation[predict_lbls].to_numpy()
    x_val = df_validation[feature_lbls].to_numpy()

    y_test = df_test[predict_lbls].to_numpy()
    x_test = df_test[feature_lbls].to_numpy()

    '''
    Se calcula el clearness index como ghi/clsky ghi
    puede que convenga quitarlo porque en las noches presenta un problema
    '''
    #To compute clearness index
    tmptr = np.array([])
    tmpval = np.array([])   
    tmpte = np.array([])   

    for i in x_train:
        if (i[1]!=0.0):
            tmptr=np.append(tmptr,i[0]/i[1])
        else:
            tmptr=np.append(tmptr,1)
            
    for i in x_val:
        if (i[1]!=0.0):
            tmpval=np.append(tmpval,i[0]/i[1])
        else:
            tmpval=np.append(tmpval,1)

    for i in x_test:
        if (i[1]!=0.0):
            tmpte=np.append(tmpte,i[0]/i[1])
        else:
            tmpte=np.append(tmpte,1)
            
    tmptr = np.reshape(tmptr,(-1,1))
    tmpval = np.reshape(tmpval,(-1,1))
    tmpte = np.reshape(tmpte,(-1,1))
    
    
    x_train = np.append(x_train,tmptr,axis=1)
    x_val = np.append(x_val,tmpval,axis=1)
    x_test = np.append(x_test,tmpte,axis=1)
    
    #escalando los datos LIMPIAR falta limpiar

    scaler_xtr = MinMaxScaler(feature_range=(0,1))
    scaler_ytr = MinMaxScaler(feature_range=(0,1))
    scaler_xval = MinMaxScaler(feature_range=(0,1))
    scaler_yval = MinMaxScaler(feature_range=(0,1))
    scaler_xtst = MinMaxScaler(feature_range=(0,1))
    scaler_ytst = MinMaxScaler(feature_range=(0,1))
    
    y_train = scaler_ytr.fit_transform(y_train)
    x_train = scaler_xtr.fit_transform(x_train)

    y_val = scaler_yval.fit_transform(y_val)
    x_val = scaler_xval.fit_transform(x_val)

    y_test = scaler_ytst.fit_transform(y_test)
    x_test = scaler_xtst.fit_transform(x_test)

    
    #one hot encoding para tipo de nube
    xtrain_encode = df[['Cloud Type']]
    xval_encode = df_validation[['Cloud Type']]
    xtest_encode = df_test[['Cloud Type']]

    encode = OneHotEncoder(categories=[np.array([ 0.,  1.,  2.,  3.,  4.,  6.,  7.,  8.,  9., 10.])])

    enctrain = np.reshape(xtrain_encode,(-1,1))
    encval = np.reshape(xval_encode,(-1,1))
    enctest = np.reshape(xtest_encode,(-1,1))

    enctrain = encode.fit_transform(enctrain).toarray()
    encval = encode.fit_transform(encval).toarray()
    enctest = encode.fit_transform(enctest).toarray()
    
    x_train = np.concatenate((x_train,enctrain),axis=1)
    x_val = np.concatenate((x_val,encval),axis=1)
    x_test = np.concatenate((x_test,enctest),axis=1)
    
    
    ## aquí cambia
    new_xtr = np.roll(x_train,WINDOW,axis=0).reshape(-1,1,17)
    new_xval = np.roll(x_val,WINDOW,axis=0).reshape(-1,1,17)
    new_xte = np.roll(x_test,WINDOW,axis=0).reshape(-1,1,17)

    #train set (place this in a single cycle)
    for i in range(1,WINDOW):
        tmp = np.roll(x_train,WINDOW-i,axis=0).reshape(-1,1,17)
        new_xtr = np.append(new_xtr,tmp,axis=1)
        
    for i in range(1,WINDOW):
        tmp = np.roll(x_val,WINDOW-i,axis=0).reshape(-1,1,17)
        new_xval = np.append(new_xval,tmp,axis=1)

    for i in range(1,WINDOW):
        tmp = np.roll(x_test,WINDOW-i,axis=0).reshape(-1,1,17)
        new_xte = np.append(new_xte,tmp,axis=1)
    
    
    new_ytr = y_train.reshape(-1,1,1)
    new_yval = y_val.reshape(-1,1,1)
    new_yte = y_test.reshape(-1,1,1)
    
    return new_xtr,new_ytr,new_xval,new_yval,new_xte,new_yte,scaler_ytr