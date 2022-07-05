# -*- coding: utf-8 -*-
import numpy as np
from Loader import MNIST_Loader
"""

Created: 2021/11/11

Author: gealach

Version: 1.0

Description:数据读取和预处理

"""
def normalize(data,method=1):
    '''
    :param data:a dict include X_train,X_test,y_train,y_test
    :return:a dict with normalized data
    '''

    X_train = np.reshape(data.get('X_train'), (data.get('X_train').shape[0], -1))
    X_test = np.reshape(data.get('X_test'), (data.get('X_test').shape[0], -1))

    X_trains_norm = np.transpose(data_normalize(np.transpose(X_train),method=method))
    X_tests_norm = np.transpose(data_normalize(np.transpose(X_test),method=method))
    y_v_train = convert_y_to_vect(data.get('y_train').astype(int),scale=10)
    #y_v_test = convert_y_to_vect(data.get('y_test'),scale=10)

    data = {
            'X_trains_norm':X_trains_norm,
            'X_tests_norm':X_tests_norm,
            'y_v_train':y_v_train,
           # 'y_v_test':y_v_test
            'y_train':data.get('y_train'),
            'y_test':data.get('y_test')
    }
    return data

def data_normalize(data,method=1):
    '''
    :param data,an array
    :param method :default 1,when method is 1 ,deal with standard normalization,when method is 2 ,just devoted by 255
    :return: data
    '''
    if method == 1:
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0,dtype=np.float32)
        return (np.array(data,dtype=np.float16) - np.array(mu,dtype=np.float16) )/ np.array(sigma,dtype=np.float16)
    elif method ==2:
        return data/255
    else:
        raise ValueError("data_normlize method just 1 or 2")


def convert_y_to_vect(y,scale:int):
    '''
    单值输出转化为向量

    :param y:
    :param scale:分类数, mnist取值为1-10 共10个取值
    :return:
    '''

    return np.eye(scale,dtype=int)[y.reshape(-1)]

def test():
    mnistLoader = MNIST_Loader(
        trainX_FileName="train-images.idx3-ubyte",
        trainy_FileName="train-labels.idx1-ubyte",
        testX_FileName="t10k-images.idx3-ubyte",
        testy_FileName="t10k-labels.idx1-ubyte",
    )
    data = mnistLoader.load()
    try:
        data=normalize(data)
    except Exception as e:
        print("Normalize Error",e)
    return data

if __name__ == "__main__":
   test_res= test()


