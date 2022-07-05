# -*- coding: utf-8 -*-
"""

Created: 2021/11/11

Author: gealach

Version: 1.0

Description:定义激活函数

"""

import numpy as np

class Activation(object):

    @staticmethod
    def func(x: np.array):
        '''
        :param x:input
        :return:f(x)
        '''
        pass

    @staticmethod
    def func_derivative(x:np.array):
        '''
        :param x:input
        :return:f'(x)
        '''
        pass

    def __new__(cls, *args, **kwargs):  # 把激活函数及其导数用class 封装,禁止创建实例
        raise NotImplementedError("There is no need to create an instance!")


class Relu(Activation):

    @staticmethod
    def func(x: np.array):
        return np.maximum(0, x)

    @staticmethod
    def func_derivative(x: np.array):
        return np.where(np.greater(x, 0), 1, 0)


class LRelu(Activation):
    '''
      leaky relu- activation
    '''

    @staticmethod
    def func(x:np.array):
        # warnings : this function will change the instance!
        x[x <= 0] = 0.01 * x
        x[x > 0] = x
        return x

    @staticmethod
    def func_derivative(x: np.array):
        x[x <= 0] = 0.01
        x[x > 0] = 1
        return x


class Sigmoid(Activation):

    @staticmethod
    def func(x:np.array):
        """

         sigmoid function /softmax function
         :param x:  input
         :return: sigmoid(x)
          注:因防止溢出而使用for
         """
        x_ravel = x.ravel()  # 将numpy数组展平
        length = len(x_ravel)
        y = []
        for index in range(length):
            if x_ravel[index] >= 0:
                y.append(1.0 / (1 + np.exp(-x_ravel[index])))
            else:
                y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
        return np.array(y).reshape(x.shape)

    @staticmethod
    def func_derivative(x:np.array):

        sigmoid_result=Sigmoid.func(x)
        return  sigmoid_result * (1 - sigmoid_result)

class Identity(Activation):
    """
        恒等映射
    """
    @staticmethod
    def func(x:np.array):
        return x

    @staticmethod
    def func_derivative(x:np.array):
        return 1
