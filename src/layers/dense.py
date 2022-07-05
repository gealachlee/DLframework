# -*- coding: utf-8 -*-
"""

Created: 2021/11/11

Author: gealach ,Abel Chan

Version: 1.0

Description:定义全连接层

"""
from layers.base_layer import Layer,LayerType
import numpy as np
from utils.activation import Activation

class FullConnectionLayer(Layer):
    layerType = LayerType.FullConnectionLayer

    def __init__(self, units: int, activation: Activation.func, **kwargs):
        self.units = units
        self.activation = activation


    def forward(self, node_in):
        '''
        # 本层的W,b与上游输出进行整合,l指的是上一层index
        z[l + 1] = W[l].dot(node_in) + b[l]
        h[l + 1] = f(z[l+1])
        :return: h[l + 1] :np.array
        '''
        self.z = np.dot(node_in, self.W.T) + self.b
        self.h = self.activation.func(self.z)

    def backward(self, h_1, delta_plus_1,**kwargs):
        '''

        Args:
            h_1: 上游输出
            delta_plus_1: 下游节点的上游误差
            **kwargs:

        Returns:

        '''
        N = delta_plus_1.shape[0]
        delta = delta_plus_1 * self.activation.func_derivative(self.z)
        dz = np.dot(delta, self.W)  # 当前层的输入梯度,即上游输出梯度dot W
        self.dW = np.dot(h_1.T, delta).T/N
        self.dz = dz
        self.delta = delta
        self.db = np.mean(delta, axis=0)


    def get_units(self):
        return self.units


class InputLayer(Layer):
    layerType = LayerType.InputLayer

    def __init__(self, units: int,**kwargs):
        self.units = units

        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, node_in):
        self.h = node_in
        return node_in

    def backward(self, *args, **kwargs):
        pass


    def get_units(self):
        return self.units


class InputLayer2D(Layer):
    layerType = LayerType.InputLayer2D
    def __init__(self, units: tuple,**kwargs):
        self.units = units
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, node_in):
        self.h = node_in
        return node_in

    def backward(self, *args, **kwargs):
        pass

    def set_initialized(self, is_initialized: bool):
        pass

    def get_units(self):
        return self.units

