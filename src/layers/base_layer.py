# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum
from abc import abstractmethod, ABCMeta

"""

Created: 2021/11/11

Author: gealach, Abel Chan

Version: 1.0

Description:定义层的基本类

"""


class LayerType(Enum):
    BatchNormalization = 'BatchNormalization'
    FullConnectionLayer = 'FullConnectionLayer'
    InputLayer = 'InputLayer'
    OutputLayer = 'OutputLayer'
    ConvolutionLayer = 'ConvolutionLayer'
    InputLayer2D = 'InputLayer2D'
    FlattenLayer = 'FlattenLayer'
    MaxPoolingLayer = 'MaxPoolingLayer'

    @staticmethod
    def get2DLayer():
        return [LayerType.InputLayer2D, LayerType.ConvolutionLayer]


class Layer(metaclass=ABCMeta):
    z: np.ndarray
    h: np.ndarray
    units: int or tuple
    activation: lambda x: x
    layerType: LayerType
    W: np.ndarray
    b: np.ndarray

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, h_1, delta_plus_1, ind, **kwargs):
        """
        Args:
            h_1: 上游节点的输出
            delta_plus_1: 下游节点的上游误差(与本层units一致)
        """
        pass

    @abstractmethod
    def get_units(self):
        pass
