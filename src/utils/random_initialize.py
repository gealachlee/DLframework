# -*- coding: utf-8 -*-

import numpy as np

"""

Created: 2021/12/11

Author: gealach

Version: 2.0

Description: 提供权值初始化方法

"""

class initialization():
    pass


class He_initialization(initialization):

    @staticmethod
    def initialize_conv_W(shape):
        """
        He's initialized method for convolution layer
        Args:
            shape:tuple such as (10,)

        Returns: np.array ,an array with random weight.

        """
        std = np.sqrt(2.0 / np.prod(shape))
        return np.random.normal(loc=0, scale=std, size=shape)

    @staticmethod
    def initialize_conv_b(shape):
        """
        He's initialized method for convolution layer
        Args:
            shape: tuple such as (10,)

        Returns: np.array ,an array with bias.

        """
        return np.zeros(shape)

    @staticmethod
    def initialize_dense_W(shape):
        """
        He's initialized method for dense(full connection) layer
        Args:
            shape: tuple,example:initialize_dense_W((32,64))

        Returns:np.array ,an array with random weight.

        """
        std = np.sqrt(2.0 / shape[0])
        return np.random.normal(size=shape, scale=std)

    @staticmethod
    def initialize_dense_b(shape):
        """
        He's initialized method for dense(full connection) layer
        Args:
            shape:  tuple,example:initialize_dense_b((32,64))

        Returns: np.array ,an array with random bias.
        """
        return np.zeros(shape)
