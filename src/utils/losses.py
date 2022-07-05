# -*- coding: utf-8 -*-
"""
定义损失函数

Created: 2021/11/11

Author: gealach

Version: 1.0

"""
import numpy as np

__all__ = [ "cross_entropy_loss"]




def cross_entropy_loss(y_predict:np.array, y_true:np.array):
    """
    交叉熵损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值,shape(N,d)
    :return:loss,  dy
    """
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1),dtype=np.float16)  # 损失函数
    dy = (y_probability - y_true)
    return loss, dy
