# -*- coding: utf-8 -*-
import numpy as np

"""

Created: 2021/11/11

Author: gealach ,Abel Chan

Version: 2.0

Description:提供降采样方法

"""


def random_sampling(data_set, per_epoch_times, batch_size):
    train_length = data_set.get('X_trains_norm').shape[0]
    for times in range(per_epoch_times):
        sampIndex = np.random.randint(low=0, high=train_length - 1, size=batch_size)
        X = data_set.get("X_trains_norm")[sampIndex]
        y = data_set.get("y_v_train")[sampIndex]
        yield X, y


def stratified_sampling(data_set, per_epoch_times, batch_size, target_num=10):
    """

    Args:
        data_set: 数据集
        per_epoch_times:  迭代轮数
        batch_size: 每次更新训练batch_size个样本
        target_num: 目标类别数

    Returns:

    """
    size = batch_size // target_num  # 1次迭代中 每类样本抽样数
    lb = lambda index: np.random.choice(data_set.get('splitDict_train_y')[index], size=size, replace=False)
    category_index = np.arange(target_num)
    X = data_set.get("X_trains_norm")
    y = data_set.get("y_v_train")
    for times in range(per_epoch_times):
        # list(range(len(X)))
        sampIndex = map(lb, category_index)
        indexList = np.array(list(sampIndex)).reshape(-1)
        np.random.shuffle(indexList)
        yield X[indexList], y[indexList]
