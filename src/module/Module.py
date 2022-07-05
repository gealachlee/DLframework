# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC, abstractmethod, ABCMeta
from collections import defaultdict
from typing import Optional
"""

Created: 2021/11/11

Author: gealach ,Abel Chan

Version: 1.0

Description:模型基本类

"""


class Module(metaclass=ABCMeta):
    structure: dict
    data_set: dict
    history: defaultdict

    @abstractmethod
    def train(self, epoch, batch_size, lr, lamb, *args, **kwargs):
        pass

    def init_params(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_test=Optional[np.array]):
        pass
