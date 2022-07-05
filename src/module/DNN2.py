# -*- coding: utf-8 -*-
"""

Created: 2021/11/11

Author: gealach ,Abel Chan

Version: 3.0

Description:定义DNN神经网络

"""

from sklearn.metrics import accuracy_score, confusion_matrix
from collections import OrderedDict, defaultdict
from optimizer import optimizer
from module.Module import Module
from Loader import MNIST_Loader
from utils.activation import Relu, Sigmoid
from utils.losses import cross_entropy_loss
from utils.random_initialize import He_initialization
from layers.dense import *
from utils import normalize, time_control, sampling,check_params


class DNN(Module):

    def __init__(self, structure: dict, data_set: dict, optimizer: optimizer, **kwargs):
        """

        Args:
            structure: Orderdict,module structure includes class layers
            data_set: dict
            **kwargs: extra params for different module
        """
        self.structure = structure  # 可省略
        self.data_set = data_set
        self.history = defaultdict(list)
        self.optimizer = optimizer
        for key, value in kwargs.items():
            setattr(self, key, value)

    #@check_params.debug_train_params
    @time_control.timing
    def train(self, epoch: int, batch_size: int, lr: float, lamb: float, loss_function, *args, **kwargs):
        """

        Args:
            lr: learning rate
            lamb: weight decay rate
            epoch: 全样本训练轮数
            batch_size: 单次权值更新所需样本数
            *args:
            **kwargs:

        Returns:

        """
        per_epoch_times = self.data_set.get('X_trains_norm').shape[0] // batch_size
        if per_epoch_times == 0:
            raise ZeroDivisionError("per_epoch_times is inf or batch size is larger than train_length")

        for per_epoch in range(1, epoch + 1):  # 对每轮进行迭代
            sampling_data = sampling.random_sampling(self.data_set, per_epoch_times, batch_size)
            for times in range(per_epoch_times):  # 对该轮的批样本进行迭代 1
                # Step1 抽样
                X, y = next(sampling_data)

                # Step2 前向计算

                for index, iter in self.structure.items():
                    print(index)
                    if index == 0:
                        iter.forward(X)
                    else:
                        iter.forward(self.structure[index - 1].h)
                    self.optimizer.forward_calculate()
                # Step3 反向传播并权值更新
                for index, iter in reversed(self.structure.items()):
                    if index == len(self.structure) - 1:
                        loss, dy = loss_function(iter.h, y)  # dy is a 2D array such as (100,10),shape=(N,output_units)
                        iter.backward(h_1=self.structure[index - 1].h, delta_plus_1=dy)
                        self.history['losses'].append(loss)
                    else:
                        if index == 0:
                            break
                        iter.backward(h_1=self.structure[index - 1].h, delta_plus_1=self.structure[index + 1].dz)
                    self.optimizer.backword_update(iter, lr, lamb)
                times_now = per_epoch_times * (per_epoch - 1) + times  #

                self.predict_during_training(times_now, per_epoch_times, perEpoch=True)
            print(f'epoch:{per_epoch}')
        y_hat = self.predict()
        accuracy = accuracy_score(self.data_set['y_test'], y_pred=y_hat)
        print(f'accuracy={accuracy}')
        print(confusion_matrix(self.data_set['y_test'], y_pred=y_hat))
        self.history['accuracyList'].append(accuracy)
        print('train_end')


    def predict_during_training(self, times_now, per_epoch_times, timesList=[], perEpoch=False,confusion=False):
        """

        Args:
            times_now: 现迭代次数
            per_epoch_times: 每个epoch迭代per_epoch_times次
            timesList: 第i次迭代时预测测试集正确率,i属于timeList
            perEpoch: 是否个Epoch结束计算一次测试集正确率

        Returns:

        """
        if times_now in timesList or (perEpoch is True and times_now % per_epoch_times == 0 and times_now != 0):
            y_hat = self.predict()
            accuracy = accuracy_score(self.data_set['y_test'], y_pred=y_hat)
            print(f'accuracy={accuracy}')
            if confusion:
                print(confusion_matrix(self.data_set['y_test'], y_pred=y_hat))
            self.history['accuracyList'].append(accuracy)

    def init_params(self, init_function):
        """
        权值初始化:1. 各层W,b初始化,2.各层梯度设置为zeros
        Args:
            init_function: 权值更新函数,from initializer such as init_funciton = random_initialize

        Returns: None

        """
        nn_structure = list(map(lambda x: x.units, self.structure.values()))
        for l,perlayer in self.structure.items():
            perlayer.W = init_function.initialize_dense_W((nn_structure[l], nn_structure[l - 1]))
            perlayer.b = init_function.initialize_dense_b((nn_structure[l],))
            print(f'--第{l}层参数已经初始化==')

    def predict(self, x_test=None) -> np.array:
        """

        Args:
            x_test: default  None, means use original data( X_test), np.array, test_data input
        Returns: np.array,predict data (predict labels if module is a classifier)

        """
        if x_test is None:
            x_test = self.data_set.get('X_tests_norm')
        # 前向计算,所有样本直接输入
        for index, iter in self.structure.items():
            if index == 0:
                iter.forward(x_test)
            else:
                iter.forward(self.structure[index - 1].h)
        y_predict = self.structure[index].h
        y_predict = np.argmax(y_predict, axis=1)
        return y_predict


if __name__ == '__main__':
    mnistLoader = MNIST_Loader()
    data = mnistLoader.load()
    del mnistLoader
    data = normalize.normalize(data)

    opt = optimizer.Momentum(mu=0.9)  # optimizer.SGD()

    structure = OrderedDict()
    structure[0] = InputLayer(784)  # 3072
    structure[1] = FullConnectionLayer(units=64, activation=Relu)
    structure[2] = FullConnectionLayer(units=32, activation=Relu)
    structure[3] = FullConnectionLayer(units=10, activation=Sigmoid)

    module = DNN(structure=structure, data_set=data, optimizer=opt)
    module.init_params(He_initialization)
    module.train(epoch=10, batch_size=20, lr=0.1, lamb=0.001, loss_function=cross_entropy_loss)

