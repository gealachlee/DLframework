from abc import ABCMeta


class Optimizer(metaclass=ABCMeta):
    params: dict

    @staticmethod
    def forward_calculate(*args, **kwargs):
        pass

    @staticmethod
    def backword_calculate(*args, **kwargs):
        pass

    @staticmethod
    def update_calculate(*args, **kwargs):
        pass

    @staticmethod
    def backword_update(*args, **kwargs):
        pass


class SGD(Optimizer):

    @staticmethod
    def forward_calculate(*args, **kwargs):
        pass

    @staticmethod
    def backword_calculate(*args, **kwargs):
        pass

    @staticmethod
    def update_calculate(iter, lr, lamb, **kwargs):
        '{W:{},b:{}'
        iter.W = iter.W - lr * (iter.dW + lamb * iter.W)
        iter.b = iter.b - lr * (iter.db)

    @staticmethod
    def backword_update(iter, lr, lamb, *args, **kwargs):
        SGD.backword_calculate(*args, **kwargs)
        SGD.update_calculate(iter, lr, lamb, **kwargs)


class Momentum(Optimizer):

    def __init__(self, mu: float):
        """

        Args:
            mu:  momentum params
        """
        self.mu = mu


    @staticmethod
    def forward_calculate(*args, **kwargs):
        pass

    @staticmethod
    def backword_calculate(*args, **kwargs):
        pass

    @staticmethod
    def update_calculate(iter, lr, lamb, **kwargs):
        iter.W += iter.momentum_params['momentum_W']
        iter.b += iter.momentum_params['momentum_b']

    def backword_update(self, iter, lr, lamb, *args, **kwargs):
        Momentum.backword_calculate(iter, lr, lamb, *args, **kwargs)

        if 'momentum_params' not in iter.__annotations__:
            iter.momentum_params = {
                'momentum_W': 0,
                'momentum_b': 0
            }
        momentum_W = iter.momentum_params.get('momentum_W')
        momentum_b = iter.momentum_params.get('momentum_b')

        momentum_W += self.mu * momentum_W - lr * (iter.dW + lamb * iter.W)
        momentum_b += self.mu * momentum_b - lr * iter.db

        iter.momentum_params['momentum_W'] = momentum_W
        iter.momentum_params['momentum_b'] = momentum_b

        Momentum.update_calculate(iter, lr, lamb, **kwargs)
