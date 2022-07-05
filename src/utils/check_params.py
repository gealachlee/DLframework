
def debug_train_params(fun):
    '''
    修饰器,用于检查输入是否为正数
    '''
    def inner(*args, **kwargs):
        for key,values in list(kwargs.items())[:4]:
            if isinstance(values,(float,int)) :
                if values <=0 :
                    raise ValueError(f"{key} can't be a negative number!")
            else:
                raise TypeError(f"{key} must be numeric!")
        fun(*args, **kwargs)
    return inner

