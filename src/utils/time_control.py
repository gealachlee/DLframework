import time



def timing(fun):
    '''
    修饰器,用于计算运行时长
    Args:
        fun:

    Returns:

    '''
    def inner(*args, **kwargs):
        print("--" * 20)
        start = time.time()
        fun(*args, **kwargs)
        end = time.time()
        print(f"-End of training-\n--Runtime: { end - start:.3f} s--")
    return inner

