# -*- coding: utf-8 -*-
import os
import warnings

import numpy as np
import struct
import pickle


class Loader:
    data: dict = {}
    path: str
    _data_exist = False


    def setPath(self,path :str):
        self.path = path

    def load(self):
        raise NotImplementedError

    def _checkFile(self, filepath: str):
        if not os.path.exists(path=filepath):
            raise FileNotFoundError(f"--文件不存在,文件位置为:{filepath}--")
        return filepath


class MNIST_Loader(Loader):
    __slots__ = ['trainX_FileName', 'trainy_FileName', 'testX_FileName', 'testy_FileName', 'path']

    def __init__(self, trainX_FileName: str = 'train-images.idx3-ubyte',
                 trainy_FileName: str = 'train-labels.idx1-ubyte',
                 testX_FileName: str = 't10k-images.idx3-ubyte',
                 testy_FileName: str = 't10k-labels.idx1-ubyte',
                 path=f'{os.path.abspath(os.path.join(os.getcwd(), "../.."))}'):

        self.path = path
        self.trainX_FileName = trainX_FileName
        self.trainy_FileName = trainy_FileName
        self.testX_FileName = testX_FileName
        self.testy_FileName = testy_FileName

    def load(self):
        '''
        :return:Dict includes {X_train,X_test,y_train,y_test}
        '''
        if self._data_exist:
            warnings.warn("---Data has been load!---")
        variableList = ['X_train', 'X_test', 'y_train', 'y_test']
        nameList = [self.trainX_FileName, self.testX_FileName, self.trainy_FileName, self.testy_FileName]
        load_funcList = [self.load_train_images, self.load_test_images, self.load_train_labels, self.load_test_labels]
        try:
            for i, name, load_func in zip(variableList, nameList, load_funcList):
                temp_path = f'{self.path}\\dataset\\MNIST\\{name}'
                self.data[i] = load_func(temp_path)
            self._data_exist = True
            return self.data
        except Exception as e:
            raise e

    @staticmethod
    def _decode_idx3_ubyte(idx3_ubyte_file):
        """解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx3_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii'  # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)  # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
        print(offset)
        fmt_image = '>' + str(
            image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
        print(fmt_image, offset, struct.calcsize(fmt_image))
        images = np.empty((num_images, num_rows, num_cols))
        # plt.figure()
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 %d' % (i + 1) + '张')
                print(offset)
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            # print(images[i])
            offset += struct.calcsize(fmt_image)
        return images

    @staticmethod
    def _decode_idx1_ubyte(idx1_ubyte_file):
        """
                解析idx1文件的通用函数
                :param idx1_ubyte_file: idx1文件路径
                :return: 数据集
                """
        # 读取二进制数据
        bin_data = open(idx1_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels

    @staticmethod
    def load_train_images(idx_ubyte_file):
        """
                TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000803(2051) magic number
                0004     32 bit integer  60000            number of images
                0008     32 bit integer  28               number of rows
                0012     32 bit integer  28               number of columns
                0016     unsigned byte   ??               pixel
                0017     unsigned byte   ??               pixel
                ........
                xxxx     unsigned byte   ??               pixel
                Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

                :param idx_ubyte_file: idx文件路径
                :return: n*row*col维np.array对象，n为图片数量
                """

        return MNIST_Loader._decode_idx3_ubyte(idx_ubyte_file)

    @staticmethod
    def load_train_labels(idx_ubyte_file):
        """
                TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                0004     32 bit integer  60000            number of items
                0008     unsigned byte   ??               label
                0009     unsigned byte   ??               label
                ........
                xxxx     unsigned byte   ??               label
                The labels values are 0 to 9.

                :param idx_ubyte_file: idx文件路径
                :return: n*1维np.array对象，n为图片数量
                """
        return MNIST_Loader._decode_idx1_ubyte(idx_ubyte_file)

    @staticmethod
    def load_test_images(idx_ubyte_file):
        """
                TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000803(2051) magic number
                0004     32 bit integer  10000            number of images
                0008     32 bit integer  28               number of rows
                0012     32 bit integer  28               number of columns
                0016     unsigned byte   ??               pixel
                0017     unsigned byte   ??               pixel
                ........
                xxxx     unsigned byte   ??               pixel
                Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

                :param idx_ubyte_file: idx文件路径
                :return: n*row*col维np.array对象，n为图片数量
                """
        return MNIST_Loader._decode_idx3_ubyte(idx_ubyte_file)

    @staticmethod
    def load_test_labels(idx_ubyte_file):
        """
                TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                0004     32 bit integer  10000            number of items
                0008     unsigned byte   ??               label
                0009     unsigned byte   ??               label
                ........
                xxxx     unsigned byte   ??               label
                The labels values are 0 to 9.

                :param idx_ubyte_file: idx文件路径
                :return: n*1维np.array对象，n为图片数量
                """
        return MNIST_Loader._decode_idx1_ubyte(idx_ubyte_file)

    def summary(self):
        for key, value in self.data.items():
            print(f"{key}: shape={value.shape},dtype={value.dtype}")


class CIFAR_10_Loader(Loader):

    __slots__ = ['trainX_FileName', 'trainy_FileName', 'testX_FileName', 'testy_FileName', 'path']

    def __init__(self, path=f'{os.path.abspath(os.path.join(os.getcwd(), "../../.."))}'):
        self.path=path

    def load_pickle(self, f):
        return pickle.load(f, encoding='latin1')

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        try:
            with open(filename, 'rb') as f:
                datadict = self.load_pickle(f)  # dict类型
                X = datadict['data']  # X, ndarray, 像素值
                Y = datadict['labels']  # Y, list, 标签, 分类

                # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
                # transpose，转置
                # astype，复制，同时指定类型
                X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
                Y = np.array(Y)
                return X, Y
        except Exception as e:
            raise e

    def load(self):
        """ load all of cifar """
        if self._data_exist:
            warnings.warn("Data has been load!")
        xs = ys = []  # list
        ROOT = self.path + '\\dataset\\' + 'cifar-10-python'
        # 训练集batch 1～5
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
            ys.append(Y)
        Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
        Ytr = np.concatenate(ys)
        del X, Y

        # 测试集
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        self._data_exist = True
        return Xtr, Ytr, Xte, Yte


def test():
    mnistLoader = MNIST_Loader(
        trainX_FileName="train-images.idx3-ubyte",
        trainy_FileName="train-labels.idx1-ubyte",
        testX_FileName="t10k-images.idx3-ubyte",
        testy_FileName="t10k-labels.idx1-ubyte",
    )
    data = mnistLoader.load()
