
from __future__ import print_function
import numpy as np
import six
import os
import paddle.dataset.common


__all__ = ['train', 'test']

URL = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1eQ2sXFTfZyovkOiOuGb9BpW8X6zrLJG5'
MD5 = 'd4accdce7a25600298819f8e28e8d593'

TRAIN_DATA = None
TEST_DATA = None






def load_data(filename, feature_num=15, ratio=0.8):
    global TRAIN_DATA, TEST_DATA
    if TRAIN_DATA is not None and TEST_DATA is not None:
        return

    data = np.fromfile(filename, sep=' ')
    data = data.reshape(data.shape[0] // feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
   
    for i in six.moves.range(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    offset = int(data.shape[0] * ratio)
    TRAIN_DATA = data[:offset]
    TEST_DATA = data[offset:]




def train(file):
    global TRAIN_DATA
    load_data(file, 15, 0.8)

    def reader():
        for d in TRAIN_DATA:
            yield d[:-1], d[-1:]

    return reader


def test(file):
    global TEST_DATA
    load_data(file, 15, 0.8)

    def reader():
        for d in TEST_DATA:
            yield d[:-1], d[-1:]

    return reader


