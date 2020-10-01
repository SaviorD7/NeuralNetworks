import six
import numpy as np
if six.PY3:
    import subprocess
    import sys
    import_cv2_proc = subprocess.Popen(
        [sys.executable, "-c", "import cv2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = import_cv2_proc.communicate()
    retcode = import_cv2_proc.poll()
    if retcode != 0:
        cv2 = None
    else:
        import cv2
else:
    try:
        import cv2
    except ImportError:
        cv2 = None
import os
import tarfile
import six.moves.cPickle as pickle
from tqdm import tqdm
import paddle
from PIL import Image

labels = ['NORMAL', 'PNEUMONIA']
img_size = 150

"""
def custom_reader_test(data_dir):

    def test():
        for label in labels: 
            path = os.path.join(data_dir, label)
            class_num = np.array([labels.index(label)])
             #print(class_num)
            for img in tqdm(os.listdir(path)):
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            # print(img_arr.shape)
                res = cv2.resize(img_arr, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC).astype(float)
                res = res / 255
                yield res, class_num

    return test


def custom_reader_train(data_dir):

    def train():
        for label in labels: 
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            #print(class_num)
            for img in tqdm(os.listdir(path)):
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            # print(img_arr.shape)
                res = cv2.resize(img_arr, dsize=(150, 150), interpolation=cv2.INTER_CUBIC).astype(float)
                res = res / 255
                yield res, class_num

    return train

"""
def upload_data(data_dir):
    new_input = []
    for label in labels: 
            path = os.path.join(data_dir, label)
            class_num = np.array([labels.index(label)]).astype('int64')
             #print(class_num)
            for img in tqdm(os.listdir(path)):
                img_arr = Image.open(os.path.join(path,img)).convert('L')
                img_arr = img_arr.resize((150, 150), Image.ANTIALIAS)
                res = np.array(img_arr).reshape( 1, 150, 150).astype(np.float32)
                res = res / 255.0 * 2.0 - 1.0
                data_dict = {}
                data_dict['data'] = res
                data_dict['label'] = class_num
                new_input.append(data_dict)
    return new_input

