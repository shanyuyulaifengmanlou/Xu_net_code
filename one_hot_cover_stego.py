import os
import numpy as np
import glob
from PIL import Image


TRAIN_DATA_DIR = './train_data' # 训练数据的路径

# 进行one—hot编码
# def text2vec(text, length=CAPTCHA_LEN):
#     text_len = len(text)
#     验证码长度校验
    # if text_len != length:
    #     raise ValueError('Error: length of captcha should be {}, but got {}'.format(length, text_len))
    #
    # vec = np.zeros((1, 2))
X_train = []
for filename in glob.glob(TRAIN_DATA_DIR + '/stego/' + '*.jpg'):
    print(filename)
    X_train.append(np.array(Image.open(filename)))

for filename in glob.glob(TRAIN_DATA_DIR + '/cover/' + '*.jpg'):
    print(filename)
    X_train.append(np.array(Image.open(filename)))

length = len(X_train)

X_train = np.array(X_train, dtype=np.float32)

one_hot = np.zeros((length, 2))
print(one_hot.shape)
half_length = length // 2
for i in range(half_length):
    one_hot[i, 0] = 1
    one_hot[half_length + i, 1] = 1


