import cv2
import numpy as np
import scipy.misc
from scipy import ndimage
# 高通滤波器是根据像素比它周围的像素更突出，就会提升它的亮度
# 常用边缘提取与增强，检测图像中物体的边缘位置
from PIL import Image
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
import glob
import pickle
import numpy as np
import tensorflow.gfile as gfile
import matplotlib.pyplot as plt

BATCH_SIZE = 25 # 批处理图片个数 默认为50 前期测试
EPOCHS = 10
IMAGE_SIZE = 1024 # 输入的是1024 x 1024
NUM_CHANNEL = 1 # 通道为1，默认是灰度图
NUM_LABELS = 2 # Labels个数，有隐写或无隐写
#BN_DECAY = 0.95
#UPDATE_OPS_COLLECTION = 'Discriminative_update_ops'
filename_str = '{}xu_net_{}_{}_bs_{}_epochs_{}{}'
MODEL_DIR = './model/train_demo/'
MODEL_FORMAT = '.h5'
HISTORY_DIR = './history/train_demo/'
HISTORY_FORMAT = '.history'

OPT = 'Nadam' # 原来Xu-net使用的是momentum优化器，Keras没有，所以选了个比较接近的Nadam
LOSS = 'categorical_crossentropy' # 二分类问题，这里我选用的是交叉熵作为损失函数


# 模型网络结构文件
MODEL_VIS_FILE = 'cover_stego_classfication' + '.png' # 模型可视化文件的存放路径
# 模型文件
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
# 训练记录文件
HISTORY_FILE = filename_str.format(HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)


TRAIN_DATA_DIR = './train_data' # 训练数据的路径
TEST_DATA_DIR = './test_data' # 测试数据的路径


kernel_3x3 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

kernel_5x5 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)

# 转换为灰度图
def rgb2gray(img):
    """
    利用公式对灰度图进行转换
    :param img:
    :return:
    """
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[:,:,:3], [0.299, 0.587, 0.114])

# X_train = rgb2gray(X_train)

def Data_Process_For_Train(TRAIN_DATA_DIR):
    """
    分别将训练数据读入X_train中
    stego和cover图像一半一半
    前50为stego
    后50为cover
    返回X_train 和经过one_hot编码的 100 x 2数组
    :param TRAIN_DATA_DIR:
    :return: X_train np.array类型
             one_hot编码 length， 2 的向量
    """
    X_train = []
    for filename in glob.glob(TRAIN_DATA_DIR + '/stego/' + '*.jpg'):
        img = np.array(Image.open(filename))
        # print(img.shape)
        img = rgb2gray(img)
        # print("wait")
        # print(img.shape)
        highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0], img.shape[1], 1)
        X_train.append(highpassed_img)


    # for filename in glob.glob(TRAIN_DATA_DIR + '/cover/' + '*.jpg'):
    #     # print(filename)
    #     img = np.array(Image.open(filename))
    #     img = rgb2gray(img)
    #     highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0)
    #     X_train.append(highpassed_img)
        # X_train.append(img)

    length = len(X_train)
    # 转换为np.array类型
    X_train = np.array(X_train, dtype=np.float32)
    one_hot = np.zeros((length, 2))
    # print(one_hot.shape)
    half_length = length // 2
    for i in range(half_length):
        one_hot[i, 0] = 1


    # print(one_hot)
    return X_train, one_hot
# test
X_train, one_hot = Data_Process_For_Train(TRAIN_DATA_DIR)
# print(one_hot)



# 绘制对应的20个灰度图，可视化
# plt.figure()
# for i in range(20):
#     plt.subplot(5,4,i+1) # 绘制前20个验证码，以5行4列子图形式展示
#     plt.tight_layout() # 自动适配子图尺寸
#     plt.imshow(X_train[i], cmap='Greys')
#     plt.xticks([]) # 删除x轴标记
#     plt.yticks([]) # 删除y轴标记
# plt.show()


def Normalize(X_train):
    """
    对数据规范化
    也就是将灰度值除以255
    :param X_train:
    :return:
    """
    return X_train / 255

# 适配 Keras 图像数据格式
def fit_keras_channels(batch, rows=IMAGE_SIZE, cols=IMAGE_SIZE):
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    return batch, input_shape

# X_train = Normalize(X_train)
# Fit Keras Channel
# X_train, input_shape = fit_keras_channels(X_train)
# (100, 1024, 1024, 1) (1024, 1024, 1)
print(X_train.shape, one_hot.shape)


