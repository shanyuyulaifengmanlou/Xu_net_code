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
from scipy import ndimage


BATCH_SIZE = 25 # 批处理图片个数 默认为50 前期测试
EPOCHS = 3 # DEFAULT 10
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

# 高通滤波器的卷积核
kernel_5x5 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)


# 模型网络结构文件
MODEL_VIS_FILE = 'xu_net_classfication' + '.png'
# 模型文件
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
# 训练记录文件
HISTORY_FILE = filename_str.format(HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)


TRAIN_DATA_DIR = './train_data' # 训练数据的路径
TEST_DATA_DIR = './test_data' # 测试数据的路径
MODEL_VIS_FILE = 'cover_stego_classfication' + '.png' # 模型可视化文件的存放路径

TXT_DIR = r'./valid_labels.txt'


def rgb2gray(img):
    """
    利用公式对灰度图进行转换
    :param img:
    :return:
    """
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def Data_Process_For_TestValidation(DATA_DIR, TXT_DIR):
    """
    处理验证的图片数据
    :param DATA_DIR: 训练集的路径
    TXT_DIR:是文本中的标签
    :return:
    """
    X_test = []
    for filename in glob.glob(DATA_DIR + '*.jpg'):
        # print(filename)
        img = np.array(Image.open(filename)) # 打开图片
        img = rgb2gray(img) # 先将图片转换成灰度图
        # 做高通滤波
        highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0], img.shape[1], 1)
        X_test.append(highpassed_img) # 将滤波后的图片添加进X_test

    test_length = len(X_test) # 图片的长度，我们要利用这个长度对应读出标签




