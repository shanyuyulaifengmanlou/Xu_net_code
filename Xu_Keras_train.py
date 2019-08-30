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

# validation_data.txt文件路径
TXT_DIR = r'./valid_labels.txt'


TRAIN_DATA_DIR = './train_data' # 训练数据的路径
TEST_DATA_DIR = './test_data' # 测试数据的路径
MODEL_VIS_FILE = 'cover_stego_classfication' + '.png' # 模型可视化文件的存放路径

# 转换为灰度图
def rgb2gray(img):
    """
    利用公式对灰度图进行转换
    :param img:
    :return:
    """
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

# X_train = rgb2gray(X_train)


def Data_Process_For_Train(DATA_DIR):
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

    # 先添加的是stego
    for filename in glob.glob(DATA_DIR + '/stego/' + '*.jpg'):
        # print(filename)
        img = np.array(Image.open(filename)) # 打开图片
        img = rgb2gray(img) # 先将图片转换成灰度图
        # 做高通滤波
        highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0], img.shape[1], 1)
        X_train.append(highpassed_img) # 将滤波后的图片添加进X_train

    # 后添加的是cover
    for filename in glob.glob(DATA_DIR + '/cover/' + '*.jpg'):
        # print(filename)
        img = np.array(Image.open(filename))
        img = rgb2gray(img)
        # 做高通滤波
        highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0], img.shape[1], 1)
        X_train.append(highpassed_img)

    length = len(X_train)
    # 转换为np.array类型
    X_train = np.array(X_train, dtype=np.float32)
    one_hot = np.zeros((length, 2))
    # print(one_hot.shape)
    half_length = length // 2
    for i in range(half_length):
        one_hot[i, 0] = 1
        one_hot[half_length + i, 1] = 1

    # [1, 0]的是stego [0， 1]的是cover


    # print(one_hot)
    return X_train, one_hot

def validation(TXT_DIR, length):
    """
    用于读取validation_data
    :param TXT_DIR: validationdata的路径
    :param length: 我们要读的长度，这里会将X_test的长度传入
    :return: np.array  validate  是一个length * 2 的向量
    """
    with open(TXT_DIR) as labels:
        # 使用splitlines去掉换行符\n
        # 使用read方法计数的时候，会把换行符给计入，所以我们要在原有的长度乘2
        a = labels.read(length*2).splitlines()
        # print(a)
        # print(len(a))
        # list 类型
        # print(type(a))
        # a = np.array(a)

    validate = []

    for i in a:
        # 由于读入的是字符串，所以判定不能直接拿数字0，而是字符0
        if i == '0':
            validate.append([0, 1])
        else :
            validate.append([1, 0])

    # print(validate)
    validate = np.array(validate)
    # print(validate.shape)
    return validate


def Data_Process_For_TestValidation(DATA_DIR, TXT_DIR):
    """
    处理验证的图片数据
    :param DATA_DIR: 训练集的路径
    TXT_DIR: 验证集标签txt文件的路径
    :return:
    """
    X_test = []

    # 先添加的是stego
    for filename in glob.glob(DATA_DIR + '/*.jpg'):
        # print(filename)
        img = np.array(Image.open(filename))  # 打开图片
        img = rgb2gray(img)  # 先将图片转换成灰度图
        # 做高通滤波
        highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0],
                                                                                              img.shape[1], 1)
        X_test.append(highpassed_img)  # 将滤波后的图片添加进X_train

    length = len(X_test) # 用于文档读取

    validate = validation(TXT_DIR, length)
    X_test = np.array(X_test, dtype=np.float32)

    return X_test, validate

# test
"""
将train目录下图片输入给X_train Y_train
"""
X_train, Y_train = Data_Process_For_Train(TRAIN_DATA_DIR)
# print(one_hot)

"""
将验证集数据输入给X_test, Y_test
"""
X_test, Y_test = Data_Process_For_TestValidation(TEST_DATA_DIR, TXT_DIR)

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


X_train = Normalize(X_train)
# Fit Keras Channel

X_train, input_shape = fit_keras_channels(X_train)
# (100, 1024, 1024, 1) (1024, 1024, 1)
# print(X_train.shape, input_shape)


# 这里先不处理测试集，我先对训练集进行建模


inputs = Input(shape=input_shape, name="inputs")

# Group1
conv1 = Conv2D(filters=8, kernel_size=(5, 5), padding= 'SAME', name='Group1Conv1')(inputs)
# 还不会写abs层
# abs1 =
bn1 = BatchNormalization()(conv1)
tan1 = Activation('tanh', name='tanh1')(bn1)
pool1 = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='SAME', name='averagepool1')(tan1)

# Group2
conv2 = Conv2D(filters=16, kernel_size=(5, 5), padding= 'SAME', name='Group2Conv1')(pool1)
bn2 = BatchNormalization()(conv2)
tan2 = Activation('tanh', name='tanh2')(bn2)
pool2 = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='SAME', name='averagepool2')(tan2)

# Group3
conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding= 'SAME', name='Group3Conv1')(pool2)
bn3 = BatchNormalization()(conv3)
relu1 = Activation('relu', name='relu1')(bn3)
pool3 = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='SAME', name='averagepool3')(relu1)


# Group4
conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding= 'SAME', name='Group4Conv1')(pool3)
bn4 = BatchNormalization()(conv4)
relu2 = Activation('relu', name='relu2')(bn4)
pool4 = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='SAME', name='averagepool4')(relu2)

# Group5
conv5 = Conv2D(filters=128, kernel_size=(1, 1), padding= 'SAME', name='Group5Conv1')(pool4)
bn4 = BatchNormalization()(conv5)
relu3 = Activation('relu', name='relu3')(bn4)
pool5 = AveragePooling2D(pool_size=(64, 64) , strides=(1, 1), padding='VALID', name='averagepool5')(relu3)

# Group6
# 先实现128的全连接层full_connect
full_connect = Dense(128, activation='relu', name="full_connected")(pool5)

faltten_layer = Flatten(name="Flatten_Layer")(full_connect)
# 最后通过softmax输出两个分类
softmax = Dense(NUM_LABELS, activation='softmax', name="Classification")(faltten_layer)

# 定义模型的输入与输出
model = Model(inputs=inputs, outputs=softmax)
model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'])

# 查看模型摘要
print(model.summary())

# 模型可视化
# Keras的可视化使用的是utils下的plot model
# plot_model(model, to_file=MODEL_VIS_FILE, show_shapes=True)

# just test
# print("wait")
# print(X_test.shape)
# print(Y_test.shape)
model.fit(X_train,
          Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2,
          validation_data=(X_test, Y_test))







