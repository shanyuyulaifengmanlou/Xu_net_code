import numpy as np
import glob
from PIL import Image
from scipy import ndimage
kernel_5x5 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)

n = 100

TXT_DIR = r'./valid_labels.txt'
TEST_DATA_DIR = './test_data' # 测试数据的路径

# def validation(TXT_DIR, length):
#     with open(TXT_DIR) as labels:
#         # 使用splitlines去掉换行符\n
#         # 使用read方法计数的时候，会把换行符给计入，所以我们要在原有的长度乘2
#         a = labels.read(length*2).splitlines()
#         print(a)
#         print(len(a))
#         # list 类型
#         # print(type(a))
#         # a = np.array(a)
#
#     validate = []
#
#     for i in a:
#         # 由于读入的是字符串，所以判定不能直接拿数字0，而是字符0
#         if i == '0':
#             validate.append([0, 1])
#         else :
#             validate.append([1, 0])
#
#     print(validate)
#     validate = np.array(validate)
#     print(validate.shape)
#
# validation(TXT_DIR, n)

def rgb2gray(img):
    """
    利用公式对灰度图进行转换
    :param img:
    :return:
    """
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])


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

    print(X_test.shape)
    return X_test, validate


X_test, Y_test = Data_Process_For_TestValidation(TEST_DATA_DIR, TXT_DIR)
print(X_test.shape)
print(Y_test.shape)
