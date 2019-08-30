import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import numpy as np
import tensorflow.contrib.slim as slim
import os
# 以下是用于设置显卡GPU

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import random
from scipy import ndimage
import scipy.io as sio

BATCH_SIZE = 10 # default:50
bs = 25
IMAGE_SIZE = 512
NUM_CHANNEL = 3 # default:1 因为原始数据集对应的都是灰度图像
NUM_LABELS = 2 # 最后输出只有0和1来代表是否隐写
NUM_ITER =2000 # default：230000
NUM_SHOWTRAIN = 200 #show result eveary epoch
NUM_SHOWTEST = 500 # default:5000
#BN_DECAY = 0.95
#UPDATE_OPS_COLLECTION = 'Discriminative_update_ops'

is_train = True
save_dir = r'F:\Xu-net\model'

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])
is_train = tf.placeholder(tf.bool, name='is_train')

# 实现高通滤波器
hpf =np.zeros([5, 5, 1, 1], dtype=np.float32)
hpf = np.zeros([5,5,1,1],dtype=np.float32)
hpf[:,:,0,0] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)
kernel0 = tf.Variable(hpf, name='kernel0')
# 初始时候先做一次高通滤波器
conv0 =tf.nn.conv2d(x, kernel0, [1,1,1,1], 'SAME', name='conv0')

# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 从服从指定正态分布的数值中取出指定个数的值
with tf.variable_scope('Group1') as scope:
    kernel1 = tf.Variable(tf.random_normal([5, 5, 1, 8], mean=0, stddev=0.01), name='kernel1')
    conv1 = tf.nn.conv2d(conv0, kernel1, [1,1,1,1], padding='SAME', name='conv1')
    abs1 = tf.abs(conv1, name='abs1')

    # tf.nn.moments是计算均值和方差，传入的参数可以指定计算的维度
    batch_mean1, batch_var1 =tf.nn.moments(abs1, [0, 1, 2], name='moments1')
    beta1 = tf.Variable(tf.zeros([8]), name='beta1')
    gamma1 =tf.Variable(tf.ones([8]), name='gamma1')
    bn1 =tf.nn.batch_normalization(abs1, batch_mean1, batch_var1, beta1, gamma1,1e-3)

    tanh1 = tf.nn.tanh(bn1, name='tanh1')
    pool1 =tf.nn.avg_pool(tanh1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

with tf.variable_scope("Group2") as scope:
    kernel2_1 = tf.Variable(tf.random_normal([5, 5, 8, 16], mean=0.0, stddev=0.01), name="kernel2_1")
    conv2_1 = tf.nn.conv2d(pool1, kernel2_1, [1, 1, 1, 1], padding="SAME", name="conv2_1")

    batch_mean2, batch_var2 = tf.nn.moments(conv2_1, [0, 1, 2], name="moments2")
    beta2 = tf.Variable(tf.zeros([16]), name="beta2")
    gamma2 = tf.Variable(tf.ones([16]), name="gamma2")
    bn2_1 = tf.nn.batch_normalization(conv2_1, batch_mean2, batch_var2, beta2, gamma2, 1e-3)

    tanh2_1 = tf.nn.tanh(bn2_1, name="tanh2_1")
    pool2 = tf.nn.avg_pool(tanh2_1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2_1")

with tf.variable_scope("Group3") as scope:
    kernel3 = tf.Variable(tf.random_normal([1, 1, 16, 32], mean=0.0, stddev=0.01), name="kernel3")
    conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding="SAME", name="conv3")

    batch_mean3, batch_var3 = tf.nn.moments(conv3, [0, 1, 2], name="moments3")
    beta3 = tf.Variable(tf.zeros([32]), name="beta3")
    gamma3 = tf.Variable(tf.ones([32]), name="gamma3")
    bn3 = tf.nn.batch_normalization(conv3, batch_mean3, batch_var3, beta3, gamma3, 1e-3)

    relu3 = tf.nn.relu(bn3, name="bn3")
    pool3 = tf.nn.avg_pool(relu3, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

with tf.variable_scope("Group4") as scope:
    kernel4_1 = tf.Variable(tf.random_normal([1, 1, 32, 64], mean=0.0, stddev=0.01), name="kernel4_1")
    conv4_1 = tf.nn.conv2d(pool3, kernel4_1, [1, 1, 1, 1], padding="SAME", name="conv4_1")

    batch_mean4, batch_var4 = tf.nn.moments(conv4_1, [0, 1, 2], name="moments4")
    beta4 = tf.Variable(tf.zeros([64]), name="beta4")
    gamma4 = tf.Variable(tf.ones([64]), name="gamma4")
    bn4_1 = tf.nn.batch_normalization(conv4_1, batch_mean4, batch_var4, beta4, gamma4, 1e-3)

    relu4_1 = tf.nn.relu(bn4_1, name="relu4_1")
    pool4 = tf.nn.avg_pool(relu4_1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4_1")

with tf.variable_scope("Group5") as scope:
    kernel5 = tf.Variable(tf.random_normal([1, 1, 64, 128], mean=0.0, stddev=0.01), name="kernel5")
    conv5 = tf.nn.conv2d(pool4, kernel5, [1, 1, 1, 1], padding="SAME", name="conv5")

    batch_mean5, batch_var5 = tf.nn.moments(conv5, [0, 1, 2], name="moments5")
    beta5 = tf.Variable(tf.zeros([128]), name="beta5")
    gamma5 = tf.Variable(tf.ones([128]), name="gamma5")
    bn5 = tf.nn.batch_normalization(conv5, batch_mean5, batch_var5, beta5, gamma5, 1e-3)

    relu5 = tf.nn.relu(bn5, name="relu5")
    pool5 = tf.nn.avg_pool(relu5, ksize=[1, 32, 32, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool5")

with tf.variable_scope("Group6") as scope:

    # get_shape().as_list() 返回一个元组，里面是张量的大小
    pool_shape =pool5.get_shape().as_list()
    pool_reshape = tf.reshape(pool5, [pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])
    weights = tf.Variable(tf.random_normal([128, 2], mean=0.0, stddev=0.01), name="weights")
    bias =tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.01), name="bias")
    # y_ 为predict值
    y_ = tf.matmul(pool_reshape, weights) + bias

vars = tf.trainable_variables()
params = [v for v in vars if ( v.name.startswith('Group1/') or  v.name.startswith('Group2/') or  v.name.startswith('Group3/') or  v.name.startswith('Group4/') or  v.name.startswith('Group5/') or  v.name.startswith('Group6/') ) ]

correct_prediction =tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# tf.cast是对数据类型进行转换
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', accuracy)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,5000, 0.9, staircase=True)
opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss,var_list=params,global_step=global_step)


data_x = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE,NUM_LABELS])
for i in range(0,bs):
    data_y[i,1] = 1
for i in range(bs,bs*2):
    data_y[i,0] = 1

saver = tf.train.Saver()
#merged = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    filelist = []
    file = r'F:\Xu-net\train_100.txt'
    with  open(file, 'r') as f:
        lines = f.readlines() # lines列表里面都是我的mat名字，如
        for i in lines:
            filelist.append(i.strip('\n'))  # 除去换行符



    count = 0
    list = [h for h in range(1, 101)] # 训练100张 [1,.......100]

#   下面的不会改了



    for i in range(1, NUM_ITER + 1):

        for j in range(bs):
            dataC = sio.loadmat(filelist[list[count]])
            cover = dataC['coefC']
            stego = dataC['coefS']
            data_x[j, :, :, 0] = cover.astype(np.float32)
            data_x[j + bs, :, :, 0] = stego.astype(np.float32)
            count = count + 1

        _, temp, l = sess.run([opt, accuracy, loss], feed_dict={x: data_x, y: data_y})

        if i == 1:
            print('9_shi_s0.2: batch result')
            print('epoch:', i)
            print('loss:', l)
            print('accuracy:', temp)
            print(' ')
        if i % 100 == 0:
            print('9_shi_s0.2: batch result')
            print('epoch:', i)
            print('loss:', l)
            print('accuracy:', temp)
            print(' ')
        if i % (5000) == 0:
            saver = tf.train.Saver()
            saver.save(sess, save_dir + str(i) + '.ckpt')

            m1 = np.zeros([8])
            v1 = np.zeros([8])

            m2 = np.zeros([16])
            v2 = np.zeros([16])

            m3 = np.zeros([32])
            v3 = np.zeros([32])

            m4 = np.zeros([64])
            v4 = np.zeros([64])

            m5 = np.zeros([128])
            v5 = np.zeros([128])

            count1 = 0  # averaging mean and variance for testing
            times = 0
            while count1 < 4000:
                for j in range(int(bs)):
                    if count1 % 4000 == 0:
                        count1 = count1 % 4000
                        random.seed(i)
                        random.shuffle(list)
                    dataC = sio.loadmat(pathI + '/' + fileList[list[count1]])
                cover = dataC['coefC']
                # dataC =  sio.loadmat(pathI+'/'+fileList[count])
                stego = dataC['coefS']
                data_x[j, :, :, 0] = cover.astype(np.float32)
                data_x[j + bs, :, :, 0] = stego.astype(np.float32)
                count1 = count1 + 1

            tm1, tv1, \
            tm2, tv2, \
            tm3, tv3, \
            tm4, tv4, \
            tm5, tv5, \
                = sess.run([batch_mean1, batch_var1, \
                            batch_mean2, batch_var2, \
                            batch_mean3, batch_var3, \
                            batch_mean4, batch_var4, \
                            batch_mean5, batch_var5], feed_dict={x: data_x, y: data_y})

            times = times + 1
            # print(times)

            m1 = m1 + tm1
            v1 = v1 + tv1
            m2 = m2 + tm2
            v2 = v2 + tv2
            m3 = m3 + tm3
            v3 = v3 + tv3
            m4 = m4 + tm4
            v4 = v4 + tv4
            m5 = m5 + tm5
            v5 = v5 + tv5

        m1 = m1 / float(times)
        v1 = v1 / float(times - 1)
        m2 = m2 / float(times)
        v2 = v2 / float(times - 1)
        m3 = m3 / float(times)
        v3 = v3 / float(times - 1)
        m4 = m4 / float(times)
        v4 = v4 / float(times - 1)
        m5 = m5 / float(times)
        v5 = v5 / float(times - 1)

        np.savez(save_dir + 'bn' + str(i) + '.npz', bm1=m1, bv1=v1, bm2=m2, bv2=v2, bm3=m3, bv3=v3, bm4=m4, bv4=v4,
                 bm5=m5, bv5=v5)

