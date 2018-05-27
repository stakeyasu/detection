
# coding: utf-8

import sys
import os
import time

from keras.datasets import cifar10
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

sys.modules['keras'] = None


def homework(train_X, train_y, test_X):
 
  rng = np.random.RandomState(1234)
    
  class Conv_BN:
    def __init__(self, filter_shape, function, strides, padding='VALID'):
        # Xavier Initialization
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6./(fan_in + fan_out)),
                        high=np.sqrt(6./(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W')

        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
        self.function = function
        self.strides = strides
        self.padding = padding
        self.avg_mean = 0.
        self.avg_var = 0.
        self.avg_cnt = 0


    def f_prop(self, x, keep_prob):
        u = tf.nn.conv2d(x, self.W, self.strides, self.padding) + self.b
        mean, var = tf.nn.moments(u, [0,1,2])
        bn = tf.nn.batch_normalization(u, mean, var, None, None, 2e-5)
        if self.function=="NON":
          return bn
        else:
          return self.function(bn)

  class dwConv_BN:
    def __init__(self, filter_shape, function, strides, padding='VALID'):
        # Xavier Initialization
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6./(fan_in + fan_out)),
                        high=np.sqrt(6./(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W')

        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
        self.function = function
        self.strides = strides
        self.padding = padding
        self.avg_mean = 0.
        self.avg_var = 0.
        self.avg_cnt = 0


    def f_prop(self, x, keep_prob):
        u = tf.nn.depthwise_conv2d(x, self.W, self.strides, self.padding) + self.b
        mean, var = tf.nn.moments(u, [0,1,2])
        bn = tf.nn.batch_normalization(u, mean, var, None, None, 2e-5)
        if self.function=="NON":
          return bn
        else:
          return self.function(bn)


  class DW_RESNET_MODULE:
    def __init__(self, filter_shape, function, strides, padding='VALID'):
        # Xavier Initialization
        self.function = function
        self.strides = strides
        self.padding = padding
        self.in_ch = filter_shape[2]
        self.out_ch = filter_shape[3]
        self.kernel = filter_shape[0]

    def f_prop(self, x, keep_prob):

        print((self.in_ch,self.out_ch))
        h=dwConv_BN((self.kernel, self.kernel, self.in_ch, 1), self.function, self.strides, self.padding).f_prop(x,keep_prob)
        h=Conv_BN((1, 1, self.in_ch, self.out_ch),"NON", [1,1,1,1], self.padding).f_prop(h,keep_prob)
        if (self.in_ch==self.out_ch)and(self.strides==[1,1,1,1]):
	        return tf.nn.relu(x+h)
        else :
                h2=Conv_BN((1, 1, self.in_ch, self.out_ch), "NON", self.strides, self.padding).f_prop(x,keep_prob)
                return tf.nn.relu(h+h2)

  class Pooling:
    def __init__(self, ksize, padding='VALID'):
        self.ksize = ksize
        self.strides = ksize
        self.padding = padding
    
    def f_prop(self, x, keep_prob):
        return tf.nn.avg_pool(x, self.ksize, self.strides, self.padding)

  class Flatten:
    def f_prop(self, x, keep_prob):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    
  class Dense:
    def __init__(self, in_dim, out_dim, function):
        # Xavier Initialization
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6./(in_dim + out_dim)),
                        high=np.sqrt(6./(in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x, keep_prob):
        return self.function(tf.matmul(x, self.W) + self.b)    

  script_name=os.path.basename(__file__)
  fp1 = open(script_name.replace(".py",".txt"), "w")
  n_ch = 64
    
  layers = [                            # (縦の次元数)x(横の次元数)x(チャネル数)
    Conv_BN((3, 3, 3, n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 32 -> 32
    DW_RESNET_MODULE((5, 5, n_ch, n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 32 -> 32
    DW_RESNET_MODULE((5, 5, n_ch, n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 32 -> 32
    DW_RESNET_MODULE((5, 5, n_ch, n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 32 -> 32
    DW_RESNET_MODULE((5, 5, n_ch, 2*n_ch), tf.nn.relu, [1,2,2,1,],"SAME"),  # 32 -> 16
    DW_RESNET_MODULE((5, 5, 2*n_ch, 2*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 16 -> 16
    DW_RESNET_MODULE((5, 5, 2*n_ch, 2*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 16 -> 16
    DW_RESNET_MODULE((5, 5, 2*n_ch, 2*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 16 -> 16
    DW_RESNET_MODULE((5, 5, 2*n_ch, 4*n_ch), tf.nn.relu, [1,2,2,1],"SAME"),  # 16 -> 8
    DW_RESNET_MODULE((5, 5, 4*n_ch, 4*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 8 -> 8
    DW_RESNET_MODULE((5, 5, 4*n_ch, 4*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 8 -> 8
    DW_RESNET_MODULE((5, 5, 4*n_ch, 4*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 8 -> 8
    DW_RESNET_MODULE((5, 5, 4*n_ch, 8*n_ch), tf.nn.relu, [1,2,2,1],"SAME"),  # 8 -> 4
    DW_RESNET_MODULE((5, 5, 8*n_ch, 8*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 4 -> 4
    DW_RESNET_MODULE((5, 5, 8*n_ch, 8*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 4 -> 4
    DW_RESNET_MODULE((5, 5, 8*n_ch, 8*n_ch), tf.nn.relu, [1,1,1,1],"SAME"),  # 4 -> 4
    Pooling((1, 4, 4, 1)),            #  4 ->  1
    Flatten(),
    Dense(8*n_ch, 10, tf.nn.softmax)
  ]

  x = tf.placeholder(tf.float32, [None, 32, 32, 3])
  t = tf.placeholder(tf.float32, [None, 10])
  keep_prob = tf.placeholder(tf.float32)

  def f_props(layers, x,keep_prob):
    for layer in layers:
        x = layer.f_prop(x,keep_prob)
    return x

  y = f_props(layers, x,keep_prob)

  cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ
#  train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
  train = tf.train.AdamOptimizer().minimize(cost)

  valid = tf.argmax(y, 1) 
  train_X /= 255.
  test_X /= 255.
  train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

  n_epochs = 30
  batch_size = 100
  n_batches = train_X.shape[0]//batch_size
  __n_batches = valid_X.shape[0]//batch_size
  _n_batches = test_X.shape[0]//batch_size
   
    
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        start_time = time.time()
        train_X, train_y = shuffle(train_X, train_y, random_state=42)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end], keep_prob: 1.0})
            
        pred_valid_y = []
        valid_cost = 0
        for i in range(__n_batches):
            start = i * batch_size
            end = start + batch_size
            _pred_valid_y, _valid_cost = sess.run([valid, cost], feed_dict={x: valid_X[start:end], t: valid_y[start:end], keep_prob: 1.0})
            pred_valid_y.extend(_pred_valid_y)
            valid_cost = valid_cost + _valid_cost
        elapsed_time = time.time() - start_time
            
        print('EPOCH:: %i, Validation cost: %.4f, Validation F1: %.4f, Time: %.4f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_valid_y, average='macro'), elapsed_time))
        fp1.write('EPOCH:: %i, Validation cost: %.4f, Validation F1: %.4f, Time: %.4f \n' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_valid_y,  average='macro'), elapsed_time))
        fp1.flush()
    print("Train Complete.")
    fp1.close()
    pred_y = []
    
    for i in range(_n_batches):
            start = i * batch_size
            end = start + batch_size
            _pred_y = sess.run(valid, feed_dict={x: test_X[start:end], keep_prob: 1.0})
            pred_y.extend(_pred_y)
    
    sess.close()

  return pred_y

def load_cifar():
    (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()

    cifar_X = np.r_[cifar_X_1, cifar_X_2]
    cifar_y = np.r_[cifar_y_1, cifar_y_2]

    cifar_X = cifar_X.astype('float32') / 255
    cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

    train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y,
                                                        test_size=10000,
                                                        random_state=42)

    return (train_X, test_X, train_y, test_y)

def validate_homework():
    train_X, test_X, train_y, test_y = load_cifar()

    # validate for small dataset
    train_X_mini = train_X[:1000]
    train_y_mini = train_y[:1000]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(np.argmax(test_y_mini, 1), pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_cifar()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(np.argmax(test_y, 1), pred_y, average='macro'))

score_homework()



