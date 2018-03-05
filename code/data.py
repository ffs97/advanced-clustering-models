import numpy as np
import tensorflow as tf


def spiral(N):
    # N number of points per class
    D = 2  # dimensionality
    K = 5  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    for j in xrange(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(2.5, 10.0, N)  # radius
        t = np.linspace(j * 1.25, (j + 1) * 1.25, N) + \
            np.random.randn(N) * 0.05  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    return X


def circular_data(len):
    data = []
    for _ in xrange(len):
        p = np.random.uniform(-np.pi, np.pi)
        data.append([np.cos(p), np.sin(p)])
    return np.array(data)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
