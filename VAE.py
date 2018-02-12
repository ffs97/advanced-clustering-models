import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data import spiral, circular_data


z_dim = 7
X_dim = 2
h_dim = 64
lr = 1e-3
epochs = 20000
size = 9000


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))

Q_mu = tf.Variable(xavier_init([X_dim, z_dim]))
Q_sigma = tf.Variable(xavier_init([X_dim, z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu + tf.matmul(X, Q_mu)
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma + tf.matmul(X, Q_sigma)
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2.) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

P_X = tf.Variable(xavier_init([z_dim, X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2 + tf.matmul(z, P_X)
    return logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
logits = P(z_sample)

# Sampling from random z
X_samples = P(z)

prob = tf.exp(-tf.reduce_sum(tf.square(logits - X), 1))
# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.square(logits - X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer(lr).minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

# data = np.random.randn(10000, 2)
# data = circular_data(size / 100)
data = spiral(size / 3)

for ep in range(epochs):
    _, loss = sess.run([solver, vae_loss], feed_dict={X: data})

    if ep % 1000 == 0:
        if raw_input("\nWhat?\n") != "":
            break
        print('Epoch: {} \t Loss: {:.4}'.format(i, loss))
        samples = sess.run(X_samples, feed_dict={
                           z: np.random.randn(1000, z_dim)})
        plt.figure(figsize=(20, 20))
        plt.figure(figsize=(20, 20))
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.rc('font', size=20)
        plt.subplots_adjust(wspace=0.25, hspace=0.25)
        plt.subplot(222)
        plt.scatter(samples[:, 0], samples[:, 1], s=5)
        plt.title("VAE Generated Data")
        plt.subplot(221)
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.title("Input Data")
        xy = np.mgrid[-15.0:15.0:0.1,
                      -15.0:15.0:0.1].reshape(2, -1).T  # 40,000 points
        s = np.zeros(np.shape(xy)[0])
        for _ in xrange(10):
            p = np.array([])
            for j in xrange(40):
                p = np.concatenate((p, sess.run(prob, feed_dict={
                                   X: xy[j * 4000:(j + 1) * 4000, :], z: np.random.randn(4000, z_dim)})), axis=0)
            s += p
        s = s / 10
        s = s / np.amax(s)
        plt.subplot(223)
        plt.scatter(xy[:, 0], xy[:, 1], c=s, s=1, cmap='Purples')
        plt.colorbar()
        plt.title("Density Net")
        plt.savefig('out/{}.png'.format(str(i).zfill(3)))
        plt.close()
        i += 1
