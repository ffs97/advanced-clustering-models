import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class VAE(object):
    """
    A VAE Class for clustering
    """

    def __init__(self, x_dim, z_dim, h_dim):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.X = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        self._init_encoder()
        self._init_decoder()
        self._init_network()

    def _init_encoder(self):
        self.Q_W1 = tf.Variable(xavier_init([self.x_dim, self.h_dim]))
        self.Q_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.Q_W2_mu = tf.Variable(xavier_init([self.h_dim, self.z_dim]))
        self.Q_b2_mu = tf.Variable(tf.zeros(shape=[self.z_dim]))

        self.Q_W2_sigma = tf.Variable(xavier_init([self.h_dim, self.z_dim]))
        self.Q_b2_sigma = tf.Variable(tf.zeros(shape=[self.z_dim]))

        self.Q_mu = tf.Variable(xavier_init([self.x_dim, self.z_dim]))
        self.Q_sigma = tf.Variable(xavier_init([self.x_dim, self.z_dim]))

    def Q(self, X):
        h = tf.nn.relu(tf.matmul(X, self.Q_W1) + self.Q_b1)
        z_mu = tf.matmul(h, self.Q_W2_mu) + self.Q_b2_mu + \
            tf.matmul(X, self.Q_mu)
        z_logvar = tf.matmul(h, self.Q_W2_sigma) + \
            self.Q_b2_sigma + tf.matmul(X, self.Q_sigma)
        return z_mu, z_logvar

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2.) * eps

    def _init_decoder(self):
        self.P_W1 = tf.Variable(xavier_init([self.z_dim, self.h_dim]))
        self.P_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.P_W2 = tf.Variable(xavier_init([self.h_dim, self.x_dim]))
        self.P_b2 = tf.Variable(tf.zeros(shape=[self.x_dim]))

        self.P_X = tf.Variable(xavier_init([self.z_dim, self.x_dim]))

    def P(self, Z):
        h = tf.nn.relu(tf.matmul(Z, self.P_W1) + self.P_b1)
        logits = tf.matmul(h, self.P_W2) + self.P_b2 + \
            tf.matmul(Z, self.P_X)
        return logits

    def _init_network(self):
        self.z_mu, self.z_logvar = self.Q(self.X)
        self.z_sample = self.sample_z(self.z_mu, self.z_logvar)
        self.logits = self.P(self.z_sample)

        self.X_samples = self.P(self.Z)

        self.prob = tf.exp(-tf.reduce_sum(tf.square(self.logits - self.X), 1))
        self.recon_loss = tf.reduce_sum(tf.square(self.logits - self.X), 1)
        self.kl_loss = 0.5 * \
            tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu **
                          2 - 1. - self.z_logvar, 1)
        # VAE loss
        self.vae_loss = tf.reduce_mean(self.recon_loss + self.kl_loss)

        self.solver = tf.train.AdamOptimizer(1e-3).minimize(self.vae_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, epochs=100):
        self.data = data

        for ep in range(epochs):
            _, loss = self.sess.run([self.solver, self.vae_loss],
                                    feed_dict={self.X: self.data})

            if ep % 1000 == 0:
                print "Epoch: {} \t Loss: {:.4}".format(ep, loss)
                self.plot("plots/{}.png".format(str(ep / 1000).zfill(3)))

    def plot(self, name):
        if not os.path.exists("plots/"):
            os.makedirs("plots/")

        samples_z = self.sess.run(self.z_sample, feed_dict={
            self.X: data
        })
        samples = self.sess.run(self.X_samples, feed_dict={
            self.Z: samples_z
        })
        print samples.shape
        plt.figure(figsize=(20, 20))
        plt.figure(figsize=(20, 20))
        plt.rc("xtick", labelsize=20)
        plt.rc("ytick", labelsize=20)
        plt.rc("font", size=20)
        plt.subplots_adjust(wspace=0.25, hspace=0.25)
        plt.subplot(222)
        plt.scatter(samples[:, 0], samples[:, 1], s=5)
        plt.title("Reconstructed Data")
        plt.subplot(221)
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.title("Input Data")
        xy = np.mgrid[-15.0:15.0:0.1,
                      -15.0:15.0:0.1].reshape(2, -1).T  # 40,000 points
        s = np.zeros(np.shape(xy)[0])
        for _ in xrange(10):
            p = np.array([])
            for j in xrange(40):
                p = np.concatenate((p, self.sess.run(self.prob, feed_dict={
                                   self.X: xy[j * 4000:(j + 1) * 4000, :], self.Z: np.random.randn(4000, self.z_dim)})), axis=0)
            s += p
        s = s / 10
        s = s / np.amax(s)
        plt.subplot(223)
        plt.scatter(xy[:, 0], xy[:, 1], c=s, s=1, cmap="Purples")
        plt.colorbar()
        plt.title("Density Net")
        plt.savefig(name)
        plt.close()

