from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

ds = tf.contrib.distributions
mnist = input_data.read_data_sets("mnist_data/")
latent_size = 2
batch_size = 100
epochs = 100
K=10

initializer = tf.contrib.layers.xavier_initializer()

X = tf.placeholder(tf.float32,[None,784])
mu = tf.Variable(tf.zeros([K,latent_size]),trainable=False)
sigma = tf.Variable(tf.ones([K,latent_size]),trainable=False)
pi = tf.Variable(tf.ones([K])/K,trainable=False)

def encoder(X):
	# Encoder network to map input to mean and log of variance of latent variable distribution
	h1 = tf.layers.dense(X,500,activation=tf.nn.relu,kernel_initializer = initializer)
	z_mean = tf.layers.dense(h1,latent_size,kernel_initializer = initializer)
	z_log_sigma = tf.layers.dense(h1,latent_size,kernel_initializer = initializer)
	return z_mean, z_log_sigma

def decoder(z,reuse=False):
	# Decoder network to map latent variable to predicted output
	with tf.variable_scope("Decoder",reuse=reuse):
		h1 = tf.layers.dense(z,500,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_1")
		output = tf.layers.dense(h1,784,kernel_initializer = initializer,name="decoder_5")
	return output, tf.nn.sigmoid(output)

def sample_z(mean,log_var):
	return mean+tf.exp(log_var/2)*tf.random_normal(shape=(batch_size,latent_size))

def gamma(z):
	c = tf.reshape(pi[0]*ds.MultivariateNormalDiag(loc=mu[0,:],scale_diag=sigma[0,:]).prob(z),[batch_size,1])
	for i in xrange(1,K):
		c = tf.concat([c,tf.reshape(ds.MultivariateNormalDiag(loc=mu[i,:],scale_diag=sigma[i,:]).prob(z),[batch_size,1])],1)
	c= c/tf.reshape(tf.reduce_sum(c,axis=1),(-1,1))
	return c

def vae_loss(gam,out,mean,log_var):
	loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X,logits=out),axis=1)
	loss += -tf.reduce_sum(gam*tf.log(gam/pi),axis=1)
	loss += 1./2 * tf.reduce_sum(1+log_var,axis=1)
	for i in xrange(K):
		loss += -1./2* gam[:,i]*tf.reduce_sum(tf.log(sigma[i])+tf.exp(log_var)/sigma[i] + tf.square(mean-mu[i])/sigma[i],axis=1)
	return tf.reduce_mean(loss)

def update_mu(mean,gam):
	m = tf.matmul(tf.reshape(gam[:,0],[1,batch_size]),mean)
	m = m/tf.reduce_sum(gam[:,0],axis=0)
	for i in xrange(1,K):
		t = tf.matmul(tf.reshape(gam[:,i],[1,batch_size]),mean)
		t = t/tf.reduce_sum(gam[:,i],axis=0)
		m = tf.concat([m,t],0)
	return m

def update_pi(gam):
	pi = tf.reduce_sum(gam,axis=0)/batch_size
	return pi

def update_var(log_var,mean,gam):
	s = tf.matmul(tf.reshape(gam[:,0],[1,batch_size]),(tf.exp(log_var) + tf.square(mean-mu[0])))/tf.reduce_sum(gam[:,0],axis=0)
	for i in xrange(1,K):
		t = tf.matmul(tf.reshape(gam[:,i],[1,batch_size]),(tf.exp(log_var) + tf.square(mean-mu[i])))/tf.reduce_sum(gam[:,i],axis=0)
		s = tf.concat([s,t],0)
	return s

mean, log_var = encoder(X)
z = sample_z(mean,log_var)
out,_ = decoder(z)
gam = gamma(z)
loss = vae_loss(gam,out,mean,log_var)

update1 = pi.assign(update_pi(gam))
update2 = mu.assign(update_mu(mean,gam))
update3 = sigma.assign(update_var(log_var,mean,gam))

train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

epoch_len = len(mnist.train.images)/batch_size

for i in xrange(epochs):
	l = 0.0
	for j in xrange(epoch_len):
		batch_xs = mnist.train.next_batch(batch_size)[0]
		g,cost,_,_,_,_ = sess.run([gam,loss,train_step,update1,update2,update3],feed_dict={X:batch_xs})
		l += cost
	print "Epoch: %s \t Loss: %s" %(i,l/epoch_len)