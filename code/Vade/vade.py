from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.manifold import TSNE

ds = tf.contrib.distributions
mnist = input_data.read_data_sets("mnist_data/")
latent_size = 2
K=10
batch_size = 100
epochs = 2000
epoch_len = len(mnist.train.images)/batch_size

initializer = tf.contrib.layers.xavier_initializer()

bs = tf.placeholder(tf.int32)
global_step = tf.Variable(0,trainable=False)

def encoder(X):
	# Encoder network to map input to mean and log of variance of latent variable distribution
	h1 = tf.layers.dense(X,500,activation=tf.nn.relu,kernel_initializer = initializer)
	h2 = tf.layers.dense(h1,500,activation=tf.nn.relu,kernel_initializer = initializer)
	h3 = tf.layers.dense(h2,2000,activation=tf.nn.relu,kernel_initializer = initializer)
	z_mean = tf.layers.dense(h3,latent_size,kernel_initializer = initializer)
	z_log_sigma = tf.layers.dense(h3,latent_size,kernel_initializer = initializer)
	return z_mean, z_log_sigma

def decoder(z,reuse=False):
	# Decoder network to map latent variable to predicted output
	with tf.variable_scope("Decoder",reuse=reuse):
		h1 = tf.layers.dense(z,2000,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_1")
		h2 = tf.layers.dense(z,500,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_2")
		h3 = tf.layers.dense(z,500,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_3")
		output = tf.layers.dense(h3,784,kernel_initializer = initializer,name="decoder_4")
	return output, tf.nn.sigmoid(output)

def sample_z(mean,log_var,eps):
	return mean+tf.exp(log_var/2)*eps

def gamma(z,mu):
	# Get gamma
	def fn(previous_output,current_input):
		i = current_input
		q = pi[i]*ds.MultivariateNormalDiag(loc=mu[i],scale_diag=sigma[i]).prob(z)
		return tf.reshape(q,[bs])

	elems = tf.Variable(tf.range(K))
	gam = tf.scan(fn,elems,initializer = tf.ones([bs]))
	gam = tf.transpose(gam)
	gam = gam/tf.reshape(tf.reduce_sum(gam,1),(-1,1))
	return gam

def vae_loss(gam,out,mean,log_var,mu):
	f = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X,logits=out),axis=1)
	f -= tf.reduce_sum(gam*tf.log(pi),axis=1)
	f += tf.reduce_sum(gam*tf.log(gam),axis=1)
	f -= 1./2 * tf.reduce_sum(1+log_var,axis=1)
	def fn(previous_output,current_input):
		i = current_input
		l = previous_output + 1.0/2 * gam[:,i]*tf.reduce_sum(tf.log(sigma[i]) + tf.exp(log_var)/sigma[i] + tf.square(mean-mu[i])/sigma[i],axis=1)
		return l

	elems = tf.Variable(tf.range(K))
	y = tf.scan(fn,elems,initializer=tf.zeros(bs))
	f += y[-1,:]
	return tf.reduce_mean(f)

def update_mu(mean,gam):
	def fn(previous_output,current_input):
		i = current_input
		t = tf.matmul(tf.reshape(gam[:,i],[1,bs]),mean)
		t = tf.reshape(t,[latent_size])/tf.reduce_sum(gam[:,i],axis=0)
		return t

	elems = tf.Variable(tf.range(K))
	m = tf.scan(fn,elems,initializer = tf.ones([latent_size]))
	return m

def update_pi(gam):
	d = tf.reduce_sum(gam,axis=0)/tf.cast(bs,tf.float32)
	return d

def update_var(log_var,mean,gam):
	s = tf.matmul(tf.reshape(gam[:,0],[1,bs]),(tf.exp(log_var) + tf.square(mean-mu[0])))/tf.reduce_sum(gam[:,0],axis=0)
	for i in xrange(1,K):
		t = tf.matmul(tf.reshape(gam[:,i],[1,bs]),(tf.exp(log_var) + tf.square(mean-mu[i])))/tf.reduce_sum(gam[:,i],axis=0)
		s = tf.concat([s,t],0)
	return s

X = tf.placeholder(tf.float32,[None,784])
mu = tf.Variable(tf.random_normal([K,latent_size]),trainable=False)
sigma = tf.Variable(tf.ones([K,latent_size]),trainable=False)
pi = tf.Variable(tf.ones([K])/K,trainable=False)
epsilon = tf.placeholder(tf.float32,[None,latent_size])

mean, log_var = encoder(X)
z = sample_z(mean,log_var,epsilon)
out,_ = decoder(z)
gam = gamma(z,mu)
loss = vae_loss(gam,out,mean,log_var,mu)
grad = tf.gradients(loss,X)

_,gen = decoder(epsilon,True)

update1 = pi.assign(update_pi(gam))
update2 = mu.assign(update_mu(mean,gam))
update3 = sigma.assign(update_var(log_var,mean,gam))

learning_rate = tf.train.exponential_decay(.002,global_step,epoch_len*10,0.9,staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

for i in xrange(epochs):
	l = 0.0
	for j in xrange(epoch_len):
		batch_xs = mnist.train.next_batch(batch_size)[0]
		cost,_,_,_,_= sess.run([loss,train_step,update1,update2,update3],feed_dict={bs:batch_size,X:batch_xs,epsilon:np.random.randn(batch_size,latent_size)})
		l += cost
	print "Epoch: %s \t Loss: %s" %(i,l/epoch_len)
	output = sess.run(gen,feed_dict={bs:100,epsilon:np.random.randn(100,latent_size)})
	figure = np.zeros((28 * 10, 28 * 10))
	for k in xrange(10):
		for j in xrange(10):
			figure[ k * 28 : ( k + 1 ) * 28, j * 28 : ( j + 1 ) * 28 ] = np.reshape(output[k*10+j,:],[28,28])
	plt.figure(figsize=(6,6))
	plt.imshow(figure, cmap="Greys_r")
	plt.savefig("plot.png")
	plt.close()

	# lat = sess.run(z,feed_dict={bs:10000,X:mnist.train.images[:10000,:],epsilon:np.random.randn(10000,latent_size)})
	# plt.scatter(lat[:,0],lat[:,1],s=1,c=mnist.train.labels[:10000],cmap="tab10")
	# plt.colorbar()
	# plt.savefig("Latent.png")
	# plt.close()

	# embed = TSNE(n_components=2).fit_transform(lat)
	# plt.scatter(embed[:,0],embed[:,1],s=1,c=mnist.train.labels[:1000],cmap="tab10")
	# plt.colorbar()
	# plt.savefig("TSNE.png")
	# plt.close()