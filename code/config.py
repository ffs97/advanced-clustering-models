n_clusters = 10

input_dim = 784
latent_dim = 10

n_epochs = 3000
batch_size = 100

regularizer = 1

encoder_hidden_size = [500, 500, 2000]
decoder_hidden_size = [2000, 500, 500]

adam_decay_steps = 10
adam_decay_rate = 0.9
adam_learning_rate = 0.002
adam_epsilon = 1e-04
