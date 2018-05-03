import os


class Config:
    def __init__(self, data="mnist"):
        if data == "mnist":
            self.n_clusters = 10

            self.input_dim = 784
            self.latent_dim = 10

            self.n_epochs = 3000
            self.batch_size = 200

            self.pretrain_vae_n_epochs = 500

            self.pretrain_gmm_n_iters = 5000
            self.pretrain_gmm_n_inits = 10

            self.regularizer = 1

            self.encoder_hidden_size = [500, 500, 2000]
            self.decoder_hidden_size = [2000, 500, 500]

            self.decay_steps = 10
            self.decay_rate = 0.9
            self.learning_rate = 0.002
            self.epsilon = 1e-04

        elif data == "spiral":
            self.n_clusters = 5

            self.input_dim = 2
            self.latent_dim = 10

            self.n_epochs = 1000
            self.batch_size = 200

            self.pretrain_vae_n_epochs = 500

            self.pretrain_gmm_n_iters = 1500
            self.pretrain_gmm_n_inits = 10

            self.regularizer = 1

            self.encoder_hidden_size = [500, 500, 2000]
            self.decoder_hidden_size = [2000, 500, 500]

            self.decay_steps = 10
            self.decay_rate = 0.9
            self.learning_rate = 0.0001
            self.epsilon = 1e-04

            if not os.path.exists("models"):
                os.makedirs("models")

            if not os.path.exists("models/spiral"):
                os.makedirs("models/spiral")
                os.makedirs("models/spiral/vae")
                os.makedirs("models/spiral/gmm")
                os.makedirs("models/spiral/vade")

            self.train_dir = "models/spiral"
