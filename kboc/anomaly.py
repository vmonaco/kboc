import numpy as np
import pandas as pd
from sklearn import svm
import tensorflow as tf


class Manhattan(object):
    """
    Manhattan distance to the mean template vector.
    """

    def fit(self, X):
        self.X = X
        self.mean = X.mean(axis=0)

    def score(self, X):
        if X.ndim == 1:
            X = X[np.newaxis, :]
        return - np.abs(X - self.mean).sum(axis=1)


class OneClassSVM(object):
    """
    One-class support vector machine
    """

    def fit(self, X):
        clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.9)
        clf.fit(X)
        self.clf = clf

    def score(self, X):
        score = self.clf.decision_function(X[np.newaxis, :]).squeeze()
        return score


class MeanEnsemble_csv(object):
    """
    A mean ensemble of csv files.
    """

    def __init__(self, fnames):
        df = pd.read_csv(fnames[0], index_col=[0, 1, 2])
        for fname in fnames[1:]:
            df += pd.read_csv(fname, index_col=[0, 1, 2])
        df /= len(fnames)
        self.scores = df.reset_index(level=0, drop=True)

    def fit(self, X, y):
        pass

    def score(self, fold, genuine):
        return self.scores.loc[fold, genuine].values.squeeze()


class MeanEnsemble_txt(object):
    """
    A mean ensemble of txt files.
    """

    def __init__(self, fnames):
        df = pd.read_csv(fnames[0], index_col=[0], sep=' ', header=None)
        for fname in fnames[1:]:
            df += pd.read_csv(fname, index_col=[0], sep=' ', header=None)
        df /= len(fnames)
        self.scores = df

    def fit(self, X, y):
        pass

    def score(self, X, users, sessions):
        idx = ['%s_%02d' % (yi, i) for yi, i in zip(users, sessions)]
        return self.scores.loc[idx].values.squeeze()


class ContractiveAutoencoder(object):
    """
    Contractive autoencoder with one hidden layer, as described by Rifai et al.
    """

    def __init__(self, n_hidden, lam):
        self.n_hidden = n_hidden
        self.lam = lam

    def fit(self, X, n_epochs=1000):
        n_examples, input_dim = X.shape
        x_in = tf.placeholder('float', [None, input_dim])

        W = tf.Variable(
            tf.random_uniform([input_dim, self.n_hidden], -1.0 / np.sqrt(input_dim), 1.0 / np.sqrt(input_dim),
                              seed=np.random.randint(0, 1e9)))
        b_x = tf.Variable(tf.zeros([self.n_hidden]))
        b_y = tf.Variable(tf.zeros([input_dim]))

        hidden = tf.nn.sigmoid(tf.matmul(x_in, W) + b_x)
        x_out = tf.matmul(hidden, tf.transpose(W)) + b_y
        # x_out_sigmoid = tf.nn.sigmoid(x_out)

        # reconstruction_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_out, x_in), 1)
        reconstruction_cost = tf.sqrt(tf.reduce_sum(tf.square(x_in - x_out), 1))

        # Jacobian cost for each training example
        jacobian_cost = tf.reduce_sum(tf.reshape(((hidden * (1 - hidden)) ** 2), (-1, 1, self.n_hidden)) * (W ** 2),
                                      (1, 2))

        # Total cost, mean over the training examples
        cost = tf.reduce_mean(reconstruction_cost + self.lam * jacobian_cost)

        # optimizer = tf.train.AdamOptimizer().minimize(cost)
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for i in range(n_epochs):
            sess.run(optimizer, feed_dict={x_in: X})

        self.x_in = x_in
        self.sess = sess
        self.cost = cost

    def score(self, X):
        error = self.sess.run(self.cost, feed_dict={self.x_in: X[np.newaxis, :]})
        return -error


class Autoencoder(object):
    """
    Basic autoencoder that uses the negative reconstruction error as a similarity score
    """

    def __init__(self, shape, n_steps=5000):
        self.shape = shape
        self.n_steps = n_steps

    @staticmethod
    def create(x, layer_sizes):
        # Build the encoding layers
        next_layer_input = x

        encoding_matrices = []
        for dim in layer_sizes:
            input_dim = int(next_layer_input.get_shape()[1])

            # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
            W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / np.sqrt(input_dim), 1.0 / np.sqrt(input_dim),
                                              seed=np.random.randint(0, 1e9)))

            # Initialize b to zero
            b = tf.Variable(tf.zeros([dim]))

            # We are going to use tied-weights so store the W matrix for later reference.
            encoding_matrices.append(W)

            output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)

            # the input into the next layer is the output of this layer
            next_layer_input = output

        # The fully encoded x value is now stored in the next_layer_input
        encoded_x = next_layer_input

        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        encoding_matrices.reverse()

        for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]):
            # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
            W = tf.transpose(encoding_matrices[i])
            b = tf.Variable(tf.zeros([dim]))
            output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)
            next_layer_input = output

        # the fully encoded and reconstructed value of x is here:
        reconstructed_x = next_layer_input

        return {
            'encoded': encoded_x,
            'decoded': reconstructed_x,
            'cost': tf.sqrt(tf.reduce_mean(tf.square(x - reconstructed_x)))
        }

    def fit(self, X):
        n_input = X[0].shape[0]

        sess = tf.Session()
        x = tf.placeholder('float', [None, n_input])
        autoencoder = Autoencoder.create(x, self.shape)
        init = tf.initialize_all_variables()
        sess.run(init)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])

        n_samples = len(X)
        for i in range(self.n_steps):
            batch = []
            for j in range(n_samples):
                batch.append(X[j])

            sess.run(train_step, feed_dict={x: np.array(batch)})

        self.x = x
        self.autoencoder = autoencoder
        self.sess = sess

    def score(self, X):
        error = self.sess.run(self.autoencoder['cost'], feed_dict={self.x: X[np.newaxis, :]})
        return -error


def xavier_init(fan_in, fan_out, constant=1):
    """
    Xavier initialization of network weights\
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32, seed=np.random.randint(0, 1e9))


class VariationalAutoencoder(object):
    """
    Variational autoencoder using a Gaussian latent space.

    Based on the implementation at: https://jmetzen.github.io/2015-11-27/vae.html

    See 'Auto-Encoding Variational Bayes' by Kingma and Welling.
    """

    def __init__(self, network_architecture, batch_size=2, transfer_fct=tf.nn.softplus, learning_rate=0.001):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _create_network(self):
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32, seed=np.random.randint(0, 1e9))
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """
        Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """
        Transform data by mapping it into the latent space.
        """
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """
        Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """
        Use VAE to reconstruct given data.
        """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})

    def fit(self, X, training_epochs=700):
        self.network_architecture['n_input'] = X[0].shape[0]

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

        n_samples = len(X)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = X[i * self.batch_size:(i * self.batch_size + self.batch_size)]
                # Fit training using batch data
                cost = self.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * self.batch_size

    def score(self, X):
        Xprime = self.reconstruct(X[np.newaxis, :])
        error = np.sqrt(np.mean((X - Xprime) ** 2))
        return -error
