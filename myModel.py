import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)

def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def _log_likelihood(mean_ts, loc_ts, std):
    """
    Args
    - mean_ts: a list with len=num_glimpses contains tensors with shape (B, 2)
    - loc_ts: a list with len=num_glimpses contains tensors with shape (B, 2), sampled location of all timesteps
    - std: scalar
    Returns
    - logll: tensor with shape (B, timesteps)
    """
    means_ts = tf.stack(mean_ts)  # [timesteps, batch_sz, loc_dim]
    loc_ts = tf.stack(loc_ts)
    gaussian = Normal(mean_ts, std)
    logll = gaussian._log_prob(x=loc_ts)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)  # reduce location(dim=2) to 1
    return tf.transpose(logll)      # [batch_sz, timesteps]

def translatedMnist(images, initImgSize=28, finalImgSize=48):
    size_diff = finalImgSize - initImgSize
    batch_size = images.shape[0]

    newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
    imgOffset = np.zeros([batch_size, 2])

    for k in xrange(batch_size):
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))

        # generate and save random coordinates
        randX_L = random.randint(0, size_diff)
        randY_L = random.randint(0, size_diff)

        randY_R = random.randint(0, size_diff)
        randX_R = random.randint(0, size_diff)

        imgOffset[k, :] = np.array([randX_R-randX_L, randY_R-randY_L]).astype(np.float32)

        # padding
        image_L = np.lib.pad(image, ((randX_L, size_diff - randX_L), (randY_L, size_diff - randY_L)), 'constant', constant_values = (0.0))
        image_R = np.lib.pad(image, ((randY_R, size_diff - randX_R), (randY_R, size_diff - randY_R)), 'constant', constant_values = (0.0))

        # newimages[k, :, 0] = np.reshape(image_L, (finalImgSize*finalImgSize))
        # newimages[k, :, 1] = np.reshape(image_R, (finalImgSize*finalImgSize))

        # residual image
        newimages[k, :] = np.reshape(image_R-image_L, (finalImgSize*finalImgSize))

    return newimages, imgOffset


class RetinaSensor(object):
    """
    A retina that extracts a `patch` around location `loc_t` from image `img_ph`.
    Args
    ----
    - img_ph: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    - loc_t: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
    - pth_size: a scalar. Size of the square glimpse patch.
    Returns
    -------
    - patch: a 4D tensor of shape (B, pth_size, pth_size, 1). The foveated glimpse of the image.
    """
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc_t):
        img = tf.reshape(img_ph, [tf.shape(img_ph)[0], self.img_size, self.img_size, 1])
        pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc_t)
        pth = tf.reshape(pth, [tf.shape(loc_t)[0], self.pth_size*self.pth_size])
        return pth


class LocationNetwork(object):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    paself.trize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args
    ----
    - hidden_size: rnn hidden size
    - loc_dim: location dim = 2
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for the current time step `t`.
    Returns
    -------
    - mean: a 2D vector of shape (B, 2). Gaussian mean of current time step.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    """
    def __init__(self, hidden_size, loc_dim, std=0.22, is_sampling=False):
        self.loc_dim = loc_dim
        self.std = std
        self.w = _weight_variable((hidden_size, loc_dim))
        self.b = _bias_variable((loc_dim,))
        self.is_sampling = is_sampling


    def __call__(self, h_t):
        # compute mean at this time step
        mean_t = tf.nn.xw_plus_b(h_t, self.w, self.b)
        mean_t = tf.clip_by_value(mean_t, -1., 1.)
        mean_t = tf.stop_gradient(mean_t)

        def is_sampling_true():
            # sample from gaussian parameterized by this mean when training
            loc_t = mean_t + tf.random_normal((tf.shape(h_t)[0], self.loc_dim), stddev=self.std)
            loc_t = tf.clip_by_value(loc_t, -1., 1.)
            return loc_t

        def is_sampling_false():
            # using mean when testing
            return  mean_t

        loc_t = tf.cond(self.is_sampling, is_sampling_true, is_sampling_false)

        loc_t = tf.stop_gradient(loc_t)

        return loc_t, mean_t


class GlimpseNetwork(object):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina `pth_t` to
    a fc layer and the glimpse location vector `loc_t`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        `glimpse_t = relu( fc( fc(pth_t) ) + fc( fc(loc_t) ) )`
    Args
    ----
    - pth_size: pth size
    - loc_dim: location dim = 2
    - g_size: hidden layer size of the fc layer for `pths`.
    - l_size: hidden layer size of the fc layer for `locs`.
    - output_size: output size of this network.
    - pth_t: a 4D Tensor of shape (B, pth_size, pth_size, 1). Current time step minibatch of pths.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    Returns
    -------
    - glimpse_t: a 2D tensor of shape (B, output_size). The glimpse representation returned by the glimpse network for the current timestep `t`.
    """
    def __init__(self, pth_size, loc_dim, g_size, l_size, output_size):
        # layer 1
        self.g1_w = _weight_variable((pth_size*pth_size, g_size))
        self.g1_b = _bias_variable((g_size,))

        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))

        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))

        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, pth_t, loc_t):
        # feed pths and locs to respective fc layers
        what  = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pth_t, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        where = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(loc_t, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        # feed to fc layer
        glimpse_t = tf.nn.relu(what + where)
        return glimpse_t


class BaseLineNetwork(object):
    """
    Regresses the baseline in the reward function to reduce the variance of the gradient update.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_ts: the hidden state vectors of the core network for the all time step `ts`.
    Returns
    -------
    - baselines: a 2D vector of shape (B, timesteps). The baseline for the all time step `ts`.
    """

    def __init__(self, hidden_size):
        self.w = _weight_variable((hidden_size, 1))
        self.b = _bias_variable((1, ))

    def __call__(self, h_ts):
        # Time independent baselines
        baselines = []
        for h_t in h_ts[1:]:
            baseline = tf.nn.xw_plus_b(h_t, self.w, self.b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)

        baselines = tf.stack(baselines)        # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)   # [batch_sz, timesteps]

        return baselines


class ClassificationNetwork(object):
    """
    Uses the internal state `h_last` of the core network to
    produce the final output classification.
    Concretely, feeds the hidden state `h_last` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.
    the network is simply a linear softmax classifier.
    Args
    ----
    - hidden_size: size of the rnn.
    - num_classes: number of classes in the dataset.
    - h_last: the hidden state vector of the core network for the last time step
    Returns
    -------
    - softmax: output probability vector over the classes.
    """
    def __init__(self, hidden_size, num_classes):
        self.w = _weight_variable((hidden_size, num_classes))
        self.b = _bias_variable((num_classes,))

    def __call__(self, h_last):
        # Take the last step only.
        logits  = tf.nn.xw_plus_b(h_last, self.w, self.b)
        pred    = tf.argmax(logits, 1)
        softmax = tf.nn.softmax(logits)

        return logits, pred, softmax


class RegressNetwork(object):
    """
    Args
    ----
    - hidden_size: size of the rnn.
    - h_last: the hidden state vector of the core network for the last time step
    Returns
    -------
    """
    def __init__(self, hidden_size, loc_dim):
        self.w = _weight_variable((hidden_size, loc_dim))
        self.b = _bias_variable((loc_dim,))

    def __call__(self, h_last):
        # Take the last step only.
        x = tf.nn.xw_plus_b(h_last, self.w, self.b)
        return x


class CoreNetwork(object):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.
    Concretely, it takes the images `img_ph` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.
    In other words:
        `h_t = relu( fc(h_t_prev) + fc(g_t) )`
    Args
    ----
    - batch_size: input size of the rnn.
    - loc_dim: location dim = 2
    - hidden_size: hidden size of the rnn.
    - num_glimpses: time steps of the rnn.
    - img_ph: a 4D tensor of shape (B, H, W, 1).
    Returns
    -------
    - h_ts: a 2D tensor of shape (B, hidden_size). The hidden state vector for the current timestep `t`.
    - loc_ts: a list of 2D tensor of shape (B, 2). The glimpse center sampled from guassian of all time steps.
    - mean_ts: a list of 2D tensor of shape (B, 2). The guassian mean of all time steps.
    """
    def __init__(self, batch_size, loc_dim, hidden_size, num_glimpses):
        self.batch_size = batch_size
        self.loc_dim = loc_dim
        self.hidden_size = hidden_size
        self.num_glimpses = num_glimpses

    def __call__(self, img_ph, location_network, retina_sensor, glimpse_network):
        # lstm cell
        cell = BasicLSTMCell(self.hidden_size)

        # helper func for feeding glimpses to every step of lstm
        # h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden state vector for the previous timestep `t-1`.
        loc_ts, mean_ts = [], []

        ## at time step t, location-->pths-->glimpse
        def loop_function(h_prev, _):
            # predict location from previous hidden state
            loc_t, mean_t = location_network(h_prev)
            loc_ts.append(loc_t)
            mean_ts.append(mean_t)

            # crop pths from image based on the predicted location
            pths_t = retina_sensor(img_ph, loc_t)

            # generate glimpse image from current pths_t and loc_t
            glimpse = glimpse_network(pths_t, loc_t)
            return glimpse

        # lstm init h_t
        init_state = cell.zero_state(self.batch_size, tf.float32)

        # lstm inputs at every step
        init_loc = tf.random_uniform((self.batch_size, self.loc_dim), minval=-1, maxval=1)
        init_pths = retina_sensor(img_ph, init_loc)
        init_glimpse = glimpse_network(init_pths, init_loc)
        rnn_inputs = [init_glimpse]
        rnn_inputs.extend([0] * self.num_glimpses)

        # get hidden state of every step from lstm
        h_ts, _ = rnn_decoder(rnn_inputs, init_state, cell, loop_function=loop_function)

        return loc_ts, mean_ts, h_ts


class RecurrentAttentionModel(object):
    """
    A Recurrent Model of Visual Attention (self. [1].
    self.is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    Args
    ----
    - pth_size: size of the square patches in the glimpses extracted by the retina.
    - g_size: hidden layer size of the fc layer for `phi`.
    - l_size: hidden layer size of the fc layer for `locs`.
    - glimpse_output_size: output size of glimpse network.
    - loc_dim: 2
    - std: standard deviation of the Gaussian policy.
    - hidden_size: hidden size of the rnn.
    - num_classes: number of classes in the dataset.
    - num_glimpses: number of glimpses to take per image, i.e. number of BPTT steps.

    - x: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
        state vector for the current timestep `t`.
    - mu: a 2D tensor of shape (B, 2). The mean that paself.trizes
        the Gaussian policy.
    - l_t: a 2D tensor of shape (B, 2). The location vector
        containing the glimpse coordinates [x, y] for the
        current timestep `t`.
    - b_t: a 2D vector of shape (B, 1). The baseline for the
        current time step `t`.
    - log_probas: a 2D tensor of shape (B, num_classes). The
        output log probability vector over the classes.
    """
    def __init__(self, img_size, pth_size, g_size, l_size, glimpse_output_size,
                 loc_dim, std, hidden_size, num_glimpses, num_classes,
                 learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch,
                 max_gradient_norm, is_training=False):
        self.training_steps_per_epoch = training_steps_per_epoch

        with tf.variable_scope('placeholder'):
            self.img_ph = tf.placeholder(tf.float32, [None, img_size*img_size])
            self.lbl_ph = tf.placeholder(tf.float32, [None, 2])  # offset
            self.is_training = tf.placeholder(tf.bool, [])

        ## init network param
        with tf.variable_scope('LocationNetwork'):
            location_network = LocationNetwork(hidden_size, loc_dim, std=std, is_sampling=self.is_training)

        with tf.variable_scope('RetinaSensor'):
            retina_sensor = RetinaSensor(img_size, pth_size)

        with tf.variable_scope('GlimpseNetwork'):
            glimpse_network = GlimpseNetwork(pth_size, loc_dim, g_size, l_size, glimpse_output_size)

        with tf.variable_scope('CoreNetwork'):
            core_network = CoreNetwork(batch_size=tf.shape(self.img_ph)[0], loc_dim=loc_dim, hidden_size=hidden_size, num_glimpses=num_glimpses)

        with tf.variable_scope('Baseline'):
            baseline_network = BaseLineNetwork(hidden_size)

        with tf.variable_scope('RegressNetwork'):
            regress_network = RegressNetwork(hidden_size, loc_dim)

        ## call all networks to build graph
        # Run the recurrent attention model for all timestep on the minibatch of images
        loc_ts, mean_ts, h_ts = core_network(self.img_ph, location_network, retina_sensor, glimpse_network)

        # baselines, approximate value function based h_ts
        baselines = baseline_network(h_ts)

        # make classify action at last time step
        self.pred_offset = regress_network(h_ts[-1])

        # training preparation
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(learning_rate, self.global_step, training_steps_per_epoch, learning_rate_decay_factor, staircase=True), min_learning_rate)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

        ## losses
        # regress loss for regress_network, core_network, glimpse_network
        self.regress_mse = tf.reduce_mean(tf.square((self.lbl_ph - self.pred_offset)))

        # RL reward for location_network
        # reward = tf.cast(tf.equal(pred, self.lbl_ph), tf.float32)
        # rewards = tf.expand_dims(reward, 1)             # [batch_sz, 1]
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        rewards = tf.reduce_mean(tf.square((self.lbl_ph - self.pred_offset)), axis=1)  # [batch_sz, 1], reduce loc_dim
        rewards = tf.tile(rewards, (1, num_glimpses))   # [batch_sz, timesteps]
        advantages = rewards - tf.stop_gradient(baselines) # (B, timesteps), baseline approximate func is trained by baseline loss only.
        self.advantage = tf.reduce_mean(advantages)
        logll = _log_likelihood(mean_ts, loc_ts, std)  # (B, timesteps)
        logllratio = tf.reduce_mean(logll * advantages) # reduce B and timesteps
        self.reward = tf.reduce_mean(reward)  # reduce batch

        # baseline loss for baseline_network, core_network, glimpse_network
        self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))

        # hybrid loss
        self.loss = -logllratio + self.regress_mse + self.baselines_mse
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def train(self, num_steps, num_MC, batch_size, mnist):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in xrange(num_steps):
                images, labels = mnist.train.next_batch(batch_size)
                images, labels = translatedMnist(images)
                images = np.tile(images, [num_MC, 1])
                labels = np.tile(labels, [num_MC])

                output_feed = [self.train_op, self.loss, self.regress_mse, self.reward, self.advantage, self.baselines_mse, self.learning_rate]
                _, loss, regress_mse, reward, advantage, baselines_mse, learning_rate = sess.run(output_feed, feed_dict={self.img_ph: images, self.lbl_ph: labels, self.is_training:True})

                # log
                if step and step % 100 == 0:
                    logging.info('step {}: lr = {:3.6f}\tloss = {:3.4f}\tregress_mse = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format( step, learning_rate, loss, xent, reward, advantage, baselines_mse))

                # Evaluation
                if step and step % self.training_steps_per_epoch == 0:
                    for dataset in [mnist.validation, mnist.test]:
                        steps_per_epoch = dataset.num_examples // batch_size
                        correct_cnt = 0
                        num_samples = steps_per_epoch * batch_size
                        for test_step in xrange(steps_per_epoch):
                            images, labels = dataset.next_batch(batch_size)
                            images, labels = translatedMnist(images)
                            labels_bak = labels
                            # Duplicate M times
                            images = np.tile(images, [num_MC, 1])
                            labels = np.tile(labels, [num_MC])
                            regress_mse = sess.run(self.regress_mse, feed_dict={self.img_ph: images, self.lbl_ph: labels, self.is_training:True})

                        if dataset == mnist.validation:
                            logging.info('valid mse = {}'.format(regress_mse))
                        else:
                            logging.info('test mse = {}'.format(regress_mse))
