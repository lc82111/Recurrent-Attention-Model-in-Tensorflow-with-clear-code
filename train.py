from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import RecurrentAttentionModel
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np



tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 9000009 "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 8, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("hidden_size", 256, "Hidden size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 6, "Number of glimpses.")
tf.app.flags.DEFINE_float("std", 0.22, "Gaussian std for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")

FLAGS = tf.app.flags.FLAGS

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=False)
training_steps_per_epoch = mnist.train.num_examples // FLAGS.batch_size


ram = RecurrentAttentionModel(img_size=28, # MNIST: 28 * 28
                              pth_size=FLAGS.patch_window_size,
                              g_size=FLAGS.g_size,
                              l_size=FLAGS.l_size,
                              glimpse_output_size=FLAGS.glimpse_output_size,
                              loc_dim=2,   # (x,y)
                              std=FLAGS.std,
                              hidden_size=FLAGS.hidden_size,
                              num_glimpses=FLAGS.num_glimpses,
                              num_classes=10,
                              learning_rate=FLAGS.learning_rate,
                              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                              min_learning_rate=FLAGS.min_learning_rate,
                              training_steps_per_epoch=training_steps_per_epoch,
                              max_gradient_norm=FLAGS.max_gradient_norm,
                              is_training=True)

ram.train(FLAGS.num_steps, FLAGS.M, FLAGS.batch_size, mnist)
