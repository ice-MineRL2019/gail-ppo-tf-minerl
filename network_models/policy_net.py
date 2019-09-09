import tensorflow as tf
import numpy
import keras

class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.conv2d(inputs=self.obs, filters=16, kernel_size=[3, 3], strides=1, padding='SAME',
                                           activation=tf.nn.relu)
                layer_2 = tf.layers.max_pooling2d(layer_1, pool_size=[2, 2], strides=2)
                layer_3 = tf.layers.conv2d(layer_2, filters=32, kernel_size=[3, 3], strides=1, padding='SAME',
                                           activation=tf.nn.relu)
                layer_4 = tf.layers.max_pooling2d(layer_3, pool_size=[2, 2], strides=2)
                layer_5 = tf.reshape(layer_4, [-1, 8192])
                layer_6 = tf.layers.dense(inputs=layer_5, units=20, activation=tf.tanh)
                layer_7 = tf.layers.dense(inputs=layer_6, units=9, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_7, units=9, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.conv2d(inputs=self.obs, filters=16, kernel_size=[3, 3], strides=1, padding='SAME',
                                           activation=tf.nn.relu)
                layer_2 = tf.layers.max_pooling2d(layer_1, pool_size=[2, 2], strides=2)
                layer_3 = tf.layers.conv2d(layer_2, filters=32, kernel_size=[3, 3], strides=1, padding='SAME',
                                           activation=tf.nn.relu)
                layer_4 = tf.layers.max_pooling2d(layer_3, pool_size=[2, 2], strides=2)
                layer_5 = tf.reshape(layer_4, [-1, 8192])
                layer_6 = tf.layers.dense(inputs=layer_5, units=20, activation=tf.tanh)
                layer_7 = tf.layers.dense(inputs=layer_6, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_7, units=1, activation=None)
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

