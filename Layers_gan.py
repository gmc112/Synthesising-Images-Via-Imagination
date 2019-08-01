import tensorflow as tf
import tensorflow.contrib.layers as tfcl


def conv_t(input, fil, name, is_train, is_final):
    with tf.name_scope(name):
        dc = tf.layers.conv2d_transpose(inputs=input, filters=fil, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        if not is_final:
            bn = tfcl.batch_norm(dc, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None)
            act = tf.nn.relu(bn)
        else:
            act = tf.nn.tanh(dc)
        tf.summary.histogram("conv_t_act", act + 1e-8)
        return act


def conv(act, fil, name, is_train):
    with tf.name_scope(name):
        c = tf.layers.conv2d(inputs=act, filters=fil, kernel_size=[5, 5], strides=[2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
        bn = tfcl.batch_norm(c, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None)
        act = tf.nn.leaky_relu(bn)
        tf.summary.histogram("conv_act", act + 1e-8)
        return act


def dropout(x, keep_prob, is_train):
    return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob), lambda: x)


