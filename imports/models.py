import tensorflow as tf

from .layers import conv, instance_norm, deconv


def discriminator(x):
    _, w, w, _ = x.shape.as_list()
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        x = conv(x, "c1", filters=64, kernel_size=5, strides=2)
        x = tf.nn.leaky_relu(x)

        x = conv(x, "c2", filters=128, kernel_size=5, strides=2)
        x = tf.nn.leaky_relu(x)

        x = conv(x, "c3", filters=256, kernel_size=5, strides=2)
        x = tf.nn.leaky_relu(x)

        x = conv(x, "c4", filters=512, kernel_size=5, strides=2)
        x = tf.nn.leaky_relu(x)

        x = conv(x, "c5", filters=1, kernel_size=w//16, strides=1)
    return tf.nn.sigmoid(x), x


def generator(x):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        with tf.name_scope("encoder"):
            e1 = conv(x, "e1", filters=64, kernel_size=5, strides=2)
            e1 = tf.nn.leaky_relu(e1)

            e2 = conv(e1, "e2", filters=128, kernel_size=5, strides=2)
            e2 = instance_norm(e2, "e2_norm")
            e2 = tf.nn.leaky_relu(e2)

            e3 = conv(e2, "e3", filters=256, kernel_size=5, strides=2)
            e3 = instance_norm(e3, "e3_norm")
            e3 = tf.nn.leaky_relu(e3)

            e4 = conv(e3, "e4", filters=512, kernel_size=5, strides=2)
            e4 = instance_norm(e4, "e4_norm")
            e4 = tf.nn.leaky_relu(e4)

            e5 = conv(e4, "e5", filters=512, kernel_size=5, strides=2)
            e5 = instance_norm(e5, "e5_norm")
            e5 = tf.nn.leaky_relu(e5)

            e6 = conv(e5, "e6", filters=512, kernel_size=5, strides=2)
            e6 = instance_norm(e6, "e6_norm")
            e6 = tf.nn.leaky_relu(e6)

            e7 = conv(e6, "e7", filters=512, kernel_size=5, strides=2)
            e7 = instance_norm(e7, "e7_norm")
            e7 = tf.nn.leaky_relu(e7)

            e8 = conv(e7, "e8", filters=512, kernel_size=5, strides=2)
            e8 = instance_norm(e8, "e8_norm")
            e8 = tf.nn.relu(e8)

        with tf.name_scope("decoder"):
            d1 = deconv(e8, "d1", filters=512, kernel_size=2, strides=2)
            d1 = instance_norm(d1, "d1_norm")
            d1 = tf.nn.dropout(d1, keep_prob=0.5)
            d1 = tf.concat([d1, e7], axis=-1)
            d1 = tf.nn.relu(d1)

            d2 = deconv(d1, "d2", filters=512, kernel_size=2, strides=2)
            d2 = instance_norm(d2, "d2_norm")
            d2 = tf.nn.dropout(d2, keep_prob=0.5)
            d2 = tf.concat([d2, e6], axis=-1)
            d2 = tf.nn.relu(d2)

            d3 = deconv(d2, "d3", filters=512, kernel_size=2, strides=2)
            d3 = instance_norm(d3, "d3_norm")
            d3 = tf.nn.dropout(d3, keep_prob=0.5)
            d3 = tf.concat([d3, e5], axis=-1)
            d3 = tf.nn.relu(d3)

            d4 = deconv(d3, "d4", filters=512, kernel_size=2, strides=2)
            d4 = instance_norm(d4, "d4_norm")
            d4 = tf.concat([d4, e4], axis=-1)
            d4 = tf.nn.relu(d4)

            d5 = deconv(d4, "d5", filters=256, kernel_size=2, strides=2)
            d5 = instance_norm(d5, "d5_norm")
            d5 = tf.concat([d5, e3], axis=-1)
            d5 = tf.nn.relu(d5)

            d6 = deconv(d5, "d6", filters=128, kernel_size=2, strides=2)
            d6 = instance_norm(d6, "d6_norm")
            d6 = tf.concat([d6, e2], axis=-1)
            d6 = tf.nn.relu(d6)

            d7 = deconv(d6, "d7", filters=64, kernel_size=2, strides=2)
            d7 = instance_norm(d7, "d7_norm")
            d7 = tf.concat([d7, e1], axis=-1)
            d7 = tf.nn.relu(d7)

            d8 = deconv(d7, "d8", filters=1, kernel_size=2, strides=2)
    return tf.nn.sigmoid(d8)
