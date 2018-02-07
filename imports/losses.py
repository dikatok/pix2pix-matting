import tensorflow as tf


def discriminator_loss(d_logits, d_logits_, labels, labels_):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=labels))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=labels_))
    return d_loss_real + d_loss_fake


def generator_loss(d_logits_, labels_, real_B, fake_B, l1_lambda=100):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=labels_)
                          + l1_lambda * tf.reduce_mean(tf.abs(real_B - fake_B)))
