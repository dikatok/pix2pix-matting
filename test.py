import tensorflow as tf
from tensorflow.python.lib.io import file_io

from imports.models import generator
import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='out.jpg', required=False)
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str)
    parser.add_argument('image')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    x = tf.keras.preprocessing.image.load_img(args.image)
    x = tf.keras.preprocessing.image.img_to_array(x)
    x = tf.expand_dims(x, axis=0)
    x = tf.image.resize_images(x, [256, 256])

    inputs = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3], name="inputs")
    outputs = generator(inputs) * inputs

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    latest_ckpt = tf.train.latest_checkpoint(args.ckpt_dir)

    with tf.Session() as sess:
        saver.restore(sess, latest_ckpt)
        x = sess.run(x)
        out = sess.run(outputs, feed_dict={inputs: x})

    out = tf.keras.preprocessing.image.array_to_img(out[0])
    out.save(args.output)

