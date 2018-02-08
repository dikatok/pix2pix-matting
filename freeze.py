import tensorflow as tf

from imports.models import generator
import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str)
    parser.add_argument('--ckpt_iter', default=-1, type=int)
    parser.add_argument('--output_graph', default="model.pb", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    ckpt_path = tf.train.latest_checkpoint(args.ckpt_dir)
    if args.ckpt_iter >= 0:
        ckpt_path = os.path.join(args.ckpt_dir, "ckpt-{iter}".format(iter=args.ckpt_iter))

    with tf.Session() as sess:
        inputs = tf.placeholder(name="inputs", dtype=tf.float32, shape=[1, 256, 256, 3])
        outputs = tf.identity(generator(inputs), name="outputs")

        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator/")
        saver = tf.train.Saver(var_list=generator_vars)
        saver.restore(sess, ckpt_path)

        convert_variables_to_constants = tf.graph_util.convert_variables_to_constants
        output_graph_def = convert_variables_to_constants(sess,  tf.get_default_graph().as_graph_def(),
                                                          ["outputs"])

        with tf.gfile.GFile(args.output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
