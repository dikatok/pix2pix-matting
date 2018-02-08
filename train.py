import tensorflow as tf
from tensorflow.python.lib.io import file_io

from imports.data_utils import create_one_shot_iterator, augment_dataset, create_initializable_iterator
from imports.losses import discriminator_loss, generator_loss
from imports.models import generator, discriminator
from imports.metrics import iou
import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', nargs='+', required=False, default="train-00001-of-00001")
    parser.add_argument('--test_files', nargs='+', required=False, default="val-00001-of-00001")
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=8, type=int)
    parser.add_argument('--resume', default=None, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    train_files = args.train_files
    test_files = args.test_files
    
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size

    num_train_samples = sum(1 for f in file_io.get_matching_files(train_files) 
                            for n in tf.python_io.tf_record_iterator(f))
    num_test_samples = sum(1 for f in file_io.get_matching_files(test_files) 
                           for n in tf.python_io.tf_record_iterator(f))
    
    num_epochs = 200 * 2

    train_iterator = create_one_shot_iterator(train_files, train_batch_size, num_epoch=num_epochs)
    test_iterator = create_initializable_iterator(test_files, batch_size=num_test_samples)

    real_A, real_B = train_iterator.get_next()
    real_A, real_B = augment_dataset(real_A, real_B, size=[256, 256])
    fake_B = generator(real_A)
    real_AB = tf.concat([real_A, real_B], axis=-1)
    fake_AB = tf.concat([real_A, fake_B], axis=-1)

    test_A, test_B = test_iterator.get_next()
    test_A, test_B = augment_dataset(test_A, test_B, size=[256, 256])
    fake_test_B = generator(test_A)

    d, d_logits = discriminator(real_AB)
    d_, d_logits_ = discriminator(fake_AB)

    d_loss = discriminator_loss(d_logits, d_logits_, tf.ones_like(d), tf.zeros_like(d_))
    g_loss = generator_loss(d_logits_, tf.ones_like(d_), real_B, fake_B)

    train_iou = iou(real_B, fake_B)
    test_iou = iou(test_B, fake_test_B)

    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator/")
    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator/")

    d_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(d_loss, var_list=discriminator_vars)
    g_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(g_loss, var_list=generator_vars)

    summary = tf.summary.FileWriter(logdir=args.log_dir)

    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)

    real_A_sum = tf.summary.image("real_A", real_A, max_outputs=2)
    real_B_sum = tf.summary.image("real_B", real_B * real_A, max_outputs=2)
    fake_B_sum = tf.summary.image("fake_B", fake_B * real_A, max_outputs=2)
    image_sum = tf.summary.merge([real_A_sum, real_B_sum, fake_B_sum])

    test_A_sum = tf.summary.image("test_A", test_A, max_outputs=2)
    test_B_sum = tf.summary.image("test_B", test_B * test_A, max_outputs=2)
    fake_test_B_sum = tf.summary.image("fake_test_B", fake_test_B * test_A, max_outputs=2)
    test_image_sum = tf.summary.merge([test_A_sum, test_B_sum, fake_test_B_sum])

    train_iou_sum = tf.summary.scalar("train_iou", train_iou)
    test_iou_sum = tf.summary.scalar("test_iou", test_iou)

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    resume = args.resume

    def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            session = session._sess
        return session


    with tf.train.MonitoredTrainingSession() as sess:
        it = 0
        if resume is not None and resume > 0:
            saver.restore(sess, os.path.join(args.ckpt_dir, "ckpt") + "-{it}".format(it=resume))
            it = resume + 1

        while not sess.should_stop():
            _, cur_d_loss_sum = sess.run([d_op, d_loss_sum])

            _, cur_g_loss_sum, cur_image_sum, cur_train_iou_sum = sess.run([g_op, g_loss_sum, image_sum,
                                                                            train_iou_sum])

            summary.add_summary(cur_d_loss_sum, it)
            summary.add_summary(cur_g_loss_sum, it)
            summary.add_summary(cur_train_iou_sum, it)

            if it % 100 == 0:
                summary.add_summary(cur_image_sum, it)

            if it % 200 == 0:
                sess.run(test_iterator.initializer)
                cur_test_image_sum, cur_test_iou_sum = sess.run([test_image_sum, test_iou_sum])
                summary.add_summary(cur_test_image_sum, it)
                summary.add_summary(cur_test_iou_sum, it)

            if it % 2000 == 0:
                ckpt_path = saver.save(get_session(sess), save_path=os.path.join(args.ckpt_dir, "ckpt"),
                                       write_meta_graph=False, global_step=it)
                print("Checkpoint saved as: {ckpt_path}".format(ckpt_path=ckpt_path))

            summary.flush()
            it += 1

        sess.run(test_iterator.initializer)
        cur_test_image_sum, cur_test_iou_sum = sess.run([test_image_sum, test_iou_sum])
        summary.add_summary(cur_test_image_sum, it)
        summary.add_summary(cur_test_iou_sum, it)
        summary.flush()

        ckpt_path = saver.save(get_session(sess), save_path=os.path.join(args.ckpt_dir, "ckpt"),
                               write_meta_graph=False, global_step=it)
        print("Checkpoint saved as: {ckpt_path}".format(ckpt_path=ckpt_path))
