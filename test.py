import tensorflow as tf
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_filename', type=str, default='model.pb')
    parser.add_argument('--output', type=str, default='out.jpg', required=False)
    parser.add_argument('image')
    return parser.parse_args()


def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="graph")

    return graph


if __name__ == '__main__':
    args = args_parser()

    graph = load_graph(args.graph_filename)

    inputs = graph.get_tensor_by_name('graph/inputs:0')
    outputs = graph.get_tensor_by_name('graph/outputs:0') * inputs

    with tf.Session(graph=graph) as sess:
        x = tf.keras.preprocessing.image.load_img(args.image)
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = tf.expand_dims(x, axis=0)
        x = tf.image.resize_images(x, [256, 256])
        x = sess.run(x)

        out = sess.run(outputs, feed_dict={inputs: x})

    out = tf.keras.preprocessing.image.array_to_img(out[0])
    out.save(args.output)

