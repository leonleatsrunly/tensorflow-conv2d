import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import argparse
import kernels
import leon_data


def init_session():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

def decode_image(a_File):
    the_producer = tf.train.string_input_producer([a_File])
    the_reader = tf.WholeFileReader()
    _,the_jpeg = the_reader.read(the_producer)
    the_image = tf.image.decode_jpeg(the_jpeg,channels=3)
    the_image = tf.cast(the_image, dtype=tf.float32)
    return the_image

def run_image(image):
    with tf.Session() as sess:
        the_Coordinator = tf.train.Coordinator()
        the_runners = tf.train.start_queue_runners(coord=the_Coordinator)
        image = sess.run(image)
        the_Coordinator.request_stop()
        the_Coordinator.join(the_runners)
    return image

def conv2d_image(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.expand_dims(image, axis=0)
    images = []
    for i, filter in enumerate(kernels.filter_list):
        conv2d = tf.nn.conv2d(image, filter=filter, strides=[1,3,3,1], padding='SAME')[0]  #2 <<< [0]
        images.append(conv2d)
    return images

def show_image(images):
    the_GridSpec = gs.GridSpec(1, len(images))
    for i, image in enumerate(images):
        image = image.reshape(image.shape[0], image.shape[1])
        plt.subplot(the_GridSpec[0,i])
        plt.imshow(image)
        plt.axis('off')
    plt.show()

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='input')
    parser.add_argument('--output', '-o', dest='output')
    args = parser.parse_args()
    a_file = leon_data.grpa_file if args.input == None else args.input
    return a_file

def main():
    init_session()
    file = arg_parse()
    image = decode_image(file)
    images = conv2d_image(image)
    images = run_image(images)
    show_image(images)
    pass

if __name__ == '__main__':
    main()