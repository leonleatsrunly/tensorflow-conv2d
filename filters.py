import numpy as np
import tensorflow as tf

f = np.zeros([3, 3, 3, 3])
f[1, 1, :, :] = 0.25
f[0, 1, :, :] = 0.125
f[1, 0, :, :] = 0.125
f[2, 1, :, :] = 0.125
f[1, 2, :, :] = 0.125
f[0, 0, :, :] = 0.0625
f[0, 2, :, :] = 0.0625
f[2, 0, :, :] = 0.0625
f[2, 2, :, :] = 0.0625

BLUR_FILTER_RGB = tf.constant(f, dtype=tf.float32)

f = np.zeros([3, 3, 1, 1])
f[1, 1, :, :] = 1.0
f[0, 1, :, :] = 1.0
f[1, 0, :, :] = 1.0
f[2, 1, :, :] = 1.0
f[1, 2, :, :] = 1.0
f[0, 0, :, :] = 1.0
f[0, 2, :, :] = 1.0
f[2, 0, :, :] = 1.0
f[2, 2, :, :] = 1.0
BLUR_FILTER = tf.constant(f, dtype=tf.float32)

f = np.zeros([3, 3, 3, 3])
f[1, 1, :, :] = 5
f[0, 1, :, :] = -1
f[1, 0, :, :] = -1
f[2, 1, :, :] = -1
f[1, 2, :, :] = -1

SHARPEN_FILTER_RGB = tf.constant(f, dtype=tf.float32)

f = np.zeros([3, 3, 1, 1])
f[1, 1, :, :] = 5
f[0, 1, :, :] = -1
f[1, 0, :, :] = -1
f[2, 1, :, :] = -1
f[1, 2, :, :] = -1

SHARPEN_FILTER = tf.constant(f, dtype=tf.float32)

EDGE_FILTER_RGB = tf.constant([
			[[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]],
            [[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
			[[ 8., 0., 0.], [ 0., 8., 0.], [ 0., 0., 8.]],
			[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]],
			[[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
			[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
			[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]]
])

f = np.zeros([3, 3, 1, 1])
f[0, 1, :, :] = -1
f[1, 0, :, :] = -1
f[1, 2, :, :] = -1
f[2, 1, :, :] = -1
f[1, 1, :, :] = 4

EDGE_FILTER = tf.constant(f, dtype=tf.float32)

f = np.zeros([3, 3, 3, 3])
f[0, :, :, :] = 1
f[0, 1, :, :] = 2 
f[2, :, :, :] = -1
f[2, 1, :, :] = -2

TOP_SOBEL_RGB = tf.constant(f, dtype=tf.float32)

f = np.zeros([3, 3, 1, 1])
f[0, :, :, :] = 1
f[0, 1, :, :] = 2 
f[2, :, :, :] = -1
f[2, 1, :, :] = -2

TOP_SOBEL = tf.constant(f, dtype=tf.float32)

f = np.zeros([3, 3, 3, 3])
f[0, 0, :, :] = -2
f[0, 1, :, :] = -1
f[1, 0, :, :] = -1
f[1, 1, :, :] = 1
f[1, 2, :, :] = 1
f[2, 1, :, :] = 1
f[2, 2, :, :] = 2

EMBOSS_FILTER_RGB = tf.constant(f, dtype=tf.float32)

f = np.zeros([3, 3, 1, 1])
f[0, 0, :, :] = -2
f[0, 1, :, :] = -1
f[1, 0, :, :] = -1
f[1, 1, :, :] = 1
f[1, 2, :, :] = 1
f[2, 1, :, :] = 1
f[2, 2, :, :] = 2
EMBOSS_FILTER = tf.constant(f, dtype=tf.float32)


f = np.zeros([3, 3, 1, 1])
f[0, :, :, :] = 5
f[1, :, :, :] = -3
f[2, :, :, :] = -3
f[1, 1, :, :] = 0

LAPLACE_FILTER = tf.constant(f, dtype=tf.float32)

filter_list = [BLUR_FILTER, SHARPEN_FILTER, EDGE_FILTER,
				TOP_SOBEL, EMBOSS_FILTER, LAPLACE_FILTER]

