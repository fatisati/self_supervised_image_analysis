import tensorflow as tf


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.per_image_standardization(img)
    # img = tf.divide(img, 255)
    return img


def read_tf_image(path):
    img = tf.io.read_file(path)
    img = decode_img(img)
    return img


def tf_ds_from_arr(arr):
    return tf.data.Dataset.from_tensor_slices(arr)
